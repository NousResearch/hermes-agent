"""Build Environment used for isolation during sdist building"""

from __future__ import annotations

import logging
import os
import pathlib
import site
import sys
import textwrap
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager as ContextManager
from contextlib import nullcontext
from io import StringIO
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, TypedDict

from pip._vendor.packaging.version import Version

from pip import __file__ as pip_location
from pip._internal.cli.spinners import open_rich_spinner, open_spinner
from pip._internal.exceptions import (
    BuildDependencyInstallError,
    DiagnosticPipError,
    InstallWheelBuildError,
    PipError,
)
from pip._internal.locations import get_platlib, get_purelib, get_scheme
from pip._internal.metadata import get_default_environment, get_environment
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.logging import VERBOSE, capture_logging
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds

if TYPE_CHECKING:
    from pip._internal.cache import WheelCache
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.operations.build.build_tracker import BuildTracker
    from pip._internal.req.req_install import InstallRequirement
    from pip._internal.resolution.base import BaseResolver

    class ExtraEnviron(TypedDict, total=False):
        extra_environ: dict[str, str]


logger = logging.getLogger(__name__)


def _dedup(a: str, b: str) -> tuple[str] | tuple[str, str]:
    return (a, b) if a != b else (a,)


class _Prefix:
    def __init__(self, path: str) -> None:
        self.path = path
        self.setup = False
        scheme = get_scheme("", prefix=path)
        self.bin_dir = scheme.scripts
        self.lib_dirs = _dedup(scheme.purelib, scheme.platlib)


def get_runnable_pip() -> str:
    """Get a file to pass to a Python executable, to run the currently-running pip.

    This is used to run a pip subprocess, for installing requirements into the build
    environment.
    """
    source = pathlib.Path(pip_location).resolve().parent

    if not source.is_dir():
        # This would happen if someone is using pip from inside a zip file. In that
        # case, we can use that directly.
        return str(source)

    return os.fsdecode(source / "__pip-runner__.py")


def _get_system_sitepackages() -> set[str]:
    """Get system site packages

    Usually from site.getsitepackages,
    but fallback on `get_purelib()/get_platlib()` if unavailable
    (e.g. in a virtualenv created by virtualenv<20)

    Returns normalized set of strings.
    """
    if hasattr(site, "getsitepackages"):
        system_sites = site.getsitepackages()
    else:
        # virtualenv < 20 overwrites site.py without getsitepackages
        # fallback on get_purelib/get_platlib.
        # this is known to miss things, but shouldn't in the cases
        # where getsitepackages() has been removed (inside a virtualenv)
        system_sites = [get_purelib(), get_platlib()]
    return {os.path.normcase(path) for path in system_sites}


class BuildEnvironmentInstaller(Protocol):
    """
    Interface for installing build dependencies into an isolated build
    environment.
    """

    def install(
        self,
        requirements: Iterable[str],
        prefix: _Prefix,
        *,
        kind: str,
        for_req: InstallRequirement | None,
    ) -> None: ...


class SubprocessBuildEnvironmentInstaller:
    """
    Install build dependencies by calling pip in a subprocess.
    """

    def __init__(
        self,
        finder: PackageFinder,
        build_constraints: list[str] | None = None,
        build_constraint_feature_enabled: bool = False,
    ) -> None:
        self.finder = finder
        self._build_constraints = build_constraints or []
        self._build_constraint_feature_enabled = build_constraint_feature_enabled

    def _deprecation_constraint_check(self) -> None:
        """
        Check for deprecation warning: PIP_CONSTRAINT affecting build environments.

        This warns when build-constraint feature is NOT enabled and PIP_CONSTRAINT
        is not empty.
        """
        if self._build_constraint_feature_enabled or self._build_constraints:
            return

        pip_constraint = os.environ.get("PIP_CONSTRAINT")
        if not pip_constraint or not pip_constraint.strip():
            return

        deprecated(
            reason=(
                "Setting PIP_CONSTRAINT will not affect "
                "build constraints in the future,"
            ),
            replacement=(
                "to specify build constraints using --build-constraint or "
                "PIP_BUILD_CONSTRAINT. To disable this warning without "
                "any build constraints set --use-feature=build-constraint or "
                'PIP_USE_FEATURE="build-constraint"'
            ),
            gone_in="26.2",
            issue=None,
        )

    def install(
        self,
        requirements: Iterable[str],
        prefix: _Prefix,
        *,
        kind: str,
        for_req: InstallRequirement | None,
    ) -> None:
        self._deprecation_constraint_check()

        finder = self.finder
        args: list[str] = [
            sys.executable,
            get_runnable_pip(),
            "install",
            "--ignore-installed",
            "--no-user",
            "--prefix",
            prefix.path,
            "--no-warn-script-location",
            "--disable-pip-version-check",
            # As the build environment is ephemeral, it's wasteful to
            # pre-compile everything, especially as not every Python
            # module will be used/compiled in most cases.
            "--no-compile",
            # The prefix specified two lines above, thus
            # target from config file or env var should be ignored
            "--target",
            "",
        ]
        if logger.getEffectiveLevel() <= logging.DEBUG:
            args.append("-vv")
        elif logger.getEffectiveLevel() <= VERBOSE:
            args.append("-v")
        for format_control in ("no_binary", "only_binary"):
            formats = getattr(finder.format_control, format_control)
            args.extend(
                (
                    "--" + format_control.replace("_", "-"),
                    ",".join(sorted(formats or {":none:"})),
                )
            )

        if finder.release_control is not None:
            # Use ordered args to preserve the user's original command-line order
            # This is important because later flags can override earlier ones
            for attr_name, value in finder.release_control.get_ordered_args():
                args.extend(("--" + attr_name.replace("_", "-"), value))

        index_urls = finder.index_urls
        if index_urls:
            args.extend(["-i", index_urls[0]])
            for extra_index in index_urls[1:]:
                args.extend(["--extra-index-url", extra_index])
        else:
            args.append("--no-index")
        for link in finder.find_links:
            args.extend(["--find-links", link])

        if finder.proxy:
            args.extend(["--proxy", finder.proxy])
        for host in finder.trusted_hosts:
            args.extend(["--trusted-host", host])
        if finder.custom_cert:
            args.extend(["--cert", finder.custom_cert])
        if finder.client_cert:
            args.extend(["--client-cert", finder.client_cert])
        if finder.prefer_binary:
            args.append("--prefer-binary")

        # Handle build constraints
        if self._build_constraint_feature_enabled:
            args.extend(["--use-feature", "build-constraint"])

        if self._build_constraints:
            # Build constraints must be passed as both constraints
            # and build constraints, so that nested builds receive
            # build constraints
            for constraint_file in self._build_constraints:
                args.extend(["--constraint", constraint_file])
                args.extend(["--build-constraint", constraint_file])

        extra_environ: ExtraEnviron = {}
        if self._build_constraint_feature_enabled and not self._build_constraints:
            # If there are no build constraints but the build constraints
            # feature is enabled then we must ignore regular constraints
            # in the isolated build environment
            extra_environ = {"extra_environ": {"_PIP_IN_BUILD_IGNORE_CONSTRAINTS": "1"}}

        if finder.uploaded_prior_to:
            args.extend(["--uploaded-prior-to", finder.uploaded_prior_to.isoformat()])
        args.append("--")
        args.extend(requirements)

        identify_requirement = (
            f" for {for_req.name}" if for_req and for_req.name else ""
        )
        with open_spinner(f"Installing {kind}") as spinner:
            call_subprocess(
                args,
                command_desc=f"installing {kind}{identify_requirement}",
                spinner=spinner,
                **extra_environ,
            )


class InprocessBuildEnvironmentInstaller:
    """
    Build dependency installer that runs in the same pip process.

    This contains a stripped down version of the install command with
    only the logic necessary for installing build dependencies. The
    finder, session, build tracker, and wheel cache are reused, but new
    instances of everything else are created as needed.

    Options are inherited from the parent install command unless
    they don't make sense for build dependencies (in which case, they
    are hard-coded, see comments below).
    """

    def __init__(
        self,
        *,
        finder: PackageFinder,
        build_tracker: BuildTracker,
        wheel_cache: WheelCache,
        build_constraints: Sequence[InstallRequirement] = (),
        verbosity: int = 0,
    ) -> None:
        from pip._internal.operations.prepare import RequirementPreparer

        self._finder = finder
        self._build_constraints = build_constraints
        self._wheel_cache = wheel_cache
        self._level = 0

        build_dir = TempDirectory(kind="build-env-install", globally_managed=True)
        self._preparer = RequirementPreparer(
            build_isolation_installer=self,
            # Inherited options or state.
            finder=finder,
            session=finder._link_collector.session,
            build_dir=build_dir.path,
            build_tracker=build_tracker,
            verbosity=verbosity,
            # This is irrelevant as it only applies to editable requirements.
            src_dir="",
            # Hard-coded options (that should NOT be inherited).
            download_dir=None,
            build_isolation=True,
            check_build_deps=False,
            progress_bar="off",
            # TODO: hash-checking should be extended to build deps, but that is
            # deferred for later as it'd be a breaking change.
            require_hashes=False,
            use_user_site=False,
            lazy_wheel=False,
            legacy_resolver=False,
        )

    def install(
        self,
        requirements: Iterable[str],
        prefix: _Prefix,
        *,
        kind: str,
        for_req: InstallRequirement | None,
    ) -> None:
        """Install entrypoint. Manages output capturing and error handling."""
        capture_logs = not logger.isEnabledFor(VERBOSE) and self._level == 0
        if capture_logs:
            # Hide the logs from the installation of build dependencies.
            # They will be shown only if an error occurs.
            capture_ctx: ContextManager[StringIO] = capture_logging()
            spinner: ContextManager[None] = open_rich_spinner(f"Installing {kind}")
        else:
            # Otherwise, pass-through all logs (with a header).
            capture_ctx, spinner = nullcontext(StringIO()), nullcontext()
            logger.info("Installing %s ...", kind)

        try:
            self._level += 1
            with spinner, capture_ctx as stream:
                self._install_impl(requirements, prefix)

        except DiagnosticPipError as exc:
            # Format similar to a nested subprocess error, where the
            # causing error is shown first, followed by the build error.
            logger.info(textwrap.dedent(stream.getvalue()))
            logger.error("%s", exc, extra={"rich": True})
            logger.info("")
            raise BuildDependencyInstallError(
                for_req, requirements, cause=exc, log_lines=None
            )

        except Exception as exc:
            logs: list[str] | None = textwrap.dedent(stream.getvalue()).splitlines()
            if not capture_logs:
                # If logs aren't being captured, then display the error inline
                # with the rest of the logs.
                logs = None
                if isinstance(exc, PipError):
                    logger.error("%s", exc)
                else:
                    logger.exception("pip crashed unexpectedly")
            raise BuildDependencyInstallError(
                for_req, requirements, cause=exc, log_lines=logs
            )

        finally:
            self._level -= 1

    def _install_impl(self, requirements: Iterable[str], prefix: _Prefix) -> None:
        """Core build dependency install logic."""
        from pip._internal.commands.install import installed_packages_summary
        from pip._internal.req import install_given_reqs
        from pip._internal.req.constructors import install_req_from_line
        from pip._internal.wheel_builder import build

        ireqs = [install_req_from_line(req, user_supplied=True) for req in requirements]
        ireqs.extend(self._build_constraints)

        resolver = self._make_resolver()
        resolved_set = resolver.resolve(ireqs, check_supported_wheels=True)
        self._preparer.prepare_linked_requirements_more(
            resolved_set.requirements.values()
        )

        reqs_to_build = [
            r for r in resolved_set.requirements_to_install if not r.is_wheel
        ]
        _, build_failures = build(reqs_to_build, self._wheel_cache, verify=True)
        if build_failures:
            raise InstallWheelBuildError(build_failures)

        installed = install_given_reqs(
            resolver.get_installation_order(resolved_set),
            prefix=prefix.path,
            # Hard-coded options (that should NOT be inherited).
            root=None,
            home=None,
            warn_script_location=False,
            use_user_site=False,
            # As the build environment is ephemeral, it's wasteful to
            # pre-compile everything since not all modules will be used.
            pycompile=False,
            progress_bar="off",
        )

        env = get_environment(list(prefix.lib_dirs))
        if summary := installed_packages_summary(installed, env):
            logger.info(summary)

    def _make_resolver(self) -> BaseResolver:
        """Create a new resolver for one time use."""
        # Legacy installer never used the legacy resolver so create a
        # resolvelib resolver directly. Yuck.
        from pip._internal.req.constructors import install_req_from_req_string
        from pip._internal.resolution.resolvelib.resolver import Resolver

        return Resolver(
            make_install_req=install_req_from_req_string,
            # Inherited state.
            preparer=self._preparer,
            finder=self._finder,
            wheel_cache=self._wheel_cache,
            # Hard-coded options (that should NOT be inherited).
            ignore_requires_python=False,
            use_user_site=False,
            ignore_dependencies=False,
            ignore_installed=True,
            force_reinstall=False,
            upgrade_strategy="to-satisfy-only",
            py_version_info=None,
        )


class BuildEnvironment:
    """Creates and manages an isolated environment to install build deps"""

    def __init__(self, installer: BuildEnvironmentInstaller) -> None:
        self.installer = installer
        temp_dir = TempDirectory(kind=tempdir_kinds.BUILD_ENV, globally_managed=True)

        self._prefixes = OrderedDict(
            (name, _Prefix(os.path.join(temp_dir.path, name)))
            for name in ("normal", "overlay")
        )

        self._bin_dirs: list[str] = []
        self._lib_dirs: list[str] = []
        for prefix in reversed(list(self._prefixes.values())):
            self._bin_dirs.append(prefix.bin_dir)
            self._lib_dirs.extend(prefix.lib_dirs)

        # Customize site to:
        # - ensure .pth files are honored
        # - prevent access to system site packages
        system_sites = _get_system_sitepackages()

        self._site_dir = os.path.join(temp_dir.path, "site")
        if not os.path.exists(self._site_dir):
            os.mkdir(self._site_dir)
        with open(
            os.path.join(self._site_dir, "sitecustomize.py"), "w", encoding="utf-8"
        ) as fp:
            fp.write(
                textwrap.dedent(
                    """
                import os, site, sys

                # First, drop system-sites related paths.
                original_sys_path = sys.path[:]
                known_paths = set()
                for path in {system_sites!r}:
                    site.addsitedir(path, known_paths=known_paths)
                system_paths = set(
                    os.path.normcase(path)
                    for path in sys.path[len(original_sys_path):]
                )
                original_sys_path = [
                    path for path in original_sys_path
                    if os.path.normcase(path) not in system_paths
                ]
                sys.path = original_sys_path

                # Second, add lib directories.
                # ensuring .pth file are processed.
                for path in {lib_dirs!r}:
                    assert not path in sys.path
                    site.addsitedir(path)
                """
                ).format(system_sites=system_sites, lib_dirs=self._lib_dirs)
            )

    def __enter__(self) -> None:
        self._save_env = {
            name: os.environ.get(name, None)
            for name in ("PATH", "PYTHONNOUSERSITE", "PYTHONPATH")
        }

        path = self._bin_dirs[:]
        old_path = self._save_env["PATH"]
        if old_path:
            path.extend(old_path.split(os.pathsep))

        pythonpath = [self._site_dir]

        os.environ.update(
            {
                "PATH": os.pathsep.join(path),
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(pythonpath),
            }
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for varname, old_value in self._save_env.items():
            if old_value is None:
                os.environ.pop(varname, None)
            else:
                os.environ[varname] = old_value

    def check_requirements(
        self, reqs: Iterable[str]
    ) -> tuple[set[tuple[str, str]], set[str]]:
        """Return 2 sets:
        - conflicting requirements: set of (installed, wanted) reqs tuples
        - missing requirements: set of reqs
        """
        missing = set()
        conflicting = set()
        if reqs:
            env = (
                get_environment(self._lib_dirs)
                if hasattr(self, "_lib_dirs")
                else get_default_environment()
            )
            for req_str in reqs:
                req = get_requirement(req_str)
                # We're explicitly evaluating with an empty extra value, since build
                # environments are not provided any mechanism to select specific extras.
                if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                    continue
                dist = env.get_distribution(req.name)
                if not dist:
                    missing.add(req_str)
                    continue
                if isinstance(dist.version, Version):
                    installed_req_str = f"{req.name}=={dist.version}"
                else:
                    installed_req_str = f"{req.name}==={dist.version}"
                if not req.specifier.contains(dist.version, prereleases=True):
                    conflicting.add((installed_req_str, req_str))
                # FIXME: Consider direct URL?
        return conflicting, missing

    def install_requirements(
        self,
        requirements: Iterable[str],
        prefix_as_string: str,
        *,
        kind: str,
        for_req: InstallRequirement | None = None,
    ) -> None:
        prefix = self._prefixes[prefix_as_string]
        assert not prefix.setup
        prefix.setup = True
        if not requirements:
            return
        self.installer.install(requirements, prefix, kind=kind, for_req=for_req)


class NoOpBuildEnvironment(BuildEnvironment):
    """A no-op drop-in replacement for BuildEnvironment"""

    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def install_requirements(
        self,
        requirements: Iterable[str],
        prefix_as_string: str,
        *,
        kind: str,
        for_req: InstallRequirement | None = None,
    ) -> None:
        raise NotImplementedError()
