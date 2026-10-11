"""``hermes plugins install``: dependency/env consent, the atomic clone-scan-publish installer core, and
the dashboard/TUI non-interactive install.

Sibling of :mod:`hermes_cli.plugins_cmd` (the facade re-exports the names other modules use and is
imported late here, never at module level).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.cli_output import line_input
from hermes_cli.plugin_install_phase import InstallPhase

logger = logging.getLogger(__name__)

# Answers to the dependency questions, reused while one plugin is installed into several profile
# homes that share one environment (memory-provider migration on ``hermes update``, #125794).
_shared_answers: ContextVar[Optional[dict]] = ContextVar("plugin_dependency_answers", default=None)


@contextmanager
def shared_dependency_answers():
    """Inside this block each dependency question is asked once; repeats reuse the first answer."""
    token = _shared_answers.set({})
    try:
        yield
    finally:
        _shared_answers.reset(token)


def _ask_yes_no(key: tuple, prompt: str, console) -> bool:
    answers = _shared_answers.get()
    if answers is not None and key in answers:
        console.print(f"  [dim]Reusing your answer for the first profile: {'yes' if answers[key] else 'no'}[/dim]")
        return answers[key]
    try:
        accepted = input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt):
        accepted = False
    if answers is not None:
        answers[key] = accepted
    return accepted


def _pc():
    """The facade, read at call time: tests patch ``plugins_cmd.<name>`` and sibling calls must see it."""
    from hermes_cli import plugins_cmd
    return plugins_cmd


class _ConsentRefusal(str):
    """A refusal reason (user-facing text) carrying its closed extension-install ``failure_class``,
    so publication classifies the refusal without matching the copy."""

    def __new__(cls, text: str, failure_class: str) -> _ConsentRefusal:
        refusal = super().__new__(cls, text)
        refusal.failure_class = failure_class
        return refusal


def _install_plugin_python_deps(
    manifest: dict, target: Path, console, *, assume_yes: bool = False
) -> tuple[bool, Optional[str]]:
    """Consent gate for plugin python deps (settled 2026-09-02; C13 rework).

    Node sidecar: y/n prompt → ``npm ci`` into the plugin's OWN
    node_modules (separate question, failure never blocks enable).
    Python deps: NO resolution here — the resolve runs inside the ONE
    admission transaction when the enable commits
    (:func:`_admit_and_save_plugin_sets`), so the environment and the
    config always change together or not at all. Returns (consented,
    reason): consented=True when the user accepted (or no prompt was
    needed); a decline/skip returns False and NOTHING is installed.
    Never raises — the caller keeps the plugin installed-but-disabled.
    *assume_yes* is ``--yes-deps``: the user's own non-interactive answer
    to the Python question, so a headless install is not refused (#122134).
    """
    from pm.plugin_declarations import read_python_declaration

    try:
        declaration = read_python_declaration(target)
        deps = declaration.install_requirements
    except Exception as exc:
        return False, _ConsentRefusal(f"invalid Python dependency declaration: {exc}", "manifest_invalid")
    has_python = declaration.is_member
    has_package_json = (target / "package.json").is_file()
    if not has_python and not has_package_json:
        return True, None  # no declared deps at all

    # Node sidecar (package.json): the npm ci executor — separate consent
    # question, same try-then-enable posture. Failure never blocks the
    # python path below.
    node_reason = None
    if has_package_json:
        console.print(f"\n[bold]{manifest.get('name', 'this plugin')}[/bold] declares Node dependencies (package.json).")
        node_ok = sys.stdin.isatty() and sys.stdout.isatty() and _ask_yes_no(
            ("node", manifest.get("name")), "  Install them into the plugin's own node_modules now? [y/N]: ", console)
        if node_ok:
            from pm.workspace import install_node_sidecar

            node_reason = install_node_sidecar(target, explicit=True)
            if node_reason:
                console.print(f"[yellow]⚠[/yellow] Node deps: {node_reason}")
        else:
            console.print("[dim]Skipped Node deps — run `hermes plugins install` again to retry.[/dim]\n")

    if not has_python:
        return True, None
    return _consent_python_deps(manifest.get("name", "this plugin"), deps, console, assume_yes=assume_yes)


def _consent_python_deps(
    plugin_name: str, deps: tuple[str, ...], console, *, assume_yes: bool = False
) -> tuple[bool, Optional[str]]:
    """The y/N gate for Python deps entering the shared environment — install,
    reinstall AND an update that declares new ones all pass through here.
    Returns (consented, reason); never raises. *assume_yes* (``--yes-deps``)
    answers the question on the user's behalf ONLY because the user passed the
    flag; the non-interactive default below stays a refusal."""
    console.print(
        f"\n[bold]{plugin_name}[/bold] declares Python dependencies:"
    )
    if deps:
        for dep in deps:
            console.print(f"  - {dep}")
    else:
        console.print("  - (declared in its pyproject.toml)")

    # An explicit --yes-deps is the user's own answer, TTY or not.
    if assume_yes:
        console.print(
            "[dim]Consent given up front — preparing them without prompting.[/dim]\n"
        )
        return True, None

    # A decline or non-interactive invocation leaves the new plugin disabled.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print(
            "[dim]Non-interactive install — skipping dependency install. "
            "Run `hermes plugins enable` when ready to prepare them.[/dim]\n"
        )
        return False, _ConsentRefusal("dependency install skipped (non-interactive)", "non_interactive")
    if not _ask_yes_no(("python", plugin_name, deps), "  Prepare these with Hermes through PM now? [y/N]: ", console):
        console.print(
            "[dim]Skipped — run `hermes plugins enable` when ready "
            "to prepare them.[/dim]\n"
        )
        return False, _ConsentRefusal("dependency install declined", "deps_declined")

    # Consent only — the python-deps resolution itself runs inside the ONE
    # admission transaction at enable-commit time (C13): env + config move
    # together or not at all.
    return True, None


def _python_dependency_summary(target: Path, warnings: list[str]) -> list[str]:
    """Read dashboard dependency details; publication and admission own installation."""
    from pm.plugin_declarations import read_python_declaration

    try:
        return list(read_python_declaration(target).install_requirements)
    except Exception as exc:
        warnings.append(f"Could not read Python dependencies: {exc}")
        return []


def _prompt_plugin_env_vars(manifest: dict, console) -> None:
    """Prompt for unset ``requires_env`` variables and save the answers to the user's ``.env``."""
    missing = _pc()._missing_env_specs(manifest)
    if not missing:
        return
    from hermes_cli.config import save_env_value
    from hermes_constants import display_hermes_home
    plugin_name = manifest.get("name", "this plugin")
    console.print(f"\n[bold]{plugin_name}[/bold] requires the following environment variables:\n")
    for spec in missing:
        name = spec["name"]
        desc = spec.get("description", "")
        url = spec.get("url", "")
        console.print(f"  {name}" + (f" — {desc}" if desc else ""))
        if url:
            console.print(f"  [dim]Get yours at: {url}[/dim]")
        try:
            value = (_pc().masked_secret_prompt if spec.get("secret", False) else line_input)(f"  {name}: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[dim]  Skipped (you can set these later in {display_hermes_home()}/.env)[/dim]")
            return

        if value:
            save_env_value(name, value)
            os.environ[name] = value
            console.print(f"  [green]✓[/green] Saved to {display_hermes_home()}/.env")
        else:
            console.print(f"  [dim]  Skipped (set {name} in {display_hermes_home()}/.env later)[/dim]")

    console.print()


def _display_after_install(plugin_dir: Path, identifier: str) -> None:
    """Show after-install.md if it exists, otherwise a default message."""
    from rich.markdown import Markdown
    from rich.panel import Panel
    console = _pc()._console()
    after_install = plugin_dir / "after-install.md"
    if after_install.exists():
        body, title = Markdown(after_install.read_text(encoding="utf-8-sig")), None
    else:
        body = f"[green bold]Plugin installed:[/] {identifier}\n[dim]Location:[/] {plugin_dir}"
        title = "✓ Installed"
    console.print()
    console.print(Panel(body, border_style="green", title=title, expand=False))
    console.print()


def _check_manifest_version(manifest: dict, plugin_name: str) -> None:
    """Reject manifests declaring a newer ``manifest_version`` than this installer supports."""
    from pm.plugin_declarations import manifest_version_error

    reason = manifest_version_error(manifest, plugin_name)
    if reason:
        from hermes_cli.config import recommended_update_command
        raise _pc().PluginOperationError(f"{reason} Run {recommended_update_command()} to update Hermes.",
                                         failure_class="incompatible")


def _read_manifest_for_install(plugin_dir: Path) -> dict:
    """A candidate's unreadable or malformed manifest must stop publication."""
    native = _pc()._native_manifest_file(plugin_dir)
    if native is not None:
        try:
            manifest = _pc()._load_yaml_manifest(native)
        except Exception as exc:
            raise _pc().PluginOperationError(f"Could not read plugin manifest {native}: {exc}",
                                             failure_class="manifest_invalid") from exc
        if not isinstance(manifest, dict):
            raise _pc().PluginOperationError(f"Plugin manifest must be a mapping: {native}",
                                             failure_class="manifest_invalid")
        return manifest
    if not _pc()._has_portable_manifest(plugin_dir):
        return {}
    try:
        from hermes_cli.agent_plugins import read_agent_plugin_manifest
        manifest, diagnostics = read_agent_plugin_manifest(plugin_dir)
    except Exception as exc:
        raise _pc().PluginOperationError(f"Portable plugin manifest validation failed: {exc}",
                                         failure_class="manifest_invalid") from exc
    for diagnostic in diagnostics:
        logger.warning("Agent Plugin install: %s", diagnostic.message)
    return manifest


def _probe_readable(path: Path) -> None:
    """Raise ``OSError`` unless *path* can actually be listed (dir) or opened for reading (file)."""
    if path.is_dir():
        os.listdir(path)
    else:
        with open(path, "rb"):
            pass


def _ensure_tree_readable(root: Path, plugins_dir: Path) -> None:
    """Refuse to ship a tree Hermes cannot read back. A clone can land unreadable (Windows ACL
    inheritance -> WinError 5, a mode-000 file) and discovery would then skip the plugin forever
    (#111804); repair ``u+rX`` where the OS supports it, otherwise fail before anything moves."""
    paths = [root]
    for dirpath, dirnames, filenames in os.walk(root):
        paths.extend(Path(dirpath) / name for name in (*dirnames, *filenames))
    for path in paths:
        try:
            _probe_readable(path)
            continue
        except OSError:
            if os.name != "nt":  # chmod only toggles the read-only bit on Windows; ACLs need icacls
                try:
                    os.chmod(path, os.stat(path).st_mode | (0o500 if path.is_dir() else 0o400))
                except OSError:
                    pass
        try:
            _probe_readable(path)
        except OSError as exc:
            fix = (f'icacls "{plugins_dir}" /grant:r "%USERNAME%":(OI)(CI)F /T' if os.name == "nt"
                   else f"chmod -R u+rX {plugins_dir}")
            raise _pc().PluginOperationError(
                f"Installed file {path.relative_to(root)} is not readable ({exc.strerror or exc}); "
                f"nothing was installed. Fix permissions on {plugins_dir} (e.g. `{fix}`) and retry.",
                failure_class="permission",
            ) from exc


def _cleanup_staging(tmp: Path) -> None:
    """Remove the .install-* staging clone on every exit path. A plain ``TemporaryDirectory``
    cleanup is a bare Windows rmtree over freshly cloned files: a held handle or a surviving
    read-only bit fails it partway and the dir leaks into the plugins dir, warning at every
    boot. ``pm.filesystem`` retries held handles and clears read-only bits; a residual failure
    is a warning so it can never mask the install's own result."""
    from pm.filesystem import remove_tree, retry_held

    try:
        retry_held(lambda: remove_tree(tmp))
    except OSError as exc:
        logger.warning("Could not remove plugin staging directory %s: %s", tmp, exc)


def _refuse_unavailable_portable_plugin(plugin_name: str, tree: Path) -> None:
    if not (tree / "plugin.json").is_file():
        return
    from hermes_cli.agent_plugins import load_agent_plugin
    from hermes_platform.declaration import gpu_label
    from hermes_platform.resolver.availability import availability

    try:
        package = load_agent_plugin(tree, tree.parent / ".hermes-install-data")
    except ValueError as exc:
        raise _pc().PluginOperationError(f"Plugin '{plugin_name}' is unavailable: {exc}.",
                                         failure_class="manifest_invalid") from exc
    for server_name, server_decl in package.server_declarations.items():
        result = availability(server_decl.declaration)
        if result.offerable:
            continue
        found = f", found version {result.version}" if result.version else ""
        if result.state == "unsupported_gpu":
            found = f", needs {gpu_label(server_decl.declaration.required_gpu)}"
        raise _pc().PluginOperationError(
            f"Plugin '{plugin_name}' server '{server_name}' is unavailable: {result.state}{found}.",
            failure_class="incompatible",
        )


def _known_issue_warnings(entry) -> list[str]:
    """Catalog ``known_issues`` are informational (#124058): surface them as warnings, never a gate.

    The guard for the traps they describe belongs at the mode-selection seam (#122341 / #123771).
    """
    return [f"Known issue: {issue}" for issue in entry.known_issues]


def _install_plugin_core(
    identifier: str,
    *,
    force: bool,
    ref: Optional[str] = None,
    scan_decision_cb=None,
    reviewed_pin: Optional[str] = None,
    python_deps: bool = True,
    assume_deps_consent: bool = False,
    catalog: Optional[dict] = None,
    allow_removed: bool = False,
    before_swap=None,
) -> tuple[Path, dict, str]:
    """Clone a Git plugin and atomically record its source and exact revision.

    *reviewed_pin* is the curated-catalog sha for this install; the scan trusts the tree
    only when the checked-out revision is exactly that sha (an annotated-tag pin is peeled to
    its commit first — HEAD can only ever be the commit). *python_deps* False refuses active
    replacements; it never bypasses PM dependency admission. *assume_deps_consent* is the
    caller's ``--yes-deps`` answer carried into the publication consent gate, so a headless
    install of an active replacement is not refused non-interactively (#122134). *catalog*
    (``{"name", "repo", "tier", "pin"}``) is recorded on the install-metadata record with the
    checked-out sha — provenance lives OUTSIDE the plugin tree, so a repo cannot forge it;
    its ``pin`` is kept only when the checkout satisfies it (a ``--ref`` install is off-pin).
    *allow_removed* records that the user knowingly bypassed the kill list.
    *before_swap(manifest, tree)* runs on the manifest-checked clone BEFORE the security scan, so
    the single scan and portable-package check admit the merged tree (file-count/size limits
    included). It may return the relative paths it merged in, which a scan block then attributes,
    and may raise :class:`PluginOperationError` to abort (re-pin consent)."""
    requested_revision = _pc()._normalize_exact_revision(ref) if ref is not None else None
    try:
        git_url, subdir = _pc()._resolve_git_url(identifier)
    except ValueError as e:
        raise _pc().PluginOperationError(str(e), failure_class="invalid_source") from e

    plugins_dir = _pc()._plugins_dir()
    source = _pc()._canonical_source(git_url, subdir)
    old_metadata = _pc()._read_install_metadata()

    # Reinstalling the same pinned source retains its pin, even if its plugin
    # directory was manually removed. Moving a pin requires an explicit --ref.
    if requested_revision is None:
        pins = [e for e in old_metadata.values() if e.get("source") == source and e.get("pinned") is True]
        if len(pins) == 1 and isinstance(pins[0].get("revision"), str):
            requested_revision = _pc()._normalize_exact_revision(pins[0]["revision"])

    tmp = Path(tempfile.mkdtemp(prefix=".install-", dir=plugins_dir))
    try:
        tmp_clone = tmp / "plugin"
        installed_revision = _pc()._clone_plugin_repo(tmp_clone, git_url, requested_revision, subdir)
        git_exe = _pc()._resolve_git_executable()
        at_reviewed_pin = bool(reviewed_pin) and installed_revision == (
            _pc()._git_resolve_commit(tmp_clone, git_exe, reviewed_pin) if git_exe and reviewed_pin else reviewed_pin)
        tmp_target = _pc()._resolve_subdir_within(tmp_clone, subdir) if subdir else tmp_clone
        _ensure_tree_readable(tmp_target, plugins_dir)
        manifest = _read_manifest_for_install(tmp_target)
        plugin_name = manifest.get("name") or (
            subdir.rstrip("/").rsplit("/", 1)[-1] if subdir else _pc()._repo_name_from_url(git_url))
        link = plugins_dir / str(plugin_name)
        from pm.filesystem import is_junction
        if "/" not in str(plugin_name) and (link.is_symlink() or is_junction(link)):
            # A provider's own installer (e.g. `mnemosyne-hermes install`) links its package here; the
            # name is fine, the slot is taken. Say so instead of blaming the manifest.
            raise _pc().PluginOperationError(
                f"Plugin '{plugin_name}' is already installed outside the catalog: {link} is a link to "
                f"{os.path.realpath(link)}. Delete that link to install the catalog version.",
                failure_class="already_installed")
        try:
            target = _pc()._sanitize_plugin_name(plugin_name, plugins_dir)
        except ValueError as e:
            raise _pc().PluginOperationError(str(e), failure_class="manifest_invalid") from e
        _check_manifest_version(manifest, plugin_name)
        prior = old_metadata.get(plugin_name)
        # `install --force --ref` is the documented way to move a pin, so a reinstall of the same
        # source is an update: it replaces the plugin's code, never the user's state.
        tracked_edits: list[str] = []
        if (before_swap is None and force and target.is_dir() and not target.is_symlink()
                and isinstance(prior, dict) and prior.get("source") == source):
            from hermes_cli.plugins_cmd_catalog import _carry_user_files, _local_changes
            local, tracked_edits = _local_changes(target)

            def _carry(_manifest, tree):
                return _carry_user_files(target, tree, local)
            before_swap = _carry
        # A callback may merge user-owned state into the candidate tree; run it first so one scan
        # admits the final bytes.
        merged = before_swap(manifest, tmp_target) if before_swap is not None else None
        # Scan BEFORE anything is moved into place; raises PluginScanBlocked when blocked.
        _pc()._scan_merged_tree(tmp_target, identifier, merged, force=force, scan_decision_cb=scan_decision_cb,
                                reviewed_pin=at_reviewed_pin)
        if not python_deps:
            from pm.workspace import enabled_plugin_dirs

            if target.resolve() in enabled_plugin_dirs(installing=target):
                raise _pc().PluginOperationError(
                    "--no-deps cannot replace an active plugin. Retry without --no-deps; "
                    "PM must prepare its dependencies before publication.", failure_class="already_installed")
        _refuse_unavailable_portable_plugin(plugin_name, tmp_target)

        if target.exists() and not force:
            from pm.environments import cli_command_name
            from pm.paths import repo_root

            raise _pc().PluginOperationError(
                f"Plugin '{plugin_name}' already exists. Use force reinstall "
                f"or run `{cli_command_name(repo_root())} plugins update {plugin_name}`.",
                failure_class="already_installed")
        if target.exists() and requested_revision is None and isinstance(prior, dict) and prior.get("pinned") is True:
            raise _pc().PluginOperationError(
                f"Plugin '{plugin_name}' is pinned. Reinstall it with an explicit "
                "--ref <40-character commit SHA> to change its source or revision.", failure_class="already_installed")

        record: dict[str, object] = {
            "pinned": requested_revision is not None,
            "revision": installed_revision,
            "source": source,
        }
        # Saved update_url tag (settled: claims vs provenance): the
        # manifest's update_url is COPIED into the row at install. Check
        # time compares manifest vs tag; a mismatch is needs-fixing and
        # only `hermes plugins trust-update-url` moves the tag.
        if manifest.get("update_url"):
            from hermes_cli.plugins_updates import https_update_url
            try:
                record["update_url"] = https_update_url(manifest["update_url"])
            except ValueError as exc:
                raise _pc().PluginOperationError(f"Plugin '{plugin_name}' {exc}", failure_class="manifest_invalid") from exc
        if catalog:
            # ``sha`` = the commit checked out; ``pin`` = the reviewed catalog sha it satisfies (the
            # annotated-tag object for a tag pin), empty when installed off-pin via ``--ref``.
            record["catalog"] = {
                **catalog,
                "sha": installed_revision,
                "pin": reviewed_pin if at_reviewed_pin else "",
            }
            from hermes_cli.plugins_cmd_catalog import write_catalog_sidecar_record
            write_catalog_sidecar_record(tmp_target, catalog, installed_revision)
        if allow_removed:
            record["allow_removed"] = True
        new_metadata = {**old_metadata, plugin_name: record}
        from hermes_cli.plugins_transaction import publish_plugin

        if tracked_edits:
            from hermes_cli.plugins_cmd_catalog import _stash_local_files
            # Outside the plugins dir: the discovery scanners recurse into every subdirectory there.
            backup = plugins_dir.parent / "plugins-backup" / f"{target.name}-{str(prior.get('revision') or 'old')[:8]}"
            _stash_local_files(target, tracked_edits, backup)
            _pc()._console().print(
                f"[yellow]Local edits to {len(tracked_edits)} tracked file(s) were not carried over; "
                f"copies are under {backup} (re-apply by hand).[/yellow]")
        try:
            publish_plugin(tmp_target, target, old_metadata, new_metadata, require_consent=True,
                           assume_consent=assume_deps_consent)
        except Exception as exc:
            raise _pc().PluginOperationError(f"Plugin '{plugin_name}' was not published: {exc}",
                                             failure_class=_publish_failure_class(exc)) from exc

    finally:
        _cleanup_staging(tmp)

    if not _pc()._looks_like_plugin_dir(target):
        logger.warning("%s has no plugin.yaml / __init__.py; may not be a valid plugin", plugin_name)
    _pc()._copy_example_files(target, _pc()._console())
    installed_manifest = _pc()._read_manifest(target)
    return target, installed_manifest, installed_manifest.get("name") or target.name


# ---- iuf c1 ----
def _publish_failure_class(exc: BaseException) -> str:
    """Why publication failed, from the exception TYPE at this raise site. Publication is the PM
    dependency sync (plugins_transaction.publish_plugin -> pm.client.sync_venv): a consent refusal
    keeps the class plugins_transaction tagged, a filesystem error keeps its kind, and everything
    else PM raised (InstallError, ResolutionConflict, AdmissionRefused, a worker error) is a
    dependency failure."""
    from hermes_cli.observability.shared_metrics_fields import tagged_failure_class

    tagged = tagged_failure_class(exc) if isinstance(exc, _pc().PluginOperationError) else None
    if tagged and tagged != "other":
        return tagged
    if isinstance(exc, PermissionError):
        return "permission"
    if isinstance(exc, OSError) and not isinstance(exc, (ConnectionError, TimeoutError)):
        return "filesystem_error"
    return "deps_failed"
# ---- end iuf c1 ----


def recorded_install(install: Callable[[], tuple], *, catalog_name: Optional[str], identifier: str) -> tuple:
    """Run one plugin install attempt (a core ``(target, manifest, installed_name)`` call) and record
    it as a shared-metrics extension install: failed when it raises, success unless it replaced an
    already-installed plugin (a reinstall is not an install)."""
    from hermes_cli.observability.shared_metrics_events import record_extension_install

    source = "catalog" if catalog_name else ("local" if identifier.startswith("file://") else "url")
    before = set(_pc()._read_install_metadata())
    try:
        result = install()
    except Exception as exc:
        record_extension_install(kind="plugin", source=source, name=catalog_name, outcome="failed", error=exc)
        raise
    if result[2] not in before:
        record_extension_install(kind="plugin", source=source, name=catalog_name, outcome="success")
    return result


def cmd_install(
    identifier: str,
    force: bool = False,
    enable: Optional[bool] = None,
    ref: Optional[str] = None,
    allow_removed: bool = False,
    no_deps: bool = False,
    yes_deps: bool = False,
) -> None:
    """Install a plugin from the curated catalog (bare name), a Git URL, or owner/repo shorthand.

    A catalog hit installs the reviewed pinned SHA and records catalog membership in the shared install
    metadata. An explicit different ``--ref`` is a custom pin. URLs/shorthand are custom sources. Every
    install is checked against the catalog kill list unless *allow_removed*.
    *enable* None prompts "Enable now? [y/N]"; True/False skip the prompt.
    *yes_deps* is ``--yes-deps``: the explicit answer to the Python-deps consent
    question, so non-interactive installs (SSH automation, CI, Docker entrypoints)
    finish in one run instead of being refused and left for an ``enable`` that
    cannot recover a refused publication (#122134). The default without it stays
    fail-closed in non-interactive sessions.
    """
    from hermes_cli import plugins_cmd_catalog as catalog
    console = _pc()._console()
    entry = None
    if catalog.looks_like_catalog_name(identifier):
        entry = catalog.resolve_catalog_name(identifier, console)
        identifier = entry.install_identifier
        console.print(f"[bold]{entry.name}[/bold] [cyan]\\[{entry.tier}][/cyan] [dim]pinned @ {entry.sha[:8]}[/dim]")
        console.print(catalog.entry_capability_summary(entry))
    else:
        console.print("[yellow]Warning:[/yellow] custom (unreviewed) source — not from the Hermes catalog.")
    if allow_removed:
        console.print(
            "[bold red]WARNING:[/bold red] [red]--allow-removed set — skipping the catalog kill-list check. "
            "This plugin may have been removed for security reasons.[/red]")

    try:
        git_url, _subdir = _pc()._resolve_git_url(identifier)
        if not allow_removed:
            catalog.raise_if_removed(identifier, git_url, *((entry.name,) if entry else ()))
    except (ValueError, _pc().PluginOperationError) as e:
        _pc()._fail(console, f"[red]Error:[/red] {e}")
    if git_url.startswith(("http://", "file://")):
        console.print(
            "[yellow]Warning:[/yellow] Using insecure/local URL scheme. "
            "Consider using https:// or git@ for production installs.")

    console.print(f"[dim]Cloning {git_url}{f' (subdir: {_subdir})' if _subdir else ''}...[/dim]")

    def _interactive_scan_decision(scan_result) -> bool:
        """Prompt the user to accept a caution-verdict plugin."""
        from tools.plugin_guard import format_scan_report
        console.print()
        console.print("[yellow]⚠ Security scan flagged this plugin:[/yellow]")
        console.print(format_scan_report(scan_result))
        return _pc()._is_tty() and _pc()._ask_yes("  Install anyway? Only continue if you trust the source. [y/N]: ")

    def _install() -> tuple:
        if entry is not None:
            return catalog.install_catalog_entry(
                entry, force=force, ref=ref, allow_removed=allow_removed, scan_decision_cb=_interactive_scan_decision,
                python_deps=not no_deps, assume_deps_consent=yes_deps)
        return _pc()._install_plugin_core(
            identifier, force=force, ref=ref, scan_decision_cb=_interactive_scan_decision,
            python_deps=not no_deps, allow_removed=allow_removed, assume_deps_consent=yes_deps)

    try:
        target, installed_manifest, installed_name = recorded_install(
            _install, catalog_name=entry.name if entry is not None else None, identifier=identifier)
    except _pc().PluginOperationError as e:
        _pc()._fail(console, f"[red]{'Blocked' if isinstance(e, _pc().PluginScanBlocked) else 'Error'}:[/red] {e}")
    if not _pc()._looks_like_plugin_dir(target):
        console.print(
            f"[yellow]Warning:[/yellow] {installed_name} doesn't contain plugin.yaml, "
            f"plugin.json, or __init__.py. It may not be a valid Hermes plugin.")
    _prompt_plugin_env_vars(installed_manifest, console)

    from pm.workspace import enabled_plugin_dirs

    # Active replacements settled consent against the staged tree before PM
    # prepared or published it. Do not present a second, ineffective veto.
    already_active = target.resolve() in enabled_plugin_dirs()
    should_enable = False if no_deps else enable
    if no_deps:
        console.print("[dim]--no-deps: skipping dependency consent; the plugin stays disabled.[/dim]")
    # A memory provider activates through memory.provider alone; the loader never reads plugins.enabled.
    # Decide from the installed tree's code (not a text match): catalog category "memory" also holds hooks,
    # context engines and skills, and a hook's docstring may name MemoryProvider.
    from plugins.memory import _defines_memory_provider
    is_memory_provider = _defines_memory_provider(target)
    if should_enable is None and not already_active:
        should_enable = _pc()._is_tty() and _pc()._ask_yes(
            f"  Use '{installed_name}' as the memory provider now? [y/N]: " if is_memory_provider
            else f"  Enable '{installed_name}' now? [y/N]: ")
    deps_ok, deps_reason = (True, None)
    if should_enable and not already_active:
        deps_ok, deps_reason = _install_plugin_python_deps(installed_manifest, target, console,
                                                           assume_yes=yes_deps)

    _pc()._display_after_install(target, identifier)

    # ONE admission transaction for the enable (C13): resolve the candidate
    # union (enabled members + this target) and commit the config in the
    # same step — env and config change together or not at all. No
    # duplicate sync here: nothing was resolved before this point.
    if should_enable and not deps_ok:
        # Consent declined/skipped: nothing was installed or changed, so
        # enabling is refused without touching config or environment.
        console.print(
            f"[red]✗[/red] Cannot enable [bold]{installed_name}[/bold]: "
            f"{deps_reason}"
        )
        console.print(
            "[dim]The plugin stays installed but disabled; re-enable "
            "after resolving the conflict.[/dim]"
        )
        should_enable = False

    # already_active is PM's selection (plugins.enabled): it settles dependency consent, not memory.provider.
    # An explicit --enable still selects the provider, repairing installs that listed it in plugins.enabled.
    if already_active and not (is_memory_provider and enable):
        console.print("[dim]Replacement installed; plugin selection was not changed.[/dim]")
    elif is_memory_provider:
        _select_memory_provider(target.name, console, select=should_enable)
    elif should_enable:
        from hermes_cli.plugins_admission import AdmissionRefused

        try:
            _pc()._set_plugin_enabled(installed_name, enable=True, console=console)
        except AdmissionRefused:
            console.print(
                "[dim]The plugin stays installed but disabled; re-enable "
                "after resolving the conflict.[/dim]"
            )
        else:
            console.print(
                f"[green]✓[/green] Plugin [bold]{installed_name}[/bold] enabled.",
            )
    else:
        console.print(
            f"[dim]Plugin installed but not enabled. "
            f"Run `hermes plugins enable {installed_name}` to activate.[/dim]")

    from hermes_cli.plugin_provider_requests import requires_auth_notice
    if notice := requires_auth_notice(installed_manifest):
        console.print(f"[yellow]{notice}[/yellow]")
    # Non-interactive installs and declines leave declared capabilities ungranted (fail closed).
    declared_caps = _pc()._declared_capabilities_from_manifest(installed_manifest, installed_name)
    if declared_caps:
        _pc()._run_capability_consent(console, installed_name, declared_caps, context="install")
    if enable and not is_memory_provider:
        # Loads it into the running gateway now (handlers live) or says what needs a restart (#87770).
        from hermes_cli.plugins_activation import activate_plugin_now, activation_hint
        console.print(f"[dim]{activation_hint(activate_plugin_now(installed_name, in_process=False))}[/dim]")
    console.print()


def _select_memory_provider(name: str, console, *, select: bool) -> None:
    """Make *name* the live memory provider (deps prepared through PM first, as ``hermes memory setup``
    does), or say how to; ``plugins enable`` cannot activate one (#119909)."""
    if not select:
        console.print(
            f"[dim]Memory provider installed but not active. Run `hermes memory setup` "
            f"(or set memory.provider: {name}) to use it.[/dim]")
        return
    from hermes_cli.memory_setup import prepare_memory_provider_dependencies
    from plugins.memory import find_provider_dir
    if find_provider_dir(name) is None:  # never save a provider name the loader cannot resolve
        console.print(f"[red]✗[/red] {name} does not resolve as a memory provider; memory.provider is unchanged.")
        return
    try:
        prepare_memory_provider_dependencies(name)
    except Exception as exc:  # resolver conflict, network, PM refusal: report, leave the selection alone
        logger.debug("memory provider %s dependency preparation failed", name, exc_info=True)
        console.print(f"[red]✗[/red] Could not prepare {name}'s dependencies: {exc}")
        console.print("[dim]memory.provider is unchanged; run `hermes memory setup` after resolving it.[/dim]")
        return
    # The loader refuses a provider parked in plugins.disabled; lift that through the same admission transaction.
    from hermes_cli.plugins_admission import AdmissionRefused
    expected_config = _pc()._plugin_selection_version()
    disabled, aliases = _pc()._get_disabled_set(), _pc()._plugin_aliases(name)
    if disabled & aliases:
        try:
            _pc()._admit_and_save_plugin_sets(_pc()._get_enabled_set(), disabled - aliases,
                                              console=console, action=f"Re-enable '{name}'",
                                              expected_config=expected_config)
        except AdmissionRefused:
            console.print("[dim]memory.provider is unchanged.[/dim]")
            return
    previous = _pc()._get_current_memory_provider()
    _pc()._save_memory_provider(name)
    console.print(
        f"[green]✓[/green] [bold]{name}[/bold] set as memory.provider"
        f"{f' (replacing {previous})' if previous and previous != name else ''}. "
        f"Run `hermes memory setup {name}` to configure it; new sessions use it.")


def _catalog_install_on_disk(catalog_name: str, ref: Optional[str]) -> Optional[tuple]:
    """``(target, manifest, installed_name)`` of the installed, not enabled tree whose install record
    names catalog entry *catalog_name*, else None. A refused enable leaves the published clone in
    place (there is no rollback), so the card's Try again, or the model asking again, stopped at
    "already exists" on a second clone and the plugin could never be turned on from the card.
    Like ``hermes plugins enable``, the tree is used at the commit it has; an explicit *ref* only
    matches a tree checked out at that commit."""
    from hermes_cli.plugins_cmd_catalog import catalog_install_record
    target = _pc()._catalog_installed_dir(catalog_name)
    if target is None:
        return None
    if ref and str(catalog_install_record(target)["sha"]).lower() != ref.lower():
        return None
    manifest = _pc()._read_manifest(target)
    installed_name = manifest.get("name") or target.name
    return None if {installed_name, target.name} & _pc()._get_enabled_set() else (target, manifest, installed_name)


def _resolve_source(identifier: str, catalog_name: Optional[str]) -> tuple:
    """``(entry, identifier, warnings, error)``: the catalog entry (None for a custom source), the
    identifier to clone, the warnings so far, and the refusal result when there is one."""
    from hermes_cli import plugins_cmd_catalog as catalog
    warnings: list[str] = []
    entry = None
    if catalog_name:
        entry = catalog.get_live_catalog_entry(catalog_name)
        if entry is None:
            return None, identifier, warnings, {"ok": False, "error": f"'{catalog_name}' is not in the Hermes plugin catalog."}
        warnings.extend(_known_issue_warnings(entry))
        identifier = entry.install_identifier
    else:
        warnings.append("Custom (unreviewed) source — not from the Hermes catalog.")
    try:
        git_url = _pc()._resolve_git_url(identifier)[0]
        if git_url.startswith(("http://", "file://")):
            warnings.append("Insecure URL scheme; prefer https:// or git@ for production installs.")
        catalog.raise_if_removed(identifier, git_url, *((entry.name,) if entry else ()))
    except ValueError:
        pass
    except _pc().PluginOperationError as exc:
        return entry, identifier, warnings, {"ok": False, "error": str(exc)}
    return entry, identifier, warnings, None


def _place_tree(entry, identifier: str, *, force: bool, ref: Optional[str], assume_deps_consent: bool,
                step: Callable[[InstallPhase], None]) -> tuple | dict:
    """``(target, manifest, installed_name)`` of the tree to enable, or the error result. A catalog
    entry already on disk and not enabled (an earlier enable was refused) is used as it is, unless
    an approved *ref* names another commit; a fresh install clones at *ref* when set, else the pin."""
    from hermes_cli import plugins_cmd_catalog as catalog
    ref = (ref or "").strip() or None
    on_disk = _catalog_install_on_disk(entry.name, ref) if entry is not None and not force else None
    if on_disk is not None:
        return on_disk

    def _install() -> tuple:
        if entry is not None:
            return catalog.install_catalog_entry(entry, force=force, ref=ref, allow_removed=False,
                                                 assume_deps_consent=assume_deps_consent)
        return _pc()._install_plugin_core(identifier, force=force, ref=ref, assume_deps_consent=assume_deps_consent)

    step(InstallPhase.downloading)
    try:
        return recorded_install(_install, catalog_name=entry.name if entry is not None else None,
                                identifier=identifier)
    except _pc().PluginScanBlocked as exc:
        fields = ("pattern_id", "severity", "category", "file", "line", "description")
        return {
            "ok": False, "error": str(exc), "scan_blocked": True,
            "scan_verdict": getattr(exc.scan_result, "verdict", "dangerous"),
            "scan_findings": [
                {k: getattr(f, k) for k in fields}
                for f in (exc.scan_result.findings if exc.scan_result is not None else ())
            ],
        }
    except _pc().PluginOperationError as exc:
        return {"ok": False, "error": str(exc)}


def _enable_placed(installed_name: str, deps, step: Callable[[InstallPhase], None]) -> Optional[dict]:
    """None once enabled, else the refusal result. Enabling admits the plugin, and admission resolves
    its Python dependencies."""
    from hermes_cli.plugins_admission import AdmissionRefused

    if deps:
        step(InstallPhase.python_packages)
    try:
        # _set_plugin_enabled serializes per home itself (parallel install-card rows).
        _pc()._set_plugin_enabled(installed_name, enable=True)
    except AdmissionRefused as exc:
        return {"ok": False, "error": f"enable refused: {exc}", "plugin_name": installed_name, "enabled": False}
    return None


def dashboard_install_plugin(
    identifier: str, *, force: bool, enable: bool, catalog_name: Optional[str] = None,
    ref: Optional[str] = None, assume_deps_consent: Optional[bool] = None,
    on_step: Optional[Callable[[InstallPhase], None]] = None,
) -> dict[str, Any]:
    """Non-interactive install for the dashboard/TUI. *catalog_name* installs a curated entry at its
    pinned SHA (identifier may be empty); *ref* pins either source to one full commit SHA (same
    contract as ``--ref``); every path enforces the kill list (no GUI bypass). *assume_deps_consent*
    is consent the caller holds for the Python deps, so no terminal is needed to answer the gate.
    None (a user's Install click) is the consent the fresh-install enable already acts on: a plugin
    the config already selects (``memory.provider``) would otherwise be refused every time, while
    the same plugin unselected installs and enables. A forced replacement still needs explicit consent.
    *on_step* hears each slow phase as it starts (the catalog card draws it under its bar)."""
    if assume_deps_consent is None:
        assume_deps_consent = enable and not force
    step = on_step or (lambda _phase: None)
    entry, identifier, warnings, error = _resolve_source(identifier, catalog_name)
    if error is not None:
        return error
    placed = _place_tree(entry, identifier, force=force, ref=ref, assume_deps_consent=assume_deps_consent, step=step)
    if isinstance(placed, dict):
        return placed
    target, installed_manifest, installed_name = placed
    deps = _pc()._python_dependency_summary(target, warnings)
    if enable and (refused := _enable_placed(installed_name, deps, step)):
        return refused
    ap = target / "after-install.md"
    # Deps first, then load: the plugin activates in this process (TUI/Desktop server subscribers see it)
    # and in the running gateway; ``activation`` says what is live now vs next session (#87770).
    from hermes_cli.plugins_activation import activate_plugin_now
    if enable:
        step(InstallPhase.loading_tools)
    activated = activate_plugin_now(installed_name) if enable else {
        "gateway_reloaded": False, "activation": None, "restart_required": False}
    return {
        "ok": True, "plugin_name": installed_name, "warnings": warnings,
        "python_dependencies": deps,
        "missing_env": [s["name"] for s in _pc()._missing_env_specs(installed_manifest)],
        "after_install_path": str(ap) if ap.exists() else None, "enabled": enable, **activated,
        "known_issues": list(entry.known_issues) if entry else [],
    }
