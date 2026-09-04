"""Package definitions for the tools hermes manages. Versions and hashes
live in pm/lock.json (written by `pm lock`), never here."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from pm.package import (
    InstallError,
    Package,
    StatePackage,
    _entry_listing,
    _missing_reason,
    _probe_reason,
)
from pm.registry import register
from pm.store import ALL_TARGETS, Store, flatten_single_dir, merge_tree
from pm.update import (
    btbn_versions,
    github_release_tags,
    llama_app_bucket_versions,
    llama_app_latest,
    martin_riedl_versions,
    node_latest_versions,
    npm_dist_tags,
    pbs_build_tags,
)

_RUST_TRIPLE = {
    "win32-x64": "x86_64-pc-windows-msvc",
    "win32-arm64": "aarch64-pc-windows-msvc",
    "linux-x64": "x86_64-unknown-linux-gnu",
    "linux-arm64": "aarch64-unknown-linux-gnu",
    "darwin-x64": "x86_64-apple-darwin",
    "darwin-arm64": "aarch64-apple-darwin",
}

_NODE_PLAT = {
    "win32-x64": "win-x64",
    "win32-arm64": "win-arm64",
    "linux-x64": "linux-x64",
    "linux-arm64": "linux-arm64",
    "darwin-x64": "darwin-x64",
    "darwin-arm64": "darwin-arm64",
}


class BinaryPackage(Package):
    """A downloaded archive exposing one binary. Covers most tools."""

    binary_rel: dict[str, str] = {}
    flatten = True
    probe_version = True
    # argv after the binary for the smoke probe. A package whose binary
    # rejects the GNU double-dash form (ffmpeg's BtbN autobuild) overrides.
    probe_args: list[str] = ["--version"]
    # Run the probe with cwd=binary.parent: dlopen'd backends (llama.cpp's
    # cudart) resolve their shared libraries from the working directory.
    probe_cwd = False

    def _rel(self, target: str) -> Optional[str]:
        win = target.startswith("win32")
        return self.binary_rel.get(target) or self.binary_rel.get(
            "win32" if win else "posix"
        )

    def stage(self, store: Store, staged: Path, version: str, target: str) -> None:
        if self.flatten:
            flatten_single_dir(staged)

    def binary(self, entry: Path, target: str) -> Optional[Path]:
        rel = self._rel(target)
        return entry / rel if rel else None

    def verify(self, entry: Path, target: str) -> str:
        """Return '' when the entry is usable on target, else why not:
        a missing binary, a wrong-arch binary, or a --version probe that
        fails to exec, times out, or exits nonzero."""
        binary = self.binary(entry, target)
        if binary is None:
            return "no binary_rel for this target"
        reason = self._binary_reason(binary, entry, target)
        if reason:
            return reason
        if not self.probe_version:
            return ""
        try:
            proc = subprocess.run(
                [str(binary), *self.probe_args],
                capture_output=True,
                timeout=60,
                cwd=str(binary.parent) if self.probe_cwd else None,
                env=self._probe_env(),
            )
        except OSError as e:
            return f"could not exec {binary} {' '.join(self.probe_args)}: {e}"
        except subprocess.TimeoutExpired:
            return f"{binary} {' '.join(self.probe_args)} timed out after 60s"
        if proc.returncode != 0:
            return _probe_reason(binary, proc)
        return ""

    def _probe_env(self) -> dict:
        """Deps' env composed in: npm's shim is `#!/usr/bin/env node` and
        must find the node it extends on PATH."""
        if not self.deps:
            return dict(os.environ)
        from pm.ensure import env_for

        return env_for(*self.deps)


@register
class Uv(BinaryPackage):
    name = "uv"
    internal = True
    binary_rel = {"win32": "uv.exe", "posix": "uv"}

    def fetch_url(self, version: str, target: str) -> str:
        triple = _RUST_TRIPLE[target]
        ext = "zip" if target.startswith("win32") else "tar.gz"
        return f"https://github.com/astral-sh/uv/releases/download/{version}/uv-{triple}.{ext}"

    def latest_versions(self, target: str, locked=None) -> list[str]:
        return github_release_tags("astral-sh/uv")


_MACOS_MANAGED_PYTHON_IDENTIFIER = "com.nousresearch.hermes.managed-python"


def _macos_sign_managed_python(python: Path) -> bool:
    """Give a downloaded Python a stable macOS code identity."""
    if platform.system() != "Darwin":
        return False

    codesign = shutil.which("codesign")
    if not codesign:
        return False

    requirement = (
        "=designated => identifier "
        f'"{_MACOS_MANAGED_PYTHON_IDENTIFIER}"'
    )
    try:
        signed = subprocess.run(
            [
                codesign,
                "--force",
                "--deep",
                "--sign",
                "-",
                "--timestamp=none",
                "--identifier",
                _MACOS_MANAGED_PYTHON_IDENTIFIER,
                "--requirements",
                requirement,
                str(python),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if signed.returncode != 0:
            return False
        verified = subprocess.run(
            [codesign, "--verify", "--deep", "--strict", str(python)],
            check=False,
            capture_output=True,
            text=True,
        )
        return verified.returncode == 0
    except Exception:
        return False


@register
class Python(BinaryPackage):
    """The payload interpreter (python-build-standalone install_only).
    Optional: dev installs use their own venv's python; bundles stage this
    and point the relocatable venv's pyvenv.cfg at it (pm adopt)."""

    name = "python"
    optional = True
    probe_version = False
    binary_rel = {"win32": "python.exe", "posix": "bin/python3"}

    def stage(self, store: Store, staged: Path, version: str, target: str) -> None:
        super().stage(store, staged, version, target)
        binary = self.binary(staged, target)
        if binary is not None:
            _macos_sign_managed_python(binary)
        # python-build-standalone ships the x64 VC runtime (vcruntime140_1.dll)
        # beside ARM64 Windows Python; it cannot load on ARM64 and would fail
        # the arch guard. Drop it HERE, before publish: the tree digest is
        # recorded over the shipped bytes, so a post-publish deletion (as the
        # old bundle-time drop did) makes doctor's re-hash mismatch.
        if target == "win32-arm64":
            (staged / "vcruntime140_1.dll").unlink(missing_ok=True)

    def fetch_url(self, version: str, target: str) -> str:
        # lock version is "<python>+<release tag>", e.g. "3.11.13+202****0807"
        pyver, _, tag = version.partition("+")
        if not tag:
            raise InstallError(self.name, f"version {version!r} needs the +<release> tag")
        triple = _RUST_TRIPLE[target]
        return (
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
            f"{tag}/cpython-{pyver}+{tag}-{triple}-install_only.tar.gz"
        )

    def latest_versions(self, target: str, locked=None) -> list[str]:
        # Stay on the locked python minor line (3.11); bump only the
        # +<build-tag>. A major/minor bump is a deliberate decision, never
        # an auto-update.
        if not locked or "+" not in locked:
            return []
        pyver = locked.partition("+")[0]
        minor = ".".join(pyver.split(".")[:2])
        return [f"{pyver}+{tag}" for tag in pbs_build_tags(minor, target)]


def _uv_lock_digest(path: Path) -> bytes:
    """sha256 of uv.lock, cached on (mtime_ns, size) — check() runs at
    every startup and uv.lock is megabyte-class."""
    import hashlib

    stat = path.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _uv_lock_digest_cache.get(path)
    if cached and cached[0] == key:
        return cached[1]
    digest = hashlib.sha256(path.read_bytes()).digest()
    _uv_lock_digest_cache[path] = (key, digest)
    return digest


_uv_lock_digest_cache: dict[Path, tuple] = {}


def uv_cache_dir() -> Path:
    """The hermes-owned uv cache: machine-scoped and shared (keyed by
    content — two profiles reuse one cache), anchored to the DEFAULT
    hermes root like partials_root(). A bundle ships a seeded copy at
    the payload root (uv-cache/); the first call on a sealed install
    copies it out to the writable machine cache (the read-only payload
    can't serve uv's working cache), and a warm `uv sync --offline`
    from it is near-free (probed: 0.4s vs 1.2s cold) — the blow-away-
    on-update contract depends on it. uv's default cache location is
    per-user/platform-opinionated and never used by pm."""
    from hermes_constants import get_default_hermes_root

    machine_cache = get_default_hermes_root() / "cache" / "uv"
    marker = machine_cache / ".seeded"
    if not marker.is_file():
        # Seed from a shipped bundle cache when present (payload root =
        # store_root().parent on a sealed install).
        try:
            from pm.paths import store_root

            payload_cache = store_root().parent / "uv-cache"
            if payload_cache.is_dir():
                machine_cache.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil

                for entry in payload_cache.iterdir():
                    if entry.name == ".seeded":
                        continue
                    dest = machine_cache / entry.name
                    if not dest.exists():
                        (
                            _shutil.copytree(entry, dest)
                            if entry.is_dir()
                            else _shutil.copy2(entry, dest)
                        )
        except OSError:
            pass  # seeding is best-effort; a cold sync still works
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("1", encoding="utf-8")
        except OSError:
            pass
    return machine_cache


def uv_env(base_env: Optional[dict] = None) -> dict[str, str]:
    """Sanitized env for pm's internal uv invocations: user-level UV
    overrides and active-venv leakage must not steer which interpreter or
    install dir uv picks (the interpreter-hijack class, #83914), and
    ambient user/system uv config (uv.toml under XDG_CONFIG_HOME/DIRS)
    must not reach the subprocess either. User config discovery is
    redirected to an empty dir; HOME is left alone so caches, credentials,
    and git keep working (the #82446 isolation contract, which lived in
    the installers' run_locked_uv_sync() before the sync moved into pm).
    UV_CACHE_DIR always points at the hermes-owned cache (shipped with
    bundles, shared machine-wide) — never the user's ambient cache."""
    env = dict(os.environ if base_env is None else base_env)
    for key in list(env):
        if key.startswith("UV_") or key in (
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONEXECUTABLE",
        ):
            env.pop(key)
    isolated = _isolated_config_dir()
    env["XDG_CONFIG_HOME"] = str(isolated)
    env["XDG_CONFIG_DIRS"] = str(isolated)
    env["UV_NO_CONFIG"] = "1"
    env["UV_CACHE_DIR"] = str(uv_cache_dir())
    return env


_ISOLATED_CONFIG_DIR: Optional[str] = None


def _isolated_config_dir() -> Path:
    """A per-process empty dir ambient uv config is redirected to. One
    shared empty dir is all that's needed (and all that ever gets
    created); it is never populated, so it cannot steer resolution."""
    import tempfile

    global _ISOLATED_CONFIG_DIR
    if _ISOLATED_CONFIG_DIR is None:
        _ISOLATED_CONFIG_DIR = tempfile.mkdtemp(prefix="pm-uv-")
    return Path(_ISOLATED_CONFIG_DIR)


@register
class Venv(StatePackage):
    """The project venv: pyproject.toml + uv.lock + enabled extras.
    Made true by `uv sync --frozen`; uv is its internal dependency."""

    name = "venv"
    deps = ("uv",)

    def project_root(self) -> Path:
        from pm.paths import repo_root

        return repo_root()

    def venv_dir(self) -> Path:
        # Sealed installs are read-only: the MUTABLE venv lives in the
        # writable machine hermes root (get_default_hermes_root()/venv —
        # per-install, beside the uv cache), seeded on first adopt by
        # copying the payload's shipped venv out. Dev/source installs
        # keep the repo-local venv (project_venv_dir) unchanged.
        from pm.ensure import sealed

        if sealed():
            from hermes_constants import get_default_hermes_root

            return get_default_hermes_root() / "venv"
        from hermes_constants import project_venv_dir

        found = project_venv_dir(self.project_root())
        return found if found else self.project_root() / "venv"

    def seed_mutable_venv(self) -> Optional[str]:
        """Bootstrap the mutable venv from the shipped payload venv
        (lazy-off installs). Returns None on success, else why not."""
        import shutil

        from pm.ensure import lazy_installs_allowed, sealed
        from pm.paths import store_root

        if not sealed():
            return None  # dev installs use the repo venv directly
        dest = self.venv_dir()
        if dest.exists():
            return None  # already bootstrapped
        if lazy_installs_allowed():
            # Lazy installs ON: build fresh from the shipped uv cache —
            # no seed copy needed (the sync is near-free from the cache).
            return None
        payload_venv = store_root().parent / "venv"
        if not payload_venv.is_dir():
            return "payload ships no venv to seed from"
        try:
            shutil.copytree(payload_venv, dest)
        except OSError as exc:
            return f"seed copy failed: {exc}"
        return None

    def expected_stamp(self, extras: list[str]) -> str:
        import hashlib
        import sys

        h = hashlib.sha256()
        h.update(_uv_lock_digest(self.project_root() / "uv.lock"))
        h.update(",".join(sorted(extras)).encode())
        h.update(f"{sys.version_info.major}.{sys.version_info.minor}".encode())
        # Plugin members union into the venv — a changed member set must
        # re-sync even when extras and core lock are unchanged.
        from pm.workspace import enabled_member_dirs, members_stamp

        h.update(members_stamp(enabled_member_dirs()).encode())
        return h.hexdigest()

    def apply(self, extras: list[str]) -> None:
        from pm.ensure import uv as pm_uv
        from pm.workspace import enabled_member_dirs, lock_and_sync

        member_dirs = enabled_member_dirs()
        if member_dirs:
            # Lazy installs OFF = the frozen bundle feature set: plugin
            # members are never installed (the bundle IS the install).
            from pm.ensure import lazy_installs_allowed

            if not lazy_installs_allowed():
                from pm.features import read_features

                if read_features() is not None:
                    raise InstallError(
                        self.name,
                        "plugin members present but lazy installs are "
                        "disabled — this bundle's feature set is frozen "
                        "(remove the plugin or enable lazy installs)",
                    )
            # Plugin deps union into the venv through the generated
            # workspace root (one lock, conflict = loud refusal). On
            # conflict, resolve_union bisects: fail-alone plugins and
            # mutual-conflict losers (incumbent wins) are dropped with
            # their resolver reasons, and the union retries.
            from pm.workspace import record_disabled_plugins, resolve_union

            import logging

            survivors, decisions = resolve_union(
                member_dirs, extras, venv_dir=self.venv_dir()
            )
            for decision in decisions:
                logging.getLogger(__name__).warning(
                    "plugin %s disabled by the venv union: %s",
                    decision["plugin"],
                    decision["reason"],
                )
            # Write bisect disables back to the plugins enabled config —
            # `hermes plugins list` must reflect reality and re-enable
            # must retry. Best-effort: a config write failure never
            # breaks the sync.
            try:
                record_disabled_plugins(decisions)
            except Exception:
                logging.getLogger(__name__).warning(
                    "could not persist bisect disable decisions", exc_info=True
                )
            # Receipt: the machine-readable surface for this rebuild
            # (same schema/dir as update receipts; the updater embeds
            # these sections via pm.receipt.snapshot()).
            try:
                from pm import receipt

                receipt.begin("sync")
                receipt.record_feature_list(sorted(extras))
                receipt.record_venv_rebuild(True)
                if decisions:
                    receipt.record_bisect(decisions)
                    receipt.finalize("bisected")
                else:
                    receipt.finalize("ok")
            except Exception:
                pass
            return

        uv_bin, env = pm_uv(venv=self.venv_dir())
        if uv_bin is None:
            raise InstallError(self.name, "uv is not installed")
        cmd = [uv_bin, "sync", "--frozen"]
        for extra in sorted(extras):
            cmd += ["--extra", extra]
        proc = subprocess.run(
            cmd,
            cwd=str(self.project_root()),
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        if proc.returncode != 0:
            raise InstallError(
                self.name, f"uv sync exited {proc.returncode}: {proc.stderr[-400:]}"
            )


@register
class Nodejs(BinaryPackage):
    name = "node"
    binary_rel = {"win32": "node.exe", "posix": "bin/node"}

    def fetch_url(self, version: str, target: str) -> str:
        plat = _NODE_PLAT[target]
        ext = "zip" if target.startswith("win32") else "tar.xz"
        return f"https://nodejs.org/dist/v{version}/node-v{version}-{plat}.{ext}"

    def latest_versions(self, target: str, locked=None) -> list[str]:
        return node_latest_versions()


@register
class Npm(BinaryPackage):
    name = "npm"
    deps = ("node",)
    binary_rel = {"win32": "npm.cmd", "posix": "bin/npm"}
    flatten = False
    probe_version = False
    url = "https://registry.npmjs.org/npm/-/npm-{version}.tgz"

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        """npm installs itself using the node it extends: a plain unpack
        resolves the cli from dirname(process.execPath) and finds node's
        bundled npm instead. --offline pins the bytes to the verified
        tarball; --ignore-scripts + a sanitized env keep user npm/node
        config out of the staging."""
        from pm.ensure import _facts, _store
        from pm.registry import get_package

        facts = _facts()
        store = _store()
        node = get_package("node")
        node_fact = facts.get("node")
        if node_fact is None:
            raise InstallError(self.name, "npm extends node, which is not installed")
        node_bin = node.binary(store.entry(node_fact["entry"]), target)
        if node_bin is None or not node_bin.is_file():
            raise InstallError(self.name, "node's entry is missing its binary")
        win = target.startswith("win32")
        bundled_cli = (
            node_bin.parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
            if win
            else node_bin.parent.parent / "lib" / "node_modules" / "npm" / "bin" / "npm-cli.js"
        )
        if not bundled_cli.is_file():
            raise InstallError(self.name, "node's entry is missing its bundled npm-cli.js")

        env = {
            key: value
            for key, value in os.environ.items()
            if not key.lower().startswith("npm_config_")
            and key not in ("NODE_OPTIONS", "NODE_PATH", "NODE_ENV")
        }
        env["npm_config_cache"] = str(archive.parent / ".npm-cache")

        staged.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [
                str(node_bin), str(bundled_cli), "install", "--global",
                "--prefix", str(staged), "--offline", "--ignore-scripts",
                "--no-audit", "--no-fund", str(archive),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            env=env,
        )
        if proc.returncode != 0:
            raise InstallError(
                self.name, f"self-install exited {proc.returncode}: {proc.stderr[-400:]}"
            )

    def latest_versions(self, target: str, locked=None) -> list[str]:
        latest = npm_dist_tags("npm").get("latest")
        return [latest] if latest else []


@register
class Git(BinaryPackage):
    """Windows only: Git for Windows carries the bash.exe contract. POSIX
    uses the system git — a deliberate gap, not an oversight. The tar.bz2
    release asset extracts with stdlib tarfile: no self-extractor, no GUI."""

    name = "git"
    optional = True
    binary_rel = {"win32": "cmd/git.exe"}
    flatten = False
    gaps = {
        "linux-x64": "POSIX uses system git by choice",
        "linux-arm64": "POSIX uses system git by choice",
        "darwin-x64": "POSIX uses system git by choice",
        "darwin-arm64": "POSIX uses system git by choice",
    }

    def fetch_url(self, version: str, target: str) -> str:
        tag, build = version.split("+")
        arch = "arm64" if target.endswith("arm64") else "64-bit"
        return (
            f"https://github.com/git-for-windows/git/releases/download/"
            f"v{tag}.windows.{build}/Git-{tag}.{build}-{arch}.tar.bz2"
        )

    def latest_versions(self, target: str, locked=None) -> list[str]:
        # Tag is v2.53.0.windows.3; the lock version is 2.53.0+3.
        out = []
        for tag in github_release_tags("git-for-windows/git", strip_prefix="v"):
            if ".windows." in tag:
                base, _, build = tag.partition(".windows.")
                out.append(f"{base}+{build}")
        return out

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        """stdlib tar extract, but skip members the data filter refuses —
        the MSYS tree ships dev/fd → /proc/self/fd style links that mean
        nothing on Windows and must not fail the install."""
        import tarfile

        staged.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            for member in tf:
                try:
                    tf.extract(member, staged, filter="data")
                except (tarfile.FilterError, OSError):
                    continue

    def env(self, entry: Path, target: str) -> dict:
        return {"PATH": [str(entry / "cmd"), str(entry / "usr" / "bin")]}


@register
class Gh(BinaryPackage):
    name = "gh"
    optional = True
    binary_rel = {"win32": "bin/gh.exe", "posix": "bin/gh"}

    def fetch_url(self, version: str, target: str) -> str:
        osname, arch = target.split("-")
        plat = {"win32": "windows", "linux": "linux", "darwin": "macOS"}[osname]
        arch = {"x64": "amd64", "arm64": "arm64"}[arch]
        ext = "zip" if osname in ("win32", "darwin") else "tar.gz"
        return (
            f"https://github.com/cli/cli/releases/download/v{version}/"
            f"gh_{version}_{plat}_{arch}.{ext}"
        )

    def latest_versions(self, target: str, locked=None) -> list[str]:
        return github_release_tags("cli/cli", strip_prefix="v")


@register
class Ffmpeg(BinaryPackage):
    """Static ffmpeg. GPLv3 builds; always bundled.
    optional=False: ffmpeg is a required runtime tool. Sealed bundles ship
    it baked into the payload (post_update skips provisioning sealed
    installs — the artifact is atomic); dev installs get it re-ensured by
    step_provision_runtimes when the pin bumps. Windows: BtbN/FFmpeg-Builds
    (dated autobuild tag; ships ffprobe too). Linux + macOS:
    ffmpeg.martin-riedl.de (uniform ZIP, published sha256; single-binary —
    no ffprobe)."""

    name = "ffmpeg"
    optional = False
    # The posix (martin-riedl) and win32 (BtbN) build streams have no shared
    # release cadence — they drift in PATCH. The lockfile version label is
    # major.minor; each target's exact patch lives in its artifact urls.
    version_style = "minor"
    # martin-riedl (posix) zips are a single `ffmpeg` file at the zip root;
    # BtbN (win32) zips carry bin/ffmpeg.exe under one top-level dir that
    # flatten hoists.
    binary_rel = {"win32": "bin/ffmpeg.exe", "posix": "ffmpeg"}
    flatten = True
    # BtbN autobuild n9.0.1-11-ge47273f4d9 rejects `--version`
    # ("Unrecognized option '-version'", exit 2880417800); `-version` works
    # and is accepted by every ffmpeg build.
    probe_args = ["-version"]

    def fetch_url(self, version: str, target: str) -> str:
        osname, arch = target.split("-")
        if osname == "win32":
            # BtbN: the newest autobuild tag whose assets carry this version.
            for tag, asset in btbn_index().get(version, [])[:1]:
                return (
                    "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
                    f"{tag}/{asset}"
                )
            # Fall back to the pinned build (index unreachable) — this must
            # match what the lockfile's own urls carry for the same version.
            return self._btbn_fallback(version, target)
        # martin-riedl: /download/<os>/<arch>/<epoch>_<version>/ffmpeg.zip
        martin = {
            ("linux", "x64"): "linux/amd64/1787074600_9.0.1",
            ("linux", "arm64"): "linux/arm64/1787072884_9.0.1",
            ("darwin", "x64"): "macos/amd64/1787081194_9.0.1",
            ("darwin", "arm64"): "macos/arm64/1787073674_9.0.1",
        }
        epoch = None
        for t, by_version in martin_riedl_index().items():
            if t == target and version in by_version:
                epoch = by_version[version]
                break
        if epoch is not None:
            osdir = "macos" if osname == "darwin" else "linux"
            return f"https://ffmpeg.martin-riedl.de/download/{osdir}/{arch}/{epoch}_{version}/ffmpeg.zip"
        # Fall back to the hardcoded pinned path (index unreachable).
        return f"https://ffmpeg.martin-riedl.de/download/{martin[(osname, arch)]}/ffmpeg.zip"

    def latest_versions(self, target: str, locked=None) -> list[str]:
        if target.startswith("win32"):
            return btbn_versions()
        return martin_riedl_versions(target)

    def _btbn_fallback(self, version: str, target: str) -> str:
        arch = "arm64" if target.endswith("arm64") else "64"
        return (
            "https://github.com/BtbN/FFmpeg-Builds/releases/download/"
            f"autobuild-2026-08-28-17-08/ffmpeg-n{version}-11-ge47273f4d9-"
            f"win{arch}-gpl-9.0.zip"
        )



@register
class Ripgrep(BinaryPackage):
    name = "ripgrep"
    binary_rel = {"win32": "rg.exe", "posix": "rg"}

    def fetch_url(self, version: str, target: str) -> str:
        triple = _RUST_TRIPLE[target].replace("-unknown-linux-gnu", "-unknown-linux-musl")
        ext = "zip" if target.startswith("win32") else "tar.gz"
        return (
            f"https://github.com/BurntSushi/ripgrep/releases/download/{version}/"
            f"ripgrep-{version}-{triple}.{ext}"
        )

    def latest_versions(self, target: str, locked=None) -> list[str]:
        return github_release_tags("BurntSushi/ripgrep")


@register
class CuaDriver(BinaryPackage):
    name = "cua-driver"
    optional = True
    binary_rel = {"win32": "cua-driver.exe", "posix": "cua-driver"}

    def fetch_url(self, version: str, target: str) -> str:
        arch = {
            "darwin-x64": "darwin-universal",
            "darwin-arm64": "darwin-universal",
            "linux-x64": "linux-x86_64",
            "linux-arm64": "linux-arm64",
            "win32-x64": "windows-x86_64",
            "win32-arm64": "windows-arm64",
        }[target]
        ext = "zip" if target.startswith("win32") else "tar.gz"
        return (
            f"https://github.com/trycua/cua/releases/download/cua-driver-rs-v{version}/"
            f"cua-driver-rs-{version}-{arch}-binary.{ext}"
        )

    def latest_versions(self, target: str, locked=None) -> list[str]:
        return github_release_tags("trycua/cua", strip_prefix="cua-driver-rs-v")

    def stage(self, store: Store, staged: Path, version: str, target: str) -> None:
        flatten_single_dir(staged)
        sdk = staged / "sdk"
        if sdk.is_dir():
            shutil.rmtree(sdk, ignore_errors=True)

    def _probe_env(self) -> dict:
        """The probe IS a first run: it must not mint telemetry state."""
        try:
            from tools.computer_use.cua_backend import cua_driver_child_env

            return cua_driver_child_env()
        except Exception:
            return dict(os.environ, CUA_DRIVER_RS_TELEMETRY_ENABLED="0")


@register
class AgentBrowser(BinaryPackage):
    name = "agent-browser"
    optional = True
    deps = ("chromium", "chromium-headless-shell")
    flatten = True
    probe_version = False
    url = "https://registry.npmjs.org/agent-browser/-/agent-browser-{version}.tgz"
    # No win32-arm64 gap: agent-browser ships only win32-x64, and Windows
    # ARM64 runs it via built-in emulation (its own postinstall falls back
    # to x64 on arm64). chromium is likewise the x64 build on win32-arm64.
    emulated_arch_targets = frozenset({"win32-arm64"})

    def latest_versions(self, target: str, locked=None) -> list[str]:
        latest = npm_dist_tags("agent-browser").get("latest")
        return [latest] if latest else []

    def _rel(self, target: str) -> Optional[str]:
        ext = ".exe" if target.startswith("win32") else ""
        # Windows ARM64 runs the x64 binary under built-in emulation:
        # agent-browser ships no native arm64 build (its own postinstall
        # falls back to x64 on arm64), so the staged name is win32-x64.
        if target == "win32-arm64":
            target = "win32-x64"
        return f"bin/agent-browser-{target}{ext}"

    def binary(self, entry: Path, target: str) -> Optional[Path]:
        # The win32-arm64 payload carries the x64 binary (emulated), so
        # resolve it under the win32-x64 name.
        return super().binary(entry, "win32-x64" if target == "win32-arm64" else target)

    def stage(self, store: Store, staged: Path, version: str, target: str) -> None:
        flatten_single_dir(staged)
        bin_dir = staged / "bin"
        if not bin_dir.is_dir():
            raise InstallError(self.name, "staged without a bin/ directory")
        keep = Path(self._rel(target)).name
        if not (bin_dir / keep).is_file():
            raise InstallError(self.name, f"{keep} missing from the staged tarball")
        for item in bin_dir.iterdir():
            if item.is_file() and item.name.startswith("agent-browser-") and item.name != keep:
                item.unlink()


class PlaywrightBrowser(Package):
    """Playwright resolves browsers by DIRECTORY NAME under one root:
    `<name with '-'→'_'>-<revision>`, no target suffix. The env points
    PLAYWRIGHT_BROWSERS_PATH at the store root itself. The entry carries
    the INSTALLATION_COMPLETE marker playwright checks.

    The version is `<playwright revision>+<chrome version>` — most targets
    download from the Chrome-for-Testing CDN by chrome version, and the
    targets CfT doesn't build (linux-arm64) come from playwright's own
    mirror by revision. The store entry is named by revision only, which
    is all playwright's resolver reads."""

    optional = True
    on_path = False
    _CDN = "https://cdn.playwright.dev"

    # Chrome-for-Testing platform names; targets absent here fall back to
    # playwright's dbazure mirror with its own platform names.
    # win32-arm64 uses the win64 (x64) build: CfT publishes no native
    # win-arm64 chromium, and Windows ARM64 runs x64 binaries via built-in
    # emulation — the same choice agent-browser's own postinstall makes.
    _CFT = {
        "linux-x64": "linux64",
        "darwin-x64": "mac-x64",
        "darwin-arm64": "mac-arm64",
        "win32-x64": "win64",
        "win32-arm64": "win64",
    }
    _MIRROR = {"linux-arm64": "linux-arm64"}
    _FILE = ""  # "chrome" / "chrome-headless-shell"
    _MIRROR_FILE = ""  # "chromium" / "chromium-headless-shell"

    def store_entry(self, version: str, target: str) -> str:
        revision = version.partition("+")[0]
        return f"{self.name.replace('-', '_')}-{revision}"

    def fetch_url(self, version: str, target: str) -> str:
        revision, _, chrome = version.partition("+")
        plat = self._CFT.get(target)
        if plat and chrome:
            return f"{self._CDN}/builds/cft/{chrome}/{plat}/{self._FILE}-{plat}.zip"
        mirror_plat = self._MIRROR[target]
        return (
            f"{self._CDN}/dbazure/download/playwright/builds/chromium/"
            f"{revision}/{self._MIRROR_FILE}-{mirror_plat}.zip"
        )

    def stage(self, store: Store, staged: Path, version: str, target: str) -> None:
        (staged / "INSTALLATION_COMPLETE").write_text("", encoding="utf-8")

    def verify(self, entry: Path, target: str) -> str:
        marker = entry / "INSTALLATION_COMPLETE"
        if marker.is_file():
            return ""
        return f"INSTALLATION_COMPLETE missing under {entry}; {_entry_listing(entry)}"

    def env(self, entry: Path, target: str) -> dict:
        return {"PLAYWRIGHT_BROWSERS_PATH": str(entry.parent)}


@register
class Chromium(PlaywrightBrowser):
    name = "chromium"
    _FILE = "chrome"
    _MIRROR_FILE = "chromium"


@register
class ChromiumHeadlessShell(PlaywrightBrowser):
    name = "chromium-headless-shell"
    _FILE = "chrome-headless-shell"
    _MIRROR_FILE = "chromium-headless-shell"


class LlamaCpp(BinaryPackage):
    """One llama.cpp backend build. Backends are dlopen'd plugins, so a
    usable engine is one archive per (target, backend) — plus, for Windows
    CUDA, the cudart archive: end users have no CUDA toolkit, and Windows
    resolves a DLL from the loading executable's own directory, so those
    DLLs must land beside llama-server.exe rather than in a second entry.

    Backend is a HARDWARE choice, not a target, so each backend is its own
    optional package and the runtime asks for the one this machine can
    use. Version is llama.cpp's rolling release tag without the `b`.
    """

    optional = True
    on_path = False
    binary_rel = {"win32": "llama-server.exe", "posix": "llama-server"}
    flatten = False
    # --version is llama-server's liveness proof AND the check that the
    # backend's shared libraries resolve: a CUDA build with no cudart
    # beside it fails here rather than at first chat.
    probe_cwd = True

    backend: str = ""
    # Release-asset infix per target, or absent where upstream ships none.
    assets: dict[str, str] = {}

    @property
    def gaps(self) -> dict[str, str]:  # type: ignore[override]
        return {
            target: f"llama.cpp publishes no {self.backend} build for {target}"
            for target in ALL_TARGETS
            if target not in self.assets
        }

    def _asset_names(self, version: str, target: str) -> list[str]:
        ext = "zip" if target.startswith("win32") else "tar.gz"
        return [f"llama-b{version}-bin-{self.assets[target]}.{ext}"]

    def fetch_urls(self, version: str, target: str) -> list[str]:
        return [
            f"https://github.com/ggml-org/llama.cpp/releases/download/b{version}/{asset}"
            for asset in self._asset_names(version, target)
        ]

    def fetch_url(self, version: str, target: str) -> str:
        return self.fetch_urls(version, target)[0]

    def latest_versions(self, target: str, locked=None) -> list[str]:
        # Resolve from the llama.app installer bucket (the installer's own
        # updater pointer + version index — no API token, no rate limit):
        # the `latest` pointer is the authoritative "next version"; the
        # bucket tree supplies the full candidate list. Artifacts still
        # come from the llama.cpp GitHub releases (1:1 tag correspondence).
        latest = llama_app_latest()
        if latest is not None:
            return [latest, *llama_app_bucket_versions()]
        return llama_app_bucket_versions() or github_release_tags(
            "ggml-org/llama.cpp", strip_prefix="b"
        )

    def known_sha256(self, version: str, url: str) -> Optional[str]:
        """GitHub's release API serves every asset's digest, so pinning a
        280 MB engine costs one API call instead of the download."""
        return _github_release_digests("ggml-org/llama.cpp", f"b{version}").get(
            url.rsplit("/", 1)[-1]
        )

    def stage(self, store: Store, staged: Path, version: str, target: str) -> None:
        """Some archives nest the binaries under build/bin; hoist them so
        binary_rel is one path for every target."""
        if (staged / self.binary(staged, target).name).is_file():
            return
        found = sorted(staged.rglob(self.binary(staged, target).name))
        if not found:
            raise InstallError(self.name, "archive contains no llama-server")
        merge_tree(found[0].parent, staged)


def _github_release_digests(repo: str, tag: str) -> dict[str, str]:
    import json
    import os
    import urllib.request

    cached = _release_digest_cache.get((repo, tag))
    if cached is not None:
        return cached
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    headers = {"User-Agent": "hermes-pm"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers),
            timeout=120,
        ) as resp:
            release = json.load(resp)
    except Exception:
        return {}
    digests = {}
    for asset in release.get("assets", []):
        digest = (asset.get("digest") or "").partition("sha256:")[2]
        if digest:
            digests[asset["name"]] = digest
    _release_digest_cache[(repo, tag)] = digests
    return digests


_release_digest_cache: dict[tuple, dict] = {}


@register
class LlamaCppCuda(LlamaCpp):
    """Windows only: upstream publishes no prebuilt Linux CUDA archive at
    current tags, so NVIDIA Linux users run the vulkan build."""

    name = "llamacpp-cuda"
    backend = "cuda"
    # CUDA 13.3 verified against 13.1/13.2 drivers; arm64 prebuilts landed
    # on 13.4 (the only CUDA line upstream builds for win-arm64).
    assets = {
        "win32-x64": "win-cuda-13.3-x64",
        "win32-arm64": "win-cuda-13.4-arm64",
    }
    _CUDART = {"win32-x64": "13.3-x64", "win32-arm64": "13.4-arm64"}

    def _asset_names(self, version: str, target: str) -> list[str]:
        return super()._asset_names(version, target) + [
            f"cudart-llama-bin-win-cuda-{self._CUDART[target]}.zip"
        ]


@register
class LlamaCppVulkan(LlamaCpp):
    name = "llamacpp-vulkan"
    backend = "vulkan"
    assets = {
        "win32-x64": "win-vulkan-x64",
        "linux-x64": "ubuntu-vulkan-x64",
        "linux-arm64": "ubuntu-vulkan-arm64",
    }


@register
class LlamaCppMetal(LlamaCpp):
    """macOS archives are unified builds with Metal compiled in."""

    name = "llamacpp-metal"
    backend = "metal"
    assets = {"darwin-x64": "macos-x64", "darwin-arm64": "macos-arm64"}


@register
class LlamaCppCpu(LlamaCpp):
    name = "llamacpp-cpu"
    backend = "cpu"
    assets = {
        "win32-x64": "win-cpu-x64",
        "win32-arm64": "win-cpu-arm64",
        "linux-x64": "ubuntu-x64",
        "linux-arm64": "ubuntu-arm64",
        "darwin-x64": "macos-x64",
        "darwin-arm64": "macos-arm64",
    }
