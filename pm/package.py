"""Package definitions: what a package IS. No versions, no hashes — those
live in lock.json, written by `pm lock`."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pm.store import Store


class InstallError(RuntimeError):
    def __init__(self, package: str, cause: str, remedy: str = ""):
        self.package = package
        self.cause = cause
        self.remedy = remedy or "retry, or run `hermes pm doctor`"
        super().__init__(f"{package}: {cause} — {self.remedy}")


def compose_env(diffs: list[dict], base: Optional[dict] = None) -> dict[str, str]:
    """Dependents win over their dependencies for every key: diffs arrive
    deps-first, later ones take precedence — npm's pinned shim must shadow
    the npm bundled inside node, and a package's exports beat inherited env.
    'PATH' values are lists of dirs, prepended."""
    env = dict(os.environ if base is None else base)
    path_dirs: list[str] = []
    for diff in reversed(diffs):
        for key, value in diff.items():
            if key == "PATH":
                dirs = value if isinstance(value, list) else [value]
                path_dirs.extend(str(d) for d in dirs if str(d) not in path_dirs)
    for diff in diffs:
        for key, value in diff.items():
            if key != "PATH":
                env[key] = str(value)
    if path_dirs:
        key = next((k for k in env if k.upper() == "PATH"), "PATH")
        existing = env.get(key, "")
        prefix = os.pathsep.join(path_dirs)
        env[key] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    return env


class Package:
    """Subclass and register. Declarative for the common case; override
    fetch_url()/unpack()/stage()/verify()/env()/migrate() for the rest.

    name: unique id.
    deps: packages installed before this one.
    optional: not part of the root closure; installed on demand.
    internal: a package manager pm uses inside install steps — never on
        PATH and never part of the root closure. uv is the single internal
        package; node/npm are shipped runtime tools (TUI, plugins), not
        install machinery.
    on_path: contributes PATH dirs.
    url: template with {version} and {target} holes. override fetch_url()
        when a platform needs a completely different url.
    gaps: targets this package does NOT exist for, with the reason
        (upstream ships no artifact). Everything else is available.
    """

    name: str = ""
    deps: tuple[str, ...] = ()
    optional: bool = False
    internal: bool = False
    on_path: bool = True
    url: str = ""
    gaps: dict[str, str] = {}
    # How the lockfile `version` field labels this package for `pm update`:
    #   "semver" — one shared version across targets (node 26.7.0, uv 0.12.3).
    #   "minor"  — the version label is major.minor; each target's exact
    #              patch lives in ITS artifact urls (ffmpeg's posix vs win32
    #              builds have no shared release cadence).
    version_style: str = "semver"
    # Targets where this package's binary is the x64 build run under
    # Windows ARM64 built-in emulation (no native arm64 artifact exists).
    # The arch guard accepts the x64 PE on these targets.
    emulated_arch_targets: frozenset[str] = frozenset()

    def missing_reason(self, target: str) -> Optional[str]:
        return self.gaps.get(target)

    def latest_versions(self, target: str, locked: Optional[str] = None) -> list[str]:
        """Newest-first candidate versions for `target` — the "how do I find
        latest" hook for `hermes pm update`. Empty list = this package has
        no auto-update source (chromium follows agent-browser; venv is a
        state; Playwright browsers are revision-pinned by playwright).

        Subclasses resolve their upstream's real release index here; the
        update driver intersects across targets (per version_style) and
        only ever pins a version every relevant target can serve. ``locked``
        is the current lockfile version when the resolver needs it (python
        keeps its 3.11 line and bumps only the +<build-tag>)."""
        return []


    def fetch_url(self, version: str, target: str) -> str:
        if not self.url:
            raise InstallError(self.name, "package has no download url")
        return self.url.format(version=version, target=target)

    def fetch_urls(self, version: str, target: str) -> list[str]:
        """Every archive this target is built from, in extraction order.
        Almost every package is one archive; override this (instead of
        fetch_url) when upstream splits a runtime across downloads that
        have to land in one directory."""
        return [self.fetch_url(version, target)]

    def store_entry(self, version: str, target: str) -> str:
        return f"{self.name}-{version}-{target}"

    def known_sha256(self, version: str, url: str) -> Optional[str]:
        """A digest the upstream already publishes, so `pm lock` does not
        have to stream the artifact to learn it. Override where a release
        API serves digests (GitHub's does); returning None means hash it."""
        return None

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        """Turn the verified archive into the staged tree. Default: extract.
        Override for self-extractors or install-style unpacks (npm).

        Called once per artifact; extract() empties its destination, so a
        multi-archive package receives each later archive in a scratch dir
        that pm merges into the staged tree.
        """
        from pm.store import extract

        extract(archive, staged)

    def stage(self, store: "Store", staged: Path, version: str, target: str) -> None:
        """Post-unpack fixups inside the scratch dir. Default: nothing."""

    def binary(self, entry: Path, target: str) -> Optional[Path]:
        return None

    def verify(self, entry: Path, target: str) -> str:
        """Return '' when the entry is usable on target, else why not.
        Every subclass check (probe, marker) must keep this shape: '' is
        the verified answer, anything else is the diagnosis."""
        binary = self.binary(entry, target)
        if binary is None:
            return ""
        return self._binary_reason(binary, entry, target)

    def _binary_reason(self, binary: Path, entry: Path, target: str) -> str:
        """'' when the binary is present and arch-plausible on target."""
        if not binary.is_file():
            return _missing_reason(binary, entry)
        if machine_matches_binary(binary, target) is False and target not in self.emulated_arch_targets:
            return f"{binary.name} is not a {target} binary"
        return ""

    def env(self, entry: Path, target: str) -> dict:
        diff: dict = {}
        if self.on_path:
            binary = self.binary(entry, target)
            if binary is not None:
                diff["PATH"] = [str(binary.parent)]
        return diff


class DebPackage(Package):
    """A .deb artifact staged by ar+tar extraction (never dpkg, never
    executing package content on the host).

    For cross-target interpreter/runtime .debs (Termux's bionic python):
    the staged binaries cannot exec on the host that stages them, so
    verify() is FILE EVIDENCE -- the control stanza carried inside the
    .deb (Package + Version) must match the pin. Digest verification of
    the downloaded bytes happens in the store, as for every package.

    unpack() is the hardened extractor: traversal, symlink-target, and
    member-type checks, shaped after the established safe-extract rules
    (symlinks allowed with in-root targets; devices/fifos refused).
    """

    # The control field this package's .deb must declare as its name
    # (defaults to the package's own name).
    deb_package: str = ""
    # Where the staged tree keeps the payload: Termux .debs carry the
    # full $PREFIX path, data/data/com.termux/files/usr. Overridable.
    prefix_rel: str = "data/data/com.termux/files/usr"

    def unpack(self, archive: Path, staged: Path, target: str) -> None:
        raw = archive.read_bytes()
        if raw[:8] != b"!<arch>\n":
            raise InstallError(self.name, f"not an ar archive: {archive.name}")
        payload = None
        offset = 8
        while offset + 60 <= len(raw):
            hdr = raw[offset:offset + 60]
            member = hdr[0:16].decode("ascii", "replace").rstrip()
            try:
                size = int(hdr[48:58].decode("ascii", "replace").strip())
            except ValueError:
                raise InstallError(self.name, f"bad ar member size in {archive.name}")
            start = offset + 60
            data = raw[start:start + size]
            if member.startswith("data.tar"):
                if member.endswith((".zst", ".lzma")):
                    raise InstallError(
                        self.name,
                        f"unsupported data compression {member} in {archive.name}",
                    )
                payload = data
                break
            offset = start + size + (size % 2)
        if payload is None:
            raise InstallError(self.name, f"no data.tar member in {archive.name}")
        self._safe_untar(payload, staged)

    def _safe_untar(self, payload: bytes, staged: Path) -> None:
        import io
        import posixpath
        import shutil
        import stat as stat_mod
        import tarfile

        real_staged = os.path.realpath(staged)

        def _contained(p: Path) -> bool:
            """True when p's REAL location (following any planted symlink
            ancestors) stays inside the staged tree."""
            real = os.path.realpath(p)
            return real == real_staged or real.startswith(real_staged + os.sep)

        deferred_links = []
        with tarfile.open(fileobj=io.BytesIO(payload)) as tf:
            for member in tf.getmembers():
                path = PurePosixPath(member.name)
                parts = tuple(q for q in path.parts if q not in ("", "."))
                if path.is_absolute() or ".." in parts:
                    raise InstallError(self.name, f"unsafe member path {member.name!r}")
                if not parts:
                    continue  # the "./" root member
                target = staged.joinpath(*parts)
                if not _contained(target):
                    raise InstallError(self.name, f"member escapes staged tree: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if member.issym():
                    if PurePosixPath(member.linkname).is_absolute() or Path(member.linkname).is_absolute():
                        raise InstallError(
                            self.name,
                            f"absolute symlink target {member.linkname!r} "
                            f"in {member.name}",
                        )
                    link_dir = "/".join(parts[:-1])
                    resolved = posixpath.normpath(posixpath.join(link_dir, member.linkname))
                    if resolved == ".." or resolved.startswith("../"):
                        raise InstallError(self.name, f"symlink escapes root: {member.name}")
                    if not _contained(target.parent / member.linkname):
                        raise InstallError(self.name, f"symlink escapes staged tree: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    try:
                        target.symlink_to(member.linkname)
                    except OSError:
                        resolved_path = staged.joinpath(*resolved.split("/"))
                        if resolved_path.is_file():
                            shutil.copy2(resolved_path, target)
                        else:
                            deferred_links.append((member.linkname, target, resolved_path))
                    continue
                if not member.isfile():
                    raise InstallError(self.name, f"unsupported member type: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not _contained(target.parent):
                    raise InstallError(
                        self.name,
                        f"member {member.name} resolves outside the staged tree "
                        "through a symlinked ancestor",
                    )
                extracted = tf.extractfile(member)
                if extracted is None:
                    raise InstallError(self.name, f"cannot read member {member.name}")
                with extracted, open(target, "wb") as dst:
                    shutil.copyfileobj(extracted, dst)
                try:
                    target.chmod(member.mode & 0o777)
                except OSError:
                    pass
        while deferred_links:
            pending = []
            for linkname, target, resolved_path in deferred_links:
                if not _contained(resolved_path):
                    raise InstallError(
                        self.name,
                        f"symlink {target.name} resolves outside the staged tree",
                    )
                if target.exists() or target.is_symlink():
                    continue
                if resolved_path.is_file():
                    shutil.copy2(resolved_path, target)
                else:
                    pending.append((linkname, target, resolved_path))
            if len(pending) == len(deferred_links):
                break
            deferred_links = pending
        # Termux debs carry owner-only modes across the whole tree (700 on
        # binaries n libs, 600 on stdlib .py files) -- postinst would
        # normalize on a real phone, but pm extracts without postinst, and
        # any uid-hostile mode breaks non-owner consumers: the dynamic
        # linker cannot read a 700 lib, the interpreter cannot read a 600
        # encoding module. Normalize EVERYTHING: a+r on all regular files,
        # a+X on anything that was executable. The deb's bytes are pinned
        # by digest; modes are not part of the pin. Symlinks are skipped:
        # chmod through one would follow it outside the staged tree.
        for root, dirs, files in os.walk(staged):
            dirs[:] = [d for d in dirs if not (Path(root) / d).is_symlink()]
            for name in files:
                f = Path(root) / name
                try:
                    mode = f.lstat().st_mode
                except OSError:
                    continue
                if stat_mod.S_ISLNK(mode):
                    continue
                if not _contained(f):
                    continue
                wanted = 0o644 | (0o111 if mode & 0o111 else 0)
                try:
                    f.chmod(wanted)
                except OSError:
                    pass

    def verify(self, entry: Path, target: str) -> str:
        """'' when the staged tree is plausible on target: the expected
        main binary is present. No exec (cross-target), no arch probe --
        the digest already proved the bytes."""
        expected = entry / self.prefix_rel / self.main_rel(target)
        if not expected.is_file() and not expected.is_symlink():
            return f"{expected.relative_to(entry)} missing under {entry}"
        return ""

    def main_rel(self, target: str) -> str:
        raise NotImplementedError


class StatePackage(Package):
    """A package that is a STATE of this install (the python venv), not a
    store entry. Verified by comparing a stamp; made true by apply()."""

    on_path = False

    def expected_stamp(self, extras: list[str]) -> str:
        raise NotImplementedError

    def apply(self, extras: list[str]) -> None:
        raise NotImplementedError


def machine_matches_binary(binary: Path, target: str) -> Optional[bool]:
    """Does this executable's architecture match the target? Reads the
    PE/ELF/Mach-O header directly. None = unknown format (scripts, shims),
    which is not a mismatch."""
    import struct

    arch = target.split("-")[1]
    try:
        with open(binary, "rb") as f:
            head = f.read(64)
            if head[:2] == b"MZ":
                f.seek(int.from_bytes(head[60:64], "little"))
                sig = f.read(6)
                if sig[:4] != b"PE\0\0":
                    return None
                machine = int.from_bytes(sig[4:6], "little")
                return machine == {"x64": 0x8664, "arm64": 0xAA64}.get(arch)
            if head[:4] == b"\x7fELF":
                machine = int.from_bytes(head[18:20], "little")
                return machine == {"x64": 0x3E, "arm64": 0xB7}.get(arch)
            if head[:4] in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf"):
                cpu = struct.unpack("<I", head[4:8])[0]
                return cpu == {"x64": 0x01000007, "arm64": 0x0100000C}.get(arch)
            if head[:4] in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
                return True  # universal binary carries both
    except OSError:
        return None
    return None


def _entry_listing(entry: Path, limit: int = 12) -> str:
    """Top-level names of a store entry, for verification diagnoses."""
    if not entry.is_dir():
        return "store entry does not exist"
    names = sorted(p.name for p in entry.iterdir())
    shown = ", ".join(names[:limit])
    if len(names) > limit:
        shown += f", … ({len(names)} entries)"
    return shown


def _missing_reason(binary: Path, entry: Path) -> str:
    """Why a package's expected binary is not where it should be — the
    diagnosis that tells you whether the pin's layout is wrong."""
    rel = binary.relative_to(entry).as_posix()
    return f"{rel} missing under {entry}; {_entry_listing(entry)}"


def _probe_reason(binary: Path, proc: "subprocess.CompletedProcess") -> str:
    """Why a --version probe failed: the exit code plus output tail."""
    out = (proc.stdout or b"") + (proc.stderr or b"")
    tail = out.decode(errors="replace").strip()[-300:]
    return f"{binary} --version exited {proc.returncode}" + (f": {tail}" if tail else "")


class Runner:
    """What ensure() hands back: a composed environment and a run mirror."""

    def __init__(self, name: str, env: dict[str, str]):
        self.name = name
        self.env = env

    def run(self, cmd, **kwargs) -> subprocess.CompletedProcess:
        kwargs.setdefault("env", self.env)
        return subprocess.run(cmd, **kwargs)
