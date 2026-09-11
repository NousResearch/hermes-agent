#!/usr/bin/env python3
"""Container-side wheelhouse build: build the android build set from sdist
against THIS container's bionic python, retag, and run the offline
completeness + import gates. Invoked by termux_build.sh inside the
digest-pinned termux/termux-docker container.

Usage: build_wheels.py --resolved RESOLVED.txt --build-set BUILD_SET.txt \
           --wheelhouse DIR --retag RETAG_SCRIPT --platform-tag TAG

Everything here runs under the container's termux python (bionic): pip
builds sdists with clang/rust from $PREFIX, wheels land in the wheelhouse,
retagging stamps the PEP 738 android tag, and the --no-index gate proves the
FULL marker-admitted lock graph installs offline from the built bytes.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolved", required=True, help="tab-separated name/spec/marker file")
    p.add_argument("--build-set", required=True, help="names needing sdist builds (one per line)")
    p.add_argument("--wheelhouse", required=True)
    p.add_argument("--retag", required=True, help="path to retag_wheel.py")
    p.add_argument("--platform-tag", required=True)
    return p.parse_args()


def normalize_reqs(export: Path, lock: Path, resolved: Path) -> None:
    """Retain both the build source and the locked offline wheel version."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    rows = []
    for line in export.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        req = Requirement(line)
        name = canonicalize_name(req.name)
        spec = str(req.specifier)
        source = req.url or ""
        if source:
            repository, separator, commit = source.removeprefix("git+").rpartition("@")
            if not source.startswith("git+") or not separator or not re.fullmatch(r"[a-f0-9]{40}", commit):
                raise ValueError(f"Source requirement needs an exact Git commit: {name}")
            matches = [p for p in packages if canonicalize_name(p["name"]) == name
                       and (git := p.get("source", {}).get("git"))
                       and urlsplit(git).fragment == commit
                       and git.split("?", 1)[0].split("#", 1)[0] == repository]
            if len(matches) != 1:
                raise ValueError(f"Source requirement does not match one locked package: {name}")
            spec = f"=={matches[0]['version']}"
        rows.append((name, spec, str(req.marker) if req.marker else "", source))
    resolved.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")


def load_entries(resolved: Path) -> dict[str, str]:
    """Select build requirements using the payload's platform markers."""
    entries: dict[str, str] = {}
    for name, spec, marker, source in _parse_full(resolved):
        if _marker_admits(marker):
            if name in entries:
                raise ValueError(f"Duplicate applicable requirement: {name}")
            entries[name] = f" @ {source}" if source else spec.strip()
    return entries


def write_reqs_file(resolved: Path, reqs: Path) -> None:
    """name==spec[ ; marker] lines for a pip/uv -r file: markers
    stay attached so the INSTALLER evaluates them on the target platform
    (win32-marker deps resolve-skip on bionic); documented build misses
    are skipped (no offline solution exists for them, by design).
    """
    out = []
    for name, spec, marker, source in _parse_full(resolved):
        if name in BUILD_MISSES:
            continue
        line = f"{name}{spec.strip()}" if spec.strip() else name
        if marker:
            line += " ; " + marker
        out.append(line)
    body = chr(10).join(out) + chr(10)
    reqs.write_text(body, encoding="utf-8")


def safe_extract(archive: Path, dest: Path) -> Path:
    """Safe-extract a tarball, rejecting traversal/symlink/device members."""
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            path = PurePosixPath(member.name)
            parts = tuple(p for p in path.parts if p not in ("", "."))
            if path.is_absolute() or ".." in parts or not parts:
                raise RuntimeError(f"unsafe archive member path: {member.name!r}")
            target = dest.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(f"unsupported archive member type: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read archive member: {member.name}")
            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)
            try:
                target.chmod(member.mode & 0o777)
            except OSError:
                pass
    roots = sorted(p for p in dest.iterdir() if p.is_dir())
    return roots[0] if roots else dest


def _dist_version(name: str, spec: str) -> str | None:
    # Exact pins carry the version ("==X.Y.Z"); anything else cannot be
    # skip-checked safely, so it always builds.
    return spec.strip().lstrip("=") if spec.strip().startswith("==") else None


def build_wheels(build_set: list[str], specs: dict[str, str], wheelhouse: Path,
                 *, python: str = sys.executable) -> None:
    for name in build_set:
        spec = specs.get(name, "")
        req = f"{name}{spec}" if spec else name
        ver = _dist_version(name, spec)
        if ver is not None and any(
            wheelhouse.glob(f"{name.replace('-', '_')}-{ver}-*.whl")
        ):
            print(f"==> {name} {ver} already in the wheelhouse (cache restore); skipping")
            continue
        if name == "uvloop":
            print(f"==> building {name} (download + extract + pre-build fixups)")
            with tempfile.TemporaryDirectory(prefix=f"hermes-build-{name}-") as tmp:
                tmp = Path(tmp)
                sdist_dir = tmp / "sdist"
                sdist_dir.mkdir()
                subprocess.run(
                    [python, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:",
                     "--no-build-isolation", "-d", str(sdist_dir), req],
                    check=True, cwd=tmp,
                )
                archives = list(sdist_dir.glob("*.tar.gz"))
                if len(archives) != 1:
                    raise RuntimeError(f"expected exactly one sdist archive for {name}, got {len(archives)}")
                src = safe_extract(archives[0], tmp / "src")
                # uvloop vendors libuv at vendor/libuv/ WITH its own
                # autogen.sh + generated configure; the build backend
                # expects libuv CONFIGURED (build artifacts present)
                # before setup.py runs it. We run configure ourselves:
                # setup.py's own ['./configure'] invocation 127s on the
                # rewritten shebang (subprocess execv semantics), but a
                # completed configure leaves artifacts setup.py reuses.
                libuv = src / "vendor" / "libuv"
                if libuv.is_dir():
                    # bash-invoked: the sdist's autogen.sh may carry a
                    # non-exec mode (extraction preserves it). autogen
                    # also (re)writes ./configure WITHOUT the exec bit
                    # (setup.py then 127s) -- chmod after bootstrap.
                    proc = subprocess.run(["bash", "autogen.sh"], cwd=libuv,
                                          check=False, capture_output=True, text=True)
                    if proc.returncode == 0 and (libuv / "configure").is_file():
                        # Run configure under the container's own sh
                        # (its #!/bin/sh shebang can't exec here).
                        cfg_proc = subprocess.run(
                            [os.environ["PREFIX"] + "/bin/sh", "./configure"],
                            cwd=libuv, check=False, capture_output=True, text=True,
                        )
                        if cfg_proc.returncode != 0:
                            print(f"FIXUP FAILED (libuv configure) for {name}")
                            print("stdout:", cfg_proc.stdout[-1500:])
                            print("stderr:", cfg_proc.stderr[-1500:])
                            raise subprocess.CalledProcessError(cfg_proc.returncode, cfg_proc.args)
                    if proc.returncode != 0:
                        # autogen needs autoreconf when configure is
                        # stale; fall back to explicit bootstrap
                        proc2 = subprocess.run(["autoreconf", "-i"], cwd=libuv,
                                               check=False, capture_output=True, text=True)
                        if proc2.returncode != 0:
                            print(f"FIXUP FAILED (libuv bootstrap) for {name}")
                            print("autogen stdout:", proc.stdout[-1200:])
                            print("autogen stderr:", proc.stderr[-1200:])
                            print("autoreconf stderr:", proc2.stderr[-1200:])
                            raise subprocess.CalledProcessError(proc2.returncode, proc2.args)
                proc = subprocess.run(
                    [python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                     "-w", str(wheelhouse), str(src)],
                    check=False, cwd=tmp, capture_output=True, text=True,
                )
                if proc.returncode != 0:
                    print(f"BUILD FAILED: {name}")
                    print("stdout:", proc.stdout[-2000:])
                    print("stderr:", proc.stderr[-2000:])
                    raise subprocess.CalledProcessError(proc.returncode, proc.args)
        else:
            # Build isolation LEFT ON: --no-build-isolation requires every
            # sdist's declared backend (pdm, hatchling, maturin...) to be
            # pre-installed, and the lock graph uses more backends than the
            # pinned toolchain covers. The invariant is the USER machine
            # never resolves/compiles -- the build container may fetch.
            print(f"==> building {name} (direct pip wheel, isolated backend)")
            proc = subprocess.run(
                [python, "-m", "pip", "wheel", "--no-deps",
                 "--no-binary", ":all:", "-w", str(wheelhouse), req],
                check=False, capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(f"BUILD FAILED: {name}")
                print("stdout:", proc.stdout[-2000:])
                print("stderr:", proc.stderr[-2000:])
                raise subprocess.CalledProcessError(proc.returncode, proc.args)


TARGET_ENV = {
    "implementation_name": "cpython",
    "implementation_version": "3.14.6",
    "os_name": "posix",
    "platform_machine": "aarch64",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.14.6",
    "python_version": "3.14",
    # 3.13+ Android CPython reports sys.platform "android" (docs: changed in
    # 3.13), so markers keying on `sys_platform == 'linux'` no longer admit
    # the termux target; `platform_system` stays "Linux" (Android kernel).
    "sys_platform": "android",
}

# Mirrors the documented misses in termux_build.sh's probe (single
# source is ideal; duplicated until the probe emits a sidecar file).
BUILD_MISSES = {"nemo-relay"}


def _marker_admits(marker: str) -> bool:
    if not marker:
        return True
    from packaging.markers import Marker
    return Marker(marker).evaluate(TARGET_ENV)


def _parse_full(resolved: Path) -> list[tuple[str, str, str, str]]:
    """Name, wheel version, marker and optional build source for each entry."""
    out: list[tuple[str, str, str, str]] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ValueError("Expected name, version, marker and source fields")
        out.append((parts[0], parts[1], parts[2], parts[3]))
    return out


def fetch_pure_wheels(build_set: list[str], specs: dict[str, str], wheelhouse: Path, resolved: Path) -> None:
    """Fetch every APPLICABLE dep's py3-none-any wheel into the wheelhouse.

    The offline completeness gate installs the FULL marker-admitted graph
    from --find-links wheelhouse under --no-index; the natives were built
    here, but the pure wheels live only on PyPI. Fetch them (the build
    container may resolve; the USER machine never does). Wheels already
    present (built natives) are skipped by pip download naturally.
    Marker-excluded deps (pywin32, winrt-*) and documented build misses
    (nemo-relay) are skipped -- the gate's reqs carry the same exclusions
    via the venv's own marker evaluation... which they do NOT: write_reqs_file
    writes the full list and uv evaluates markers from the RUNNING platform.
    The gate reqs keep all lines; pip/uv on bionic skips the win32 ones.
    """
    reqs = []
    for name, spec, marker, source in _parse_full(resolved):
        if name in build_set or name in BUILD_MISSES:
            continue
        if not _marker_admits(marker):
            continue
        reqs.append(f"{name}{spec.strip()}" if spec.strip() else name)
    # Build-backend deps the deb-venv's app install resolves offline:
    # the app dir install (even --no-deps) prepares metadata through
    # build-system.requires, which needs setuptools in the find-links.
    reqs.append("setuptools==83.0.0")
    if not reqs:
        return
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps",
         "--only-binary", ":all:", "-d", str(wheelhouse), *reqs],
        check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("PURE-WHEEL FETCH FAILED")
        print("stdout:", proc.stdout[-1500:])
        print("stderr:", proc.stderr[-1500:])
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    fetched = len(list(wheelhouse.glob("*.whl")))
    print(f"  pure wheels fetched; wheelhouse now holds {fetched} wheels")


def retag_all(wheelhouse: Path, retag_script: Path, platform_tag: str) -> None:
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"no wheels built into {wheelhouse}")
    # maturin-built wheels (cryptography, pydantic-core, jiter, watchfiles)
    # emerge ANDROID-tagged already (ANDROID_API_LEVEL drives the build);
    # only the setuptools/distutils-built ones carry linux_aarch64 tags and
    # need the retag. Retagging an android wheel again is a hard error.
    linux_tagged = [w for w in wheels if "linux" in w.name and "none-any" not in w.name]
    android_tagged = [w for w in wheels if "android" in w.name]
    if linux_tagged:
        subprocess.run(
            [sys.executable, str(retag_script), "--platform-tag", platform_tag, *linux_tagged],
            check=True,
        )
    retagged = list(wheelhouse.glob(f"*{platform_tag}*.whl"))
    total_android = len(retagged)
    if total_android == 0:
        raise RuntimeError(f"no wheels retagged to {platform_tag}")
    print(f"  retagged {len(linux_tagged)} linux-tagged wheels; {len(android_tagged)} were android-tagged at build; {total_android} android wheels total")


def import_native_modules(build_set: list[str]) -> None:
    import importlib

    modules = {
        "ruamel-yaml-clib": "_ruamel_yaml", "cffi": "_cffi_backend",
        "pillow": "PIL._imaging", "pyyaml": "yaml._yaml", "firecrawl-anydoc": "anydoc",
    }
    for name in build_set:
        module = modules.get(name, name.replace("-", "_"))
        importlib.import_module(module)
        print("  imported", module, flush=True)


def wheelhouse_gates(resolved: Path, wheelhouse: Path, build_set: list[str]) -> None:
    """Offline install of every marker-admitted dep into ONE clean venv,
    shared by both gates (the full offline install runs once, not twice).

    1. completeness: --no-index makes completeness a build-time invariant
       -- a dep missing from the wheelhouse fails loudly. The container's
       own bionic python evaluates the markers exactly as the phone will
       (sys_platform/platform_machine are real here), so deps the android
       target excludes (pywin32, winrt-*) are skipped by pip itself.
    2. imports: every NATIVE wheel we built must import (the py3-none-any
       wheels are PyPI bytes; OUR builds are what this gate proves).
    """
    with tempfile.TemporaryDirectory(prefix="hermes-wheelhouse-gate-") as tmp:
        tmp = Path(tmp)
        from pm import build_requirements_environment

        reqs = tmp / "reqs.txt"
        write_reqs_file(resolved, reqs)
        vp = build_requirements_environment(
            reqs.read_text(encoding="utf-8").splitlines(), out=tmp / "venv",
            python=Path(sys.executable), wheelhouse=wheelhouse, offline=True, explicit=True,
        )
        print("  completeness gate: offline install of the marker-admitted graph OK")
        subprocess.run(
            [str(vp), str(Path(__file__).resolve()), "--import-modules", *build_set],
            check=True,
        )
        print("  import gate: every built native wheel imports")



def main() -> int:
    if sys.argv[1:2] == ["--normalize"]:
        normalize_reqs(*(Path(value) for value in sys.argv[2:]))
        return 0
    if sys.argv[1:2] == ["--import-modules"]:
        import_native_modules(sys.argv[2:])
        return 0
    args = parse_args()
    resolved = Path(args.resolved)
    build_set = [l.strip() for l in Path(args.build_set).read_text(encoding="utf-8").splitlines() if l.strip()]
    wheelhouse = Path(args.wheelhouse)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    specs = load_entries(resolved)

    build_wheels(build_set, specs, wheelhouse)
    fetch_pure_wheels(build_set, specs, wheelhouse, Path(args.resolved))
    retag_all(wheelhouse, Path(args.retag), args.platform_tag)
    from python_linkage import repair_wheel

    library = Path(sys.base_prefix) / "lib" / sysconfig.get_config_var("LDLIBRARY")
    if not library.is_file():
        raise RuntimeError(f"payload libpython missing: {library}")
    for wheel in sorted(wheelhouse.glob("*.whl")):
        repaired = repair_wheel(wheel, library)
        if repaired:
            print(f"  linked {repaired} native extensions to {library.name}: {wheel.name}")
    wheelhouse_gates(resolved, wheelhouse, build_set)
    print(f"wheelhouse complete: {len(list(wheelhouse.glob('*.whl')))} wheels in {wheelhouse}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
