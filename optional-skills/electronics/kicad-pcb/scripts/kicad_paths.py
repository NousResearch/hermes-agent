#!/usr/bin/env python3
"""Cross-platform path resolution for KiCad/Freerouting tools — no pcbnew import needed.

Used by demo_autoroute.py (re-exec shim + pipeline) and check_env.py (preflight).
Resolution order in every function: environment variable -> known per-OS locations -> PATH.
All pathlib; no hardcoded separators in strings.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

IS_WIN = os.name == "nt"
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

FOOTPRINT_PROBE = "Resistor_SMD.pretty"  # marker that official libs are installed


class ResolveError(Exception):
    """Component not found — .fix carries the per-OS repair instruction."""

    def __init__(self, component: str, fix: str):
        self.component, self.fix = component, fix
        super().__init__(f"{component}: not found. {fix}")


# ---------------------------------------------------------------- python+pcbnew

def python_with_pcbnew_candidates() -> list[Path]:
    """Interpreters that are likely to have the pcbnew bindings."""
    out: list[Path] = []
    env = os.environ.get("KICAD_PYTHON")
    if env:
        out.append(Path(env))
    if IS_WIN:
        for pat in (
            r"C:\Program Files\KiCad\*\bin\python.exe",
            str(Path.home() / r"AppData\Local\Programs\KiCad\*\bin\python.exe"),
        ):
            out += sorted(Path(p).resolve() for p in __import__("glob").glob(pat))
    elif IS_MAC:
        out += sorted(Path("/Applications/KiCad").glob(
            "KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3"))
    else:  # linux: distro/PPA packages install pcbnew into the system python
        out.append(Path(sys.executable))
        out += [Path(p) for p in ("/usr/bin/python3",)]
    return out


def verify_pcbnew(python: Path, timeout: int = 60) -> str | None:
    """Runs `import pcbnew` in the interpreter; returns its version or None."""
    try:
        r = subprocess.run(
            [str(python), "-c", "import pcbnew; print(pcbnew.GetBuildVersion())"],
            capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok"
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


# ---------------------------------------------------------------- footprints

def footprints_dir() -> Path:
    """Official footprint libraries dir (contains Resistor_SMD.pretty)."""
    envs = ("KICAD10_FOOTPRINT_DIR", "KICAD9_FOOTPRINT_DIR", "KICAD_FOOTPRINT_DIR")
    for e in envs:
        if os.environ.get(e):
            p = Path(os.environ[e])
            if (p / FOOTPRINT_PROBE).is_dir():
                return p
    roots: list[Path] = [Path("/usr/share/kicad"),
                         Path("/usr/local/share/kicad"),
                         Path.home() / ".local/share/kicad"]
    if IS_WIN:
        # "+ '/'" anchors the drive: a bare Path("C:") is drive-RELATIVE on Windows
        roots = [Path(os.environ.get("SystemDrive", "C:") + "/") / "Program Files" / "KiCad",
                 Path.home() / "AppData" / "Local" / "Programs" / "KiCad"]
    elif IS_MAC:
        roots = [Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport")]
    # expand one version level (KiCad/10.0, kicad/10.0) + the root itself
    bases: list[Path] = []
    for r in roots:
        if r.is_dir():
            bases += sorted(r.glob("*")) + [r]
    for b in bases:
        for p in (b / "share" / "kicad" / "footprints", b / "footprints"):
            if (p / FOOTPRINT_PROBE).is_dir():
                return p
    raise ResolveError("footprints dir", "install the official KiCad 10 libraries and/or set "
                       f"{'|'.join(envs)}. Roots searched: " + ", ".join(str(r) for r in roots))


# ---------------------------------------------------------------- kicad-cli

def kicad_cli() -> Path:
    env = os.environ.get("KICAD_CLI")
    if env and Path(env).exists():
        return Path(env)
    if IS_WIN:
        hits: list[Path] = []
        for root in (Path("C:/Program Files/KiCad"),
                     Path.home() / "AppData/Local/Programs/KiCad"):
            if root.is_dir():
                hits.extend(root.glob("*/bin/kicad-cli.exe"))
        if hits:
            # Installed numeric version takes priority over a stale PATH entry.
            return max(hits, key=lambda p: tuple(int(n) for n in
                       re.findall(r"\d+", p.parent.parent.name)) or (0,))
    if IS_MAC:
        p = Path("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli")
        if p.exists():
            return p
    w = shutil.which("kicad-cli.exe") or shutil.which("kicad-cli")
    if w:
        return Path(w)
    raise ResolveError("kicad-cli", "install KiCad 10 (kicad.kicad.org) or set KICAD_CLI")


# ---------------------------------------------------------------- java

def java_version(java: Path) -> int | None:
    try:
        r = subprocess.run([str(java), "-version"], capture_output=True, text=True, timeout=30)
        m = re.search(r'version "(\d+)', r.stderr + r.stdout)
        return int(m.group(1)) if m else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def java(min_major: int = 25) -> Path:
    """Java >= min_major (freerouting-2.4.1.jar = class file 69 => Java 25)."""
    w = shutil.which("java.exe") or shutil.which("java")
    if w:
        p = Path(w)
        if (v := java_version(p)) and v >= min_major:
            return p
    raise ResolveError(
        f"java >= {min_major}",
        f"the Freerouting 2.4.1 jar requires Java {min_major}+. Recommended: use the OS bundle "
        f"with embedded runtime (freerouting-*-linux-x64.zip / -windows-x64.msi) — no system "
        f"Java needed. Alternative: install Eclipse Temurin {min_major} (adoptium.net).")


# ---------------------------------------------------------------- freerouting

def freerouting() -> tuple[str, list[str]]:
    """Returns (mode, argv): mode 'exe' (bundle launcher/PATH) or 'jar' (java -jar ...)."""
    if os.environ.get("FREEROUTING_EXE"):
        p = Path(os.environ["FREEROUTING_EXE"])
        if p.exists():
            return "exe", [str(p)]
    if os.environ.get("FREEROUTING_JAR"):
        j = Path(os.environ["FREEROUTING_JAR"])
        if j.exists():
            return "jar", [str(java()), "-jar", str(j)]

    home = Path.home()
    exe_globs: list[Path] = []
    if IS_WIN:
        # MSI 2.4.1 default is a per-user jpackage app-image in AppData\Local\freerouting
        exe_globs += [Path("C:/Program Files/Freerouting/**/freerouting.exe"),
                      home / "AppData/Local/freerouting/**/freerouting.exe",
                      home / "AppData/Local/Programs/Freerouting/**/freerouting.exe",
                      home / "Work/tools/freerouting*/**/freerouting.exe"]
    elif IS_MAC:
        exe_globs += [Path("/Applications/Freerouting.app/Contents/MacOS/*"),
                      home / "Applications/Freerouting.app/Contents/MacOS/*",
                      home / "Work/tools/freerouting*/**/freerouting"]
    else:
        exe_globs += [home / "Work/tools/freerouting*/bin/freerouting",
                      home / "Work/tools/freerouting*/**/freerouting",
                      Path("/opt/freerouting*/bin/freerouting")]
    import glob as _g
    for pat in exe_globs:
        for hit in sorted(_g.glob(str(pat), recursive=True)):
            if hit.lower().endswith((".exe", "freerouting")) and Path(hit).is_file():
                try:  # unzipped bundles may come without the exec bit
                    Path(hit).chmod(Path(hit).stat().st_mode | 0o111)
                except OSError:
                    pass
                return "exe", [hit]
    w = shutil.which("freerouting")
    if w:
        return "exe", [w]

    for jar_pat in (home / "Work/tools/freerouting*.jar",
                    Path.cwd() / "freerouting*.jar",
                    home / "Downloads/freerouting*.jar"):
        hits = sorted(jar_pat.parent.glob(jar_pat.name))
        if hits:
            return "jar", [str(java()), "-jar", str(hits[-1])]
    raise ResolveError(
        "freerouting",
        "download the OS bundle with embedded runtime from "
        "https://github.com/freerouting/freerouting/releases (linux-x64.zip | windows-x64.msi | "
        "macos-*.dmg), extract it under ~/Work/tools/ (default searched location) or set "
        "FREEROUTING_EXE / FREEROUTING_JAR.")


if __name__ == "__main__":  # quick debug: python3 kicad_paths.py
    for fn in (footprints_dir, kicad_cli, java, freerouting):
        try:
            print(f"{fn.__name__}: {fn()}")
        except ResolveError as e:
            print(f"{fn.__name__}: FAILED — {e}")
