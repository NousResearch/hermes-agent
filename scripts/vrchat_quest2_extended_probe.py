"""Extended read-only VR stack probes for Quest2 + VD + VRChat diagnosis."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import winreg
from pathlib import Path


def read_reg(path: str, name: str = "") -> dict | None:
    hive_map = {
        "HKLM": winreg.HKEY_LOCAL_MACHINE,
        "HKCU": winreg.HKEY_CURRENT_USER,
    }
    parts = path.split("\\", 1)
    hive = hive_map.get(parts[0])
    if hive is None:
        return None
    subkey = parts[1]
    try:
        with winreg.OpenKey(hive, subkey) as key:
            if name:
                val, typ = winreg.QueryValueEx(key, name)
                return {"path": path, "name": name, "value": val, "type": typ}
            out = {}
            i = 0
            while True:
                try:
                    n, v, t = winreg.EnumValue(key, i)
                    out[n] = v
                    i += 1
                except OSError:
                    break
            return {"path": path, "values": out}
    except OSError as exc:
        return {"path": path, "error": str(exc)}


def main() -> int:
    openxr_paths = [
        r"HKLM\SOFTWARE\Khronos\OpenXR\1\ActiveRuntime",
        r"HKCU\SOFTWARE\Khronos\OpenXR\1\ActiveRuntime",
        r"HKLM\SOFTWARE\WOW6432Node\Khronos\OpenXR\1\ActiveRuntime",
    ]
    meta_paths = [
        r"HKLM\SOFTWARE\Oculus VR, LLC\Oculus",
        r"HKLM\SOFTWARE\Meta",
        r"HKLM\SOFTWARE\WOW6432Node\Oculus VR, LLC\Oculus",
    ]

    vrchat_dir = Path.home() / "AppData/LocalLow/VRChat/VRChat"
    config_json = vrchat_dir / "config.json"
    osc_dir = vrchat_dir / "OSC"

    report: dict = {
        "openxr": [read_reg(p) for p in openxr_paths],
        "meta_oculus_registry": [read_reg(p) for p in meta_paths],
        "vrchat_config_exists": config_json.is_file(),
        "vrchat_config_size": config_json.stat().st_size if config_json.is_file() else 0,
        "vrchat_osc_dir": str(osc_dir),
        "vrchat_osc_files": sorted(p.name for p in osc_dir.glob("*")) if osc_dir.is_dir() else [],
        "vrchat_top_level": sorted(p.name for p in vrchat_dir.iterdir()) if vrchat_dir.is_dir() else [],
    }

    if config_json.is_file():
        text = config_json.read_text(encoding="utf-8", errors="replace")
        keywords = ("osc", "input", "controller", "vr", "openxr", "steamvr")
        hits = []
        for i, line in enumerate(text.splitlines(), 1):
            low = line.lower()
            if any(k in low for k in keywords):
                hits.append({"line": i, "text": line.strip()[:200]})
        report["vrchat_config_keyword_hits"] = hits[:40]

    vd_candidates = [
        Path(r"C:\Program Files\Virtual Desktop Streamer\VirtualDesktop.Streamer.exe"),
        Path(r"C:\Program Files (x86)\Virtual Desktop Streamer\VirtualDesktop.Streamer.exe"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Virtual Desktop" / "VirtualDesktop.Streamer.exe",
    ]
    report["vd_streamer_paths"] = [
        {"path": str(p), "exists": p.is_file()} for p in vd_candidates
    ]

    # SteamVR OpenXR json
    svr_openxr = Path(r"C:\Program Files (x86)\Steam\steamapps\common\SteamVR\steamxr_win64.json")
    report["steamvr_openxr_manifest"] = {
        "path": str(svr_openxr),
        "exists": svr_openxr.is_file(),
        "preview": svr_openxr.read_text(encoding="utf-8", errors="replace")[:500] if svr_openxr.is_file() else None,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
