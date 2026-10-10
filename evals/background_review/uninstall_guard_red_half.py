"""Red half of the card-``t_0cf0aa7e`` guard: the same contract against the PRE-FIX module.

``tests/hermes_cli/test_uninstall_windows_registry_isolation.py`` asserts that, under test
isolation, the uninstaller's two ``HKCU\\Environment`` mutators never open the registry. A
green run only means something if the same call fails on a tree without the guard -- so this
probe builds that tree's module *in memory*:

1. read the current ``hermes_cli/uninstall.py`` and strip the two guard blocks
   (``if os.environ.get("HERMES_TEST_ISOLATION"): return []``);
2. import the stripped copy as its own module and call both mutators with the REAL default
   home (the dangerous configuration), with ``winreg.OpenKey`` refusing to open anything;
3. report which calls were refused -- i.e. which functions the guard is what makes inert.

The stripped copy is derived from the current file rather than from a git rev, so the probe
stays reproducible after the guard is committed (a rev pin would expire at that commit).

Nothing is ever written: ``OpenKey`` raises before any read or write can happen, and the real
stored value is snapshotted before and after.

Usage:  python evals/background_review/uninstall_guard_red_half.py [--out out.json]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import types
import winreg
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "hermes_cli" / "uninstall.py"
sys.path.insert(0, str(REPO))

GUARD_BLOCK = '    if os.environ.get("HERMES_TEST_ISOLATION"):\n        return []\n'

OUT = None
for i, a in enumerate(sys.argv):
    if a == "--out" and i + 1 < len(sys.argv):
        OUT = sys.argv[i + 1]


def raw_user_path() -> tuple[str, int]:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_QUERY_VALUE) as k:
        value, kind = winreg.QueryValueEx(k, "Path")
    return str(value), int(kind)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def real_default_home() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "hermes"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


BEFORE, BEFORE_KIND = raw_user_path()
source = SOURCE.read_text(encoding="utf-8")
stripped = source.replace(GUARD_BLOCK, "")
guard_blocks = source.count(GUARD_BLOCK)
stripped_blocks = stripped.count(GUARD_BLOCK)

scratch = Path(tempfile.mkdtemp(prefix="hermes-uninstall-guard-"))
pre_path = scratch / "uninstall_pre.py"
pre_path.write_text(stripped, encoding="utf-8", newline="\n")

pre = load_module(pre_path, "uninstall_pre")
fixed = load_module(SOURCE, "uninstall_fixed")

report: dict = {
    "repo": str(REPO),
    "source": str(SOURCE),
    "guard_blocks_in_current_file": guard_blocks,
    "guard_blocks_in_stripped_copy": stripped_blocks,
    "stripped_copy": str(pre_path),
    "stripped_sha16": sha(stripped),
    "real_default_home": str(real_default_home()),
    "calls": [],
}


def call(label: str, module, fn_name: str, *args, isolation: bool, **kwargs) -> dict:
    """Call ``fn`` with every open of the registry refused. Report what happened.

    ``isolation`` sets/clears ``HERMES_TEST_ISOLATION``, so the four combinations (guard
    present or stripped) x (marker set or not) can be read side by side.
    """
    refused: list[str] = []
    real_open = winreg.OpenKey

    def refuse(*a, **k):
        refused.append("OpenKey")
        raise AssertionError("HKCU\\Environment opened during a test run")

    if isolation:
        os.environ["HERMES_TEST_ISOLATION"] = str(scratch)
    else:
        os.environ.pop("HERMES_TEST_ISOLATION", None)
    winreg.OpenKey = refuse  # type: ignore[assignment]
    try:
        try:
            result = getattr(module, fn_name)(*args, **kwargs)
            outcome, detail = "returned", repr(result)
        except AssertionError as exc:
            outcome, detail = "refused", str(exc)
        except Exception as exc:  # health: allow BLE001 -- 探针要把「这一步炸了」如实记进证据，不能因为被测函数抛了别的异常就把 4 组合证据整轮丢掉
            outcome, detail = f"raised {type(exc).__name__}", str(exc)
    finally:
        winreg.OpenKey = real_open  # type: ignore[assignment]
        os.environ.pop("HERMES_TEST_ISOLATION", None)
    return {
        "call": label,
        "HERMES_TEST_ISOLATION": isolation,
        "outcome": outcome,
        "detail": detail,
        "registry_opens_refused": len(refused),
    }


home = real_default_home()
report["calls"] = [
    call("PRE-FIX (guard stripped) remove_path_from_windows_registry(real root, +managed bin), no marker",
         pre, "remove_path_from_windows_registry", home, isolation=False, include_managed_bin=True),
    call("PRE-FIX (guard stripped) remove_path_from_windows_registry(real root, +managed bin), marker set",
         pre, "remove_path_from_windows_registry", home, isolation=True, include_managed_bin=True),
    call("PRE-FIX (guard stripped) remove_hermes_env_vars_windows(), marker set",
         pre, "remove_hermes_env_vars_windows", isolation=True),
    call("FIXED   remove_path_from_windows_registry(real root, +managed bin), marker set",
         fixed, "remove_path_from_windows_registry", home, isolation=True, include_managed_bin=True),
    call("FIXED   remove_hermes_env_vars_windows(), marker set",
         fixed, "remove_hermes_env_vars_windows", isolation=True),
    call("FIXED   remove_path_from_windows_registry(real root, +managed bin), NO marker (a real uninstall)",
         fixed, "remove_path_from_windows_registry", home, isolation=False, include_managed_bin=True),
]

AFTER, AFTER_KIND = raw_user_path()
report["real_path"] = {
    "before": {"chars": len(BEFORE), "kind": BEFORE_KIND, "sha16": sha(BEFORE)},
    "after": {"chars": len(AFTER), "kind": AFTER_KIND, "sha16": sha(AFTER)},
    "identical": (BEFORE == AFTER and BEFORE_KIND == AFTER_KIND),
}

if OUT:
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

print(f"guard blocks: {guard_blocks} in the current file, {stripped_blocks} after stripping")
print(f"stripped copy: {pre_path}")
print()
for row in report["calls"]:
    print(f"  {row['call']}")
    print(f"    -> {row['outcome']}: {row['detail'][:110]}")
print()
print(f"real HKCU\\Environment\\Path: {report['real_path']['before']['sha16']} ->"
      f" {report['real_path']['after']['sha16']}"
      f"  identical={report['real_path']['identical']}")
if OUT:
    print(f"\njson: {OUT}")
