"""What does ``hermes_cli/uninstall.py::remove_path_from_windows_registry`` actually do?

Card ``t_0cf0aa7e``. The third persistent-PATH write point: the *deleter* (it strips
Hermes-owned prefixes from ``HKCU\\Environment\\Path``), the only one of the three that had
no ``HERMES_TEST_ISOLATION`` guard before this card.

Two module variants are loaded side by side, because once the guard lands a single variant
can no longer answer the question:

* **live** -- ``hermes_cli/uninstall.py`` as it is on disk (guarded);
* **pre-fix** -- the same file with the two ``HERMES_TEST_ISOLATION`` guard blocks stripped.
  Built in memory rather than from a git rev, so the probe stays reproducible after the
  guard is committed (a rev pin would expire at that commit).

Two kinds of evidence, deliberately separated:

* REAL arms -- the real function, the real registry key, no interception. Their job is to
  show that the shape pytest creates is a **no-op with and without the guard**: ``removed``
  comes back empty, ``SetValueEx`` is never reached, and the stored value + its type stay
  byte-identical. Any arm whose markers *would* match a real entry is SKIPPED, never run --
  that is the destructive configuration (real root + ``include_managed_bin=True``), and it
  deletes the operator's ``hermes`` launcher entry.
* SYNTHETIC arms -- the real ``edit()`` closure and the real marker derivation, but the
  registry is a fake module in ``sys.modules`` and the input value is injected. This is the
  only way to answer "which entry would it delete, under which configuration?" without ever
  putting a real entry at risk.

The synthetic input value defaults to the REAL stored PATH read at start, so the arms show
the per-entry diff on the machine's actual data rather than on invented strings.

Read-only against the registry: the REAL arms open the key with KEY_WRITE, but the only
``SetValueEx`` call site sits behind ``if removed:``, each arm is pre-checked to be a no-op,
and the probe re-reads the raw value + type after every arm (restoring from the snapshot and
aborting if anything ever changed).

Usage:  python evals/background_review/uninstall_path_stripper_probe.py [--out out.json]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import winreg
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = None
for i, a in enumerate(sys.argv):
    if a == "--out" and i + 1 < len(sys.argv):
        OUT = sys.argv[i + 1]

GUARD_BLOCK = '    if os.environ.get("HERMES_TEST_ISOLATION"):\n        return []\n'
SOURCE = REPO / "hermes_cli" / "uninstall.py"
REAL_DEFAULT_HOME = Path(os.environ.get("LOCALAPPDATA", "")) / "hermes"
SANDBOX = Path(tempfile.mkdtemp(prefix="hermes-uninstallprobe-")) / "hermes_test"


def reg_query_raw() -> str:
    """The two ``reg query`` originals the card asks for, byte-for-byte."""
    p = subprocess.run(["reg.exe", "query", r"HKCU\Environment", "/v", "Path"], capture_output=True,
                       timeout=30)
    return p.stdout.decode("utf-8", "replace")


def raw_user_path() -> tuple[str, int]:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_QUERY_VALUE) as k:
        value, kind = winreg.QueryValueEx(k, "Path")
    return str(value), int(kind)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def entries(value: str) -> list[str]:
    return [e for e in value.split(";") if e]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


source_text = SOURCE.read_text(encoding="utf-8")
stripped_text = source_text.replace(GUARD_BLOCK, "")
guard_blocks = source_text.count(GUARD_BLOCK)
assert stripped_text != source_text, "no guard block found to strip -- did the guard move?"

pre_path = Path(tempfile.mkdtemp(prefix="hermes-uninstallpre-")) / "uninstall_pre.py"
pre_path.write_text(stripped_text, encoding="utf-8", newline="\n")
pre = load_module(pre_path, "uninstall_pre")
live = load_module(SOURCE, "uninstall_live")

BEFORE_VALUE, BEFORE_KIND = raw_user_path()
BEFORE_RAW = reg_query_raw()

report: dict = {
    "repo": str(REPO),
    "guard_blocks_in_live_file": guard_blocks,
    "stripped_copy": str(pre_path),
    "real_path": {"chars": len(BEFORE_VALUE), "entries": len(entries(BEFORE_VALUE)),
                  "kind": BEFORE_KIND, "sha16": sha(BEFORE_VALUE)},
    "sandbox_home": str(SANDBOX),
    "real_default_home": str(REAL_DEFAULT_HOME),
    "arms_real": [],
    "arms_synthetic": [],
}


def predicted(root: Path, flag: bool, value: str) -> list[str]:
    """Which entries the marker set would remove -- computed without touching the key."""
    markers = tuple(m.lower() for m in live._hermes_path_markers(root, include_managed_bin=flag))
    return [e for e in entries(value) if e.rstrip("\\/").lower().startswith(markers)]


def real_arm(label: str, module, root: Path, flag: bool, *, marker: bool) -> None:
    """Run the real function against the REAL key -- only when provably a no-op."""
    would = predicted(root, flag, BEFORE_VALUE)
    if would:
        report["arms_real"].append({
            "arm": label, "root": str(root), "include_managed_bin": flag,
            "HERMES_TEST_ISOLATION": marker, "outcome": "SKIPPED (destructive)",
            "would_remove": would,
            "why": "the markers match real PATH entries; running it would delete them",
        })
        return
    if marker:
        os.environ["HERMES_TEST_ISOLATION"] = str(SANDBOX)
    else:
        os.environ.pop("HERMES_TEST_ISOLATION", None)
    before = raw_user_path()
    try:
        removed = module.remove_path_from_windows_registry(root, include_managed_bin=flag)
        error = None
    except Exception as exc:  # health: allow BLE001 -- 探针要把「这一步炸了」如实记进证据，不能因为被测函数抛了别的异常就把整轮 REAL arm 证据丢掉
        removed, error = None, f"{type(exc).__name__}: {exc}"
    finally:
        os.environ.pop("HERMES_TEST_ISOLATION", None)
    after = raw_user_path()
    if after != before:  # never expected -- restore from the snapshot and stop
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, "Path", 0, BEFORE_KIND, BEFORE_VALUE)
        raise SystemExit(f"!! REGISTRY CHANGED in {label} -- restored, stopping")
    report["arms_real"].append({
        "arm": label, "root": str(root), "include_managed_bin": flag,
        "HERMES_TEST_ISOLATION": marker, "outcome": "ran",
        "markers": live._hermes_path_markers(root, include_managed_bin=flag),
        "removed": removed, "error": error, "registry_unchanged": True,
        "before": {"chars": len(before[0]), "kind": before[1], "sha16": sha(before[0])},
        "after": {"chars": len(after[0]), "kind": after[1], "sha16": sha(after[0])},
    })


for label, module, root, flag, marker in (
    ("R1 live    sandbox root, flag=False, marker set (pytest shape)", live, SANDBOX, False, True),
    ("R2 live    sandbox root, flag=True,  marker set (pytest shape)", live, SANDBOX, True, True),
    ("R3 pre-fix sandbox root, flag=False, marker set (pytest shape)", pre, SANDBOX, False, True),
    ("R4 pre-fix sandbox root, flag=True,  marker set (pytest shape)", pre, SANDBOX, True, True),
    ("R5 pre-fix REAL root,    flag=False, no marker", pre, REAL_DEFAULT_HOME, False, False),
    ("R6 live    REAL root,    flag=True,  marker set (what the guard buys)",
     live, REAL_DEFAULT_HOME, True, True),
    ("R7 pre-fix REAL root,    flag=True,  marker set", pre, REAL_DEFAULT_HOME, True, True),
    ("R8 pre-fix REAL root,    flag=True,  no marker (a real uninstall)",
     pre, REAL_DEFAULT_HOME, True, False),
):
    real_arm(label, module, root, flag, marker=marker)


def synthetic(label: str, module, root: Path, flag: bool, value: str, *, marker: bool = False) -> None:
    """Real ``edit()`` closure + real markers, fake ``winreg``, injected value."""
    recorded: list[str] = []
    fake = types.ModuleType("winreg")
    for name, attr in (("HKEY_CURRENT_USER", winreg.HKEY_CURRENT_USER), ("Environment", "Environment"),
                       ("KEY_READ", winreg.KEY_READ), ("KEY_WRITE", winreg.KEY_WRITE),
                       ("REG_SZ", winreg.REG_SZ), ("REG_EXPAND_SZ", winreg.REG_EXPAND_SZ)):
        setattr(fake, name, attr)
    setattr(fake, "QueryValueEx", lambda key, name: (value, winreg.REG_EXPAND_SZ))
    setattr(fake, "SetValueEx", lambda key, name, reserved, kind, new: recorded.append(str(new)))

    class _Key:
        def __enter__(self):
            return object()

        def __exit__(self, *exc):
            return False

    setattr(fake, "OpenKey", lambda *a, **k: _Key())

    real = sys.modules["winreg"]
    sys.modules["winreg"] = fake
    if marker:
        os.environ["HERMES_TEST_ISOLATION"] = str(SANDBOX)
    else:
        os.environ.pop("HERMES_TEST_ISOLATION", None)
    try:
        removed = module.remove_path_from_windows_registry(root, include_managed_bin=flag)
    finally:
        sys.modules["winreg"] = real
        os.environ.pop("HERMES_TEST_ISOLATION", None)

    before_entries = entries(value)
    after_entries = entries(recorded[0]) if recorded else before_entries
    report["arms_synthetic"].append({
        "arm": label, "root": str(root), "include_managed_bin": flag,
        "HERMES_TEST_ISOLATION": marker, "writes": len(recorded),
        "entries_before": len(before_entries), "entries_after": len(after_entries),
        "removed": removed,
        "lost": [e for e in before_entries if e not in after_entries],
    })


LEGACY = [str(REAL_DEFAULT_HOME) + r"\git\cmd", str(REAL_DEFAULT_HOME) + r"\node"]
LEAK = str(SANDBOX) + r"\bin"

synthetic("S1 pre-fix REAL root,    flag=True,  REAL PATH value", pre, REAL_DEFAULT_HOME, True, BEFORE_VALUE)
synthetic("S2 pre-fix REAL root,    flag=False, REAL PATH value", pre, REAL_DEFAULT_HOME, False, BEFORE_VALUE)
synthetic("S3 pre-fix REAL root,    flag=True,  REAL value + legacy git\\cmd/node", pre, REAL_DEFAULT_HOME,
          True, BEFORE_VALUE + ";" + ";".join(LEGACY))
synthetic("S4 pre-fix sandbox root, flag=True,  REAL value + sandbox\\bin leak", pre, SANDBOX, True,
          BEFORE_VALUE + ";" + LEAK)
synthetic("S5 live    REAL root,    flag=True,  REAL value, marker set", live, REAL_DEFAULT_HOME, True,
          BEFORE_VALUE, marker=True)

AFTER_VALUE, AFTER_KIND = raw_user_path()
AFTER_RAW = reg_query_raw()
report["real_path_after_probe"] = {
    "chars": len(AFTER_VALUE), "entries": len(entries(AFTER_VALUE)), "kind": AFTER_KIND,
    "sha16": sha(AFTER_VALUE),
    "identical": (AFTER_VALUE, AFTER_KIND) == (BEFORE_VALUE, BEFORE_KIND),
    "reg_query_byte_identical": AFTER_RAW == BEFORE_RAW,
}

if OUT:
    out_dir = Path(OUT).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    (out_dir / "regquery-BEFORE.txt").write_bytes(BEFORE_RAW.encode("utf-8"))
    (out_dir / "regquery-AFTER.txt").write_bytes(AFTER_RAW.encode("utf-8"))

print(f"guard blocks in the live file: {guard_blocks}; pre-fix copy: {pre_path}")
print(f"REAL PATH: {report['real_path']['chars']} chars / {report['real_path']['entries']} entries"
      f" / kind={BEFORE_KIND} / sha16={sha(BEFORE_VALUE)}")
print()
print("REAL ARMS (real function, real registry, no interception)")
for a in report["arms_real"]:
    print(f"  {a['arm']}")
    if a["outcome"] == "SKIPPED (destructive)":
        print(f"      -> SKIPPED; would have removed {a['would_remove']}")
    else:
        print(f"      -> removed={a['removed']}  registry_unchanged={a['registry_unchanged']}"
              f"  sha {a['before']['sha16']} -> {a['after']['sha16']}")
print()
print("SYNTHETIC ARMS (real edit() logic, injected value, SetValueEx recorded)")
for a in report["arms_synthetic"]:
    print(f"  {a['arm']}")
    print(f"      writes={a['writes']}  entries {a['entries_before']}->{a['entries_after']}"
          f"  removed={a['removed']}")
    for lost in a["lost"]:
        print(f"        WOULD LOSE: {lost}")
print()
print(f"REAL PATH after probe: {report['real_path_after_probe']['chars']} chars /"
      f" {report['real_path_after_probe']['entries']} entries /"
      f" sha16={report['real_path_after_probe']['sha16']}"
      f"  identical={report['real_path_after_probe']['identical']}"
      f"  reg_query_byte_identical={report['real_path_after_probe']['reg_query_byte_identical']}")
if OUT:
    print(f"\njson: {OUT}")
