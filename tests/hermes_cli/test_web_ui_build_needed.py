"""Regression test for HOTFIX-HERMES-UPDATER-01: web UI build freshness.

Covers the content-hash based skip logic so a ``git pull``/``npm`` mtime
rewrite can no longer skip a needed dashboard rebuild:

  - ``_web_ui_build_input_hash(web_dir)``
  - ``_web_ui_build_stamp_path(web_dir)``
  - ``_web_ui_build_needed(web_dir, *, force)``
  - ``_write_web_ui_build_stamp(web_dir)``

Strategy: load ``hermes_cli/main.py`` via importlib against a small temp
fixture (PROJECT_ROOT redirected) so the functions run without the heavy
module startup. Falls back to py_compile only if the import itself fails.
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO = (THIS_DIR / ".." / "..").resolve()
MAIN = REPO / "hermes_cli" / "main.py"

failures = []
checks = 0


def check(name, cond):
    global checks
    checks += 1
    if cond:
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name}")
        failures.append(name)


# --- load main.py, redirect PROJECT_ROOT to a temp fixture ---
spec = importlib.util.spec_from_file_location("hermes_cli_main_test", MAIN)
mod = importlib.util.module_from_spec(spec)
tmp = Path(tempfile.mkdtemp(prefix="webui-stamp-"))
web_dir = tmp / "web"
web_dir.mkdir(parents=True)
dist_dir = tmp / "hermes_cli" / "web_dist"
dist_dir.mkdir(parents=True)
(web_dir / "package.json").write_text(json.dumps({"name": "web"}), encoding="utf-8")
src = web_dir / "src"
src.mkdir()
(src / "app.tsx").write_text("export const x = 1;", encoding="utf-8")
(tmp / "package-lock.json").write_text("{}", encoding="utf-8")
(dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")

try:
    spec.loader.exec_module(mod)
except Exception as exc:  # pragma: no cover
    print(f"MODUL-IMPORT FEHLGESCHLAGEN: {exc!r}")
    print("Falle auf py_compile zurück.")
    import subprocess

    subprocess.run([sys.executable, "-m", "py_compile", str(MAIN)], check=False)
    sys.exit(2)

mod.PROJECT_ROOT = tmp

print("=== Test 1: force=True erzwingt immer Rebuild ===")
check(
    "needed(force=True) -> True",
    mod._web_ui_build_needed(web_dir, force=True) is True,
)

print("=== Test 2: kein Stamp + frischer dist -> mtime-Heuristik greift ===")
need = mod._web_ui_build_needed(web_dir, force=False)
check("needed(force=False) ohne Stamp liefert bool", isinstance(need, bool))

print("=== Test 3: Stamp schreiben + erneut prüfen ===")
mod._write_web_ui_build_stamp(web_dir)
stamp_path = mod._web_ui_build_stamp_path(web_dir)
check("Stamp-Datei existiert", stamp_path.is_file())
stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
check("Stamp hat contentHash", bool(stamp.get("contentHash")))
check("Stamp hat builtAt", bool(stamp.get("builtAt")))
check(
    "needed(force=False) mit passendem Stamp -> False",
    mod._web_ui_build_needed(web_dir, force=False) is False,
)

print("=== Test 4: Quelländerung nach Stamp -> needed=True ===")
(src / "app.tsx").write_text("export const x = 2;", encoding="utf-8")
os.utime(src / "app.tsx", (10_000_000, 10_000_000))  # alte mtime
check(
    "needed(force=False) nach Quelländerung -> True",
    mod._web_ui_build_needed(web_dir, force=False) is True,
)

print("=== Test 5: neuer Stamp nach Änderung -> wieder needed=False ===")
mod._write_web_ui_build_stamp(web_dir)
check(
    "needed(force=False) nach neuem Stamp -> False",
    mod._web_ui_build_needed(web_dir, force=False) is False,
)

print("=== Test 6: fehlender dist -> needed=True ===")
for f in dist_dir.glob("*"):
    f.unlink()
check(
    "needed(force=False) ohne dist -> True",
    mod._web_ui_build_needed(web_dir, force=False) is True,
)

print()
if failures:
    print(f"FEHLGESCHLAGEN ({len(failures)}/{checks}): {failures}")
    sys.exit(1)
print(f"ALLE {checks} CHECKS OK")
