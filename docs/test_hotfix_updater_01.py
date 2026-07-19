"""Lokaler Validierungstest für HOTFIX-HERMES-UPDATER-01.

Testet die drei neuen/geänderten Funktionen aus hermes_cli/main.py
unabhängig vom schwer importierbaren Modul-Startup:

  - _web_ui_build_input_hash(web_dir)
  - _web_ui_build_stamp_path(web_dir)
  - _web_ui_build_needed(web_dir, *, force)
  - _write_web_ui_build_stamp(web_dir)

Strategie: wir laden main.py als Modul über importlib mit einem
PROVISORISCHEN PROJECT_ROOT (temp-Verzeichnis), damit die Funktionen
gegen ein sauberes, kleines Fixture laufen. main.py liest PROJECT_ROOT
nur als Modul-Global (kein Startup-Import von schwerem Krams beim
reinen Laden ohne __main__-Ausführung).

Falls der reine Import scheitert, fällt der Test auf eine
Py-compile-Prüfung + manuelle Logik-Aussage zurück und meldet das.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/zigytb-007/.hermes/hermes-agent")
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


# --- main.py laden, aber PROJECT_ROOT auf ein temp-Fixture umlenken ---
spec = importlib.util.spec_from_file_location("hermes_cli_main_test", MAIN)
mod = importlib.util.module_from_spec(spec)
# PROJECT_ROOT vor dem Ausführen des Modul-Codes setzen, damit die
# workspace-root-Helfer im Test-Fixture arbeiten.
tmp = Path(tempfile.mkdtemp(prefix="webui-stamp-"))
# web/ als Subdir + hermes_cli/web_dist als Output
web_dir = tmp / "web"
web_dir.mkdir(parents=True)
dist_dir = tmp / "hermes_cli" / "web_dist"
dist_dir.mkdir(parents=True)
# minimale web/package.json + eine Quelldatei
(web_dir / "package.json").write_text(json.dumps({"name": "web"}), encoding="utf-8")
src = web_dir / "src"
src.mkdir()
(src / "app.tsx").write_text("export const x = 1;", encoding="utf-8")
# root lockfile (wie im echten Workspace)
(tmp / "package-lock.json").write_text("{}", encoding="utf-8")
# fresh dist outputs
(dist_dir / "index.html").write_text("<html></html>", encoding="utf-8")

try:
    spec.loader.exec_module(mod)
except Exception as exc:  # pragma: no cover
    print(f"MODUL-IMPORT FEHLGESCHLAGEN: {exc!r}")
    print("Falle auf py_compile zurück.")
    import subprocess

    subprocess.run([sys.executable, "-m", "py_compile", str(MAIN)], check=False)
    sys.exit(2)

# PROJECT_ROOT für die Funktionen überschreiben (sie nutzen das Modul-Global)
mod.PROJECT_ROOT = tmp

print("=== Test 1: force=True erzwingt immer Rebuild ===")
check("needed(force=True) -> True", mod._web_ui_build_needed(web_dir, force=True) is True)

print("=== Test 2: kein Stamp + frischer dist -> mtime-Heuristik greift ===")
# dist ist frisch (jetzt), web/src ist älter (mkdtemp vorher) -> braucht
# keinen Rebuild laut mtime, aber KEIN stamp -> fallthrough mtime.
need = mod._web_ui_build_needed(web_dir, force=False)
check("needed(force=False) ohne Stamp liefert bool", isinstance(need, bool))

print("=== Test 3: Stamp schreiben + erneut prüfen ===")
mod._write_web_ui_build_stamp(web_dir)
stamp_path = mod._web_ui_build_stamp_path(web_dir)
check("Stamp-Datei existiert", stamp_path.is_file())
stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
check("Stamp hat contentHash", bool(stamp.get("contentHash")))
check("Stamp hat builtAt", bool(stamp.get("builtAt")))
# Nach Stamp-Schreiben + unveränderten Quellen -> needed=False
check(
    "needed(force=False) mit passendem Stamp -> False",
    mod._web_ui_build_needed(web_dir, force=False) is False,
)

print("=== Test 4: Quelländerung nach Stamp -> needed=True ===")
# Quelldatei ändern (Inhalt + mtime) -> Hash ändert sich
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
