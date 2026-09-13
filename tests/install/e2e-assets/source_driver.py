"""Read-only source-install acceptance, before a CLI launch can repair it.

The driver owns this file; probes import only the installed tree's passive
readers. No bootstrap, installer, build, PM worker, or application entry point.

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def desktop_outputs(root: Path) -> list[Path]:
    desktop = root / "apps/desktop"
    return [path for pattern in (
        "release/*/resources/app.asar.unpacked/dist",
        "release/mac*/Hermes.app/Contents/Resources/app.asar.unpacked/dist",
    ) for path in desktop.glob(pattern)]


def verify_products(root: Path, desktop: str, node: Path | None = None) -> None:
    app = root / "apps/desktop"
    outputs = desktop_outputs(root)
    has_desktop = (app / "dist/index.html").exists() or any((app / "release").glob("*"))
    if desktop == "absent" and has_desktop:
        raise RuntimeError("unexpected desktop output in a no-desktop scenario")
    if desktop == "present":
        if not outputs or not any((out / "index.html").is_file() for out in outputs):
            raise RuntimeError("desktop packaged renderer is missing or incomplete")
        executables = [path for pattern in (
            "release/*/Hermes.exe", "release/*/Hermes", "release/*/hermes",
            "release/mac*/Hermes.app/Contents/MacOS/Hermes",
        ) for path in app.glob(pattern) if path.is_file() and path.stat().st_size]
        if not executables:
            raise RuntimeError("desktop executable is missing or empty")
    if node is None:
        return  # Historical builds have no compiler receipt contract.
    products = [("tui", root / "ui-tui/dist"), ("web", root / "hermes_cli/web_dist")]
    if desktop == "present":
        products.extend(("desktop", out) for out in outputs)
    for product, out in products:
        result = subprocess.run(
            [str(node), str(root / "scripts/build/freshness.mjs"),
             "--source", str(root), "--product", product, "--out", str(out)],
            cwd=root, capture_output=True, text=True, check=True, timeout=120,
        )
        if result.stdout.strip() != "true":
            raise RuntimeError(f"{product} output is missing, stale, or damaged: {out}")


def probe_pm(root: Path, desktop: str, command: list[str]) -> None:
    # Use the launcher's Python ABI, but deliberately do NOT execute its
    # bootstrap: that would complete dependencies or recover markers for it.
    sys.path.insert(0, str(root))
    from hermes_cli._launchers import runtime_command
    from hermes_cli.runtime_paths import selected_venv, site_packages

    if command != runtime_command(root):
        raise RuntimeError("published launcher belongs to another installation or Python")
    if any((root / marker).exists() for marker in (".update-incomplete", ".lazy-refresh-incomplete")):
        raise RuntimeError("source update left an incomplete marker; refusing launch-time recovery")
    sys.path.insert(1, str(site_packages(selected_venv(root))))
    from pm.ensure import installed_package, venv_is_current

    if not venv_is_current(project_root=root):
        raise RuntimeError("installed dependency generation is not current")
    node = installed_package("node")
    if node is None or node.binary is None:
        raise RuntimeError("installed Node is missing or does not match the target pin")
    verify_products(root, desktop, node.binary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--desktop", choices=("absent", "present"), required=True)
    parser.add_argument("--runtime-command", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.runtime_command:
        probe_pm(root, args.desktop, json.loads(args.runtime_command))
    elif (root / "pm/lock.json").is_file():
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HERMES_DISABLE_LAZY_INSTALLS="1")
        result = subprocess.run([str(args.launcher), "--print-runtime-command"],
                                capture_output=True, text=True, check=True, env=env, timeout=30)
        command = json.loads(result.stdout)
        if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
            raise RuntimeError("invalid published runtime command")
        subprocess.run([command[0], "-I", "-B", str(Path(__file__).resolve()),
                        "--root", str(root), "--launcher", str(args.launcher),
                        "--desktop", args.desktop, "--runtime-command", json.dumps(command)],
                       env=env, check=True, timeout=300)
    else:
        verify_products(root, args.desktop)
        print("Historical install: artifact presence only (no PM/compiler receipt capability)")
    print(f"Read-only source verification passed: {root} (desktop={args.desktop})")


if __name__ == "__main__":
    main()