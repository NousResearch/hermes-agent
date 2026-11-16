#!/usr/bin/env python3
"""Package the current compiled Hermes Desktop without editing its source tree.

Creates a fresh local build, never replaces an installed app or changes login data.
Run the upstream desktop build first after updating Hermes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop-source", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument("--user-data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.desktop_source.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error("output must be a new directory; previous builds are preserved")
    stamp = json.loads((source / "build/install-stamp.json").read_text())
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if stamp.get("dirty") or not stamp.get("commit"):
        parser.error("compiled build is not stamped from a clean commit; rebuild upstream first")
    # Backend-only commits need not force a desktop rebuild.
    unchanged = subprocess.run(["git", "-C", str(source.parent.parent), "diff", "--quiet",
        stamp["commit"], head, "--", "apps/desktop", "apps/shared", "package.json", "package-lock.json"])
    if unchanged.returncode:
        parser.error("desktop inputs changed since the compiled build; rebuild upstream first")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        parser.error("source has uncommitted changes; preserve and resolve them before packaging")
    for required in ("dist/electron-main.mjs", "dist/index.html", "assets/icon.icns"):
        if not (source / required).is_file():
            parser.error(f"missing compiled asset: {required}")
    output.mkdir(parents=True)
    stage = output / "workspace/apps/desktop"
    stage.mkdir(parents=True)
    for directory in ("dist", "assets", "public", "build", "scripts", "electron"):
        shutil.copytree(source / directory, stage / directory)
    # Keep build tools available without installing or changing dependencies.
    for modules, target in ((source / "node_modules", stage / "node_modules"),
                            (source.parent.parent / "node_modules", stage.parent.parent / "node_modules")):
        if modules.is_dir():
            target.symlink_to(modules, target_is_directory=True)
    package = json.loads((source / "package.json").read_text())
    package["productName"] = "Hermes Desktop"
    package["main"] = "dist/companion-launch.mjs"
    build = package["build"]
    build["productName"] = "Hermes Desktop"
    build["executableName"] = "Hermes Desktop"
    build["directories"]["output"] = str(output / "release")
    build["mac"]["extendInfo"].update({key: "Hermes Desktop" for key in
        ("CFBundleName", "CFBundleDisplayName", "CFBundleExecutable")})
    # This is a personal local build: no upload, distribution identity, or notarization.
    build["mac"]["identity"] = "-"
    build.pop("afterSign", None)
    (stage / "package.json").write_text(json.dumps(package, indent=2) + "\n")
    settings = {"HERMES_DESKTOP_APP_NAME": "Hermes Desktop",
                "HERMES_HOME": str(args.hermes_home.resolve()),
                "HERMES_DESKTOP_USER_DATA_DIR": str(args.user_data_dir.resolve())}
    bootstrap = "// Personal Mac launcher; preserves existing data and uses supported settings.\n"
    for key, value in settings.items():
        bootstrap += f"process.env[{json.dumps(key)}] = {json.dumps(value)};\n"
    bootstrap += "await import('./electron-main.mjs');\n"
    (stage / "dist/companion-launch.mjs").write_text(bootstrap)
    # Resolve the original builder so hoisted workspace tooling remains usable.
    resolver = "const p=require.resolve('electron-builder/package.json');const b=require(p).bin;console.log(require('path').resolve(require('path').dirname(p),typeof b==='string'?b:b['electron-builder']));console.log(require('path').join(require('path').dirname(require.resolve('electron/package.json')),'dist'));"
    builder, electron_dist = subprocess.check_output(["node", "-e", resolver], cwd=source, text=True).strip().splitlines()
    subprocess.run(["node", builder, "--dir", "--mac", "--arm64", "--publish", "never",
                    "-c.electronDist=" + electron_dist], cwd=stage, check=True)
    app = output / "release/mac-arm64/Hermes Desktop.app"
    if not app.is_dir():
        raise SystemExit("Expected Hermes Desktop.app was not produced")
    # Preserve the hardened-runtime entitlements supplied by upstream.
    subprocess.run(["codesign", "--verify", "--deep", "--strict", str(app)], check=True)
    receipt = {"built_at": datetime.now(timezone.utc).isoformat(), "source_head": head,
               "upstream_build": stamp, "bundle": str(app), "display_name": "Hermes Desktop",
               "scope": "personal local package; no authenticated runtime acceptance implied"}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified local bundle: {app}")


if __name__ == "__main__":
    main()
