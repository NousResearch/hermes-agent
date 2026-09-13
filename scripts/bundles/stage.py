"""Select complete frontend products and stage the runnable native agent."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.bundles.native import stage_native


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--cache", type=Path, help="persistent uv build cache")
    parser.add_argument("--tui", type=Path)
    parser.add_argument("--web", type=Path)
    args = parser.parse_args(argv)
    if bool(args.tui) != bool(args.web):
        parser.error("supply both --tui and --web products, or neither to build them")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    if args.tui is not None:
        args.frontends = {"tui": args.tui.resolve(), "web": args.web.resolve()}
        return stage_native(args)
    node = shutil.which("node")
    if node is None:
        raise FileNotFoundError("Node is required to build frontend products")
    with tempfile.TemporaryDirectory(prefix="hermes-products-") as temp:
        products = Path(temp)
        source = products / "source"
        from scripts.bundles.payload import snapshot
        snapshot(ROOT, args.ref, source)
        commands = [
            ["scripts/build/node-deps.mjs", "--source", str(source), "--workspace", "ui-tui", "--workspace", "web"],
            ["scripts/generate-icons.mjs", "--source", str(source), "--out", str(products / "icons")],
            ["scripts/build/tui.mjs", "--source", str(source), "--out", str(products / "tui")],
            ["scripts/build/web.mjs", "--source", str(source), "--icons", str(products / "icons"), "--out", str(products / "web")],
        ]
        for command in commands:
            subprocess.run([node, *command], cwd=ROOT, check=True)
        args.frontends = {"tui": products / "tui", "web": products / "web"}
        return stage_native(args)


if __name__ == "__main__":
    raise SystemExit(main())
