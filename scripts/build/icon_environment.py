"""Prepare the locked icon-only environment through PM, then run the generator."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

# The Node wrapper can run this file from a separately prepared source tree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pm


def prepare_icon_environment(source: Path, out: Path, cache: Path, *, explicit: bool = True) -> Path:
    """Prepare a fresh locked build-only interpreter without generating assets."""
    return pm.build_environment(
        source=source.resolve(), out=out.resolve(), cache=cache.resolve(),
        groups=["icon-build"], only_groups=True, explicit=explicit,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--on-demand", action="store_true")
    args, _ = parser.parse_known_args(argv)
    if args.on_demand:
        argv.remove("--on-demand")
    source = args.source.resolve()
    with TemporaryDirectory(prefix="hermes-icon-build-") as temporary:
        python = prepare_icon_environment(
            source, Path(temporary) / "venv", source / ".cache/icon-build",
            explicit=not args.on_demand,
        )
        return subprocess.run(
            [str(python), "-I", str(ROOT / "scripts/generate_icons.py"), *argv],
            cwd=source,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
