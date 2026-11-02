"""Source-build entrypoint for the independently PM-owned DMG supplier."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def prepare_dmgbuild(out: Path, cache: Path) -> Path:
    import pm
    from pm import paths

    target = pm.current_target()
    if not target.startswith("darwin-"):
        raise ValueError("dmgbuild preparation requires native macOS")
    store = pm.prepare_tools(["dmgbuild"], out=out, target=target, cache=cache)
    package = pm.get_package("dmgbuild")
    version = pm.Lockfile(paths.lockfile_path()).version(package.name)
    assert version is not None  # prepare_tools cannot succeed without a lock pin
    binary = package.binary(store / package.store_entry(version, target), target)
    assert binary is not None  # dmgbuild declares its launcher
    return binary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_dmgbuild(args.out, args.cache))
