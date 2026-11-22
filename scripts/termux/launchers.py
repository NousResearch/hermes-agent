"""APT maintainer hooks; agent entrypoints belong to scripts.build.launchers."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def write_maintainer_scripts(control: Path, names: list[str]) -> None:
    names_text = " ".join(shlex.quote(name) for name in names)
    postinst = '''#!/data/data/com.termux/files/usr/bin/sh
set -eu
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
mkdir -p "$PREFIX/bin"
for name in __NAMES__; do
    link="$PREFIX/bin/$name"
    target="../lib/hermes-agent/bin/$name"
    if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
        continue
    fi
    if [ -e "$link" ] || [ -L "$link" ]; then
        printf 'Refusing to replace foreign launcher: %s\\n' "$link" >&2
        exit 1
    fi
    ln -s "$target" "$link"
done
'''
    prerm = '''#!/data/data/com.termux/files/usr/bin/sh
set -eu
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
case "${1:-remove}" in
    remove|deconfigure) ;;
    *) exit 0 ;;
esac
for name in __NAMES__; do
    link="$PREFIX/bin/$name"
    if [ -L "$link" ] && [ "$(readlink "$link")" = "../lib/hermes-agent/bin/$name" ]; then
        rm -f "$link"
    fi
done
'''
    for name, script in (("postinst", postinst), ("prerm", prerm)):
        path = control / name
        path.write_text(script.replace("__NAMES__", names_text), encoding="utf-8")
        path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    from scripts.build.inputs import project_entries
    entries = project_entries(args.payload / "app/pyproject.toml")
    write_maintainer_scripts(args.control, list(entries))


if __name__ == "__main__":
    main()
