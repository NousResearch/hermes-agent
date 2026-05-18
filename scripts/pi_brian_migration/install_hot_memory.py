#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def install_file(source: Path, target: Path) -> None:
	target.parent.mkdir(parents=True, exist_ok=True)
	if target.exists():
		backup = target.with_suffix(target.suffix + ".bak")
		shutil.copy2(target, backup)
	shutil.copy2(source, target)


def main() -> int:
	parser = argparse.ArgumentParser(description="Install reviewed Hermes hot-memory drafts")
	parser.add_argument("--hermes-home", required=True, help="Target HERMES_HOME directory")
	parser.add_argument("--user-file", required=True, help="Reviewed USER.md file to install")
	parser.add_argument("--memory-file", required=True, help="Reviewed MEMORY.md file to install")
	args = parser.parse_args()

	hermes_home = Path(args.hermes_home).expanduser().resolve()
	memories_dir = hermes_home / "memories"
	install_file(Path(args.user_file).expanduser().resolve(), memories_dir / "USER.md")
	install_file(Path(args.memory_file).expanduser().resolve(), memories_dir / "MEMORY.md")
	print(f"Installed USER.md and MEMORY.md into {memories_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
