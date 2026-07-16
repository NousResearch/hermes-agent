"""Mutation self-test support for refactor-equivalence goldens."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Mutation:
    name: str
    apply: Callable[[Path], None]


class MutationMissed(AssertionError):
    """Raised when a mutation does not make golden replay fail."""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r} in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def run_mutations(module_path: str | Path, mutations: list[Mutation], verify_cmd: list[str]) -> None:
    path = Path(module_path)
    original = path.read_text(encoding="utf-8")
    for mutation in mutations:
        try:
            _clear_bytecode(path)
            mutation.apply(path)
            _clear_bytecode(path)
            result = subprocess.run(verify_cmd, cwd=Path.cwd(), text=True, capture_output=True)
            if result.returncode == 0:
                raise MutationMissed(f"mutation was not detected: {mutation.name}")
        finally:
            path.write_text(original, encoding="utf-8")
            _clear_bytecode(path)


def _clear_bytecode(path: Path) -> None:
    cache = path.parent / "__pycache__"
    if not cache.exists():
        return
    stem = path.stem
    for compiled in cache.glob(f"{stem}.*.pyc"):
        compiled.unlink()


def relay_header_mutations() -> list[Mutation]:
    return [
        Mutation(
            "return-value: provider gate accepts direct anthropic",
            lambda p: replace_once(
                p,
                '_POOL_AFFINITY_PROVIDERS = frozenset({"claude-apr"})',
                '_POOL_AFFINITY_PROVIDERS = frozenset({"claude-apr", "anthropic"})',
            ),
        ),
        Mutation(
            "message-emit: drop session header",
            lambda p: replace_once(
                p,
                'out["x-hermes-session"] = sid',
                'out["x-hermes-session"] = "mutated-session"',
            ),
        ),
        Mutation(
            "DB-write: invert delegated lane branch",
            lambda p: replace_once(
                p,
                'return "background" if (delegated or noninteractive) else "interactive"',
                'return "interactive" if (delegated or noninteractive) else "background"',
            ),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--verify-cmd", action="store_true", required=True)
    args, verify_cmd = parser.parse_known_args(argv)
    module = args.module
    mutations = relay_header_mutations() if module.endswith("relay_headers.py") else []
    if not mutations:
        raise SystemExit(f"no registered mutations for {module}")
    if verify_cmd and verify_cmd[0] == "--":
        verify_cmd = verify_cmd[1:]
    if not verify_cmd:
        raise SystemExit("--verify-cmd requires a command")
    run_mutations(module, mutations, verify_cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
