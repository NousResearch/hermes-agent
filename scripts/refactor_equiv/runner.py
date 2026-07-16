"""Golden capture/replay runner for pure refactor equivalence checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from .determinism import Determinism
from .equiv_normalize import normalize


class EquivalenceError(AssertionError):
    """Raised when replay diverges from a captured golden."""


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def input_hash(case: dict[str, Any]) -> str:
    return hashlib.sha256(stable_json(case).encode("utf-8")).hexdigest()


def import_ref(spec: str) -> Any:
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"expected module:attribute, got {spec!r}")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data["cases"])
    return list(data)


def run_entrypoint(entrypoint: Callable[..., Any], case: dict[str, Any]) -> Any:
    if "input" in case and isinstance(case["input"], dict):
        payload = case["input"]
        return entrypoint(*payload.get("args", []), **payload.get("kwargs", {}))
    return entrypoint(case)


def capture(cases: list[dict[str, Any]], runner: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with Determinism():
        for case in cases:
            key = input_hash(case)
            out[key] = {
                "name": case.get("name", key),
                "input": case,
                "output": normalize(runner(case)),
            }
    return out


def write_golden(
    corpus_path: str | Path,
    golden_path: str | Path,
    runner: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    golden = capture(load_cases(corpus_path), runner)
    path = Path(golden_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return golden


def verify_golden(
    corpus_path: str | Path,
    golden_path: str | Path,
    runner: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    actual = capture(load_cases(corpus_path), runner)
    expected = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    if actual != expected:
        raise EquivalenceError(_first_diff(expected, actual))
    return actual


def _first_diff(expected: Any, actual: Any, path: str = "$") -> str:
    if expected == actual:
        return "no diff"
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual), key=str):
            if key not in expected:
                return f"{path}.{key}: unexpected field {actual[key]!r}"
            if key not in actual:
                return f"{path}.{key}: missing field {expected[key]!r}"
            if expected[key] != actual[key]:
                return _first_diff(expected[key], actual[key], f"{path}.{key}")
    if isinstance(expected, list) and isinstance(actual, list):
        for idx, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                return _first_diff(left, right, f"{path}[{idx}]")
        return f"{path}: length {len(expected)} != {len(actual)}"
    return f"{path}: expected {expected!r}, got {actual!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("capture", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", required=True)
        p.add_argument("--golden", required=True)
        p.add_argument("--runner", required=True)
    args = parser.parse_args(argv)
    runner = import_ref(args.runner)
    if args.cmd == "capture":
        write_golden(args.corpus, args.golden, runner)
    else:
        verify_golden(args.corpus, args.golden, runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

