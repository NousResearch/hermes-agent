"""Validate local Hermes operator training inputs before launching Axolotl."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_BASE_MODEL_RE = re.compile(r"^\s*base_model:\s*(?P<value>.+?)\s*$", re.MULTILINE)


def _jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object record")
            count += 1
    return count


def _read_base_model(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    match = _BASE_MODEL_RE.search(text)
    if not match:
        raise ValueError(f"{config_path}: missing base_model")
    return match.group("value").strip().strip("\"'")


def _looks_like_local_path(value: str) -> bool:
    return (
        value.startswith((".", "/", "\\"))
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
        or value.startswith("training/")
        or value.startswith("training\\")
    )


def validate_ready(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []

    if not args.sft.exists():
        errors.append(f"SFT file not found: {args.sft}")
    else:
        try:
            sft_count = _jsonl_count(args.sft)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if sft_count < args.min_sft_rows:
                errors.append(f"SFT file has {sft_count} row(s), expected at least {args.min_sft_rows}")

    if args.dpo:
        if not args.dpo.exists():
            errors.append(f"DPO file not found: {args.dpo}")
        else:
            try:
                dpo_count = _jsonl_count(args.dpo)
            except ValueError as exc:
                errors.append(str(exc))
            else:
                if dpo_count < args.min_dpo_rows:
                    errors.append(f"DPO file has {dpo_count} row(s), expected at least {args.min_dpo_rows}")

    if not args.qlora_config.exists():
        errors.append(f"QLoRA config not found: {args.qlora_config}")
    else:
        try:
            base_model = _read_base_model(args.qlora_config)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if base_model.lower().endswith(".gguf"):
                errors.append("base_model points to a GGUF. Use the matching HF checkpoint, then export GGUF after merge.")
            elif _looks_like_local_path(base_model) and not Path(base_model).expanduser().exists():
                errors.append(f"local base_model path not found: {base_model}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Hermes operator Axolotl inputs before training.")
    parser.add_argument("--sft", type=Path, default=Path("training/corpora/hermes_operator_sft.jsonl"))
    parser.add_argument("--dpo", type=Path)
    parser.add_argument("--qlora-config", type=Path, default=Path("training/qlora_config.yaml"))
    parser.add_argument("--min-sft-rows", type=int, default=1)
    parser.add_argument("--min-dpo-rows", type=int, default=1)
    args = parser.parse_args(argv)

    errors = validate_ready(args)
    if errors:
        print("training readiness: failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("training readiness: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
