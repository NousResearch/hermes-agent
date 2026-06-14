"""Prepare a local HF export for causal-LM LoRA training without mutating it."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def prepare_checkpoint(source: Path, output: Path, *, link_files: bool = False) -> None:
    source = source.expanduser()
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"source checkpoint not found: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    copy_function = os.link if link_files else shutil.copy2
    shutil.copytree(source, output, copy_function=copy_function)

    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        for key, value in text_config.items():
            if key not in config:
                config[key] = value
        if config.get("architectures") == ["Qwen3_5ForConditionalGeneration"]:
            config["architectures"] = ["Qwen3_5ForCausalLM"]
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a causal-LM training copy of a local HF export.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--link-files", action="store_true", help="Hardlink files instead of copying them.")
    args = parser.parse_args(argv)

    prepare_checkpoint(args.source, args.output, link_files=args.link_files)
    info = inspect_checkpoint(args.output)
    print(json.dumps(info, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
