"""Render machine-local Hermes operator Axolotl configs from safe templates."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def _env_file_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _get_value(name: str, env_file: dict[str, str], override: str | None = None) -> str:
    value = override or os.environ.get(name) or env_file.get(name) or ""
    value = value.strip()
    if not value or value.startswith("<"):
        raise ValueError(f"{name} is required")
    return value


def _replace_yaml_scalar(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^(?P<prefix>\s*{re.escape(key)}:\s*).*$", re.MULTILINE)
    if not pattern.search(text):
        raise ValueError(f"missing YAML key: {key}")
    normalized = value.replace("\\", "/")
    return pattern.sub(rf"\g<prefix>{normalized}", text)


def render_config(
    template: Path,
    output: Path,
    *,
    base_model: str,
    sft_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    text = template.read_text(encoding="utf-8")
    text = _replace_yaml_scalar(text, "base_model", base_model)
    if sft_path is not None:
        text = re.sub(
            r"(?m)^(\s*-\s+path:\s*).*$",
            lambda match: f"{match.group(1)}{str(sft_path).replace(chr(92), '/')}",
            text,
            count=1,
        )
    if output_dir is not None:
        text = _replace_yaml_scalar(text, "output_dir", str(output_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a local Axolotl config for Hermes operator post-training.")
    parser.add_argument("--template", type=Path, default=Path("training/qlora_config.yaml"))
    parser.add_argument("--output", type=Path, default=Path("training/local_qlora_config.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path("training/local.env"))
    parser.add_argument("--base-model")
    parser.add_argument("--sft", type=Path, default=Path("training/corpora/hermes_operator_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/runs/hermes-operator-qlora"))
    args = parser.parse_args(argv)

    env_values = _env_file_values(args.env_file)
    base_model = _get_value("HERMES_OPERATOR_BASE_MODEL", env_values, args.base_model)
    render_config(args.template, args.output, base_model=base_model, sft_path=args.sft, output_dir=args.output_dir)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
