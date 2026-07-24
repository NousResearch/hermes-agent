#!/usr/bin/env python3
"""Emit an openWakeWord automatic-training YAML for a custom phrase.

Writes the config openWakeWord's ``train.py`` consumes, pre-filled with the
target phrase, model name, and the positives/negatives/output paths. It covers
the common fields; always cross-check against openWakeWord's ``custom_model.yml``
for your installed version, since the schema evolves.
"""

from __future__ import annotations

import argparse
import re


def slugify(phrase: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", phrase.strip().lower())
    return s.strip("_") or "wake_word"


def build_config(
    phrase: str,
    output_dir: str,
    *,
    positives_dir: str | None = None,
    background_dir: str = "./audioset_16k",
    rir_dir: str = "./mit_rirs",
    false_positive_dir: str = "./fma",
    n_samples: int = 5_000,
    n_samples_val: int = 1_000,
    steps: int = 10_000,
) -> dict:
    """Build the openWakeWord training-config dict.

    When *positives_dir* is given (OpenAI-TTS clips already generated), the
    trainer uses those instead of synthesizing its own with Piper.
    """
    model_name = slugify(phrase)
    cfg: dict = {
        "target_phrase": [phrase.strip()],
        "model_name": model_name,
        "output_dir": output_dir,
        "n_samples": n_samples,
        "n_samples_val": n_samples_val,
        "steps": steps,
        "target_accuracy": 0.7,
        "target_recall": 0.5,
        # Augmentation + negative sources (see references/platforms.md).
        "rir_paths": [rir_dir],
        "background_paths": [background_dir],
        "false_positive_validation_data_path": false_positive_dir,
        "augmentation_rounds": 1,
        "layer_size": 32,
    }
    if positives_dir:
        # Reuse pre-generated positive clips rather than Piper-synthesizing them.
        cfg["custom_positive_samples_dir"] = positives_dir
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phrase", required=True, help='e.g. "hey morgane"')
    ap.add_argument("--out", required=True, help="path to write the YAML config")
    ap.add_argument("--output-dir", default="./oww_out", help="trainer output dir")
    ap.add_argument("--positives-dir", default=None, help="pre-generated positives")
    ap.add_argument("--steps", type=int, default=10_000)
    args = ap.parse_args(argv)

    import yaml

    cfg = build_config(
        args.phrase,
        args.output_dir,
        positives_dir=args.positives_dir,
        steps=args.steps,
    )
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote training config for {cfg['model_name']!r} to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
