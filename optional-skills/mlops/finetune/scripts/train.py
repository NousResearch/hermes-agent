#!/usr/bin/env python3
"""
Training orchestrator for the finetune pipeline.

Generates per-cluster Axolotl configs, launches QLoRA training, and handles
merge + quantize of finished adapters.

Usage:
    python train.py [--cluster CLUSTER_ID] [--base-model MODEL] [--dry-run]
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from common import (
    ADAPTERS_DIR, CLUSTERS_DIR, CLUSTER_STATE_PATH, LOGS_DIR, MODELS_DIR,
    ensure_dirs, load_config, load_json, save_json, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Skill directory (where templates live)
SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = SKILL_DIR / "templates" / "base_qlora.yaml"

# Maturity-stage overrides
MATURITY_OVERRIDES = {
    "nascent": {
        "lora_dropout": 0.1,
        "learning_rate": 5e-5,
        "num_epochs": 2,
    },
    "established": {
        # Standard config, no overrides
    },
    "mature": {
        # Standard config, no overrides
    },
}


def _load_template() -> dict:
    """Load the base QLoRA config template."""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Training template not found: {TEMPLATE_PATH}")
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _next_version(cluster_dir: Path) -> str:
    """Determine the next version number for a cluster's adapter."""
    existing = sorted(
        (d.name for d in cluster_dir.iterdir() if d.is_dir() and d.name.startswith("v")),
        key=lambda x: int(x[1:]) if x[1:].isdigit() else 0,
    ) if cluster_dir.exists() else []
    if not existing:
        return "v1"
    last = existing[-1]
    num = int(last[1:]) if last[1:].isdigit() else 0
    return f"v{num + 1}"


class TrainingOrchestrator:
    """Generate configs, launch training, merge and quantize adapters."""

    def __init__(self, config: dict = None):
        self.config = config or load_config().get("training", {})
        self.base_model = self.config.get("base_model", "~/programs/carnice/Carnice-9b-Q8_0.gguf")
        self.chat_template = self.config.get("chat_template", "chatml")
        self.quantization = self.config.get("quantization", "Q5_K_M")

    def generate_config(
        self,
        cluster_id: str,
        version: str,
        maturity: str = "established",
    ) -> Path:
        """Generate a per-cluster Axolotl training config."""
        ensure_dirs()
        template = _load_template()

        # Override base model and chat template from user config
        template["base_model"] = self.base_model
        template["chat_template"] = self.chat_template

        # Apply maturity-stage overrides
        overrides = MATURITY_OVERRIDES.get(maturity, {})
        template.update(overrides)

        # Set data paths
        data_dir = CLUSTERS_DIR / cluster_id
        train_path = data_dir / "train.jsonl"
        eval_path = data_dir / "eval.jsonl"

        template["datasets"] = [{
            "path": str(train_path),
            "type": "sharegpt",
            "conversation": "chatml",
        }]

        if eval_path.exists():
            template["val_set_size"] = 0  # We provide our own eval split
            template["datasets_eval"] = [{
                "path": str(eval_path),
                "type": "sharegpt",
                "conversation": "chatml",
            }]

        # Set output path
        output_dir = ADAPTERS_DIR / cluster_id / version
        output_dir.mkdir(parents=True, exist_ok=True)
        template["output_dir"] = str(output_dir / "adapter_model")

        # Write config
        config_path = output_dir / "config.yml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(template, f, default_flow_style=False)

        # Save dataset manifest
        save_json(output_dir / "dataset_manifest.json", {
            "cluster_id": cluster_id,
            "version": version,
            "train_path": str(train_path),
            "eval_path": str(eval_path) if eval_path.exists() else None,
            "train_size": sum(1 for _ in open(train_path)) if train_path.exists() else 0,
            "eval_size": sum(1 for _ in open(eval_path)) if eval_path.exists() else 0,
            "base_model": self.base_model,
            "maturity": maturity,
            "generated_at": datetime.now().isoformat(),
        })

        logger.info("Generated config: %s", config_path)
        return config_path

    def train(
        self,
        cluster_id: str,
        version: str = None,
        dry_run: bool = False,
    ) -> Optional[Path]:
        """
        Launch training for a cluster.

        Returns the adapter output directory on success, None on failure.
        """
        cluster_state = load_json(CLUSTER_STATE_PATH, {})
        cluster_info = cluster_state.get("clusters", {}).get(cluster_id, {})
        maturity = cluster_info.get("maturity", "established")

        # Check maturity gate
        if maturity == "embryonic":
            logger.warning(
                "Cluster %s is embryonic (< 50 good turns). Skipping training.",
                cluster_id,
            )
            return None

        # Check training data exists
        train_path = CLUSTERS_DIR / cluster_id / "train.jsonl"
        if not train_path.exists():
            logger.error("No training data for cluster %s", cluster_id)
            return None

        train_size = sum(1 for _ in open(train_path))
        if train_size == 0:
            logger.error("Training data is empty for cluster %s", cluster_id)
            return None

        # Determine version
        cluster_dir = ADAPTERS_DIR / cluster_id
        if version is None:
            version = _next_version(cluster_dir)

        # Generate config
        config_path = self.generate_config(cluster_id, version, maturity)

        if dry_run:
            logger.info("[DRY RUN] Would train %s %s with %d examples",
                        cluster_id, version, train_size)
            return config_path.parent

        # Launch training. accelerate ships as a console script in
        # venv/bin/accelerate — it can't be invoked via `python -m accelerate`
        # because the package doesn't define __main__. We resolve the binary
        # in the same venv as the current Python interpreter so virtual envs
        # work correctly.
        log_path = LOGS_DIR / f"train_{cluster_id}_{version}_{datetime.now():%Y%m%d_%H%M%S}.log"
        accelerate_bin = Path(sys.executable).parent / "accelerate"
        if not accelerate_bin.exists():
            logger.error(
                "accelerate not found at %s — install with: pip install accelerate axolotl",
                accelerate_bin,
            )
            return None

        cmd = [
            str(accelerate_bin), "launch",
            "-m", "axolotl.cli.train",
            str(config_path),
        ]

        logger.info("Launching training: %s", " ".join(cmd))
        logger.info("Log: %s", log_path)

        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(SKILL_DIR),
                    timeout=3600 * 24,  # 24h max
                )

            if result.returncode != 0:
                logger.error(
                    "Training failed for %s %s (exit code %d). See %s",
                    cluster_id, version, result.returncode, log_path,
                )
                return None

            logger.info("Training complete: %s %s", cluster_id, version)
            return config_path.parent

        except subprocess.TimeoutExpired:
            logger.error("Training timed out for %s %s", cluster_id, version)
            return None
        except FileNotFoundError:
            logger.error(
                "accelerate not found. Install with: pip install accelerate axolotl"
            )
            return None

    def merge_and_quantize(
        self, cluster_id: str, version: str,
    ) -> Optional[Path]:
        """
        Merge LoRA adapter into base model and quantize to GGUF.

        Returns path to merged GGUF on success.
        """
        adapter_dir = ADAPTERS_DIR / cluster_id / version
        config_path = adapter_dir / "config.yml"
        adapter_model_dir = adapter_dir / "adapter_model"

        if not adapter_model_dir.exists():
            logger.error("Adapter model not found: %s", adapter_model_dir)
            return None

        # Step 1: Merge LoRA
        merged_dir = adapter_dir / "merged_model"
        merge_cmd = [
            sys.executable, "-m", "axolotl.cli.merge_lora",
            str(config_path),
            "--lora_model_dir", str(adapter_model_dir),
            "--output_dir", str(merged_dir),
        ]

        logger.info("Merging LoRA adapter...")
        try:
            result = subprocess.run(merge_cmd, capture_output=True, text=True, timeout=1800)
            if result.returncode != 0:
                logger.error("Merge failed: %s", result.stderr[:500])
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error("Merge failed: %s", e)
            return None

        # Step 2: Quantize to GGUF
        gguf_path = adapter_dir / "merged.gguf"
        quant_cmd = ["llama-quantize", str(merged_dir), str(gguf_path), self.quantization]

        logger.info("Quantizing to %s...", self.quantization)
        try:
            result = subprocess.run(quant_cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                # Try higher quant level
                logger.warning(
                    "Quantization at %s failed, trying Q8_0...", self.quantization
                )
                quant_cmd[-1] = "Q8_0"
                result = subprocess.run(quant_cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode != 0:
                    logger.error("Quantization failed: %s", result.stderr[:500])
                    # Keep unmerged adapter for LoRA loading
                    return None
        except FileNotFoundError:
            logger.warning(
                "llama-quantize not found. Keeping unmerged adapter for LoRA loading."
            )
            return None

        # Clean up merged model dir to save space
        if gguf_path.exists() and merged_dir.exists():
            shutil.rmtree(merged_dir)
            logger.info("Cleaned up merged model dir (GGUF available)")

        logger.info("Merged GGUF: %s", gguf_path)
        return gguf_path

    def train_eligible(self, dry_run: bool = False) -> List[str]:
        """Train all eligible clusters (non-embryonic with training data)."""
        cluster_state = load_json(CLUSTER_STATE_PATH, {})
        clusters = cluster_state.get("clusters", {})
        trained = []

        for cid, info in clusters.items():
            if info.get("maturity") == "embryonic":
                logger.info("Skipping embryonic cluster: %s", cid)
                continue

            result = self.train(cid, dry_run=dry_run)
            if result:
                trained.append(cid)

        return trained


def main():
    parser = argparse.ArgumentParser(description="Train QLoRA adapters")
    parser.add_argument("--cluster", type=str, default=None,
                        help="Specific cluster to train (default: all eligible)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Override base model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate configs without launching training")
    args = parser.parse_args()

    config = load_config().get("training", {})
    if args.base_model:
        config["base_model"] = args.base_model

    orchestrator = TrainingOrchestrator(config=config)

    if args.cluster:
        result = orchestrator.train(args.cluster, dry_run=args.dry_run)
        if result:
            print(f"Training {'config generated' if args.dry_run else 'complete'}: {result}")
        else:
            print(f"Training skipped or failed for {args.cluster}")
    else:
        trained = orchestrator.train_eligible(dry_run=args.dry_run)
        print(f"Trained {len(trained)} clusters: {trained}")


if __name__ == "__main__":
    main()
