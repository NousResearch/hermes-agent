#!/usr/bin/env python3
"""
Training orchestrator for the finetune pipeline.

Generates per-cluster Axolotl configs and launches QLoRA training.
(GGUF conversion for serving is handled by manage.py's redeploy flow.)

Usage:
    python train.py [--cluster CLUSTER_ID] [--base-model MODEL] [--dry-run]
"""

import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from common import (
    ADAPTERS_DIR, CLUSTERS_DIR, CLUSTER_STATE_PATH, LOGS_DIR, MODELS_DIR,
    ensure_dirs, load_config, load_json, pipeline_lock, save_json, logger,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# How long a standalone `python train.py` waits for the pipeline lock
# before giving up. See main() — the lock is coarse on purpose.
LOCK_TIMEOUT = 30.0

# Skill directory (where templates live)
SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = SKILL_DIR / "templates" / "base_qlora.yaml"

# Datasets below this record count get val_set_size 0 (no eval split):
# a 10% split of a tiny personal dataset is near-zero records, which
# axolotl rejects outright.
MIN_RECORDS_FOR_EVAL_SPLIT = 50

# Keys that only make sense when an eval split exists.
EVAL_CONFIG_KEYS = (
    "eval_steps",
    "eval_strategy",
    "evaluation_strategy",
    "early_stopping_patience",
    "metric_for_best_model",
    "greater_is_better",
    "load_best_model_at_end",
)


def _kill_process_group(proc: "subprocess.Popen") -> None:
    """SIGTERM (then SIGKILL) the whole process group of a Popen launched
    with start_new_session=True, so training workers don't outlive the
    accelerate launcher and squat on the GPU."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def _count_lines(path: Path) -> int:
    """Count lines in a file without leaking the handle."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


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
    """Generate configs and launch QLoRA training."""

    def __init__(self, config: dict = None):
        self.config = config or load_config().get("training", {})
        self.base_model = self.config.get("base_model", "kai-os/Carnice-9b")
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

        # Override base model and chat template from user config.
        # Axolotl needs an HF repo ID or a local safetensors directory —
        # a GGUF path would produce a config that fails at model load,
        # so reject it here with an actionable message.
        base_model = os.path.expanduser(str(self.base_model))
        if base_model.lower().endswith(".gguf"):
            raise ValueError(
                f"finetune.training.base_model points at a GGUF file "
                f"({base_model}), which axolotl cannot train against. "
                "Per SKILL.md: \"Axolotl trains against HuggingFace "
                "safetensors, NOT GGUF. Set finetune.training.base_model "
                "to the HF repo ID (e.g. kai-os/Carnice-9b), not the GGUF "
                "path. The GGUF path is only for inference-time serving "
                "via llama.cpp.\""
            )
        template["base_model"] = base_model
        template["chat_template"] = self.chat_template

        # Apply maturity-stage overrides
        overrides = MATURITY_OVERRIDES.get(maturity, {})
        template.update(overrides)

        # Set data paths
        data_dir = CLUSTERS_DIR / cluster_id
        train_path = data_dir / "train.jsonl"
        eval_path = data_dir / "eval.jsonl"

        # Modern axolotl deprecated `type: sharegpt` in favor of
        # `type: chat_template`. Our format.py emits ShareGPT-style records
        # ({"conversations": [{"from": "human", "value": "..."}, ...]}) so we
        # tell axolotl how to map those fields onto its expected schema:
        #   - field_messages: which JSONL key holds the turn list
        #   - message_property_mappings: which turn keys map to role/content
        # The default `roles` mapping in axolotl already aliases ShareGPT
        # role names (human → user, gpt → assistant), so we don't override it.
        template["datasets"] = [{
            "path": str(train_path),
            "type": "chat_template",
            "chat_template": template.get("chat_template", "chatml"),
            "field_messages": "conversations",
            "message_property_mappings": {
                "role": "from",
                "content": "value",
            },
        }]

        # Have axolotl carve a 10% eval split from train.jsonl internally.
        # We still write our own eval.jsonl to disk via format.py for the
        # lightweight eval gate (eval.py), but axolotl's training loop
        # uses its own internal split. This avoids the val_set_size==0 +
        # eval_steps validation conflict in modern axolotl, where
        # `datasets_eval` is not a recognized key.
        #
        # Exception: tiny personal datasets. Below ~50 records a 10% split
        # is near-zero examples and axolotl rejects an empty eval set, so
        # we disable the split and drop the eval-dependent keys entirely.
        train_size = _count_lines(train_path)
        eval_size = _count_lines(eval_path)
        if train_size < MIN_RECORDS_FOR_EVAL_SPLIT:
            logger.info(
                "Dataset has %d records (< %d) — disabling eval split "
                "(val_set_size 0) for cluster %s",
                train_size, MIN_RECORDS_FOR_EVAL_SPLIT, cluster_id,
            )
            template["val_set_size"] = 0
            for key in EVAL_CONFIG_KEYS:
                template.pop(key, None)
        else:
            template["val_set_size"] = 0.1

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
            "train_size": train_size,
            "eval_size": eval_size,
            "base_model": base_model,
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

        train_size = _count_lines(train_path)
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
        # next to the current Python interpreter first (so virtual envs work
        # correctly), then fall back to whatever is on PATH.
        log_path = LOGS_DIR / f"train_{cluster_id}_{version}_{datetime.now():%Y%m%d_%H%M%S}.log"
        accelerate_bin = Path(sys.executable).parent / "accelerate"
        if not accelerate_bin.exists():
            which_accelerate = shutil.which("accelerate")
            if which_accelerate:
                accelerate_bin = Path(which_accelerate)
            else:
                logger.error(
                    "accelerate not found at %s or on PATH — install with: "
                    "pip install accelerate axolotl",
                    accelerate_bin,
                )
                return None

        cmd = [
            str(accelerate_bin), "launch",
            "-m", "axolotl.cli.train",
            str(config_path),
        ]

        # Disable axolotl telemetry. Beyond the privacy concern, current
        # axolotl wheels ship with a broken telemetry whitelist path that
        # crashes on import unless telemetry is explicitly disabled. Setting
        # both env vars covers older and newer axolotl versions.
        train_env = os.environ.copy()
        train_env["AXOLOTL_DO_NOT_TRACK"] = "1"
        train_env["DO_NOT_TRACK"] = "1"

        # Reduce CUDA allocator fragmentation. On 12GB cards with an active
        # desktop session, the difference between fitting and OOMing often
        # comes down to fragmentation rather than absolute memory usage.
        # PyTorch's expandable_segments allocator coalesces freed regions
        # so a large eval-time allocation (~3-4 GiB for the LoRA kernel
        # workspace) doesn't fail on a fragmented heap.
        if "PYTORCH_CUDA_ALLOC_CONF" not in train_env:
            train_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        logger.info("Launching training: %s", " ".join(cmd))
        logger.info("Log: %s", log_path)

        # Launch in a new session (own process group) so a timeout can take
        # down the whole training tree — accelerate spawns worker processes
        # that would otherwise keep the GPU after the launcher dies.
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(SKILL_DIR),
                    env=train_env,
                    start_new_session=True,
                )
                try:
                    returncode = proc.wait(timeout=3600 * 24)  # 24h max
                except subprocess.TimeoutExpired:
                    logger.error(
                        "Training timed out for %s %s — killing process group %d",
                        cluster_id, version, proc.pid,
                    )
                    _kill_process_group(proc)
                    return None

            if returncode != 0:
                logger.error(
                    "Training failed for %s %s (exit code %d). See %s",
                    cluster_id, version, returncode, log_path,
                )
                return None

            logger.info("Training complete: %s %s", cluster_id, version)
            return config_path.parent

        except FileNotFoundError:
            logger.error(
                "accelerate not found. Install with: pip install accelerate axolotl"
            )
            return None

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

    # Coarse inter-process lock, taken at the CLI entry point only (never
    # inside train()) so manage.py run_pipeline — which already holds the
    # same lock for the whole pipeline — can call the orchestrator
    # in-process without deadlocking. This serializes version allocation
    # (_next_version) + adapter-dir writes against concurrent cron/manual
    # runs; coarse on purpose (see common.pipeline_lock).
    try:
        with pipeline_lock(timeout=LOCK_TIMEOUT):
            if args.cluster:
                result = orchestrator.train(args.cluster, dry_run=args.dry_run)
                if result:
                    print(f"Training {'config generated' if args.dry_run else 'complete'}: {result}")
                else:
                    print(f"Training skipped or failed for {args.cluster}")
            else:
                trained = orchestrator.train_eligible(dry_run=args.dry_run)
                print(f"Trained {len(trained)} clusters: {trained}")
    except TimeoutError as e:
        print(f"Another finetune operation is running — {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
