"""Tinker run normalization and runtime script generation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.config import get_research_config
from research.models import RunSpec, now_utc


@dataclass(slots=True)
class TinkerRuntimePaths:
    runtime_dir: Path
    config_path: Path
    runner_path: Path
    status_path: Path
    log_path: Path
    pid_path: Path
    last_checkpoint_path: Path
    last_sampler_path: Path


def build_runtime_paths(project_dir: Path, run_id: str) -> TinkerRuntimePaths:
    runtime_dir = project_dir / "runtime" / run_id
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return TinkerRuntimePaths(
        runtime_dir=runtime_dir,
        config_path=runtime_dir / "config.json",
        runner_path=runtime_dir / "runner.py",
        status_path=runtime_dir / "status.json",
        log_path=runtime_dir / "runner.log",
        pid_path=runtime_dir / "runner.pid",
        last_checkpoint_path=runtime_dir / "last_checkpoint.txt",
        last_sampler_path=runtime_dir / "last_sampler.txt",
    )


def normalize_run_spec(
    raw_config: dict[str, Any],
    project_id: str,
    cwd: str | Path,
    experiment_id: str | None = None,
    run_id: str | None = None,
) -> RunSpec:
    """Validate and normalize a Tinker run spec."""
    cfg = get_research_config()
    tinker_defaults = cfg.get("tinker", {})
    normalized = dict(raw_config or {})
    normalized["project_id"] = project_id
    if experiment_id:
        normalized["experiment_id"] = experiment_id
    if run_id:
        normalized["run_id"] = run_id
    normalized.setdefault("method", tinker_defaults.get("default_method", "sft"))
    normalized.setdefault("base_model", tinker_defaults.get("default_base_model", "meta-llama/Llama-3.1-8B"))
    normalized.setdefault("lora_rank", tinker_defaults.get("default_lora_rank", 32))
    normalized.setdefault("learning_rate", tinker_defaults.get("default_learning_rate", 1e-4))
    normalized.setdefault("batch_size", 32)
    normalized.setdefault("num_epochs", 1)
    normalized.setdefault("save_every", 50)
    normalized.setdefault("field_map", {})
    normalized.setdefault("sampling_params", {})
    normalized.setdefault("optimizer", {})
    normalized.setdefault("metadata", {})

    dataset_path = normalized.get("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path is required")
    dataset = Path(dataset_path)
    if not dataset.is_absolute():
        dataset = Path(cwd).resolve() / dataset
    normalized["dataset_path"] = str(dataset.resolve())

    method = str(normalized.get("method", "sft")).strip().lower()
    field_map = dict(normalized.get("field_map") or {})
    if method == "sft":
        field_map.setdefault("prompt", "prompt")
        field_map.setdefault("completion", "completion")
    elif method == "dpo":
        field_map.setdefault("prompt", "prompt")
        field_map.setdefault("chosen", "chosen")
        field_map.setdefault("rejected", "rejected")
    elif method == "ppo":
        field_map.setdefault("prompt", "prompt")
        field_map.setdefault("completion", "completion")
        field_map.setdefault("logprobs", "logprobs")
        if "advantage" not in field_map and "reward" not in field_map:
            field_map["reward"] = "reward"
    else:
        raise ValueError(f"Unsupported training method: {method}")
    normalized["field_map"] = field_map

    if method == "ppo" and "logprobs" not in field_map:
        raise ValueError("ppo runs require field_map.logprobs")

    return RunSpec.model_validate(normalized)


def estimate_run_cost(run_spec: RunSpec | dict[str, Any], config: dict[str, Any] | None = None) -> float:
    """Estimate the USD cost of a training run.

    Formula: (num_rows / batch_size) * num_epochs * seconds_per_step * overhead * (gpu_hour_cost / 3600)
    DPO gets a 2.5x multiplier, PPO gets 3x.
    """
    if config is None:
        config = get_research_config()
    cost_cfg = config.get("cost_estimation", {})
    gpu_hour_cost = float(cost_cfg.get("gpu_hour_cost_usd", 2.50))
    overhead = float(cost_cfg.get("overhead_multiplier", 1.1))

    spec = run_spec if isinstance(run_spec, dict) else run_spec.model_dump(mode="json")
    batch_size = max(int(spec.get("batch_size", 32)), 1)
    num_epochs = max(int(spec.get("num_epochs", 1)), 1)
    method = str(spec.get("method", "sft")).lower()

    # Estimate total rows — if dataset is available, try to count; otherwise use 1000 as fallback
    dataset_path = spec.get("dataset_path", "")
    num_rows = 0
    if dataset_path:
        from pathlib import Path as _Path
        dp = _Path(dataset_path)
        if dp.exists():
            try:
                content = dp.read_text(encoding="utf-8")
                num_rows = sum(1 for line in content.splitlines() if line.strip())
            except Exception:
                num_rows = 1000
        else:
            num_rows = 1000
    else:
        num_rows = 1000

    steps_per_epoch = max(num_rows // batch_size, 1)
    total_steps = steps_per_epoch * num_epochs

    # Rough estimate: ~2 seconds per step for a 7-8B model with LoRA
    seconds_per_step = 2.0
    total_seconds = total_steps * seconds_per_step * overhead

    # Method multipliers
    if method == "dpo":
        total_seconds *= 2.5
    elif method == "ppo":
        total_seconds *= 3.0

    cost_usd = total_seconds * (gpu_hour_cost / 3600)
    return round(cost_usd, 2)


def launch_background_run(paths: TinkerRuntimePaths, cwd: str | Path) -> subprocess.Popen[str]:
    """Launch a generated Tinker runner in the background."""
    log_handle = paths.log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(paths.runner_path)],
        cwd=str(cwd),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    paths.pid_path.write_text(str(process.pid), encoding="utf-8")
    return process


def build_runner_script(run_spec: RunSpec, paths: TinkerRuntimePaths) -> str:
    """Render a self-contained Tinker runner script."""
    config_json = json.dumps(run_spec.model_dump(mode="json"), indent=2, ensure_ascii=False)
    payload = {
        "config_path": str(paths.config_path),
        "status_path": str(paths.status_path),
        "checkpoint_path": str(paths.last_checkpoint_path),
        "sampler_path": str(paths.last_sampler_path),
        "generated_at": now_utc(),
    }
    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    return f'''#!/usr/bin/env python3
import json
import math
import traceback
from pathlib import Path


RUNTIME = {payload_json}
CONFIG = {config_json}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_status(status: str, **extra) -> None:
    payload = {{
        "run_id": CONFIG["run_id"],
        "project_id": CONFIG["project_id"],
        "status": status,
        **extra,
    }}
    write_json(Path(RUNTIME["status_path"]), payload)


def load_rows(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {{path}}")
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must contain a list of records")
        return payload
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def batched(rows, size):
    for idx in range(0, len(rows), size):
        yield rows[idx : idx + size]


def make_supervised_datum(prompt: str, completion: str, tokenizer, types_module):
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)
    return types_module.Datum(
        model_input=types_module.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={{
            "weights": weights[1:],
            "target_tokens": all_tokens[1:],
        }},
    )


def build_sft_batch(rows, tokenizer, types_module):
    field_map = CONFIG["field_map"]
    return [
        make_supervised_datum(str(row[field_map["prompt"]]), str(row[field_map["completion"]]), tokenizer, types_module)
        for row in rows
    ]


def build_dpo_batch(rows, tokenizer, types_module):
    field_map = CONFIG["field_map"]
    datums = []
    for row in rows:
        prompt = str(row[field_map["prompt"]])
        datums.append(make_supervised_datum(prompt, str(row[field_map["chosen"]]), tokenizer, types_module))
        datums.append(make_supervised_datum(prompt, str(row[field_map["rejected"]]), tokenizer, types_module))
    return datums


def build_ppo_batch(rows, tokenizer, types_module):
    field_map = CONFIG["field_map"]
    reward_field = field_map.get("reward")
    advantage_field = field_map.get("advantage")
    rewards = []
    if not advantage_field and reward_field:
        rewards = [float(row.get(reward_field, 0.0)) for row in rows]
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    else:
        reward_mean = 0.0

    datums = []
    for row in rows:
        prompt = str(row[field_map["prompt"]])
        completion = str(row[field_map["completion"]])
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
        logprobs = row[field_map["logprobs"]]
        if len(logprobs) != len(completion_tokens):
            raise ValueError("logprobs length must match completion token length for ppo")
        advantage = float(row.get(advantage_field, 0.0)) if advantage_field else float(row.get(reward_field, 0.0)) - reward_mean
        all_tokens = prompt_tokens + completion_tokens
        ob_len = len(prompt_tokens) - 1
        datums.append(
            types_module.Datum(
                model_input=types_module.ModelInput.from_ints(tokens=all_tokens[:-1]),
                loss_fn_inputs={{
                    "target_tokens": all_tokens[1:],
                    "weights": ([0] * len(prompt_tokens) + [1] * len(completion_tokens))[1:],
                    "logprobs": ([0.0] * ob_len) + [float(x) for x in logprobs],
                    "advantages": ([0.0] * ob_len) + ([advantage] * len(completion_tokens)),
                }},
            )
        )
    return datums


def save_checkpoint(training_client, name):
    state_result = training_client.save_state(name=name).result()
    sampler_result = training_client.save_weights_for_sampler(name=name).result()
    Path(RUNTIME["checkpoint_path"]).write_text(str(state_result.path), encoding="utf-8")
    Path(RUNTIME["sampler_path"]).write_text(str(sampler_result.path), encoding="utf-8")
    return str(state_result.path), str(sampler_result.path)


def run():
    import tinker

    rows = load_rows(CONFIG["dataset_path"])
    if not rows:
        raise ValueError("dataset is empty")

    service = tinker.ServiceClient()
    training_client = service.create_lora_training_client(
        base_model=CONFIG["base_model"],
        rank=CONFIG["lora_rank"],
    )
    if CONFIG.get("resume_from"):
        training_client.load_state(CONFIG["resume_from"]).result()

    tokenizer = training_client.get_tokenizer()
    batch_size = int(CONFIG.get("batch_size", 32))
    save_every = int(CONFIG.get("save_every", 50))
    learning_rate = float(CONFIG.get("learning_rate", 1e-4))
    total_steps = 0
    start_step = int(CONFIG.get("resume_step", 0) or 0)
    write_status("running", step=start_step, total_rows=len(rows))

    if CONFIG["method"] == "dpo":
        import torch
        import torch.nn.functional as F

        reference_client = service.create_lora_training_client(
            base_model=CONFIG.get("reference_model_name") or CONFIG["base_model"],
            rank=CONFIG["lora_rank"],
        ).save_weights_and_get_sampling_client("reference")

        for epoch_idx in range(int(CONFIG.get("num_epochs", 1))):
            for batch_idx, batch_rows in enumerate(batched(rows, batch_size)):
                total_steps += 1
                if total_steps <= start_step:
                    continue
                datums = build_dpo_batch(batch_rows, tokenizer, tinker.types)
                full_sequences = []
                for datum in datums:
                    targets = list(datum.loss_fn_inputs["target_tokens"])
                    full_sequences.append(datum.model_input.append_int(int(targets[-1])))

                ref_logprobs = []
                for sequence in full_sequences:
                    raw = reference_client.compute_logprobs(sequence).result()
                    if hasattr(raw, "logprobs"):
                        raw = raw.logprobs
                    ref_logprobs.append(torch.tensor(list(raw)[1:]).float())

                def dpo_loss_fn(data, logprobs_list):
                    chosen_data = [datum for idx, datum in enumerate(data) if idx % 2 == 0]
                    rejected_data = [datum for idx, datum in enumerate(data) if idx % 2 == 1]
                    chosen_logprobs = []
                    chosen_ref_logprobs = []
                    rejected_logprobs = []
                    rejected_ref_logprobs = []

                    for idx in range(len(chosen_data)):
                        c_lp = torch.as_tensor(logprobs_list[idx * 2]).float()
                        c_ref = ref_logprobs[idx * 2]
                        c_weights = torch.tensor(list(chosen_data[idx].loss_fn_inputs["weights"])).float()
                        chosen_logprobs.append(torch.dot(c_lp, c_weights))
                        chosen_ref_logprobs.append(torch.dot(c_ref, c_weights))

                        r_lp = torch.as_tensor(logprobs_list[idx * 2 + 1]).float()
                        r_ref = ref_logprobs[idx * 2 + 1]
                        r_weights = torch.tensor(list(rejected_data[idx].loss_fn_inputs["weights"])).float()
                        rejected_logprobs.append(torch.dot(r_lp, r_weights))
                        rejected_ref_logprobs.append(torch.dot(r_ref, r_weights))

                    beta = float(CONFIG.get("beta", 0.1))
                    chosen_delta = torch.stack([lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)])
                    rejected_delta = torch.stack([lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)])
                    losses = -F.logsigmoid(beta * (chosen_delta - rejected_delta))
                    loss = losses.mean()
                    return loss, {{
                        "dpo_loss": float(loss.item()),
                        "accuracy": float((chosen_delta > rejected_delta).float().mean().item()),
                        "margin": float((chosen_delta - rejected_delta).mean().item()),
                    }}

                training_client.forward_backward_custom(datums, dpo_loss_fn).result()
                training_client.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result()
                if save_every > 0 and total_steps % save_every == 0:
                    checkpoint_path, sampler_path = save_checkpoint(training_client, f"step_{{total_steps}}")
                    write_status("running", step=total_steps, epoch=epoch_idx, batch=batch_idx, last_checkpoint=checkpoint_path, last_sampler=sampler_path)

    elif CONFIG["method"] == "ppo":
        ppo_config = CONFIG.get("optimizer", {{}}).get("loss_fn_config", {{"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}})
        for epoch_idx in range(int(CONFIG.get("num_epochs", 1))):
            for batch_idx, batch_rows in enumerate(batched(rows, batch_size)):
                total_steps += 1
                if total_steps <= start_step:
                    continue
                datums = build_ppo_batch(batch_rows, tokenizer, tinker.types)
                training_client.forward_backward(datums, loss_fn="ppo", loss_fn_config=ppo_config).result()
                training_client.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result()
                if save_every > 0 and total_steps % save_every == 0:
                    checkpoint_path, sampler_path = save_checkpoint(training_client, f"step_{{total_steps}}")
                    write_status("running", step=total_steps, epoch=epoch_idx, batch=batch_idx, last_checkpoint=checkpoint_path, last_sampler=sampler_path)

    else:
        for epoch_idx in range(int(CONFIG.get("num_epochs", 1))):
            for batch_idx, batch_rows in enumerate(batched(rows, batch_size)):
                total_steps += 1
                if total_steps <= start_step:
                    continue
                batch = build_sft_batch(batch_rows, tokenizer, tinker.types)
                training_client.forward_backward(batch, loss_fn="cross_entropy").result()
                training_client.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result()
                if save_every > 0 and total_steps % save_every == 0:
                    checkpoint_path, sampler_path = save_checkpoint(training_client, f"step_{{total_steps}}")
                    write_status("running", step=total_steps, epoch=epoch_idx, batch=batch_idx, last_checkpoint=checkpoint_path, last_sampler=sampler_path)

    checkpoint_path, sampler_path = save_checkpoint(training_client, CONFIG.get("checkpoint_name", "final"))
    write_status("completed", step=max(total_steps, start_step), last_checkpoint=checkpoint_path, last_sampler=sampler_path)


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        write_status("failed", error=f"{{type(exc).__name__}}: {{exc}}", traceback=traceback.format_exc())
        raise
'''
