from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_GPUS = {"T4", "L4", "G4", "H100", "A100"}
ALLOWED_TPUS = {"v5e1", "v6e1"}
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 6
DEFAULT_STATUS_TIMEOUT_SECONDS = 20


STATUS_SCHEMA = {
    "description": "Report Google Colab CLI availability and platform support.",
    "type": "object",
    "properties": {
        "probe_sessions": {
            "type": "boolean",
            "description": "Also run a read-only colab sessions probe.",
            "default": False,
        },
        "auth": {
            "type": "string",
            "enum": ["adc", "oauth2"],
            "description": "Colab CLI auth mode.",
            "default": "adc",
        },
        "config_path": {
            "type": "string",
            "description": "Optional isolated colab-cli sessions config path.",
        },
        "allow_windows_native": {
            "type": "boolean",
            "description": "Allow unsupported native Windows colab executable if explicitly requested.",
            "default": False,
        },
    },
}

SESSIONS_SCHEMA = {
    "description": "Run read-only colab sessions through the Google Colab CLI.",
    "type": "object",
    "properties": {
        "auth": {"type": "string", "enum": ["adc", "oauth2"], "default": "adc"},
        "config_path": {"type": "string"},
        "allow_windows_native": {"type": "boolean", "default": False},
    },
}

RUN_SCHEMA = {
    "description": "Run a local Python script through colab run after explicit confirmation.",
    "type": "object",
    "properties": {
        "script_path": {
            "type": "string",
            "description": "Local Python script to execute remotely.",
        },
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Arguments forwarded to the remote script.",
        },
        "gpu": {
            "type": "string",
            "enum": sorted(ALLOWED_GPUS),
            "description": "Optional GPU accelerator.",
        },
        "tpu": {
            "type": "string",
            "enum": sorted(ALLOWED_TPUS),
            "description": "Optional TPU accelerator.",
        },
        "keep": {
            "type": "boolean",
            "description": "Keep the VM after the script exits.",
            "default": False,
        },
        "session_name": {
            "type": "string",
            "description": "Optional stable session name passed to colab run.",
        },
        "auth": {
            "type": "string",
            "enum": ["adc", "oauth2"],
            "default": "adc",
        },
        "config_path": {
            "type": "string",
            "description": "Optional isolated colab-cli sessions config path.",
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Subprocess timeout for the local colab command.",
            "default": DEFAULT_TIMEOUT_SECONDS,
        },
        "confirmed": {
            "type": "boolean",
            "description": "Must be true because colab run can allocate billable compute.",
            "default": False,
        },
        "allow_windows_native": {
            "type": "boolean",
            "description": "Allow unsupported native Windows colab executable if explicitly requested.",
            "default": False,
        },
    },
    "required": ["script_path", "confirmed"],
}

SFT_TEMPLATE_SCHEMA = {
    "description": "Write a Hermes-oriented TRL SFT/QLoRA Colab job template.",
    "type": "object",
    "properties": {
        "output_path": {
            "type": "string",
            "description": "Destination Python script path.",
        },
        "model_id": {
            "type": "string",
            "description": "Base model id.",
            "default": "Qwen/Qwen3-0.6B",
        },
        "dataset_name": {
            "type": "string",
            "description": "Hugging Face dataset name.",
            "default": "trl-lib/Capybara",
        },
        "dataset_split": {
            "type": "string",
            "default": "train",
        },
        "output_dir": {
            "type": "string",
            "default": "./hermes-sft-adapter",
        },
        "max_steps": {
            "type": "integer",
            "default": 120,
        },
        "push_to_hub_repo": {
            "type": "string",
            "description": "Optional repo id for adapter upload.",
        },
    },
    "required": ["output_path"],
}


@dataclass(frozen=True)
class ColabBackend:
    mode: str
    available: bool
    command: list[str]
    reason: str = ""
    wsl: bool = False


def to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _is_windows() -> bool:
    return os.name == "nt"


def _colab_exe() -> str | None:
    return shutil.which("colab")


def _wsl_exe() -> str | None:
    return shutil.which("wsl.exe") or shutil.which("wsl")


def _run_command(
    command: list[str],
    *,
    timeout_seconds: int | float,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": None,
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"timed out after {timeout_seconds} seconds",
        }
    except OSError as exc:
        return {
            "ok": False,
            "exit_code": None,
            "command": command,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
        }


def _wsl_run(script: str, timeout_seconds: int | float) -> dict[str, Any]:
    wsl = _wsl_exe()
    if not wsl:
        return {
            "ok": False,
            "exit_code": None,
            "command": ["wsl", "-e", "bash", "-lc", script],
            "stdout": "",
            "stderr": "",
            "error": "WSL executable was not found.",
        }
    return _run_command([wsl, "-e", "bash", "-lc", script], timeout_seconds=timeout_seconds)


def _wsl_colab_available() -> bool:
    result = _wsl_run("command -v colab >/dev/null 2>&1", DEFAULT_STATUS_TIMEOUT_SECONDS)
    return bool(result["ok"])


def resolve_backend(*, allow_windows_native: bool = False) -> ColabBackend:
    native = _colab_exe()
    if _is_windows():
        if _wsl_exe() and _wsl_colab_available():
            return ColabBackend(
                mode="wsl",
                available=True,
                command=[_wsl_exe() or "wsl.exe", "-e", "bash", "-lc"],
                wsl=True,
            )
        if native and allow_windows_native:
            return ColabBackend(
                mode="native-windows-unsupported",
                available=True,
                command=[native],
                reason="Native Windows is not officially supported by google-colab-cli.",
            )
        reason = (
            "google-colab-cli currently supports Linux and macOS. "
            "On Windows, install it inside WSL or pass allow_windows_native after accepting the unsupported path."
        )
        return ColabBackend(mode="unavailable", available=False, command=[], reason=reason)

    if native:
        return ColabBackend(mode="native", available=True, command=[native])
    return ColabBackend(
        mode="unavailable",
        available=False,
        command=[],
        reason="The colab executable was not found on PATH.",
    )


def _quote_for_wsl(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


def _wslpath(path: Path) -> str:
    result = _wsl_run(
        "wslpath -a " + shlex.quote(str(path)),
        DEFAULT_STATUS_TIMEOUT_SECONDS,
    )
    if result["ok"]:
        converted = result["stdout"].strip().splitlines()
        if converted:
            return converted[-1]
    # Fallback for default WSL mount layout.
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = str(resolved)[3:].replace("\\", "/")
    if drive and tail:
        return f"/mnt/{drive}/{tail}"
    return str(path)


def _global_args(values: dict[str, Any], *, wsl: bool) -> list[str]:
    args: list[str] = []
    auth = str(values.get("auth") or "adc").strip()
    if auth:
        args.append(f"--auth={auth}")
    config_path = values.get("config_path")
    if config_path:
        config = Path(str(config_path)).expanduser()
        args.extend(["--config", _wslpath(config) if wsl else str(config)])
    return args


def _run_colab(
    values: dict[str, Any],
    colab_args: list[str],
    *,
    timeout_seconds: int | float,
) -> dict[str, Any]:
    backend = resolve_backend(allow_windows_native=bool(values.get("allow_windows_native")))
    if not backend.available:
        return {
            "ok": False,
            "backend": backend.mode,
            "error": backend.reason,
            "platform": platform.platform(),
        }

    full_args = _global_args(values, wsl=backend.wsl) + colab_args
    if backend.wsl:
        shell_script = "colab " + _quote_for_wsl(full_args)
        result = _wsl_run(shell_script, timeout_seconds)
    else:
        result = _run_command(backend.command + full_args, timeout_seconds=timeout_seconds)
    result["backend"] = backend.mode
    return result


def check_available() -> bool:
    return resolve_backend().available


def status_payload(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    backend = resolve_backend(allow_windows_native=bool(values.get("allow_windows_native")))
    payload: dict[str, Any] = {
        "ok": backend.available,
        "available": backend.available,
        "backend": backend.mode,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "paths": {
            "colab": _colab_exe(),
            "wsl": _wsl_exe(),
        },
        "notes": [],
    }
    if backend.reason:
        payload["notes"].append(backend.reason)
    if backend.available:
        payload["version"] = _run_colab(
            values,
            ["version"],
            timeout_seconds=DEFAULT_STATUS_TIMEOUT_SECONDS,
        )
    if values.get("probe_sessions") and backend.available:
        payload["sessions"] = sessions_payload(values)
    return payload


def sessions_payload(values: dict[str, Any] | None = None) -> dict[str, Any]:
    values = values or {}
    return _run_colab(
        values,
        ["sessions"],
        timeout_seconds=DEFAULT_STATUS_TIMEOUT_SECONDS,
    )


def _validate_accelerator(values: dict[str, Any]) -> tuple[bool, str | None]:
    gpu = values.get("gpu")
    tpu = values.get("tpu")
    if gpu and tpu:
        return False, "Choose either gpu or tpu, not both."
    if gpu and str(gpu) not in ALLOWED_GPUS:
        return False, f"Unsupported GPU '{gpu}'. Supported: {', '.join(sorted(ALLOWED_GPUS))}."
    if tpu and str(tpu) not in ALLOWED_TPUS:
        return False, f"Unsupported TPU '{tpu}'. Supported: {', '.join(sorted(ALLOWED_TPUS))}."
    return True, None


def run_job(values: dict[str, Any]) -> dict[str, Any]:
    if not values.get("confirmed"):
        return {
            "ok": False,
            "confirmation_required": True,
            "reason": "colab run can allocate billable compute. Re-run with confirmed=true.",
        }

    valid, error = _validate_accelerator(values)
    if not valid:
        return {"ok": False, "error": error}

    raw_script = values.get("script_path")
    if not raw_script:
        return {"ok": False, "error": "script_path is required."}
    script_path = Path(str(raw_script)).expanduser()
    if not script_path.is_file():
        return {"ok": False, "error": f"Script not found: {script_path}"}

    backend = resolve_backend(allow_windows_native=bool(values.get("allow_windows_native")))
    remote_script = _wslpath(script_path) if backend.wsl else str(script_path)
    args: list[str] = ["run"]
    if values.get("gpu"):
        args.extend(["--gpu", str(values["gpu"])])
    if values.get("tpu"):
        args.extend(["--tpu", str(values["tpu"])])
    if values.get("keep"):
        args.append("--keep")
    if values.get("session_name"):
        args.extend(["-s", str(values["session_name"])])
    args.append(remote_script)
    for item in values.get("args") or []:
        args.append(str(item))

    timeout_seconds = int(values.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    result = _run_colab(values, args, timeout_seconds=timeout_seconds)
    result["script_path"] = str(script_path)
    result["accelerator"] = {"gpu": values.get("gpu"), "tpu": values.get("tpu")}
    result["kept_vm"] = bool(values.get("keep"))
    return result


def _sft_template_source(values: dict[str, Any]) -> str:
    model_id = values.get("model_id") or "Qwen/Qwen3-0.6B"
    dataset_name = values.get("dataset_name") or "trl-lib/Capybara"
    dataset_split = values.get("dataset_split") or "train"
    output_dir = values.get("output_dir") or "./hermes-sft-adapter"
    max_steps = int(values.get("max_steps") or 120)
    push_to_hub_repo = values.get("push_to_hub_repo") or ""
    return textwrap.dedent(
        f"""
        #!/usr/bin/env python3
        from __future__ import annotations

        import os
        import subprocess
        import sys

        REQUIRED = [
            "accelerate",
            "bitsandbytes>=0.46.1",
            "datasets",
            "huggingface_hub",
            "peft",
            "safetensors",
            "torch",
            "transformers",
            "trl",
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *REQUIRED])

        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer

        MODEL_ID = {model_id!r}
        DATASET_NAME = {dataset_name!r}
        DATASET_SPLIT = {dataset_split!r}
        OUTPUT_DIR = {output_dir!r}
        MAX_STEPS = {max_steps!r}
        PUSH_TO_HUB_REPO = {push_to_hub_repo!r}

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        print("Loading dataset:", DATASET_NAME, DATASET_SPLIT)
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

        def normalize(example):
            if "messages" in example or ("prompt" in example and "completion" in example) or "text" in example:
                return example
            if "instruction" in example and "output" in example:
                return {{"prompt": example["instruction"], "completion": example["output"]}}
            raise ValueError(
                "Dataset must contain messages, prompt+completion, text, or instruction+output columns."
            )

        dataset = dataset.map(normalize)

        print("Loading model:", MODEL_ID)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=hf_token,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

        assistant_only_loss = "messages" in dataset.column_names
        args = SFTConfig(
            output_dir=OUTPUT_DIR,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            save_strategy="steps",
            save_steps=max(20, MAX_STEPS // 3),
            report_to="none",
            assistant_only_loss=assistant_only_loss,
        )

        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Saved adapter:", OUTPUT_DIR)

        if PUSH_TO_HUB_REPO:
            from huggingface_hub import HfApi

            if not hf_token:
                raise RuntimeError("HF_TOKEN or HUGGINGFACE_TOKEN is required to push to the Hub.")
            api = HfApi(token=hf_token)
            api.create_repo(PUSH_TO_HUB_REPO, repo_type="model", exist_ok=True)
            api.upload_folder(
                repo_id=PUSH_TO_HUB_REPO,
                repo_type="model",
                folder_path=OUTPUT_DIR,
            )
            print("Uploaded adapter:", PUSH_TO_HUB_REPO)
        """
    ).lstrip()


def write_sft_template(values: dict[str, Any]) -> dict[str, Any]:
    raw_output = values.get("output_path")
    if not raw_output:
        return {"ok": False, "error": "output_path is required."}
    output_path = Path(str(raw_output)).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = _sft_template_source(values)
    output_path.write_text(source, encoding="utf-8", newline="\n")
    return {
        "ok": True,
        "file_path": str(output_path),
        "bytes": output_path.stat().st_size,
        "run_hint": f"hermes google-colab run --gpu T4 --confirm {output_path}",
        "notes": [
            "For gated models or Hub upload, make HF_TOKEN available inside the Colab runtime.",
            "Use T4 or L4 first unless the account has higher accelerator entitlement.",
        ],
    }


def handle_slash(raw_args: str) -> str:
    parts = shlex.split(raw_args or "")
    action = parts[0] if parts else "status"
    if action == "status":
        return to_json(status_payload({}))
    if action == "sessions":
        return to_json(sessions_payload({}))
    if action == "sft-template":
        output = parts[1] if len(parts) > 1 else "hermes_colab_sft.py"
        return to_json(write_sft_template({"output_path": output}))
    return to_json(
        {
            "ok": False,
            "error": "Supported /colab actions: status, sessions, sft-template.",
        }
    )
