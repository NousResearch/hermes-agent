"""Fallback PEFT/QLoRA trainer for the redacted Hermes operator SFT corpus.

Axolotl remains the preferred training path. This script exists so the same
redacted corpus can still be trained in an environment that has Transformers,
PEFT, TRL/datasets, and bitsandbytes but does not have Axolotl installed.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any


REQUIRED_MODULES = ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
DEFAULT_OUTPUT_DIR = Path("training/runs/hermes-operator-peft")
DEFAULT_SMOKE_OUTPUT_DIR = Path("training/runs/hermes-operator-peft-smoke")


def check_dependencies() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(value)
    return rows


def _chat_text(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        except Exception:
            pass
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def _load_text_rows(path: Path, tokenizer: Any, *, limit: int | None = None) -> list[dict[str, str]]:
    rows = []
    for record in _read_jsonl(path):
        messages = record.get("messages")
        if isinstance(messages, list):
            rows.append({"text": _chat_text(tokenizer, messages)})
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _dependency_imports() -> dict[str, Any]:
    missing = check_dependencies()
    if missing:
        raise RuntimeError(f"missing training dependencies: {', '.join(missing)}")

    torch = importlib.import_module("torch")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")

    return {
        "torch": torch,
        "LoraConfig": peft.LoraConfig,
        "PeftModel": peft.PeftModel,
        "get_peft_model": peft.get_peft_model,
        "prepare_model_for_kbit_training": peft.prepare_model_for_kbit_training,
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoConfig": transformers.AutoConfig,
        "AutoTokenizer": transformers.AutoTokenizer,
        "BitsAndBytesConfig": transformers.BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": transformers.DataCollatorForLanguageModeling,
        "Trainer": transformers.Trainer,
        "TrainingArguments": transformers.TrainingArguments,
    }


class TextTokenDataset:
    def __init__(self, rows: list[dict[str, str]], tokenizer: Any, sequence_len: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.rows[index]["text"],
            truncation=True,
            max_length=self.sequence_len,
            padding=False,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded


def _load_tokenizer(auto_tokenizer: Any, base_model: str) -> Any:
    try:
        return auto_tokenizer.from_pretrained(base_model, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        return auto_tokenizer.from_pretrained(base_model, trust_remote_code=True)


def run_tokenize_smoke(args: argparse.Namespace) -> int:
    deps = _dependency_imports()
    tokenizer = _load_tokenizer(deps["AutoTokenizer"], args.base_model)
    rows = _load_text_rows(args.sft, tokenizer, limit=args.limit)
    if not rows:
        raise RuntimeError(f"no trainable SFT rows in {args.sft}")
    lengths = [len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) for row in rows]
    print(f"tokenize smoke: rows={len(rows)} min={min(lengths)} max={max(lengths)}")
    over_limit = sum(1 for length in lengths if length > args.sequence_len)
    if over_limit:
        print(f"tokenize smoke: {over_limit} row(s) exceed sequence_len={args.sequence_len} and will be truncated")
    return 0


def run_env_report() -> int:
    deps = _dependency_imports()
    torch = deps["torch"]
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "modules": {name: "ok" for name in REQUIRED_MODULES},
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "memory_free_mib": int(free // (1024 * 1024)),
            "memory_total_mib": int(total // (1024 * 1024)),
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def run_config_smoke(args: argparse.Namespace) -> int:
    deps = _dependency_imports()
    accelerate = importlib.import_module("accelerate")
    config = deps["AutoConfig"].from_pretrained(args.base_model, trust_remote_code=True)
    missing = [
        key for key in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads") if not hasattr(config, key)
    ]
    if missing:
        raise RuntimeError(f"model config is missing required key(s): {', '.join(missing)}")
    with accelerate.init_empty_weights():
        deps["AutoModelForCausalLM"].from_config(config, trust_remote_code=True)
    print(
        json.dumps(
            {
                "config_smoke": "ok",
                "model_type": getattr(config, "model_type", None),
                "vocab_size": getattr(config, "vocab_size", None),
                "hidden_size": getattr(config, "hidden_size", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_train(args: argparse.Namespace) -> int:
    deps = _dependency_imports()
    torch = deps["torch"]
    tokenizer = _load_tokenizer(deps["AutoTokenizer"], args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = _load_text_rows(args.sft, tokenizer, limit=args.limit)
    if not rows:
        raise RuntimeError(f"no trainable SFT rows in {args.sft}")

    tokenized = TextTokenDataset(rows, tokenizer, args.sequence_len)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    quant_config = deps["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = deps["AutoModelForCausalLM"].from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = deps["prepare_model_for_kbit_training"](model)
    lora_config = deps["LoraConfig"](
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    )
    model = deps["get_peft_model"](model, lora_config)
    training_args = deps["TrainingArguments"](
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        bf16=bool(torch.cuda.is_available()),
        fp16=not bool(torch.cuda.is_available()),
        report_to=[],
    )
    collator = deps["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False)
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"wrote adapter: {args.output_dir}")
    return 0


def run_merge(args: argparse.Namespace) -> int:
    deps = _dependency_imports()
    torch = deps["torch"]
    tokenizer = _load_tokenizer(deps["AutoTokenizer"], args.base_model)
    model = deps["AutoModelForCausalLM"].from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_model = deps["PeftModel"].from_pretrained(model, args.adapter_dir)
    merged = peft_model.merge_and_unload()
    args.merged_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.merged_dir)
    merged.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    print(f"wrote merged model: {args.merged_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fallback PEFT/QLoRA trainer for Hermes operator SFT data.")
    parser.add_argument(
        "command",
        choices=("check-deps", "env-report", "config-smoke", "tokenize-smoke", "train-smoke", "train", "merge"),
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft", type=Path, default=Path("training/corpora/hermes_operator_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adapter-dir", type=Path, default=Path("training/runs/hermes-operator-peft"))
    parser.add_argument("--merged-dir", type=Path, default=Path("training/runs/hermes-operator-peft-merged"))
    parser.add_argument("--sequence-len", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    args = parser.parse_args(argv)

    if args.command == "check-deps":
        missing = check_dependencies()
        if missing:
            print("missing: " + ", ".join(missing))
            return 1
        print("training dependencies: ok")
        return 0
    if args.command == "env-report":
        return run_env_report()
    if args.command == "config-smoke":
        return run_config_smoke(args)
    if args.command == "tokenize-smoke":
        return run_tokenize_smoke(args)
    if args.command == "train-smoke":
        args.limit = 1
        args.max_steps = 1
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = DEFAULT_SMOKE_OUTPUT_DIR
        args.sequence_len = min(args.sequence_len, 2048)
        return run_train(args)
    if args.command == "train":
        return run_train(args)
    if args.command == "merge":
        return run_merge(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
