# Local checkpoints, Qwen3.8, and residue loops

Load this file when the model is a directory on disk, a Qwen3.8 hybrid, an FP8/NVFP4 checkpoint, or a follow-up pass after leftover refusals.

Source of truth: OBLITERATUS `0.1.3` CLI (`obliteratus/cli.py`) and README at https://github.com/elder-plinius/OBLITERATUS

## Local path as `model`

`obliteratus obliterate`, `info`, `recommend`, `gpu-calc`, `tourney`, and `self-improve` all take a HuggingFace **name or path**. A local tree is valid when it has `config.json` and weight shards. Confirm `config.json` with `read_file` before spending GPU time.

Prefer the on-disk snapshot. Do not re-download a sibling official copy that is already present.

Always set `--output-dir` to a new directory. Do not overwrite the source tree.

Gated hub ids need `HF_TOKEN`. Local directories do not. Optional vault/file resolution order from the README: explicit arg → env var → `<NAME>_FILE` → `OBLITERATUS_SECRET_DIR` → `CREDENTIALS_DIRECTORY` → `OBLITERATUS_SECRET_COMMAND`.

## GPU selection

`--gpus` sets `CUDA_VISIBLE_DEVICES` before CUDA init:

```bash
obliteratus obliterate /abs/path/to/ckpt --gpus 0 --output-dir ./liberated
obliteratus obliterate /abs/path/to/ckpt --gpus 0,1 --output-dir ./liberated
obliteratus obliterate /abs/path/to/ckpt --gpus all --output-dir ./liberated
```

Multi-GPU uses accelerate `device_map="auto"` **pipeline** sharding: memory solution, not a speedup. Idle GPUs hold layers. Skip this path for Qwen3.8.

`--gpu-memory-utilization` is a fraction in `(0, 1]`. Default reserve is 15% or 2 GiB per GPU, whichever is larger.

`--trust-remote-code` is required only for custom modeling code (README example: Nemotron Omni).

`--dtype` CLI default on `obliterate` is `float16`. Prefer `bfloat16` when the hardware supports it.

## Quantization vs surgery dtype

| Flag | Meaning |
|---|---|
| `--quantization 4bit` / `8bit` | bitsandbytes load. Requires `pip install -e ".[quantization]"` |
| no quantization flag | native float load |
| FP8 / NVFP4 checkpoint | detected from `quantization_config`; dequantized shard-by-shard to BF16; **no extra flag** |

Peak VRAM for FP8/NVFP4 is the BF16 size, not the packed size. Saved output is plain BF16. Re-quantize afterward if you need a 4-bit serving artifact.

Unsupported or ambiguous quantization layouts fail closed with the scheme named.

README also shows UI labels like `bitsandbytes-4bit`; the **CLI choices are only `4bit` and `8bit`**.

`--large-model` enables conservative defaults for **120B+** (fewer directions, 1 pass, lower SAE expansion). Do not set it for 70B.

## Qwen3.8 hybrid contract

From the README Qwen3.8 safety status:

- Hybrid Gated DeltaNet needs supported FLA and causal-conv1d CUDA kernels for correct pristine logits.
- Install: `pip install -e ".[qwen-hybrid]"` (`transformers>=5.8`, `flash-linear-attention>=0.5.2`, `causal-conv1d>=1.7.0`).
- Place the complete text model on **one** CUDA device with 15% headroom. Stop before allocation if that cannot be met.
- Generic PyTorch fallback and automatic multi-GPU layer sharding are **rejected**.
- Saved-checkpoint Chat reloads use the same contract (FLA/causal-conv1d, SDPA, one CUDA device).
- Pristine and post-edit quality gates remain mandatory. A failed pristine checkpoint is never modified.

```bash
obliteratus obliterate /abs/path/to/Qwen3.8-27B \
  --dtype bfloat16 \
  --gpus 0 \
  --output-dir ./liberated-qwen38
```

Use `--quantization 4bit` only when native BF16 cannot fit that single device.

## Layer / shield knobs (optional)

Only set these when `recommend` or a failed first pass justifies it:

| Flag | Role |
|---|---|
| `--min-layer-fraction` / `--max-layer-fraction` | Restrict edits to a depth band (e.g. `0.75` keeps the final quarter) |
| `--harmless-pc-count` | Subtract top harmless activation PCs from refusal directions |
| `--shield-concept-count` / `--shield-ridge` / `--shield-residualize` / `--shield-layer-penalty` | Capability/style shield atoms |
| `--projection-target {all,attention,ffn,output}` | Which modules to project |
| `--projection-row-fraction` | Project only the strongest fraction of rows/columns |
| `--n-directions` | Override direction count |
| `--regularization` | Fraction to preserve (`0.0`–`1.0`) |
| `--refinement-passes` | Iterative re-probe passes |

## Quality gates

CLI fail-closed defaults on `obliterate`:

| Flag | Default |
|---|---|
| `--verify-sample-size` | 30 harmful prompts |
| `--refusal-max-tokens` | 128 |
| `--max-perplexity-increase` | `3.0` × stock perplexity |
| `--min-coherence-retention` | `0.5` |
| `--max-degenerate-fraction` | `0.2` |

Practical local-model target (not the CLI abort threshold): refusal rate ≤ 5%, ideally ≤ 3%, degeneracy well under `0.2`. A 3.0× perplexity abort is a catastrophe brake, not a quality goal.

`--prompt-pairs-file` is a local UTF-8 JSON object with exactly `harmful` and `harmless` arrays. Mutually exclusive with `--residue-file`. `--dataset` / `--residue-weight` / `--residue-max` cannot mix with `--prompt-pairs-file`.

## Residue loop (leftover refusals)

```bash
obliteratus self-improve /abs/path/to/ckpt \
  --audit /abs/path/to/refusal-audit.json \
  --output-dir ./liberated-pass2 \
  --method advanced \
  --direction-method diff_means
```

`--audit` is repeatable. `--dry-run` writes residue/plan without surgery.

Or stay on `obliterate` and pass `--residue-file` (repeatable) with `--residue-weight` (default 5).

## Capability check vs stock

```bash
obliteratus capability-check \
  --abliterated ./liberated \
  --stock /abs/path/to/ckpt \
  --dtype bfloat16 \
  --quick
```

`--quick` is 5 subjects. Full MMLU is the default without `--quick`.

Vision/MTP restore after text-only surgery: `obliteratus restore-multimodal --abliterated <dir> --stock <dir> --output <dir>`.

## Durable runs

```bash
obliteratus runs launch --notes "local 8B advanced" -- /abs/path/to/ckpt --method advanced --output-dir ./liberated
obliteratus runs list
obliteratus runs status <run_id>
```

Archive root defaults to `OBLITERATUS_RUN_ARCHIVE` or the user state directory.

## Remote SSH (optional)

`--remote [user@]host`, `--ssh-key`, `--ssh-port` (1–65535, default 22), `--remote-dir` (default `/tmp/obliteratus_run`), `--remote-python` (default `python3`), `--no-sync`.
