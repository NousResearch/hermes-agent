# Checkpointing in TorchTitan

TorchTitan uses PyTorch Distributed Checkpoint (DCP) for fault-tolerant, interoperable checkpointing.

## Basic Configuration

```toml
[checkpoint]
enable = true
folder = "checkpoint"
interval = 500
```

## Save Model Only (Smaller Checkpoints)

Exclude optimizer state and training metadata:

```toml
[checkpoint]
enable = true
last_save_model_only = true
export_dtype = "bfloat16"  # Optional: export in lower precision
```

## Excluding Keys from Loading

Partial checkpoint loading for modified settings:

```toml
[checkpoint]
enable = true
exclude_from_loading = ["data_loader", "lr_scheduler"]
```

CLI equivalent:
```bash
--checkpoint.exclude_from_loading data_loader,lr_scheduler
```

## Creating Seed Checkpoints

Required for Pipeline Parallelism to ensure consistent initialization:

```bash
NGPU=1 CONFIG_FILE=<path_to_config> ./run_train.sh \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.context_parallel_degree 1 \
  --parallelism.expert_parallel_degree 1
```

This initializes on single CPU for reproducible initialization across any GPU count.

## Async Checkpointing

Reduce checkpoint overhead with async writes:

```toml
[checkpoint]
enable = true
async_mode = "async"  # Options: "disabled", "async", "async_with_pinned_mem"
```

## HuggingFace Conversion

### During Training

Save directly in HuggingFace format:

```toml
[checkpoint]
last_save_in_hf = true
last_save_model_only = true
```

Load from HuggingFace:

```toml
[checkpoint]
initial_load_in_hf = true

[model]
hf_assets_path = "./path/to/hf/checkpoint"
```

### Offline Conversion

Convert without running training:

```bash
# HuggingFace -> TorchTitan
python ./scripts/checkpoint_conversion/convert_from_hf.py \
  <input_dir> <output_dir> \
  --model_name llama3 \
  --model_flavor 8B

# TorchTitan -> HuggingFace
python ./scripts/checkpoint_conversion/convert_to_hf.py \
  <input_dir> <output_dir> \
  --hf_assets_path ./assets/hf/Llama3.1-8B \
  --model_name llama3 \
  --model_flavor 8B
```

### Example

```bash
python ./scripts/convert_from_hf.py \
  ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/ \
  ./initial_load_path/ \
  --model_name llama3 \
  --model_flavor 8B
```

## Converting to Single .pt File

Convert DCP sharded checkpoint to single PyTorch file:

```bash
python -m torch.distributed.checkpoint.format_utils \
  dcp_to_torch \
  torchtitan/outputs/checkpoint/step-1000 \
  checkpoint.pt
```

## Checkpoint Structure

DCP saves sharded checkpoints that can be resharded for different parallelism configurations:

```
checkpoint/
├── step-500/
│   ├── .metadata
│   ├── __0_0.distcp
│   ├── __0_1.distcp
│   └── ...
└── step-1000/
    └── ...
```

## Resume Training

Training auto-resumes from the latest checkpoint in the configured folder. To resume from a specific step:

```toml
[checkpoint]
load_step = 500  # Resume from step 500
```

## Interoperability with TorchTune

Checkpoints saved with `last_save_model_only = true` can be loaded directly into [torchtune](https://github.com/pytorch/torchtune) for fine-tuning.

## Full Configuration Example

```toml
[checkpoint]
enable = true
folder = "checkpoint"
interval = 500
load_step = -1  # -1 = latest, or specify step number
last_save_model_only = true
export_dtype = "bfloat16"
async_mode = "async"
exclude_from_loading = []
last_save_in_hf = false
initial_load_in_hf = false
create_seed_checkpoint = false
```

## Best Practices

1. **Large models**: Use `async_mode = "async"` to overlap checkpoint saves with training
2. **Fine-tuning export**: Enable `last_save_model_only` and `export_dtype = "bfloat16"` for smaller files
3. **Pipeline parallelism**: Always create seed checkpoint first
4. **Debugging**: Save frequent checkpoints during development, reduce for production
5. **HF interop**: Use conversion scripts for offline conversion, direct save/load for training workflows
