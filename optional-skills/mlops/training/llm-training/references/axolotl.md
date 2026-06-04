# Axolotl Reference

Axolotl is a YAML-driven LLM fine-tuning tool supporting LoRA, QLoRA, DPO, KTO, ORPO, and GRPO methods with DeepSpeed and FSDP support.

## Installation

```bash
pip install axolotl[deepspeed] torch transformers datasets peft accelerate
```

## Core Concepts

- **YAML-first**: All training is configured via YAML files — no Python coding required
- **100+ models**: Supports Llama, Mistral, Qwen, Gemma, Phi, and more
- **Distributed training**: DeepSpeed ZeRO stages 1-3, FSDP
- **Multimodal**: Vision-language model fine-tuning
- **Compressed saving**: Save models in compressed format for vLLM/llmcompressor

## Quick Reference Patterns

### NCCL Connectivity Test

Validate data transfer speeds for multi-GPU training:

```
./build/all_reduce_perf -b 8 -e 128M -f 2 -g 3
```

### FSDP Configuration

```yaml
fsdp_version: 2
fsdp_config:
  offload_params: true
  state_dict_type: FULL_STATE_DICT
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: LlamaDecoderLayer
  reshard_after_forward: true
```

### Context Parallelism

```yaml
context_parallel_size: 4  # Must be divisor of total GPUs
```

With 8 GPUs and context_parallel_size=4: 2 different batches per step (each split across 4 GPUs). If per-GPU micro_batch_size=2, global batch size decreases from 16 to 4.

### Compressed Model Saving

```yaml
save_compressed: true  # ~40% disk reduction, compatible with vLLM/llmcompressor
```

### Custom Integrations

Place integrations anywhere — just install the package in your Python environment.

Example: https://github.com/axolotl-ai-cloud/diff-transformer

### Handling Long Sequences

```python
from axolotl.utils.trainer import drop_long_seq
# Single example: sample['input_ids'] is list[int]
# Batched: sample['input_ids'] is list[list[int]]
drop_long_seq(sample, sequence_len=2048, min_sequence_len=2)
```

## Core Classes

### AxolotlTrainer

```python
from axolotl.core.trainers.base import AxolotlTrainer

trainer = AxolotlTrainer(
    *_args,
    bench_data_collator=None,
    eval_data_collator=None,
    dataset_tags=None,
    **kwargs
)
trainer.log(logs, start_time=None)
```

### Cloud Integration

```python
from axolotl.cli.cloud.modal_ import ModalCloud

cloud = ModalCloud(config, app=None)
ModalCloud.run_cmd(cmd, run_folder, volumes=None)
```

### Prompt Strategies

```python
from axolotl.prompt_strategies.input_output import RawInputOutputPrompter
```

## Dataset Formats

Supports instruction datasets, preference data (chosen/rejected for DPO), and multimodal datasets.

## Resources

- GitHub: https://github.com/axolotl-ai-cloud/axolotl
- Docs: https://axolotl-ai-cloud.github.io/axolotl/