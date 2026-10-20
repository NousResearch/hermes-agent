# slime Troubleshooting Guide

## Common Issues and Solutions

### SGLang Issues

#### Issue: SGLang Engine Crash

**Symptoms**: Inference engine dies mid-training, connection errors

**Solutions**:

1. **Enable fault tolerance**:
```bash
--use-fault-tolerance
```

2. **Increase memory allocation**:
```bash
--sglang-mem-fraction-static 0.85  # Increase from 0.8
```

3. **Reduce batch size**:
```bash
--rollout-batch-size 16  # Reduce from 32
```

4. **Disable CUDA graphs** (for debugging):
```bash
--sglang-disable-cuda-graph
```

#### Issue: SGLang Router Load Imbalance

**Symptoms**: Some SGLang engines overloaded while others idle

**Solutions**:

1. **Adjust routing strategy**:
```bash
--sglang-router-strategy round_robin
```

2. **Increase number of engines**:
```bash
--rollout-num-gpus-per-engine 1  # More engines, less GPUs each
```

### Weight Synchronization Issues

#### Issue: Weight Sync Timeout

**Symptoms**: Training hangs after rollout, timeout errors

**Solutions**:

1. **Increase sync interval** (async mode):
```bash
--update-weights-interval 5  # Increase from 2
```

2. **Use colocated mode** (eliminates network transfer):
```bash
--colocate
```

3. **Check network bandwidth**:
```bash
# Verify InfiniBand is enabled
ibstat
```

#### Issue: Weight Sync Failures in Multi-Node

**Symptoms**: Nodes fail to receive updated weights

**Solutions**:

1. **Set NCCL environment**:
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
```

2. **Increase timeout**:
```bash
export NCCL_TIMEOUT=1800
```

### Memory Issues

#### Issue: OOM During Training

**Symptoms**: CUDA OOM in backward pass

**Solutions**:

1. **Enable gradient checkpointing**:
```bash
--recompute-activations
```

2. **Reduce micro-batch size**:
```bash
--micro-batch-size 1
```

3. **Enable sequence parallelism**:
```bash
--sequence-parallel
```

4. **Reduce global batch size**:
```bash
--global-batch-size 128  # Reduce from 256
```

#### Issue: OOM in Colocated Mode

**Symptoms**: OOM when both training and inference run on same GPUs

**Solutions**:

1. **Reduce SGLang memory**:
```bash
--sglang-mem-fraction-static 0.4  # Reduce from 0.8
```

2. **Enable offloading**:
```bash
--offload-optimizer-states
```

3. **Use smaller sequence length**:
```bash
--seq-length 2048  # Reduce from 4096
```

### Data Loading Issues

#### Issue: Slow Data Loading

**Symptoms**: GPU idle during data fetch, low GPU utilization

**Solutions**:

1. **Increase data workers**:
```bash
--num-data-workers 4
```

2. **Use streaming dataset**:
```bash
--streaming-data
```

3. **Pre-tokenize data**:
```python
# Pre-process data offline
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model_path")
# Save tokenized data
```

#### Issue: Data Format Errors

**Symptoms**: KeyError, missing fields, parsing failures

**Solutions**:

1. **Verify data format**:
```python
import json
with open("data.jsonl") as f:
    for line in f:
        data = json.loads(line)
        assert "prompt" in data, "Missing prompt field"
        assert "label" in data, "Missing label field"
```

2. **Check key names**:
```bash
--input-key prompt  # Must match your data
--label-key label   # Must match your data
```

### Training Stability Issues

#### Issue: Loss Explosion / NaN

**Symptoms**: Loss becomes NaN or explodes

**Solutions**:

1. **Reduce learning rate**:
```bash
--lr 1e-6  # Reduce from 5e-6
```

2. **Enable gradient clipping**:
```bash
--clip-grad 1.0
```

3. **Check for data issues**:
```python
# Verify no empty prompts or responses
for sample in dataset:
    assert len(sample["prompt"]) > 0
```

4. **Use BF16 instead of FP16**:
```bash
--bf16  # More numerically stable
```

#### Issue: Reward Collapse

**Symptoms**: Reward drops to zero, model outputs garbage

**Solutions**:

1. **Increase KL penalty**:
```bash
--kl-loss-coef 0.01  # Increase from 0.001
```

2. **Reduce number of samples**:
```bash
--n-samples-per-prompt 4  # Reduce from 8
```

3. **Verify reward function**:
```python
# Test reward function independently
from custom_rm import reward_func
sample = Sample(prompt="test", response="test response")
reward = reward_func(args, sample)
print(f"Reward: {reward}")  # Should be reasonable
```

### Async Training Issues

#### Issue: Async Training Not Supported with Colocate

**Symptoms**: Error when using `--colocate` with `train_async.py`

**Solution**: Colocated mode is NOT supported for async training. Use separate GPUs:
```bash
# Remove --colocate flag
python train_async.py \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    # No --colocate
```

#### Issue: Stale Weights in Async Mode

**Symptoms**: Policy divergence, inconsistent behavior

**Solutions**:

1. **Reduce async buffer size**:
```bash
--async-buffer-size 2  # Reduce from 4
```

2. **Increase weight update frequency**:
```bash
--update-weights-interval 1  # Sync every rollout
```

### Multi-Turn Training Issues

#### Issue: Tool Responses Included in Loss

**Symptoms**: Model learns to output tool responses verbatim

**Solution**: Properly set loss mask in custom generate function:
```python
def build_loss_mask(sample):
    """Create loss mask that excludes tool responses."""
    mask = []
    for i, token in enumerate(sample.tokens):
        if is_tool_response(token, sample.metadata):
            mask.append(0)  # Don't compute loss
        else:
            mask.append(1)  # Compute loss
    return mask
```

#### Issue: Multi-Turn Context Too Long

**Symptoms**: OOM or truncation in multi-turn conversations

**Solutions**:

1. **Limit conversation history**:
```python
# In custom generate function
conversation = sample.prompt[-10:]  # Keep last 10 turns
```

2. **Increase context length**:
```bash
--sglang-context-length 16384
```

### Checkpoint Issues

#### Issue: Checkpoint Loading Fails

**Symptoms**: Cannot load saved checkpoint

**Solutions**:

1. **Verify checkpoint path**:
```bash
ls -la /path/to/checkpoint/
```

2. **Check parallelism matches**:
```bash
# Checkpoint was saved with TP=2, must load with TP=2
--tensor-model-parallel-size 2
```

3. **Convert HuggingFace to Megatron** (if needed):
```bash
python tools/convert_hf_to_megatron.py \
    --hf_model_path /path/to/hf/model \
    --save_path /path/to/megatron/checkpoint
```

### Debugging Tips

#### Enable Verbose Logging

```bash
--log-level DEBUG
export SLIME_DEBUG=1
```

#### Check GPU Utilization

```bash
watch -n 1 nvidia-smi
```

#### Monitor Training

```bash
tensorboard --logdir outputs/
```

#### Test Custom Functions Independently

```python
# Test reward function
import asyncio
from custom_rm import reward_func

async def test():
    sample = Sample(prompt="test", response="test", label="expected")
    reward = await reward_func(args, sample)
    print(f"Reward: {reward}")

asyncio.run(test())
```

## Constraint Reference

Key constraint to remember:

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

Example: `32 × 8 = 256 × 1`

## Resources

- GitHub Issues: https://github.com/THUDM/slime/issues
- Documentation: https://thudm.github.io/slime/
- Examples: `examples/` directory
