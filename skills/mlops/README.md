# MLOps Skills

Machine learning operations, model training, inference, evaluation, and deployment workflows for Hermes Agent.

## Overview

This category contains 13 skills for the complete ML lifecycle — from training and fine-tuning models to serving them in production, evaluating performance, and managing experiments. Whether you're fine-tuning LLMs, running benchmarks, or deploying high-throughput inference servers, these skills provide professional MLOps workflows.

## Available Skills

### Model Hub & Distribution

#### **huggingface-hub**
HuggingFace CLI for searching, downloading, and uploading models and datasets.

**Use when:** Managing models and datasets from Hugging Face Hub programmatically.

**Key features:**
- Search models and datasets
- Download and upload artifacts
- Manage model cards and metadata
- CLI-based workflows (`hf` command)

---

### Training & Fine-Tuning

#### **unsloth**
2-5x faster LoRA/QLoRA fine-tuning with reduced VRAM usage.

**Use when:** Fine-tuning LLMs efficiently on consumer hardware or when speed matters.

**Key features:**
- 2-5x faster than standard training
- Reduced VRAM requirements
- LoRA and QLoRA support
- Compatible with popular models

---

#### **axolotl**
YAML-based LLM fine-tuning framework supporting LoRA, DPO, and GRPO.

**Use when:** You need declarative, reproducible fine-tuning configurations.

**Key features:**
- YAML configuration for reproducibility
- LoRA, DPO, GRPO support
- Multi-dataset training
- Extensive model compatibility

---

#### **fine-tuning-with-trl** (TRL)
Transformer Reinforcement Learning — SFT, DPO, PPO, GRPO, and reward modeling for RLHF.

**Use when:** Implementing RLHF workflows, preference optimization, or reward modeling.

**Key features:**
- Supervised Fine-Tuning (SFT)
- Direct Preference Optimization (DPO)
- PPO and GRPO for RL
- Reward model training
- Complete RLHF pipeline

---

### Inference & Serving

#### **vllm** (serving-llms-vllm)
High-throughput LLM serving with OpenAI-compatible API and quantization support.

**Use when:** Deploying LLMs in production requiring high throughput and low latency.

**Key features:**
- PagedAttention for efficient memory
- OpenAI API compatibility
- Quantization (AWQ, GPTQ)
- Batching and streaming
- Production-grade serving

---

#### **llama-cpp**
Local GGUF model inference with Hugging Face Hub integration.

**Use when:** Running quantized models locally on CPU/GPU with minimal dependencies.

**Key features:**
- GGUF format support
- CPU and GPU inference
- Low memory footprint
- HuggingFace Hub discovery
- Cross-platform compatibility

---

#### **outlines**
Structured generation — enforce JSON schemas, regex patterns, or Pydantic models in LLM outputs.

**Use when:** You need guaranteed output formats (JSON, structured data) from LLMs.

**Key features:**
- JSON schema enforcement
- Regex-based generation
- Pydantic model integration
- Constrained decoding
- Type-safe outputs

---

#### **obliteratus** (OBLITERATUS)
Remove LLM refusals using difference-in-means ablation technique.

**Use when:** You need uncensored models for research or want to study alignment mechanisms.

**Key features:**
- Abliteration via diff-in-means
- Removes safety refusals
- Research and analysis tool
- Works with various LLMs

**Note:** Use responsibly for research purposes.

---

### Evaluation & Monitoring

#### **evaluating-llms-harness** (lm-evaluation-harness)
Benchmark LLMs on standard tasks (MMLU, GSM8K, HumanEval, etc.).

**Use when:** Evaluating model performance on academic benchmarks or comparing models.

**Key features:**
- 60+ evaluation tasks
- MMLU, GSM8K, HumanEval, HellaSwag, etc.
- Few-shot evaluation
- Reproducible benchmarks
- Community-standard metrics

---

#### **weights-and-biases** (W&B)
Experiment tracking, hyperparameter sweeps, model registry, and dashboards.

**Use when:** Managing ML experiments, tracking metrics, or collaborating on model development.

**Key features:**
- Experiment logging and visualization
- Hyperparameter sweeps
- Model versioning and registry
- Team collaboration
- Custom dashboards

---

### Research & Advanced Techniques

#### **dspy**
Declarative LM programs — automatically optimize prompts, build RAG systems, and chain models.

**Use when:** Building complex LM pipelines with automatic prompt optimization.

**Key features:**
- Declarative program specification
- Automatic prompt optimization
- RAG pipeline building
- Multi-hop reasoning
- Compositional programs

---

### Specialized Models

#### **audiocraft-audio-generation** (AudioCraft)
Text-to-music (MusicGen) and text-to-sound (AudioGen) generation.

**Use when:** Generating music or sound effects from text descriptions.

**Key features:**
- MusicGen for music generation
- AudioGen for sound effects
- Text-to-audio synthesis
- Controllable generation
- High-quality outputs

---

#### **segment-anything-model** (SAM)
Zero-shot image segmentation using points, boxes, or masks as prompts.

**Use when:** Segmenting objects in images without training data.

**Key features:**
- Zero-shot segmentation
- Point, box, and mask prompts
- High-quality masks
- Fast inference
- Versatile object detection

---

## Quick Start

### Example: Fine-Tune an LLM

```bash
# 1. Fast fine-tuning with Unsloth
/unsloth "Fine-tune Llama 3.1 on my custom dataset with LoRA, 4-bit quantization"

# 2. Or use Axolotl for reproducible config
/axolotl "Create YAML config for DPO training on preference dataset"

# 3. Track with W&B
/weights-and-biases "Initialize W&B logging for fine-tuning run"
```

### Example: Deploy LLM Inference

```bash
# 1. High-throughput production serving
/vllm "Deploy Llama 3.1 8B with vLLM, enable AWQ quantization, expose OpenAI API"

# 2. Or local CPU inference
/llama-cpp "Run Mistral 7B GGUF Q4 quantization locally"

# 3. Enforce JSON output format
/outlines "Generate structured user profiles as JSON schema from Llama 3.1"
```

### Example: Evaluate Model Performance

```bash
# 1. Run standard benchmarks
/lm-eval-harness "Evaluate my fine-tuned model on MMLU, GSM8K, HumanEval"

# 2. Track experiments
/weights-and-biases "Log evaluation results to W&B project: llama-finetune"
```

### Example: Research Workflow

```bash
# 1. Build RAG system with DSPy
/dspy "Create RAG pipeline with automatic prompt optimization for Q&A"

# 2. Fine-tune with preferences
/trl "Train reward model and run DPO on preference dataset"
```

## Skill Combinations

**Complete Fine-Tuning Pipeline:**
1. Use `huggingface-hub` to download base model
2. Use `unsloth` or `axolotl` for efficient fine-tuning
3. Use `weights-and-biases` to track experiments
4. Use `lm-eval-harness` to benchmark results
5. Use `vllm` to deploy the best checkpoint

**Production Inference Stack:**
1. Use `huggingface-hub` to fetch model
2. Use `vllm` for high-throughput serving
3. Use `outlines` for structured outputs
4. Use `weights-and-biases` to monitor production metrics

**Research & Alignment:**
1. Use `dspy` to build complex LM programs
2. Use `trl` for RLHF training (SFT → Reward → PPO/DPO)
3. Use `lm-eval-harness` to measure alignment
4. Use `obliteratus` to study refusal mechanisms (research only)

**Specialized Workflows:**
1. Use `audiocraft` for music/sound generation
2. Use `segment-anything` for vision tasks
3. Combine with LLMs for multimodal applications

## Choosing the Right Tool

**For fine-tuning:**
- Fast iteration → `unsloth`
- Reproducible configs → `axolotl`
- RLHF pipeline → `trl`

**For inference:**
- Production/high-throughput → `vllm`
- Local/low-resource → `llama-cpp`
- Structured outputs → `outlines`

**For evaluation:**
- Academic benchmarks → `lm-eval-harness`
- Experiment tracking → `weights-and-biases`

**For research:**
- Prompt optimization → `dspy`
- Alignment methods → `trl`
- Ablation studies → `obliteratus`

## Common Workflows

### LLM Fine-Tuning on Custom Data

```bash
# 1. Download base model
/huggingface-hub "Download meta-llama/Llama-3.1-8B-Instruct"

# 2. Fine-tune with Unsloth (fast!)
/unsloth "LoRA fine-tune on custom instructions dataset, 4-bit, 10 epochs"

# 3. Evaluate on benchmarks
/lm-eval-harness "Run MMLU and GSM8K on fine-tuned model"

# 4. Deploy if performance is good
/vllm "Serve fine-tuned model with vLLM, quantize to AWQ"
```

### RLHF Training Pipeline

```bash
# 1. Supervised Fine-Tuning
/trl "SFT on instruction dataset"

# 2. Train reward model
/trl "Train reward model on preference pairs"

# 3. Preference optimization
/trl "Run DPO using preference dataset and SFT model"

# 4. Evaluate alignment
/lm-eval-harness "Test on HHH (helpful, honest, harmless) evals"
```

### Production Deployment

```bash
# 1. Fetch optimized model
/huggingface-hub "Download TheBloke/Llama-3.1-8B-AWQ"

# 2. Deploy with vLLM
/vllm "Start OpenAI-compatible server, enable batching, AWQ quantization"

# 3. Enforce output schemas
/outlines "Wrap vLLM with Outlines for JSON schema enforcement"

# 4. Monitor in production
/weights-and-biases "Log inference latency, throughput, error rates"
```

## Best Practices

**Fine-Tuning:**
- Start with `unsloth` for speed, use `axolotl` for reproducibility
- Always track experiments with W&B
- Benchmark before and after with `lm-eval-harness`
- Use LoRA/QLoRA for efficient training

**Inference:**
- `vllm` for production (high QPS), `llama-cpp` for local/edge
- Use quantization (AWQ, GPTQ) to reduce memory
- `outlines` when you need guaranteed output formats
- Monitor latency and throughput

**Evaluation:**
- Run standard benchmarks (`lm-eval-harness`) for reproducibility
- Use multiple metrics (accuracy, perplexity, human eval)
- Track all experiments in W&B
- Compare against baselines

**Research:**
- Use `dspy` for complex multi-step reasoning
- `trl` for alignment research
- Document all hyperparameters
- Version models and datasets

## Integration Tips

**Environment Setup:**
Most skills require Python environments with specific dependencies. Consider using:
```bash
conda create -n mlops python=3.10
conda activate mlops
```

**GPU Requirements:**
- Training: 24GB+ VRAM recommended (or use Unsloth/QLoRA for less)
- Inference: 8-24GB depending on model size and quantization
- Evaluation: Can run on CPU for most benchmarks

**Model Storage:**
Models are typically cached in `~/.cache/huggingface/`. Ensure sufficient disk space (10-100GB+ for large models).

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue describing the improvement
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **software-development/** - Debugging and testing workflows
- **research/** - Research paper analysis and writing
- **creative/** - Generative art and media creation
- **data-science/** - Data analysis and visualization

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
