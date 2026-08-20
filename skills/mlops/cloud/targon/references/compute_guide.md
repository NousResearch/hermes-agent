# Targon Compute Guide

Reference documentation for Targon (Bittensor Subnet 4) — decentralized confidential GPU cloud by Manifold Labs.

---

## Architecture Overview

### What is Targon?

Targon is **Bittensor Subnet 4 (SN4)**, operated by Manifold Labs. It functions as a decentralized AI cloud marketplace where:

- **Miners** contribute GPU resources (H200s, RTX 4090s, CPUs) and are rewarded in TAO (Bittensor's native token).
- **Validators** verify that miners are performing genuine compute and not cheating.
- **Users** pay in USD (credit card or crypto) to rent compute capacity.

Unlike centralized clouds (AWS, GCP, Azure), no single entity controls the hardware. Demand and supply are matched on-chain, and miner rewards are algorithmically determined by the Bittensor protocol.

### Targon Virtual Machine (TVM)

The TVM is Targon's privacy layer. Every rental runs inside a **hardware-enforced confidential enclave**:

```
┌─────────────────────────────────────────────────────────────┐
│  Targon Virtual Machine (TVM)                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Your workload (model weights, data, code)           │   │
│  │  ← encrypted in memory, invisible to host OS         │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Confidential Computing Layer                        │   │
│  │  Intel TDX / AMD SEV / NVIDIA Confidential Computing │   │
│  └──────────────────────────────────────────────────────┘   │
│                Host (miner) hardware                        │
└─────────────────────────────────────────────────────────────┘
```

The miner who owns the GPU **cannot read your data** — the enclave is isolated at the hardware level. This is the key differentiator from RunPod, Vast.ai, and other peer-to-peer GPU marketplaces.

### Bittensor SN4 Connection

Targon integrates with Bittensor's incentive mechanism:

1. Miners register on SN4 and stake TAO.
2. Validators periodically send synthetic workloads and verify results.
3. Miners are scored on uptime, latency, and correctness.
4. TAO emissions reward high-scoring miners, creating economic incentives for quality hardware.

Users benefit indirectly: miners are economically incentivized to maintain 99%+ uptime and low-latency connections, with the Bittensor protocol as the enforcement layer.

---

## GPU Types Available

### NVIDIA H200

| Spec | Value |
|------|-------|
| VRAM | 141 GB HBM3e |
| Memory Bandwidth | 4.8 TB/s |
| FP8 Tensor TOPS | 3,958 |
| BF16 Tensor TFLOPS | 1,979 |
| Confidential Computing | Yes (NVIDIA CC) |
| Recommended For | Large model training (70B+), multi-GPU inference, fine-tuning Llama 3 405B |
| Price | ~$2/hr |

The H200 is the flagship GPU on Targon. Its 141 GB HBM3e VRAM makes it the best choice for running large models without quantization or for training at full precision.

### NVIDIA RTX 4090

| Spec | Value |
|------|-------|
| VRAM | 24 GB GDDR6X |
| Memory Bandwidth | 1,008 GB/s |
| FP8 Tensor TOPS | 1,321 |
| Confidential Computing | No (consumer card) |
| Recommended For | Inference, fine-tuning models up to 13B, diffusion models, RLHF experiments |
| Price | Competitive with market |

The 4090 is the cost-effective option for workloads that fit within 24 GB. It lacks hardware confidential computing, so sensitive data should run on H200s.

---

## Serverless vs. Rental

| Feature | Serverless | Rental (SSH) |
|---------|-----------|--------------|
| Access method | HTTP endpoint | SSH + direct GPU |
| Billing | Per-request / per-second | Per-hour |
| Cold starts | Yes (30–60s first call) | No (persistent) |
| Idle cost | None (scales to zero) | Full hourly rate |
| GPU persistence | Ephemeral | Persistent until stopped |
| Custom CUDA libs | Docker image | Full system access |
| Multi-GPU training | Not supported | Supported |
| Use case | Inference APIs, batch jobs | Training, fine-tuning, research |

**Choose Serverless when:**
- Running inference with variable or low traffic
- Deploying a model as an API with auto-scaling
- Minimizing costs by paying only for actual compute time

**Choose Rental when:**
- Running multi-day training jobs
- Needing persistent storage or custom kernel builds
- Debugging interactively over SSH
- Using multi-GPU setups (DDP, FSDP)

---

## Pricing Overview

| Resource | Price | Notes |
|----------|-------|-------|
| H200 GPU | ~$2/hr | 141 GB HBM3e, confidential compute |
| RTX 4090 | Market rate | 24 GB GDDR6X |
| CPU Instances | From $0.10/hr | For data preprocessing, orchestration |
| Serverless | Per-second | Only pay when function runs |

**Promo code**: `starter10` — get **$10 free credits** at https://targon.com. No credit card required to start.

Pricing is denominated in USD. Payments can be made by credit card or crypto. Manifold Labs may adjust rates based on market conditions; always check https://targon.com/pricing for current rates.

---

## Security Model

### Intel TDX (Trust Domain Extensions)

Intel TDX is a CPU-level isolation mechanism available on 4th/5th-gen Xeon Scalable processors. A TDX Trust Domain (TD) is a hardware-isolated VM whose memory is encrypted with a key that the host OS, hypervisor, and other VMs cannot access. The TDX module is part of the CPU microcode, not the OS — the miner's kernel cannot intercept or inspect VM memory.

**Remote attestation**: TDX generates a cryptographically signed attestation report (backed by Intel SGX infrastructure) that proves to users which code is running inside the TD and that it hasn't been tampered with.

### AMD SEV (Secure Encrypted Virtualization)

AMD SEV and SEV-SNP (Secure Nested Paging) encrypt VM memory using keys held exclusively by the AMD Secure Processor. SEV-SNP adds integrity protection, preventing the hypervisor from replaying or remapping pages. Like TDX, SEV supports remote attestation via AMD's Key Distribution Service (KDS).

### NVIDIA Confidential Computing (CC)

NVIDIA H100/H200 GPUs support CC mode, which establishes an encrypted channel between the CPU enclave (TDX/SEV) and the GPU. GPU VRAM is encrypted, PCIe DMA transfers are protected, and the host cannot read GPU memory. This is critical for AI workloads: model weights and activations in GPU VRAM are inaccessible to the miner.

Without NVIDIA CC, even if the CPU is inside a TDX enclave, data copied to the GPU is exposed on the PCIe bus and in GPU VRAM. H200 rentals on Targon use NVIDIA CC to close this gap.

### Full Stack Privacy

```
User's model/data
      │
      ▼
[Encrypted with TDX/SEV key] ── CPU Memory
      │
[NVIDIA CC encrypted channel] ── PCIe Bus
      │
      ▼
[Encrypted GPU VRAM] ── H200 GPU
```

At no point does the miner's host OS or other system software have access to plaintext model weights, training data, or intermediate activations.

---

## Common Workflows

### LLM Training (H200, 70B+ models)

```bash
# 1. Rent an H200
targon run \
  --gpu H200 \
  --image nvcr.io/nvidia/pytorch:24.01-py3 \
  --name llm-training

# 2. SSH in
ssh <RENTAL_ID>@ssh.deployments.targon.com

# 3. Clone your training code and start
git clone https://github.com/your-org/training-repo
cd training-repo
torchrun --nproc_per_node=1 train.py \
  --model_name_or_path meta-llama/Llama-3-70b \
  --dataset_path /data/train.jsonl \
  --output_dir /workspace/checkpoints
```

### Inference Server (vLLM on H200)

```bash
# Rent and deploy vLLM
targon run \
  --gpu H200 \
  --image vllm/vllm-openai:latest \
  --cmd "python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8b-instruct \
    --host 0.0.0.0 --port 8000" \
  --name vllm-server

# Test the endpoint (via SSH tunnel)
ssh -L 8000:localhost:8000 <RENTAL_ID>@ssh.deployments.targon.com -N &
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3-8b-instruct", "prompt": "Hello", "max_tokens": 50}'
```

### Fine-tuning with LoRA (4090)

```bash
# Cost-effective fine-tuning on 4090
targon run \
  --gpu 4090 \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  --name lora-finetune

ssh <RENTAL_ID>@ssh.deployments.targon.com

pip install transformers peft accelerate datasets trl
python finetune_lora.py \
  --base_model mistralai/Mistral-7B-v0.1 \
  --data_path data.jsonl \
  --output_dir /workspace/lora-adapter
```

### Serverless Batch Inference

```python
# batch_inference.py
from targon import endpoint

@endpoint(gpu="H200", timeout=300)
def batch_classify(texts: list[str]) -> list[dict]:
    from transformers import pipeline
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    return classifier(texts, batch_size=32)
```

```bash
targon deploy --file batch_inference.py --name batch-classifier

# Invoke
curl -X POST https://api.targon.com/v1/functions/batch-classifier \
  -H "Authorization: Bearer $TARGON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["great product", "terrible experience", "okay I guess"]}'
```

### Diffusion Model Image Generation (4090)

```bash
targon run \
  --gpu 4090 \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  --name sdxl-gen

ssh <RENTAL_ID>@ssh.deployments.targon.com
pip install diffusers accelerate transformers

python3 << 'EOF'
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("A photorealistic mountain landscape at sunset").images[0]
image.save("output.png")
EOF
```

---

## Resources

- **Targon Docs**: https://docs.targon.com
- **Targon Dashboard**: https://targon.com
- **Manifold Labs GitHub**: https://github.com/manifold-inc/targon
- **Bittensor Subnet 4**: https://taostats.io/subnet/4
- **NVIDIA Confidential Computing**: https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/
- **Intel TDX Overview**: https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html
- **AMD SEV-SNP**: https://www.amd.com/en/developer/sev.html
