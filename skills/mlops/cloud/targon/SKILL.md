---
name: targon-compute
description: Provision and manage confidential GPU compute on Targon (Bittensor SN4). Use when you need decentralized GPU rentals (H200, 4090), serverless function deployment, or privacy-preserving AI workloads with hardware-level confidential computing.
version: 1.0.0
author: het4rk
license: MIT
dependencies: [targon-sdk>=0.1.0]
metadata:
  hermes:
    tags: [Compute, GPU, Bittensor, Targon, Decentralized, Confidential]
required_environment_variables: [TARGON_API_KEY]
platforms: [macos, linux]

---

# Targon Confidential GPU Compute (Bittensor SN4)

Targon (Subnet 4) by Manifold Labs is a decentralized, confidential AI cloud built on the Bittensor network. It combines hardware-level confidential computing (Intel TDX, AMD SEV, NVIDIA Confidential Computing) with decentralized GPU supply to provide private, verifiable AI infrastructure.

**Free credits**: Use promo code `starter10` at https://targon.com to get **$10 free credits** — no credit card required to start.

## When to Use

**Use Targon when:**
- You need GPU rentals with hardware-enforced privacy guarantees (TVM — Targon Virtual Machine)
- Running sensitive ML workloads where data must not be visible to the host
- You want decentralized, censorship-resistant compute (no single cloud vendor lock-in)
- Deploying serverless inference endpoints or batch jobs on H200s or RTX 4090s
- You need SSH access to a dedicated GPU node for training, fine-tuning, or custom setups
- Cost-sensitive workloads: H200s from $2/hr, competitive 4090 rates

**Use alternatives instead:**
- **Modal / RunPod**: For purely centralized workloads where confidentiality is not a concern
- **Vast.ai**: For lowest-cost 4090 rentals without confidentiality requirements
- **AWS/GCP**: For deep cloud ecosystem integration (managed databases, IAM, etc.)

## Quick Reference

| Task | Command |
|------|---------|
| Install SDK | `pip install targon-sdk` |
| Authenticate | `targon setup` |
| List available GPUs | `targon inventory` |
| Rent a GPU | `targon run --gpu H200 --image <docker-image>` |
| Deploy serverless | `targon deploy --file app.py` |
| SSH into rental | `ssh <RENTAL_ID>@ssh.deployments.targon.com` |
| Check active rentals | `targon list` |
| Stop a rental | `targon stop <RENTAL_ID>` |

## Procedure

### 1. Install and Authenticate

```bash
pip install targon-sdk

# Interactive setup — opens browser for API key retrieval
targon setup
```

Or set the API key directly:

```bash
export TARGON_API_KEY="your_api_key_here"
```

Get your API key at https://targon.com (use promo code `starter10` for $10 free credits).

### 2. Check Available GPU Inventory

```bash
# List available GPU types and counts
targon inventory

# Filter by GPU type
targon inventory --gpu H200
targon inventory --gpu 4090
```

### 3. Provision a GPU Rental

```bash
# Rent an H200 with a PyTorch Docker image
targon run \
  --gpu H200 \
  --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
  --name my-training-job

# Rent a 4090 with custom startup command
targon run \
  --gpu 4090 \
  --image nvcr.io/nvidia/cuda:12.3.0-base-ubuntu22.04 \
  --cmd "bash -c 'nvidia-smi && sleep infinity'" \
  --name my-inference-node
```

Once provisioned, retrieve the rental ID:

```bash
targon list
# Output: RENTAL_ID  GPU    STATUS   STARTED
#         abc123     H200   running  2 min ago
```

### 4. SSH into a Rental

```bash
# Direct SSH access — no VPN or bastion required
ssh abc123@ssh.deployments.targon.com

# Copy files to rental
scp model.py abc123@ssh.deployments.targon.com:/workspace/
```

Your SSH public key is registered during `targon setup`. You can add additional keys in the Targon dashboard.

### 5. Deploy a Serverless Function

```python
# app.py — Targon serverless function
from targon import endpoint

@endpoint(gpu="H200")
def run_inference(payload: dict) -> dict:
    import torch
    # your model inference logic here
    result = {"output": "inference result"}
    return result
```

```bash
# Deploy to serverless
targon deploy --file app.py --name my-inference-api

# Output: Deployed at https://api.targon.com/v1/functions/my-inference-api
```

### 6. Monitor Usage and Billing

```bash
# Check credit balance and usage
targon usage

# View active and past rentals
targon list --all

# Stop a rental (billing stops immediately)
targon stop abc123
```

You can also run the included status script:

```bash
python scripts/targon_status.py
```

### 7. Use the Python SDK Directly

```python
import os
from targon import TargonClient

client = TargonClient(api_key=os.environ["TARGON_API_KEY"])

# List inventory
inventory = client.inventory.list()
for gpu in inventory:
    print(f"{gpu.type}: {gpu.available} available @ ${gpu.price_per_hour}/hr")

# Provision a rental
rental = client.rentals.create(
    gpu_type="H200",
    image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
    name="my-job",
)
print(f"Rental ID: {rental.id}, Status: {rental.status}")
```

## Pitfalls

- **Billing continues until you stop the rental.** Always run `targon stop <RENTAL_ID>` when done. There is no automatic shutdown.
- **SSH key must be added before provisioning.** Run `targon setup` first to register your public key. Adding keys after provisioning requires a restart.
- **Docker image must be CUDA-compatible.** Use `nvidia/cuda` or framework images (PyTorch, TensorFlow). Plain Ubuntu images won't have GPU drivers.
- **Serverless cold starts**: First invocation may take 30–60 seconds while the container launches. Subsequent calls are fast (<50ms).
- **API key scope**: The `TARGON_API_KEY` environment variable must be set for both the CLI and SDK. `targon setup` writes it to `~/.targon/config`.
- **Confidential computing overhead**: TVM adds a small overhead (~2–5%) vs. standard GPU access. This is expected and the trade-off for hardware-level privacy.

## Verification

After provisioning a rental, verify everything is working:

```bash
# 1. Confirm rental is running
targon list
# Should show your rental with status "running"

# 2. SSH in and check GPU
ssh <RENTAL_ID>@ssh.deployments.targon.com
nvidia-smi
# Should show your H200 or 4090

# 3. Verify confidential compute is active (H200 only)
# nvidia-smi conf-compute --query-cc-settings
# Should show "CC mode: ON"

# 4. Run a quick CUDA sanity check
python3 -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

## References

- **[Compute Guide](references/compute_guide.md)** — Architecture, GPU specs, pricing, security model, and common workflows
- **[Status Script](scripts/targon_status.py)** — Check CLI health, active rentals, inventory, and billing
- **Targon Docs**: https://docs.targon.com
- **Targon Site**: https://targon.com
- **GitHub**: https://github.com/manifold-inc/targon
