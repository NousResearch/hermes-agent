# Lambda Labs Advanced Usage Guide

## Multi-Node Distributed Training

### PyTorch DDP across nodes

```python
# train_multi_node.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    # Environment variables set by launcher
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup_distributed()

    model = MyModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Training loop with synchronized gradients
    for epoch in range(num_epochs):
        train_one_epoch(model, dataloader)

        # Save checkpoint on rank 0 only
        if rank == 0:
            torch.save(model.module.state_dict(), f"checkpoint_{epoch}.pt")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Launch on multiple instances

```bash
# On Node 0 (master)
export MASTER_ADDR=<NODE0_PRIVATE_IP>
export MASTER_PORT=29500

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_multi_node.py

# On Node 1
export MASTER_ADDR=<NODE0_PRIVATE_IP>
export MASTER_PORT=29500

torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_multi_node.py
```

### FSDP for large models

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Wrap policy for transformer models
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer}
)

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    device_id=local_rank,
)
```

### DeepSpeed ZeRO

```python
# ds_config.json
{
    "train_batch_size": 64,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": true},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    }
}
```

```bash
# Launch with DeepSpeed
deepspeed --num_nodes=2 \
    --num_gpus=8 \
    --hostfile=hostfile.txt \
    train.py --deepspeed ds_config.json
```

### Hostfile for multi-node

```bash
# hostfile.txt
node0_ip slots=8
node1_ip slots=8
```

## API Automation

### Auto-launch training jobs

```python
import os
import time
import lambda_cloud_client
from lambda_cloud_client.models import LaunchInstanceRequest

class LambdaJobManager:
    def __init__(self, api_key: str):
        self.config = lambda_cloud_client.Configuration(
            host="https://cloud.lambdalabs.com/api/v1",
            access_token=api_key
        )

    def find_available_gpu(self, gpu_types: list[str], regions: list[str] = None):
        """Find first available GPU type across regions."""
        with lambda_cloud_client.ApiClient(self.config) as client:
            api = lambda_cloud_client.DefaultApi(client)
            types = api.instance_types()

            for gpu_type in gpu_types:
                if gpu_type in types.data:
                    info = types.data[gpu_type]
                    for region in info.regions_with_capacity_available:
                        if regions is None or region.name in regions:
                            return gpu_type, region.name

        return None, None

    def launch_and_wait(self, instance_type: str, region: str,
                        ssh_key: str, filesystem: str = None,
                        timeout: int = 900) -> dict:
        """Launch instance and wait for it to be ready."""
        with lambda_cloud_client.ApiClient(self.config) as client:
            api = lambda_cloud_client.DefaultApi(client)

            request = LaunchInstanceRequest(
                region_name=region,
                instance_type_name=instance_type,
                ssh_key_names=[ssh_key],
                file_system_names=[filesystem] if filesystem else [],
            )

            response = api.launch_instance(request)
            instance_id = response.data.instance_ids[0]

            # Poll until ready
            start = time.time()
            while time.time() - start < timeout:
                instance = api.get_instance(instance_id)
                if instance.data.status == "active":
                    return {
                        "id": instance_id,
                        "ip": instance.data.ip,
                        "status": "active"
                    }
                time.sleep(30)

            raise TimeoutError(f"Instance {instance_id} not ready after {timeout}s")

    def terminate(self, instance_ids: list[str]):
        """Terminate instances."""
        from lambda_cloud_client.models import TerminateInstanceRequest

        with lambda_cloud_client.ApiClient(self.config) as client:
            api = lambda_cloud_client.DefaultApi(client)
            request = TerminateInstanceRequest(instance_ids=instance_ids)
            api.terminate_instance(request)


# Usage
manager = LambdaJobManager(os.environ["LAMBDA_API_KEY"])

# Find available H100 or A100
gpu_type, region = manager.find_available_gpu(
    ["gpu_8x_h100_sxm5", "gpu_8x_a100_80gb_sxm4"],
    regions=["us-west-1", "us-east-1"]
)

if gpu_type:
    instance = manager.launch_and_wait(
        gpu_type, region,
        ssh_key="my-key",
        filesystem="training-data"
    )
    print(f"Ready: ssh ubuntu@{instance['ip']}")
```

### Batch job submission

```python
import subprocess
import paramiko

def run_remote_job(ip: str, ssh_key_path: str, commands: list[str]):
    """Execute commands on remote instance."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username="ubuntu", key_filename=ssh_key_path)

    for cmd in commands:
        stdin, stdout, stderr = client.exec_command(cmd)
        print(stdout.read().decode())
        if stderr.read():
            print(f"Error: {stderr.read().decode()}")

    client.close()

# Submit training job
commands = [
    "cd /lambda/nfs/storage/project",
    "git pull",
    "pip install -r requirements.txt",
    "nohup torchrun --nproc_per_node=8 train.py > train.log 2>&1 &"
]

run_remote_job(instance["ip"], "~/.ssh/lambda_key", commands)
```

### Monitor training progress

```python
def monitor_job(ip: str, ssh_key_path: str, log_file: str = "train.log"):
    """Stream training logs from remote instance."""
    import time

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(ip, username="ubuntu", key_filename=ssh_key_path)

    # Tail log file
    stdin, stdout, stderr = client.exec_command(f"tail -f {log_file}")

    try:
        for line in stdout:
            print(line.strip())
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
```

## 1-Click Cluster Workflows

### Slurm job submission

```bash
#!/bin/bash
#SBATCH --job-name=llm-training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Set up distributed environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch training
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
    --config config.yaml
```

### Interactive cluster session

```bash
# Request interactive session
srun --nodes=1 --ntasks=1 --gpus=8 --time=4:00:00 --pty bash

# Now on compute node with 8 GPUs
nvidia-smi
python train.py
```

### Monitoring cluster jobs

```bash
# View job queue
squeue

# View job details
scontrol show job <JOB_ID>

# Cancel job
scancel <JOB_ID>

# View node status
sinfo

# View GPU usage across cluster
srun --nodes=4 nvidia-smi --query-gpu=name,utilization.gpu --format=csv
```

## Advanced Filesystem Usage

### Data staging workflow

```bash
# Stage data from S3 to filesystem (one-time)
aws s3 sync s3://my-bucket/dataset /lambda/nfs/storage/datasets/

# Or use rclone
rclone sync s3:my-bucket/dataset /lambda/nfs/storage/datasets/
```

### Shared filesystem across instances

```python
# Instance 1: Write checkpoints
checkpoint_path = "/lambda/nfs/shared/checkpoints/model_step_1000.pt"
torch.save(model.state_dict(), checkpoint_path)

# Instance 2: Read checkpoints
model.load_state_dict(torch.load(checkpoint_path))
```

### Filesystem best practices

```bash
# Organize for ML workflows
/lambda/nfs/storage/
├── datasets/
│   ├── raw/           # Original data
│   └── processed/     # Preprocessed data
├── models/
│   ├── pretrained/    # Base models
│   └── fine-tuned/    # Your trained models
├── checkpoints/
│   └── experiment_1/  # Per-experiment checkpoints
├── logs/
│   └── tensorboard/   # Training logs
└── outputs/
    └── inference/     # Inference results
```

## Environment Management

### Custom Python environments

```bash
# Don't modify system Python, create venv
python -m venv ~/myenv
source ~/myenv/bin/activate

# Install packages
pip install torch transformers accelerate

# Save to filesystem for reuse
cp -r ~/myenv /lambda/nfs/storage/envs/myenv
```

### Conda environments

```bash
# Install miniconda (if not present)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Create environment
~/miniconda3/bin/conda create -n ml python=3.10 pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Activate
source ~/miniconda3/bin/activate ml
```

### Docker containers

```bash
# Pull and run NVIDIA container
docker run --gpus all -it --rm \
    -v /lambda/nfs/storage:/data \
    nvcr.io/nvidia/pytorch:24.01-py3

# Run training in container
docker run --gpus all -d \
    -v /lambda/nfs/storage:/data \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/pytorch:24.01-py3 \
    python /workspace/train.py
```

## Monitoring and Observability

### GPU monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# GPU utilization over time
nvidia-smi dmon -s u -d 1

# Detailed GPU info
nvidia-smi -q
```

### System monitoring

```bash
# CPU and memory
htop

# Disk I/O
iostat -x 1

# Network
iftop

# All resources
glances
```

### TensorBoard integration

```bash
# Start TensorBoard
tensorboard --logdir /lambda/nfs/storage/logs --port 6006 --bind_all

# SSH tunnel from local machine
ssh -L 6006:localhost:6006 ubuntu@<IP>

# Access at http://localhost:6006
```

### Weights & Biases integration

```python
import wandb

# Initialize with API key
wandb.login(key=os.environ["WANDB_API_KEY"])

# Start run
wandb.init(
    project="lambda-training",
    config={"learning_rate": 1e-4, "epochs": 100}
)

# Log metrics
wandb.log({"loss": loss, "accuracy": acc})

# Save artifacts to filesystem + W&B
wandb.save("/lambda/nfs/storage/checkpoints/best_model.pt")
```

## Cost Optimization Strategies

### Checkpointing for interruption recovery

```python
import os

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(path, model, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    return 0, float('inf')

# Save every N steps to filesystem
checkpoint_path = "/lambda/nfs/storage/checkpoints/latest.pt"
if step % 1000 == 0:
    save_checkpoint(model, optimizer, epoch, loss, checkpoint_path)
```

### Instance selection by workload

```python
def recommend_instance(model_params: int, batch_size: int, task: str) -> str:
    """Recommend Lambda instance based on workload."""

    if task == "inference":
        if model_params < 7e9:
            return "gpu_1x_a10"  # $0.75/hr
        elif model_params < 13e9:
            return "gpu_1x_a6000"  # $0.80/hr
        else:
            return "gpu_1x_h100_pcie"  # $2.49/hr

    elif task == "fine-tuning":
        if model_params < 7e9:
            return "gpu_1x_a100"  # $1.29/hr
        elif model_params < 13e9:
            return "gpu_4x_a100"  # $5.16/hr
        else:
            return "gpu_8x_h100_sxm5"  # $23.92/hr

    elif task == "pretraining":
        return "gpu_8x_h100_sxm5"  # Maximum performance

    return "gpu_1x_a100"  # Default
```

### Auto-terminate idle instances

```python
import time
from datetime import datetime, timedelta

def auto_terminate_idle(api_key: str, idle_threshold_hours: float = 2):
    """Terminate instances idle for too long."""
    manager = LambdaJobManager(api_key)

    with lambda_cloud_client.ApiClient(manager.config) as client:
        api = lambda_cloud_client.DefaultApi(client)
        instances = api.list_instances()

        for instance in instances.data:
            # Check if instance has been running without activity
            # (You'd need to track this separately)
            launch_time = instance.launched_at
            if datetime.now() - launch_time > timedelta(hours=idle_threshold_hours):
                print(f"Terminating idle instance: {instance.id}")
                manager.terminate([instance.id])
```

## Security Best Practices

### SSH key rotation

```bash
# Generate new key pair
ssh-keygen -t ed25519 -f ~/.ssh/lambda_key_new -C "lambda-$(date +%Y%m)"

# Add new key via Lambda console or API
# Update authorized_keys on running instances
ssh ubuntu@<IP> "echo '$(cat ~/.ssh/lambda_key_new.pub)' >> ~/.ssh/authorized_keys"

# Test new key
ssh -i ~/.ssh/lambda_key_new ubuntu@<IP>

# Remove old key from Lambda console
```

### Firewall configuration

```bash
# Lambda console: Only open necessary ports
# Recommended:
# - 22 (SSH) - Always needed
# - 6006 (TensorBoard) - If using
# - 8888 (Jupyter) - If using
# - 29500 (PyTorch distributed) - For multi-node only
```

### Secrets management

```bash
# Don't hardcode API keys in code
# Use environment variables
export HF_TOKEN="hf_..."
export WANDB_API_KEY="..."

# Or use .env file (add to .gitignore)
source .env

# On instance, store in ~/.bashrc
echo 'export HF_TOKEN="..."' >> ~/.bashrc
```
