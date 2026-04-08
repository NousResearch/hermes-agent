# Lambda Labs Troubleshooting Guide

## Instance Launch Issues

### No instances available

**Error**: "No capacity available" or instance type not listed

**Solutions**:
```bash
# Check availability via API
curl -u $LAMBDA_API_KEY: \
  https://cloud.lambdalabs.com/api/v1/instance-types | jq '.data | to_entries[] | select(.value.regions_with_capacity_available | length > 0) | .key'

# Try different regions
# US regions: us-west-1, us-east-1, us-south-1
# International: eu-west-1, asia-northeast-1, etc.

# Try alternative GPU types
# H100 not available? Try A100
# A100 not available? Try A10 or A6000
```

### Instance stuck launching

**Problem**: Instance shows "booting" for over 20 minutes

**Solutions**:
```bash
# Single-GPU: Should be ready in 3-5 minutes
# Multi-GPU (8x): May take 10-15 minutes

# If stuck longer:
# 1. Terminate the instance
# 2. Try a different region
# 3. Try a different instance type
# 4. Contact Lambda support if persistent
```

### API authentication fails

**Error**: `401 Unauthorized` or `403 Forbidden`

**Solutions**:
```bash
# Verify API key format (should start with specific prefix)
echo $LAMBDA_API_KEY

# Test API key
curl -u $LAMBDA_API_KEY: \
  https://cloud.lambdalabs.com/api/v1/instance-types

# Generate new API key from Lambda console if needed
# Settings > API keys > Generate
```

### Quota limits reached

**Error**: "Instance limit reached" or "Quota exceeded"

**Solutions**:
- Check current running instances in console
- Terminate unused instances
- Contact Lambda support to request quota increase
- Use 1-Click Clusters for large-scale needs

## SSH Connection Issues

### Connection refused

**Error**: `ssh: connect to host <IP> port 22: Connection refused`

**Solutions**:
```bash
# Wait for instance to fully initialize
# Single-GPU: 3-5 minutes
# Multi-GPU: 10-15 minutes

# Check instance status in console (should be "active")

# Verify correct IP address
curl -u $LAMBDA_API_KEY: \
  https://cloud.lambdalabs.com/api/v1/instances | jq '.data[].ip'
```

### Permission denied

**Error**: `Permission denied (publickey)`

**Solutions**:
```bash
# Verify SSH key matches
ssh -v -i ~/.ssh/lambda_key ubuntu@<IP>

# Check key permissions
chmod 600 ~/.ssh/lambda_key
chmod 644 ~/.ssh/lambda_key.pub

# Verify key was added to Lambda console before launch
# Keys must be added BEFORE launching instance

# Check authorized_keys on instance (if you have another way in)
cat ~/.ssh/authorized_keys
```

### Host key verification failed

**Error**: `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!`

**Solutions**:
```bash
# This happens when IP is reused by different instance
# Remove old key
ssh-keygen -R <IP>

# Then connect again
ssh ubuntu@<IP>
```

### Timeout during SSH

**Error**: `ssh: connect to host <IP> port 22: Operation timed out`

**Solutions**:
```bash
# Check if instance is in "active" state

# Verify firewall allows SSH (port 22)
# Lambda console > Firewall

# Check your local network allows outbound SSH

# Try from different network/VPN
```

## GPU Issues

### GPU not detected

**Error**: `nvidia-smi: command not found` or no GPUs shown

**Solutions**:
```bash
# Reboot instance
sudo reboot

# Reinstall NVIDIA drivers (if needed)
wget -nv -O- https://lambdalabs.com/install-lambda-stack.sh | sh -
sudo reboot

# Check driver status
nvidia-smi
lsmod | grep nvidia
```

### CUDA out of memory

**Error**: `torch.cuda.OutOfMemoryError: CUDA out of memory`

**Solutions**:
```python
# Check GPU memory
import torch
print(torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

# Clear cache
torch.cuda.empty_cache()

# Reduce batch size
batch_size = batch_size // 2

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(**inputs)

# Use larger GPU instance
# A100-40GB → A100-80GB → H100
```

### CUDA version mismatch

**Error**: `CUDA driver version is insufficient for CUDA runtime version`

**Solutions**:
```bash
# Check versions
nvidia-smi  # Shows driver CUDA version
nvcc --version  # Shows toolkit version

# Lambda Stack should have compatible versions
# If mismatch, reinstall Lambda Stack
wget -nv -O- https://lambdalabs.com/install-lambda-stack.sh | sh -
sudo reboot

# Or install specific PyTorch version
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

### Multi-GPU not working

**Error**: Only one GPU being used

**Solutions**:
```python
# Check all GPUs visible
import torch
print(f"GPUs available: {torch.cuda.device_count()}")

# Verify CUDA_VISIBLE_DEVICES not set restrictively
import os
print(os.environ.get("CUDA_VISIBLE_DEVICES", "not set"))

# Use DataParallel or DistributedDataParallel
model = torch.nn.DataParallel(model)
# or
model = torch.nn.parallel.DistributedDataParallel(model)
```

## Filesystem Issues

### Filesystem not mounted

**Error**: `/lambda/nfs/<name>` doesn't exist

**Solutions**:
```bash
# Filesystem must be attached at launch time
# Cannot attach to running instance

# Verify filesystem was selected during launch

# Check mount points
df -h | grep lambda

# If missing, terminate and relaunch with filesystem
```

### Slow filesystem performance

**Problem**: Reading/writing to filesystem is slow

**Solutions**:
```bash
# Use local SSD for temporary/intermediate files
# /home/ubuntu has fast NVMe storage

# Copy frequently accessed data to local storage
cp -r /lambda/nfs/storage/dataset /home/ubuntu/dataset

# Use filesystem for checkpoints and final outputs only

# Check network bandwidth
iperf3 -c <filesystem_server>
```

### Data lost after termination

**Problem**: Files disappeared after instance terminated

**Solutions**:
```bash
# Root volume (/home/ubuntu) is EPHEMERAL
# Data there is lost on termination

# ALWAYS use filesystem for persistent data
/lambda/nfs/<filesystem_name>/

# Sync important local files before terminating
rsync -av /home/ubuntu/outputs/ /lambda/nfs/storage/outputs/
```

### Filesystem full

**Error**: `No space left on device`

**Solutions**:
```bash
# Check filesystem usage
df -h /lambda/nfs/storage

# Find large files
du -sh /lambda/nfs/storage/* | sort -h

# Clean up old checkpoints
find /lambda/nfs/storage/checkpoints -mtime +7 -delete

# Increase filesystem size in Lambda console
# (may require support request)
```

## Network Issues

### Port not accessible

**Error**: Cannot connect to service (TensorBoard, Jupyter, etc.)

**Solutions**:
```bash
# Lambda default: Only port 22 is open
# Configure firewall in Lambda console

# Or use SSH tunneling (recommended)
ssh -L 6006:localhost:6006 ubuntu@<IP>
# Access at http://localhost:6006

# For Jupyter
ssh -L 8888:localhost:8888 ubuntu@<IP>
```

### Slow data download

**Problem**: Downloading datasets is slow

**Solutions**:
```bash
# Check available bandwidth
speedtest-cli

# Use multi-threaded download
aria2c -x 16 <URL>

# For HuggingFace models
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install hf_transfer

# For S3, use parallel transfer
aws s3 sync s3://bucket/data /local/data --quiet
```

### Inter-node communication fails

**Error**: Distributed training can't connect between nodes

**Solutions**:
```bash
# Verify nodes in same region (required)

# Check private IPs can communicate
ping <other_node_private_ip>

# Verify NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available

# Check firewall allows distributed ports
# Need: 29500 (PyTorch), or configured MASTER_PORT
```

## Software Issues

### Package installation fails

**Error**: `pip install` errors

**Solutions**:
```bash
# Use virtual environment (don't modify system Python)
python -m venv ~/myenv
source ~/myenv/bin/activate
pip install <package>

# For CUDA packages, match CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Clear pip cache if corrupted
pip cache purge
```

### Python version issues

**Error**: Package requires different Python version

**Solutions**:
```bash
# Install alternate Python (don't replace system Python)
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create venv with specific Python
python3.11 -m venv ~/py311env
source ~/py311env/bin/activate
```

### ImportError or ModuleNotFoundError

**Error**: Module not found despite installation

**Solutions**:
```bash
# Verify correct Python environment
which python
pip list | grep <module>

# Ensure virtual environment is activated
source ~/myenv/bin/activate

# Reinstall in correct environment
pip uninstall <package>
pip install <package>
```

## Training Issues

### Training hangs

**Problem**: Training stops progressing, no output

**Solutions**:
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# If GPUs at 0%, likely data loading bottleneck
# Increase num_workers in DataLoader

# Check for deadlocks in distributed training
export NCCL_DEBUG=INFO

# Add timeouts
dist.init_process_group(..., timeout=timedelta(minutes=30))
```

### Checkpoint corruption

**Error**: `RuntimeError: storage has wrong size` or similar

**Solutions**:
```python
# Use safe saving pattern
checkpoint_path = "/lambda/nfs/storage/checkpoint.pt"
temp_path = checkpoint_path + ".tmp"

# Save to temp first
torch.save(state_dict, temp_path)
# Then atomic rename
os.rename(temp_path, checkpoint_path)

# For loading corrupted checkpoint
try:
    state = torch.load(checkpoint_path)
except:
    # Fall back to previous checkpoint
    state = torch.load(checkpoint_path + ".backup")
```

### Memory leak

**Problem**: Memory usage grows over time

**Solutions**:
```python
# Clear CUDA cache periodically
torch.cuda.empty_cache()

# Detach tensors when logging
loss_value = loss.detach().cpu().item()

# Don't accumulate gradients unintentionally
optimizer.zero_grad(set_to_none=True)

# Use gradient accumulation properly
if (step + 1) % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

## Billing Issues

### Unexpected charges

**Problem**: Bill higher than expected

**Solutions**:
```bash
# Check for forgotten running instances
curl -u $LAMBDA_API_KEY: \
  https://cloud.lambdalabs.com/api/v1/instances | jq '.data[].id'

# Terminate all instances
# Lambda console > Instances > Terminate all

# Lambda charges by the minute
# No charge for stopped instances (but no "stop" feature - only terminate)
```

### Instance terminated unexpectedly

**Problem**: Instance disappeared without manual termination

**Possible causes**:
- Payment issue (card declined)
- Account suspension
- Instance health check failure

**Solutions**:
- Check email for Lambda notifications
- Verify payment method in console
- Contact Lambda support
- Always checkpoint to filesystem

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `No capacity available` | Region/GPU sold out | Try different region or GPU type |
| `Permission denied (publickey)` | SSH key mismatch | Re-add key, check permissions |
| `CUDA out of memory` | Model too large | Reduce batch size, use larger GPU |
| `No space left on device` | Disk full | Clean up or use filesystem |
| `Connection refused` | Instance not ready | Wait 3-15 minutes for boot |
| `Module not found` | Wrong Python env | Activate correct virtualenv |

## Getting Help

1. **Documentation**: https://docs.lambda.ai
2. **Support**: https://support.lambdalabs.com
3. **Email**: support@lambdalabs.com
4. **Status**: Check Lambda status page for outages

### Information to Include

When contacting support, include:
- Instance ID
- Region
- Instance type
- Error message (full traceback)
- Steps to reproduce
- Time of occurrence
