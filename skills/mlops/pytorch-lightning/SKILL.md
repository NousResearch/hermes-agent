---
name: pytorch-lightning
description: High-level PyTorch framework with Trainer class, automatic distributed training (DDP/FSDP/DeepSpeed), callbacks system, and minimal boilerplate. Scales from laptop to supercomputer with same code. Use when you want clean training loops with built-in best practices.
version: 1.0.0
author: Orchestra Research
license: MIT
tags: [PyTorch Lightning, Training Framework, Distributed Training, DDP, FSDP, DeepSpeed, High-Level API, Callbacks, Best Practices, Scalable]
dependencies: [lightning, torch, transformers]
---

# PyTorch Lightning - High-Level Training Framework

## Quick start

PyTorch Lightning organizes PyTorch code to eliminate boilerplate while maintaining flexibility.

**Installation**:
```bash
pip install lightning
```

**Convert PyTorch to Lightning** (3 steps):

```python
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Step 1: Define LightningModule (organize your PyTorch code)
class LitModel(L.LightningModule):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)  # Auto-logged to TensorBoard
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Step 2: Create data
train_loader = DataLoader(train_dataset, batch_size=32)

# Step 3: Train with Trainer (handles everything else!)
trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=2)
model = LitModel()
trainer.fit(model, train_loader)
```

**That's it!** Trainer handles:
- GPU/TPU/CPU switching
- Distributed training (DDP, FSDP, DeepSpeed)
- Mixed precision (FP16, BF16)
- Gradient accumulation
- Checkpointing
- Logging
- Progress bars

## Common workflows

### Workflow 1: From PyTorch to Lightning

**Original PyTorch code**:
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
model.to('cuda')

for epoch in range(max_epochs):
    for batch in train_loader:
        batch = batch.to('cuda')
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**Lightning version**:
```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)  # No .to('cuda') needed!
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Train
trainer = L.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(LitModel(), train_loader)
```

**Benefits**: 40+ lines → 15 lines, no device management, automatic distributed

### Workflow 2: Validation and testing

```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Train with validation
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

# Test
trainer.test(model, test_loader)
```

**Automatic features**:
- Validation runs every epoch by default
- Metrics logged to TensorBoard
- Best model checkpointing based on val_loss

### Workflow 3: Distributed training (DDP)

```python
# Same code as single GPU!
model = LitModel()

# 8 GPUs with DDP (automatic!)
trainer = L.Trainer(
    accelerator='gpu',
    devices=8,
    strategy='ddp'  # Or 'fsdp', 'deepspeed'
)

trainer.fit(model, train_loader)
```

**Launch**:
```bash
# Single command, Lightning handles the rest
python train.py
```

**No changes needed**:
- Automatic data distribution
- Gradient synchronization
- Multi-node support (just set `num_nodes=2`)

### Workflow 4: Callbacks for monitoring

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Create callbacks
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    filename='model-{epoch:02d}-{val_loss:.2f}'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Add to Trainer
trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint, early_stop, lr_monitor]
)

trainer.fit(model, train_loader, val_loader)
```

**Result**:
- Auto-saves best 3 models
- Stops early if no improvement for 5 epochs
- Logs learning rate to TensorBoard

### Workflow 5: Learning rate scheduling

```python
class LitModel(L.LightningModule):
    # ... (training_step, etc.)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Update per epoch
                'frequency': 1
            }
        }

# Learning rate auto-logged!
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader)
```

## When to use vs alternatives

**Use PyTorch Lightning when**:
- Want clean, organized code
- Need production-ready training loops
- Switching between single GPU, multi-GPU, TPU
- Want built-in callbacks and logging
- Team collaboration (standardized structure)

**Key advantages**:
- **Organized**: Separates research code from engineering
- **Automatic**: DDP, FSDP, DeepSpeed with 1 line
- **Callbacks**: Modular training extensions
- **Reproducible**: Less boilerplate = fewer bugs
- **Tested**: 1M+ downloads/month, battle-tested

**Use alternatives instead**:
- **Accelerate**: Minimal changes to existing code, more flexibility
- **Ray Train**: Multi-node orchestration, hyperparameter tuning
- **Raw PyTorch**: Maximum control, learning purposes
- **Keras**: TensorFlow ecosystem

## Common issues

**Issue: Loss not decreasing**

Check data and model setup:
```python
# Add to training_step
def training_step(self, batch, batch_idx):
    if batch_idx == 0:
        print(f"Batch shape: {batch[0].shape}")
        print(f"Labels: {batch[1]}")
    loss = ...
    return loss
```

**Issue: Out of memory**

Reduce batch size or use gradient accumulation:
```python
trainer = L.Trainer(
    accumulate_grad_batches=4,  # Effective batch = batch_size × 4
    precision='bf16'  # Or 'fp16', reduces memory 50%
)
```

**Issue: Validation not running**

Ensure you pass val_loader:
```python
# WRONG
trainer.fit(model, train_loader)

# CORRECT
trainer.fit(model, train_loader, val_loader)
```

**Issue: DDP spawns multiple processes unexpectedly**

Lightning auto-detects GPUs. Explicitly set devices:
```python
# Test on CPU first
trainer = L.Trainer(accelerator='cpu', devices=1)

# Then GPU
trainer = L.Trainer(accelerator='gpu', devices=1)
```

## Advanced topics

**Callbacks**: See [references/callbacks.md](references/callbacks.md) for EarlyStopping, ModelCheckpoint, custom callbacks, and callback hooks.

**Distributed strategies**: See [references/distributed.md](references/distributed.md) for DDP, FSDP, DeepSpeed ZeRO integration, multi-node setup.

**Hyperparameter tuning**: See [references/hyperparameter-tuning.md](references/hyperparameter-tuning.md) for integration with Optuna, Ray Tune, and WandB sweeps.

## Hardware requirements

- **CPU**: Works (good for debugging)
- **Single GPU**: Works
- **Multi-GPU**: DDP (default), FSDP, or DeepSpeed
- **Multi-node**: DDP, FSDP, DeepSpeed
- **TPU**: Supported (8 cores)
- **Apple MPS**: Supported

**Precision options**:
- FP32 (default)
- FP16 (V100, older GPUs)
- BF16 (A100/H100, recommended)
- FP8 (H100)

## Resources

- Docs: https://lightning.ai/docs/pytorch/stable/
- GitHub: https://github.com/Lightning-AI/pytorch-lightning ⭐ 29,000+
- Version: 2.5.5+
- Examples: https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples
- Discord: https://discord.gg/lightning-ai
- Used by: Kaggle winners, research labs, production teams


