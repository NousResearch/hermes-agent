# PyTorch Lightning Callbacks

## Overview

Callbacks add functionality to training without modifying the LightningModule. They capture **non-essential logic** like checkpointing, early stopping, and logging.

## Built-In Callbacks

### 1. ModelCheckpoint

**Saves best models during training**:

```python
from lightning.pytorch.callbacks import ModelCheckpoint

# Save top 3 models based on validation loss
checkpoint = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    save_last=True,  # Also save last epoch
    verbose=True
)

trainer = L.Trainer(callbacks=[checkpoint])
trainer.fit(model, train_loader, val_loader)
```

**Configuration options**:
```python
checkpoint = ModelCheckpoint(
    monitor='val_acc',        # Metric to monitor
    mode='max',               # 'max' for accuracy, 'min' for loss
    save_top_k=5,             # Keep best 5 models
    save_last=True,           # Save last epoch separately
    every_n_epochs=1,         # Save every N epochs
    save_on_train_epoch_end=False,  # Save on validation end instead
    filename='best-{epoch}-{val_acc:.3f}',  # Naming pattern
    auto_insert_metric_name=False  # Don't auto-add metric to filename
)
```

**Load checkpoint**:
```python
# Load best model
best_model_path = checkpoint.best_model_path
model = LitModel.load_from_checkpoint(best_model_path)

# Resume training
trainer = L.Trainer(callbacks=[checkpoint])
trainer.fit(model, train_loader, val_loader, ckpt_path='checkpoints/last.ckpt')
```

### 2. EarlyStopping

**Stops training when metric stops improving**:

```python
from lightning.pytorch.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,               # Wait 5 epochs
    mode='min',
    min_delta=0.001,          # Minimum change to qualify as improvement
    verbose=True,
    strict=True,              # Crash if monitored metric not found
    check_on_train_epoch_end=False  # Check on validation end
)

trainer = L.Trainer(callbacks=[early_stop])
trainer.fit(model, train_loader, val_loader)
# Stops automatically if no improvement for 5 epochs
```

**Advanced usage**:
```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.0,
    verbose=True,
    mode='min',
    stopping_threshold=0.1,   # Stop if val_loss < 0.1
    divergence_threshold=5.0, # Stop if val_loss > 5.0
    check_finite=True         # Stop on NaN/Inf
)
```

### 3. LearningRateMonitor

**Logs learning rate**:

```python
from lightning.pytorch.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(
    logging_interval='epoch',  # Or 'step'
    log_momentum=True          # Also log momentum
)

trainer = L.Trainer(callbacks=[lr_monitor])
# Learning rate automatically logged to TensorBoard/WandB
```

### 4. TQDMProgressBar

**Customizes progress bar**:

```python
from lightning.pytorch.callbacks import TQDMProgressBar

progress_bar = TQDMProgressBar(
    refresh_rate=10,  # Update every 10 batches
    process_position=0
)

trainer = L.Trainer(callbacks=[progress_bar])
```

### 5. GradientAccumulationScheduler

**Dynamic gradient accumulation**:

```python
from lightning.pytorch.callbacks import GradientAccumulationScheduler

# Accumulate more gradients as training progresses
accumulator = GradientAccumulationScheduler(
    scheduling={
        0: 8,   # Epochs 0-4: accumulate 8 batches
        5: 4,   # Epochs 5-9: accumulate 4 batches
        10: 2   # Epochs 10+: accumulate 2 batches
    }
)

trainer = L.Trainer(callbacks=[accumulator])
```

### 6. StochasticWeightAveraging (SWA)

**Averages weights for better generalization**:

```python
from lightning.pytorch.callbacks import StochasticWeightAveraging

swa = StochasticWeightAveraging(
    swa_lrs=1e-2,  # SWA learning rate
    swa_epoch_start=0.8,  # Start at 80% of training
    annealing_epochs=10,  # Annealing period
    annealing_strategy='cos'  # 'cos' or 'linear'
)

trainer = L.Trainer(callbacks=[swa])
```

## Custom Callbacks

### Basic Custom Callback

```python
from lightning.pytorch.callbacks import Callback

class PrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done!")

    def on_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} ended")

# Use it
trainer = L.Trainer(callbacks=[PrintingCallback()])
```

### Advanced Custom Callback

```python
class MetricsCallback(Callback):
    """Logs custom metrics every N batches."""

    def __init__(self, log_every_n_batches=100):
        self.log_every_n_batches = log_every_n_batches
        self.metrics = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_batches == 0:
            # Compute custom metric
            metric = self.compute_metric(outputs)
            self.metrics.append(metric)

            # Log to Lightning
            pl_module.log('custom_metric', metric)

    def compute_metric(self, outputs):
        # Your custom logic
        return outputs['loss'].item()

    def state_dict(self):
        """Save callback state in checkpoint."""
        return {'metrics': self.metrics}

    def load_state_dict(self, state_dict):
        """Restore callback state from checkpoint."""
        self.metrics = state_dict['metrics']
```

### Gradient Monitoring Callback

```python
class GradientMonitorCallback(Callback):
    """Monitor gradient norms."""

    def on_after_backward(self, trainer, pl_module):
        # Compute gradient norm
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Log
        pl_module.log('grad_norm', total_norm)

        # Warn if exploding
        if total_norm > 100:
            print(f"Warning: Large gradient norm: {total_norm:.2f}")
```

### Model Inspection Callback

```python
class ModelInspectionCallback(Callback):
    """Inspect model activations during training."""

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:  # First batch of epoch
            # Register hooks
            self.activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    self.activations[name] = output.detach()
                return hook

            # Attach to specific layers
            pl_module.model.layer1.register_forward_hook(get_activation('layer1'))
            pl_module.model.layer2.register_forward_hook(get_activation('layer2'))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            # Log activation statistics
            for name, activation in self.activations.items():
                mean = activation.mean().item()
                std = activation.std().item()
                pl_module.log(f'{name}_mean', mean)
                pl_module.log(f'{name}_std', std)
```

## Callback Hooks

**All available hooks**:

```python
class MyCallback(Callback):
    # Setup/Teardown
    def setup(self, trainer, pl_module, stage):
        """Called at beginning of fit/test/predict."""
        pass

    def teardown(self, trainer, pl_module, stage):
        """Called at end of fit/test/predict."""
        pass

    # Training
    def on_train_start(self, trainer, pl_module):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass

    def on_train_end(self, trainer, pl_module):
        pass

    # Validation
    def on_validation_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        pass

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        pass

    # Test (same structure as validation)
    def on_test_start(self, trainer, pl_module):
        pass
    # ... (test_epoch_start, test_batch_start, etc.)

    # Predict
    def on_predict_start(self, trainer, pl_module):
        pass
    # ... (predict_epoch_start, predict_batch_start, etc.)

    # Backward
    def on_before_backward(self, trainer, pl_module, loss):
        pass

    def on_after_backward(self, trainer, pl_module):
        pass

    # Optimizer
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        pass

    # Checkpointing
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Add data to checkpoint."""
        pass

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Restore data from checkpoint."""
        pass
```

## Combining Multiple Callbacks

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Create all callbacks
checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
lr_monitor = LearningRateMonitor(logging_interval='epoch')
custom_callback = MyCustomCallback()

# Add all to Trainer
trainer = L.Trainer(
    callbacks=[checkpoint, early_stop, lr_monitor, custom_callback]
)

trainer.fit(model, train_loader, val_loader)
```

**Execution order**: Callbacks execute in the order they're added

## Best Practices

### 1. Keep Callbacks Independent

**Bad** (dependent on other callback):
```python
class BadCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        # Assumes ModelCheckpoint is present
        best_path = trainer.checkpoint_callback.best_model_path  # Fragile!
```

**Good** (self-contained):
```python
class GoodCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        # Find checkpoint callback if present
        for callback in trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                best_path = callback.best_model_path
                break
```

### 2. Use State Dict for Persistence

```python
class StatefulCallback(Callback):
    def __init__(self):
        self.counter = 0
        self.history = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.counter += 1
        self.history.append(outputs['loss'].item())

    def state_dict(self):
        """Save state."""
        return {
            'counter': self.counter,
            'history': self.history
        }

    def load_state_dict(self, state_dict):
        """Restore state."""
        self.counter = state_dict['counter']
        self.history = state_dict['history']
```

### 3. Handle Distributed Training

```python
class DistributedCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only run on main process
        if trainer.is_global_zero:
            print("This only prints once in distributed training")

        # Run on all processes
        loss = outputs['loss']
        # ... do something with loss on each GPU
```

## Resources

- Callback API: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html
- Built-in callbacks: https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks
- Examples: https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/callbacks
