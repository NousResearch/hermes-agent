# Hyperparameter Tuning with PyTorch Lightning

## Integration with Tuning Frameworks

Lightning integrates seamlessly with popular hyperparameter tuning libraries.

### 1. Ray Tune Integration

**Installation**:
```bash
pip install ray[tune]
pip install lightning
```

**Basic Ray Tune example**:

```python
import lightning as L
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

class LitModel(L.LightningModule):
    def __init__(self, lr, batch_size):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.model = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 1))

    def training_step(self, batch, batch_idx):
        loss = self.model(batch).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        val_loss = self.model(batch).mean()
        self.log('val_loss', val_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train_fn(config):
    """Training function for Ray Tune."""
    model = LitModel(lr=config["lr"], batch_size=config["batch_size"])

    # Add callback to report metrics to Tune
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[TuneReportCallback({"loss": "val_loss"}, on="validation_end")]
    )

    trainer.fit(model, train_loader, val_loader)

# Define search space
config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128])
}

# Run hyperparameter search
analysis = tune.run(
    train_fn,
    config=config,
    num_samples=20,  # 20 trials
    resources_per_trial={"gpu": 1}
)

# Best hyperparameters
best_config = analysis.get_best_config(metric="loss", mode="min")
print(f"Best config: {best_config}")
```

**Advanced: Population-Based Training (PBT)**:

```python
from ray.tune.schedulers import PopulationBasedTraining

# PBT scheduler
scheduler = PopulationBasedTraining(
    time_attr='training_iteration',
    metric='val_loss',
    mode='min',
    perturbation_interval=5,  # Perturb every 5 epochs
    hyperparam_mutations={
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": [16, 32, 64, 128]
    }
)

analysis = tune.run(
    train_fn,
    config=config,
    num_samples=8,  # Population size
    scheduler=scheduler,
    resources_per_trial={"gpu": 1}
)
```

### 2. Optuna Integration

**Installation**:
```bash
pip install optuna
pip install optuna-integration
```

**Optuna example**:

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 64, 512, step=64)

    # Create model
    model = LitModel(lr=lr, n_layers=n_layers, hidden_size=hidden_size)

    # Pruning callback (early stopping for bad trials)
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")

    trainer = L.Trainer(
        max_epochs=20,
        callbacks=[pruning_callback],
        enable_progress_bar=False,
        logger=False
    )

    trainer.fit(model, train_loader, val_loader)

    return trainer.callback_metrics["val_loss"].item()

# Create study
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner()  # Prune bad trials early
)

# Optimize
study.optimize(objective, n_trials=50, timeout=3600)

# Best params
print(f"Best trial: {study.best_trial.params}")
print(f"Best value: {study.best_value}")

# Visualization
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

**Optuna with distributed training**:

```python
import optuna

# Shared database for distributed optimization
storage = optuna.storages.RDBStorage(
    url='postgresql://user:pass@localhost/optuna'
)

study = optuna.create_study(
    study_name='distributed_study',
    storage=storage,
    load_if_exists=True,
    direction='minimize'
)

# Run on multiple machines
study.optimize(objective, n_trials=50)
```

### 3. Weights & Biases (WandB) Sweeps

**Installation**:
```bash
pip install wandb
```

**WandB sweep config** (`sweep.yaml`):
```yaml
program: train.py
method: bayes
metric:
  name: val_loss
  goal: minimize
parameters:
  lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.1
  batch_size:
    values: [16, 32, 64, 128]
  optimizer:
    values: ['adam', 'sgd', 'adamw']
  dropout:
    distribution: uniform
    min: 0.0
    max: 0.5
```

**Training script** (`train.py`):
```python
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger

def train():
    # Initialize wandb
    wandb.init()
    config = wandb.config

    # Create model with sweep params
    model = LitModel(
        lr=config.lr,
        batch_size=config.batch_size,
        optimizer=config.optimizer,
        dropout=config.dropout
    )

    # WandB logger
    wandb_logger = WandbLogger(project='hyperparameter-sweep')

    trainer = L.Trainer(
        max_epochs=20,
        logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    train()
```

**Launch sweep**:
```bash
# Initialize sweep
wandb sweep sweep.yaml
# Output: wandb: Created sweep with ID: abc123

# Run agent (can run on multiple machines)
wandb agent your-entity/your-project/abc123
```

### 4. Hyperopt Integration

**Installation**:
```bash
pip install hyperopt
```

**Hyperopt example**:

```python
from hyperopt import hp, fmin, tpe, Trials

def objective(params):
    model = LitModel(
        lr=params['lr'],
        batch_size=int(params['batch_size']),
        hidden_size=int(params['hidden_size'])
    )

    trainer = L.Trainer(
        max_epochs=10,
        enable_progress_bar=False,
        logger=False
    )

    trainer.fit(model, train_loader, val_loader)

    # Return loss (minimize)
    return trainer.callback_metrics["val_loss"].item()

# Define search space
space = {
    'lr': hp.loguniform('lr', np.log(1e-5), np.log(1e-1)),
    'batch_size': hp.quniform('batch_size', 16, 128, 16),
    'hidden_size': hp.quniform('hidden_size', 64, 512, 64)
}

# Optimize
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # Tree-structured Parzen Estimator
    max_evals=50,
    trials=trials
)

print(f"Best hyperparameters: {best}")
```

## Built-In Lightning Tuning

### Auto Learning Rate Finder

```python
class LitModel(L.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.model = nn.Linear(10, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss = self.model(batch).mean()
        return loss

# Find optimal learning rate
model = LitModel()
trainer = L.Trainer(auto_lr_find=True)

# This runs LR finder before training
trainer.tune(model, train_loader)

# Or manually
from lightning.pytorch.tuner import Tuner
tuner = Tuner(trainer)
lr_finder = tuner.lr_find(model, train_loader)

# Plot results
fig = lr_finder.plot(suggest=True)
fig.show()

# Get suggested LR
suggested_lr = lr_finder.suggestion()
print(f"Suggested LR: {suggested_lr}")

# Update model
model.lr = suggested_lr

# Train with optimal LR
trainer.fit(model, train_loader)
```

### Auto Batch Size Finder

```python
class LitModel(L.LightningModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.model = nn.Linear(10, 1)

    def train_dataloader(self):
        return DataLoader(dataset, batch_size=self.batch_size)

model = LitModel()
trainer = L.Trainer(auto_scale_batch_size='binsearch')

# Find optimal batch size
trainer.tune(model)

print(f"Optimal batch size: {model.batch_size}")

# Train with optimal batch size
trainer.fit(model, train_loader)
```

## Advanced Tuning Strategies

### 1. Multi-Fidelity Optimization (Successive Halving)

```python
from ray.tune.schedulers import ASHAScheduler

# ASHA: Asynchronous Successive Halving Algorithm
scheduler = ASHAScheduler(
    max_t=100,  # Max epochs
    grace_period=10,  # Min epochs before stopping
    reduction_factor=2  # Halve resources each round
)

analysis = tune.run(
    train_fn,
    config=config,
    num_samples=64,
    scheduler=scheduler,
    resources_per_trial={"gpu": 1}
)
```

**How it works**:
- Start 64 trials
- After 10 epochs, stop bottom 50% (32 trials remain)
- After 20 epochs, stop bottom 50% (16 trials remain)
- After 40 epochs, stop bottom 50% (8 trials remain)
- After 80 epochs, stop bottom 50% (4 trials remain)
- Run remaining 4 trials to completion (100 epochs)

### 2. Bayesian Optimization

```python
from ray.tune.search.bayesopt import BayesOptSearch

search = BayesOptSearch(
    metric="val_loss",
    mode="min"
)

analysis = tune.run(
    train_fn,
    config=config,
    num_samples=50,
    search_alg=search,
    resources_per_trial={"gpu": 1}
)
```

### 3. Grid Search

```python
from ray import tune

# Exhaustive grid search
config = {
    "lr": tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
    "batch_size": tune.grid_search([16, 32, 64, 128]),
    "optimizer": tune.grid_search(['adam', 'sgd', 'adamw'])
}

# Total trials: 4 × 4 × 3 = 48
analysis = tune.run(train_fn, config=config)
```

### 4. Random Search

```python
config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "dropout": tune.uniform(0.0, 0.5),
    "hidden_size": tune.randint(64, 512)
}

# Random sampling
analysis = tune.run(
    train_fn,
    config=config,
    num_samples=100  # 100 random samples
)
```

## Best Practices

### 1. Start Simple

```python
# Phase 1: Coarse search (fast)
coarse_config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.choice([32, 64])
}
coarse_analysis = tune.run(train_fn, config=coarse_config, num_samples=10, max_epochs=5)

# Phase 2: Fine-tune around best (slow)
best_lr = coarse_analysis.best_config["lr"]
fine_config = {
    "lr": tune.uniform(best_lr * 0.5, best_lr * 2),
    "batch_size": tune.choice([16, 32, 64, 128])
}
fine_analysis = tune.run(train_fn, config=fine_config, num_samples=20, max_epochs=20)
```

### 2. Use Checkpointing

```python
def train_fn(config, checkpoint_dir=None):
    model = LitModel(lr=config["lr"])

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={"loss": "val_loss"},
                filename="checkpoint",
                on="validation_end"
            )
        ]
    )

    # Resume from checkpoint if exists
    ckpt_path = None
    if checkpoint_dir:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
```

### 3. Monitor Resource Usage

```python
import GPUtil

def train_fn(config):
    # Before training
    GPUs = GPUtil.getGPUs()
    print(f"GPU memory before: {GPUs[0].memoryUsed} MB")

    # Train
    model = LitModel(lr=config["lr"], batch_size=config["batch_size"])
    trainer.fit(model, train_loader)

    # After training
    GPUs = GPUtil.getGPUs()
    print(f"GPU memory after: {GPUs[0].memoryUsed} MB")
```

## Common Issues

### Issue: Trials Running Out of Memory

**Solution**: Reduce concurrent trials or batch size
```python
analysis = tune.run(
    train_fn,
    config=config,
    resources_per_trial={"gpu": 0.5},  # 2 trials per GPU
    max_concurrent_trials=2  # Limit concurrent trials
)
```

### Issue: Slow Hyperparameter Search

**Solution**: Use early stopping scheduler
```python
from ray.tune.schedulers import ASHAScheduler

scheduler = ASHAScheduler(
    max_t=100,
    grace_period=5,  # Stop bad trials after 5 epochs
    reduction_factor=3
)
```

### Issue: Can't Reproduce Best Trial

**Solution**: Set seeds in training function
```python
def train_fn(config):
    L.seed_everything(42, workers=True)
    # Rest of training...
```

## Resources

- Ray Tune + Lightning: https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
- Optuna: https://optuna.readthedocs.io/
- WandB Sweeps: https://docs.wandb.ai/guides/sweeps
- Lightning Tuner: https://lightning.ai/docs/pytorch/stable/tuning.html
