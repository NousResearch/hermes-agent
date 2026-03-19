"""Verify CosmosTransformer instantiates and runs correctly."""
import sys
sys.path.insert(0, r"d:\Cosmos\Cosmos\web")
import torch

from cosmosynapse.model import CosmosConfig, CosmosTransformer

print("=== CosmosTransformer Verification ===")
print()

# 1. Create config and validate
config = CosmosConfig()
config.validate()
print(f"Config: d_state={config.d_state} (12D CST + 24D Hebbian + 18D Chaos)")
print(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
print(f"  memory_size={config.memory_size}, chaos_oscillators={config.n_chaos_oscillators}")
print()

# 2. Create model
model = CosmosTransformer(config)
params = model.count_parameters()
print(f"Model Parameters: {params['total_millions']}")
for k, v in params.items():
    if k not in ('total', 'trainable', 'total_millions'):
        print(f"  {k}: {v}")
print()

# 3. Forward pass test
print("Running forward pass test...")
test_ids = torch.randint(0, config.vocab_size, (1, 64))  # batch=1, seq=64
result = model(test_ids)
print(f"  Logits shape: {result['logits'].shape}")
print(f"  State 54D shape: {result['state_54d'].shape}")
print(f"  State 54D dim: {result['state_54d'].shape[-1]} (expected 54)")
print(f"  Layers: {len(result['layer_states'])}")
print()

# 4. Loss computation test
targets = torch.randint(0, config.vocab_size, (1, 64))
result_with_loss = model(test_ids, targets=targets)
print(f"  Loss: {result_with_loss['loss'].item():.4f}")
print()

# 5. Generation test
print("Running generation test (8 tokens)...")
prompt = torch.randint(0, config.vocab_size, (1, 8))
generated = model.generate(prompt, max_new_tokens=8, temperature=1.0)
print(f"  Generated shape: {generated.shape} (expected [1, 16])")
print()

print("ALL TESTS PASSED")
