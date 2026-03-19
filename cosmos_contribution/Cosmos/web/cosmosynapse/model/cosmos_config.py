"""
COSMOS 54D Model Configuration.

The 12D Cosmic Davis Hebbian Transformer — Configuration
Encodes the hyperparameters for the 54D state-space transformer.

Dimension Breakdown:
  12D — CST (Cosmic Synapse Theory) geometric phase space
  24D — Hebbian plasticity state (synaptic weight evolution)
  18D — Chaos oscillator state (7 Lorenz + 4 Rössler attractors × 3D each)
  ─────
  54D — Total state dimensionality per token position
"""

from dataclasses import dataclass, field



@dataclass
class CosmosConfig:
    """Configuration for the 54D CosmosTransformer."""

    # --- Core Transformer ---
    vocab_size: int = 50257            # GPT-2 tokenizer vocab size (tiktoken)
    d_model: int = 512                 # Model hidden dimension
    n_layers: int = 6                  # Number of transformer layers
    n_heads: int = 8                   # Multi-head attention heads
    d_ff: int = 2048                   # Feed-forward hidden dimension
    max_seq_len: int = 2048            # Maximum sequence length
    dropout: float = 0.1              # Dropout rate

    # --- 54D State Space ---
    d_state: int = 54                  # Total state dimensionality (12 + 24 + 18)
    d_cst: int = 12                    # CST geometric phase dimensions
    d_hebbian: int = 24                # Hebbian plasticity dimensions
    d_chaos: int = 18                  # Chaos oscillator dimensions

    # --- Hebbian Plasticity ---
    hebbian_lr: float = 0.01           # Hebbian learning rate (fire-together/wire-together)
    hebbian_decay: float = 0.999       # Synaptic weight decay (prevents runaway growth)
    hebbian_momentum: float = 0.9      # Momentum for synaptic trace

    # --- Chaos Oscillators ---
    n_chaos_oscillators: int = 6       # Number of coupled Lorenz oscillators (6 × 3D = 18D)
    chaos_sigma: float = 10.0          # Lorenz σ parameter
    chaos_rho: float = 28.0            # Lorenz ρ parameter
    chaos_beta: float = 8.0 / 3.0     # Lorenz β parameter
    chaos_dt: float = 0.01             # Integration timestep
    chaos_coupling: float = 0.05       # Inter-oscillator coupling strength

    # --- Persistent Memory Bank ---
    memory_size: int = 256             # Number of episodic memory slots
    memory_dim: int = 512              # Dimension per memory slot (= d_model)
    memory_heads: int = 4              # Attention heads for memory read/write
    memory_decay: float = 0.995        # Memory slot relevance decay

    # --- Phase-Conjugate Residuals ---
    phase_modulation: bool = True      # Enable geometric phase gating on residuals
    phase_freq_base: float = 10000.0   # Base frequency for phase encoding (like RoPE)

    # --- Training ---
    learning_rate: float = 3e-4        # Adam learning rate
    weight_decay: float = 0.01         # AdamW weight decay
    warmup_steps: int = 1000           # Linear warmup
    max_steps: int = 100000            # Max training steps

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self):
        """Validate configuration consistency."""
        assert self.d_state == self.d_cst + self.d_hebbian + self.d_chaos, \
            f"d_state ({self.d_state}) must equal d_cst ({self.d_cst}) + d_hebbian ({self.d_hebbian}) + d_chaos ({self.d_chaos})"
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_chaos_oscillators * 3 <= self.d_chaos, \
            f"n_chaos_oscillators * 3 ({self.n_chaos_oscillators * 3}) must fit in d_chaos ({self.d_chaos})"
        return True

    def to_dict(self) -> dict:
        """Serialize to dict for checkpoint saving."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "d_state": self.d_state,
            "d_cst": self.d_cst,
            "d_hebbian": self.d_hebbian,
            "d_chaos": self.d_chaos,
            "max_seq_len": self.max_seq_len,
            "memory_size": self.memory_size,
            "n_chaos_oscillators": self.n_chaos_oscillators,
        }
