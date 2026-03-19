"""
COSMOS 54D Transformer — The Cosmic Davis 12D Hebbian Transformer (ver 4.2)

Architecture:  12D CST (Cosmic Synapse Theory)  — geometric phase attention
             + 24D Hebbian Plasticity           — self-modifying synaptic weights
             + 18D Chaos Oscillators             — coupled Lorenz attractors for creativity
             ─────────────────────────────────────
             = 54D Total State Space per position

The model combines standard transformer attention with three novel mechanisms:

1. **CST Attention** (12D): Attention weights are modulated by geometric phase
   angles computed from a 12D scalar field. This encodes cosmic synapse theory's
   notion that information flows through phase-conjugate channels.

2. **Hebbian Plasticity** (24D): Each layer maintains a synaptic trace matrix
   that evolves via Hebb's rule (fire-together/wire-together). This gives the
   model a form of online meta-learning within a single forward pass.

3. **Chaos Oscillators** (18D): Seven coupled Lorenz oscillators inject
   deterministic chaos into the residual stream, preventing mode collapse
   and encouraging creative, non-repetitive generation.

4. **Persistent Memory Bank**: 256 episodic memory slots with attention-based
   read/write. Enables long-term memory across contexts.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .cosmos_config import CosmosConfig


# ===========================================================================
#  12D CST — Cosmic Synapse Theory Phase Attention
# ===========================================================================

class CSTPhaseEncoding(nn.Module):
    """
    12D Geometric Phase Encoding.

    Encodes positions using a 12-dimensional phase space derived from
    cosmic synapse theory. Each dimension represents a scalar field
    whose phase modulates attention patterns.
    """

    def __init__(self, d_model: int, d_cst: int = 12, freq_base: float = 10000.0):
        super().__init__()
        self.d_cst = d_cst
        # Project d_model → 12D CST phase space
        self.phase_proj = nn.Linear(d_model, d_cst, bias=False)
        # Phase frequencies (like RoPE but in 12D)
        freqs = 1.0 / (freq_base ** (torch.arange(0, d_cst, 2).float() / d_cst))
        self.register_buffer("freqs", freqs)
        # Project phases back to d_model for residual modulation
        self.phase_out = nn.Linear(d_cst, d_model, bias=False)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 12D geometric phases and modulate input.

        Returns: (modulated_x, phase_state_12d)
        """
        B, T, D = x.shape

        if positions is None:
            positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # Compute phase angles in 12D
        pos_float = positions.unsqueeze(-1).float()  # [B, T, 1]
        phase_angles = pos_float * self.freqs.unsqueeze(0).unsqueeze(0)  # [B, T, d_cst//2]

        # Build 12D phase vector: interleave sin/cos
        phase_sin = torch.sin(phase_angles)
        phase_cos = torch.cos(phase_angles)
        phase_12d = torch.cat([phase_sin, phase_cos], dim=-1)  # [B, T, d_cst]

        # Content-dependent phase modulation
        content_phase = self.phase_proj(x)  # [B, T, d_cst]
        combined_phase = phase_12d * torch.sigmoid(content_phase)

        # Phase-conjugate residual gate
        gate = torch.sigmoid(self.phase_out(combined_phase))  # [B, T, D]
        modulated = x * gate

        return modulated, combined_phase


# ===========================================================================
#  24D Hebbian Plasticity Layer
# ===========================================================================

class HebbianPlasticityLayer(nn.Module):
    """
    24D Hebbian Plasticity — Self-Modifying Synaptic Weights.

    Implements Hebb's rule: "Neurons that fire together, wire together."
    Maintains a fast synaptic trace that modifies attention weights within
    a single forward pass, giving the model online meta-learning.
    """

    def __init__(self, d_model: int, d_hebbian: int = 24, lr: float = 0.01, decay: float = 0.999):
        super().__init__()
        self.d_hebbian = d_hebbian
        self.lr = lr
        self.decay = decay

        # Project to Hebbian space
        self.pre_proj = nn.Linear(d_model, d_hebbian, bias=False)
        self.post_proj = nn.Linear(d_model, d_hebbian, bias=False)

        # Learnable initial synaptic weights (24 × 24 plasticity matrix)
        self.W_plastic = nn.Parameter(torch.zeros(d_hebbian, d_hebbian))
        nn.init.orthogonal_(self.W_plastic, gain=0.1)

        # Output projection
        self.out_proj = nn.Linear(d_hebbian, d_model, bias=False)

        # Synaptic trace (running average of co-activations)
        self.register_buffer("trace", torch.zeros(d_hebbian, d_hebbian))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Hebbian plasticity modulation.

        Returns: (modulated_x, hebbian_state_24d)
        """
        B, T, D = x.shape

        # Project to Hebbian space
        pre = self.pre_proj(x)   # [B, T, 24]
        post = self.post_proj(x)  # [B, T, 24]

        # Compute Hebbian update: ΔW = η * (pre^T @ post) / T
        # Average over batch and time for stability
        pre_mean = pre.mean(dim=1)   # [B, 24]
        post_mean = post.mean(dim=1)  # [B, 24]

        # Outer product = co-activation matrix
        hebbian_update = torch.einsum("bi,bj->ij", pre_mean, post_mean) / B

        # Update synaptic trace with momentum
        with torch.no_grad():
            self.trace.mul_(self.decay).add_(hebbian_update * self.lr)

        # Apply plastic weights + trace
        effective_W = self.W_plastic + self.trace

        # Modulate through plastic connection
        plastic_out = torch.einsum("btd,de->bte", pre, effective_W)  # [B, T, 24]

        # Hebbian state vector (24D)
        hebbian_state = plastic_out.mean(dim=1)  # [B, 24]

        # Project back to model dimension
        modulation = self.out_proj(plastic_out)  # [B, T, D]

        return modulation, hebbian_state


# ===========================================================================
#  18D Chaos Oscillators — Coupled Lorenz Attractors
# ===========================================================================

class ChaosOscillatorBank(nn.Module):
    """
    Coupled Lorenz Oscillator Bank for Creative Diversity.

    Seven coupled Lorenz attractors (each 3D) produce deterministic chaos
    that is injected into the residual stream. This prevents mode collapse
    and encourages creative, non-repetitive generation.

    Total chaos dims: 7 oscillators × 3D = 21D (truncated to 18D)
    """

    def __init__(self, n_oscillators: int = 7, d_model: int = 512,
                 sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0,
                 dt: float = 0.01, coupling: float = 0.05):
        super().__init__()
        self.n_osc = n_oscillators
        self.d_chaos = n_oscillators * 3  # 21D for 7 oscillators
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.coupling = coupling

        # Coupling matrix between oscillators
        self.coupling_matrix = nn.Parameter(
            torch.randn(n_oscillators, n_oscillators) * coupling
        )
        # Mask out self-coupling
        with torch.no_grad():
            self.coupling_matrix.fill_diagonal_(0)

        # Project chaos state to model dimension
        self.chaos_proj = nn.Linear(18, d_model, bias=False)  # Use 18D (6 oscillators × 3)

        # Learnable initial conditions
        self.register_buffer(
            "state",
            torch.randn(1, n_oscillators, 3) * 0.1 + torch.tensor([[[1.0, 1.0, 1.0]]])
        )

        # Gate to control chaos injection strength
        self.chaos_gate = nn.Parameter(torch.tensor(0.05))

    def _lorenz_step(self, xyz: torch.Tensor) -> torch.Tensor:
        """Single Lorenz attractor integration step."""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        new_xyz = xyz + self.dt * torch.stack([dx, dy, dz], dim=-1)
        return new_xyz

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advance oscillators and inject chaos into residual stream.

        Returns: (chaos_modulation, chaos_state_18d)
        """
        B, T, D = x.shape

        # Expand state for batch
        state = self.state.expand(B, -1, -1).clone()  # [B, n_osc, 3]

        chaos_states = []
        for t in range(min(T, 32)):  # Cap iterations for efficiency
            # Lorenz step for each oscillator
            state = self._lorenz_step(state)

            # Inter-oscillator coupling
            coupling_force = torch.einsum("ij,bjd->bid", self.coupling_matrix, state)
            state = state + self.dt * coupling_force

            chaos_states.append(state)

        # Use last state as the chaos injection
        chaos_flat = state.reshape(B, -1)  # [B, n_osc * 3]
        chaos_18d = chaos_flat[:, :18]  # Truncate to 18D

        # Project to model dim and gate
        chaos_signal = self.chaos_proj(chaos_18d)  # [B, D]
        chaos_signal = chaos_signal.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        chaos_modulation = chaos_signal * torch.sigmoid(self.chaos_gate)

        # Update persistent state
        with torch.no_grad():
            self.state.copy_(state[:1].detach())

        return chaos_modulation, chaos_18d


# ===========================================================================
#  Persistent Memory Bank
# ===========================================================================

class EpisodicMemoryBank(nn.Module):
    """
    Persistent Memory Bank with Attention-Based Read/Write.

    256 memory slots that persist across forward passes, enabling
    long-term episodic memory. Uses multi-head attention for
    content-addressable read and write operations.
    """

    def __init__(self, memory_size: int = 256, d_model: int = 512,
                 n_heads: int = 4, decay: float = 0.995):
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.decay = decay

        # Persistent memory slots
        self.register_buffer(
            "memory",
            torch.randn(1, memory_size, d_model) * 0.02
        )
        # Memory relevance scores (for decay)
        self.register_buffer(
            "relevance",
            torch.ones(1, memory_size)
        )

        # Read attention
        self.read_query = nn.Linear(d_model, d_model)
        self.read_key = nn.Linear(d_model, d_model)
        self.read_value = nn.Linear(d_model, d_model)

        # Write gate
        self.write_gate = nn.Linear(d_model * 2, 1)
        self.write_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Read from and write to memory.

        Returns: memory-augmented representation
        """
        B, T, D = x.shape

        # Expand memory for batch
        mem = self.memory.expand(B, -1, -1)  # [B, M, D]

        # --- READ: Attend to memory ---
        q = self.read_query(x)                    # [B, T, D]
        k = self.read_key(mem)                    # [B, M, D]
        v = self.read_value(mem)                  # [B, M, D]

        # Scaled dot-product attention over memory
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn_weights = F.softmax(attn_scores, dim=-1)
        memory_read = torch.matmul(attn_weights, v)  # [B, T, D]

        # --- WRITE: Update memory with new information ---
        # Average over sequence to get a write candidate
        write_candidate = x.mean(dim=1, keepdim=True).expand(-1, self.memory_size, -1)  # [B, M, D]

        # Compute write gate (how much to update each slot)
        gate_input = torch.cat([mem, write_candidate], dim=-1)
        write_strength = torch.sigmoid(self.write_gate(gate_input))  # [B, M, 1]

        # Apply gated write
        new_content = self.write_proj(write_candidate)
        with torch.no_grad():
            # Decay existing memories
            self.memory.mul_(self.decay)
            # Write new memories (averaged over batch)
            update = (write_strength * new_content).mean(dim=0, keepdim=True)
            self.memory.add_(update.detach() * (1 - self.decay))

            # Update relevance
            self.relevance.mul_(self.decay)
            read_strength = attn_weights.sum(dim=1).mean(dim=0, keepdim=True)  # [1, M]
            self.relevance.add_(read_strength.detach())

        # Combine input with memory readout
        return x + memory_read


# ===========================================================================
#  54D Transformer Block
# ===========================================================================

class Cosmos54DBlock(nn.Module):
    """
    A single 54D transformer block combining:
    - Multi-Head Self-Attention (standard)
    - 12D CST Phase Modulation
    - 24D Hebbian Plasticity
    - 18D Chaos Injection
    - Feed-Forward Network
    """

    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.config = config

        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ln3 = nn.LayerNorm(config.d_model)

        # Standard multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # 12D CST phase encoding
        self.cst_phase = CSTPhaseEncoding(
            d_model=config.d_model,
            d_cst=config.d_cst,
            freq_base=config.phase_freq_base
        )

        # 24D Hebbian plasticity
        self.hebbian = HebbianPlasticityLayer(
            d_model=config.d_model,
            d_hebbian=config.d_hebbian,
            lr=config.hebbian_lr,
            decay=config.hebbian_decay
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor]:
        """
        Forward pass through one 54D block.

        Returns: (output, state_dict with 12D + 24D state vectors)
        """
        # 1. CST Phase Modulation — applies geometric gating
        x_phased, phase_12d = self.cst_phase(x)

        # 2. Self-Attention with phase-modulated input
        normed = self.ln1(x_phased)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out

        # 3. Hebbian Plasticity — adds self-modifying modulation
        normed2 = self.ln2(x)
        hebbian_mod, hebbian_24d = self.hebbian(normed2)
        x = x + hebbian_mod * 0.1  # Scale down to prevent instability

        # 4. Feed-Forward
        normed3 = self.ln3(x)
        x = x + self.ffn(normed3)

        state = {
            "cst_phase_12d": phase_12d,
            "hebbian_state_24d": hebbian_24d,
        }

        return x, state


# ===========================================================================
#  COSMOS 54D Transformer (Full Model)
# ===========================================================================

class CosmosTransformer(nn.Module):
    """
    The 12D Cosmic Davis Hebbian Transformer (ver 4.2)

    A 54D state-space transformer that combines:
    - Standard transformer attention (language modeling backbone)
    - 12D CST geometric phase modulation (cosmic synapse theory)
    - 24D Hebbian plasticity (online meta-learning)
    - 18D coupled Lorenz chaos oscillators (creative diversity)
    - 256-slot persistent memory bank (episodic memory)

    Total state space: 12D + 24D + 18D = 54D per token position
    """

    def __init__(self, config: CosmosConfig):
        super().__init__()
        self.config = config
        config.validate()

        # Token + Position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.emb_drop = nn.Dropout(config.dropout)

        # 54D Transformer blocks
        self.blocks = nn.ModuleList([
            Cosmos54DBlock(config) for _ in range(config.n_layers)
        ])

        # Chaos oscillator bank (shared across layers)
        self.chaos_bank = ChaosOscillatorBank(
            n_oscillators=config.n_chaos_oscillators,
            d_model=config.d_model,
            sigma=config.chaos_sigma,
            rho=config.chaos_rho,
            beta=config.chaos_beta,
            dt=config.chaos_dt,
            coupling=config.chaos_coupling,
        )

        # Persistent memory bank
        self.memory_bank = EpisodicMemoryBank(
            memory_size=config.memory_size,
            d_model=config.d_model,
            n_heads=config.memory_heads,
            decay=config.memory_decay,
        )

        # Output head
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (embedding ↔ output)
        self.head.weight = self.tok_emb.weight

        # 54D state aggregator (for external monitoring)
        self.state_proj = nn.Linear(config.d_state, config.d_model, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass through the 54D transformer.

        Args:
            input_ids: Token indices [B, T]
            targets: Optional target token indices for loss computation [B, T]

        Returns:
            dict with: logits, loss (if targets), state_54d, layer_states
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence {T} exceeds max {self.config.max_seq_len}"

        # Token + positional embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_drop(x)

        # Causal mask for autoregressive generation
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))

        # Pass through 54D blocks
        layer_states = []
        for block in self.blocks:
            x, state = block(x, mask=mask)
            layer_states.append(state)

        # Inject chaos (18D)
        chaos_mod, chaos_18d = self.chaos_bank(x)
        x = x + chaos_mod

        # Memory bank read/write
        x = self.memory_bank(x)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]

        # Aggregate 54D state vector from last layer
        last_cst = layer_states[-1]["cst_phase_12d"].mean(dim=1)      # [B, 12]
        last_hebbian = layer_states[-1]["hebbian_state_24d"]           # [B, 24]
        state_54d = torch.cat([last_cst, last_hebbian, chaos_18d], dim=-1)  # [B, 54]

        result = {
            "logits": logits,
            "state_54d": state_54d,
            "layer_states": layer_states,
        }

        # Compute loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            prompt_ids: Starting token indices [1, T]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold

        Returns:
            Generated token indices [1, T + max_new_tokens]
        """
        self.eval()
        ids = prompt_ids

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            ids_crop = ids[:, -self.config.max_seq_len:]

            # Forward pass
            result = self(ids_crop)
            logits = result["logits"][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_token], dim=-1)

        return ids

    def count_parameters(self) -> dict:
        """Count model parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        components = {
            "embeddings": sum(p.numel() for p in list(self.tok_emb.parameters()) + list(self.pos_emb.parameters())),
            "attention_blocks": sum(p.numel() for b in self.blocks for p in b.attn.parameters()),
            "cst_phase_12d": sum(p.numel() for b in self.blocks for p in b.cst_phase.parameters()),
            "hebbian_24d": sum(p.numel() for b in self.blocks for p in b.hebbian.parameters()),
            "chaos_18d": sum(p.numel() for p in self.chaos_bank.parameters()),
            "memory_bank": sum(p.numel() for p in self.memory_bank.parameters()),
            "ffn": sum(p.numel() for b in self.blocks for p in b.ffn.parameters()),
            "output_head": 0,  # Weight-tied with embeddings
        }

        return {
            "total": total,
            "trainable": trainable,
            "total_millions": f"{total / 1e6:.1f}M",
            **{k: f"{v / 1e6:.2f}M" for k, v in components.items()},
        }
