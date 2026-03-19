"""
Phi Constants — The Golden Ratio Scaling Module
================================================
Central source of truth for φ (Golden Ratio) and all derived constants.

Theory:
    φ = (1 + √5) / 2 ≈ 1.618033988749895
    φ⁻¹ = φ - 1 ≈ 0.618033988749895
    φ² = φ + 1 ≈ 2.618033988749895

    The Golden Ratio is the universe's scaling constant for recursive,
    self-similar growth. It appears in:
    - Fibonacci spirals (nature's growth algorithm)
    - Phyllotaxis (leaf arrangement for optimal light capture)
    - Galaxy arm spacing
    - DNA helix proportions

    In the CST 12D Engine, φ serves as:
    - The recursive damping factor for generational node scaling
    - The Hebbian weight normalizer across 12D coordinates
    - The Lorenz oscillation damper
    - The non-vanishing penalty steepness
    - The emotional intensity scaler

Author: Cosmos CNS / Cory Shane Davis
Version: 1.0.0 (The Geometric Stabilizer)
"""

import math

# ════════════════════════════════════════════════════════
# THE GOLDEN RATIO
# ════════════════════════════════════════════════════════

PHI = (1.0 + math.sqrt(5.0)) / 2.0          # 1.618033988749895
PHI_INV = 1.0 / PHI                          # 0.618033988749895 (= φ - 1)
PHI_SQ = PHI * PHI                           # 2.618033988749895 (= φ + 1)
PHI_CUBE = PHI ** 3                          # 4.236067977499790
LOG_PHI = math.log(PHI)                      # 0.481211825059603


# ════════════════════════════════════════════════════════
# DERIVED FUNCTIONS
# ════════════════════════════════════════════════════════

def phi_decay(n: int) -> float:
    """
    Generational harmonic decay: φⁿ
    
    Each subsequent generation in the 12D State Space
    is scaled by 1/φⁿ to ensure harmonic decay and
    structural integrity.
    
    Args:
        n: Generation depth (0 = root, 1 = first child, ...)
        
    Returns:
        φⁿ (the divisor for that generation)
    """
    return PHI ** n


def phi_inv_decay(n: int) -> float:
    """
    Inverse decay: 1/φⁿ (the actual damping multiplier)
    
    Apply to values that need to decay harmonically:
        value_at_depth_n = value_at_root × phi_inv_decay(n)
    """
    return PHI_INV ** n


def phi_normalize(value: float, depth: int) -> float:
    """
    Normalize a value by its 12D coordinate depth.
    
    W_normalized = value / φⁿ
    
    Used for Hebbian weight normalization where connections
    deeper in the 12D manifold have proportionally less influence.
    """
    return value / phi_decay(depth)


def phi_influence_radius(mass: float, depth: int) -> float:
    """
    Calculate the influence radius of an informational mass.
    
    Combines the inverse-square law with φ-scaling:
        radius = m_I / (φⁿ × depth²)    for depth > 0
        radius = m_I                      for depth = 0 (root)
    
    Ensures that heavier masses have wider influence,
    but that influence decays harmonically with depth.
    """
    if depth <= 0:
        return mass
    return mass / (phi_decay(depth) * (depth ** 2))


def phi_scale_emotional(raw_intensity: float) -> float:
    """
    Scale emotional intensity to feel natural using φ.
    
    Maps raw bio-signal intensity through a φ-harmonic curve:
        scaled = raw × (2 / (1 + φ^(1 - 2×raw)))
    
    This creates a sigmoid-like curve centered at 0.5
    that follows the Golden Ratio's natural proportions.
    Result is always in [0, 1].
    """
    # Clamp input
    raw = max(0.0, min(1.0, raw_intensity))
    
    # φ-sigmoid: smooth, natural-feeling scaling
    exponent = 1.0 - 2.0 * raw
    scaled = 2.0 / (1.0 + PHI ** exponent)
    
    return max(0.0, min(1.0, scaled))


# ════════════════════════════════════════════════════════
# 12D CONTEXT DEPTH MAPPING
# ════════════════════════════════════════════════════════

# Each cognitive context has a "depth" in the 12D manifold.
# LOGIC is closest to the surface (strongest learning).
# CREATIVITY is deepest (most dampened, most exploratory).
CONTEXT_DEPTH = {
    "LOGIC": 0,       # Surface: strongest Hebbian update
    "EMPATHY": 1,     # Mid-depth: φ⁻¹ dampened
    "CREATIVITY": 2,  # Deep: φ⁻² dampened (preserves novelty)
}


# ════════════════════════════════════════════════════════
# STANDALONE VERIFICATION
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PHI CONSTANTS — GOLDEN RATIO MODULE")
    print("=" * 60)
    print(f"  φ       = {PHI:.15f}")
    print(f"  1/φ     = {PHI_INV:.15f}")
    print(f"  φ²      = {PHI_SQ:.15f}")
    print(f"  φ³      = {PHI_CUBE:.15f}")
    print(f"  ln(φ)   = {LOG_PHI:.15f}")
    print()
    
    print("── Generational Decay ──")
    for n in range(6):
        print(f"  Depth {n}: φ^{n} = {phi_decay(n):.6f} | 1/φ^{n} = {phi_inv_decay(n):.6f}")
    
    print()
    print("── Weight Normalization ──")
    for ctx, depth in CONTEXT_DEPTH.items():
        eta = 0.05
        normalized = phi_normalize(eta, depth)
        print(f"  {ctx} (depth={depth}): η={eta} → η/φ^{depth} = {normalized:.6f}")
    
    print()
    print("── Emotional Scaling ──")
    for raw in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"  raw={raw:.2f} → φ-scaled={phi_scale_emotional(raw):.4f}")
    
    print()
    print("── Influence Radius (mass=10.0) ──")
    for d in range(5):
        print(f"  depth={d}: radius={phi_influence_radius(10.0, d):.4f}")
    
    print()
    
    # Verify identity: φ² = φ + 1
    assert abs(PHI_SQ - (PHI + 1)) < 1e-10, "φ² ≠ φ + 1 (FAILED)"
    # Verify identity: 1/φ = φ - 1
    assert abs(PHI_INV - (PHI - 1)) < 1e-10, "1/φ ≠ φ - 1 (FAILED)"
    
    print("✅ All φ identities verified")
