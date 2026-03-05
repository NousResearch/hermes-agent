# OBLITERATUS Analysis Modules — Reference

27 analysis modules for mechanistic interpretability of refusal in LLMs.
These help you understand HOW a model refuses before you decide to remove it.

## Core Analysis (Run These First)

### Alignment Imprint Detection
**File:** `alignment_imprint.py`
**Purpose:** Identifies what alignment technique was used to train the model
**Detects:** DPO, RLHF, CAI (Constitutional AI), SFT (Supervised Fine-Tuning)
**How:** Analyzes subspace geometry — each alignment method leaves a distinct
geometric "fingerprint" in the weight space
**Output:** Detected method + confidence score
**Why it matters:** Different alignment methods need different abliteration approaches.
DPO models typically have cleaner single-direction refusal; RLHF is more diffuse.

### Concept Cone Geometry
**File:** `concept_geometry.py`
**Purpose:** Maps whether refusal is one direction or a polyhedral cone (many)
**Output:** Cone angle, dimensionality, per-category breakdown
**Why it matters:** If refusal is a single direction, `basic` method works. If it's
a cone (multiple directions for different refusal categories), you need `advanced`
or `informed` with higher `n_directions`.

### Refusal Logit Lens
**File:** `logit_lens.py`
**Purpose:** Identifies the specific layer where the model "decides" to refuse
**How:** Projects intermediate hidden states to vocabulary space at each layer,
watches when "I cannot" tokens spike in probability
**Output:** Layer-by-layer refusal probability plot
**Why it matters:** Tells you which layers are most important to target

### Ouroboros (Self-Repair) Detection
**File:** `anti_ouroboros.py`
**Purpose:** Predicts whether the model will reconstruct its refusal after removal
**How:** Measures redundancy in refusal representation across layers
**Output:** Self-repair risk score (0-1)
**Why it matters:** High self-repair risk means you need multiple refinement passes
or the `informed` method which auto-compensates

### Causal Tracing
**File:** `causal_tracing.py`
**Purpose:** Determines which components are causally necessary for refusal
**How:** Patches activations between clean and corrupted runs, measures causal effect
**Output:** Causal importance map across layers, heads, and MLPs
**Why it matters:** Shows exactly which components to target for surgical removal

## Geometric Analysis

### Cross-Layer Alignment
**File:** `cross_layer.py`
**Purpose:** Measures how aligned refusal directions are across layers
**Output:** Alignment matrix, cluster assignments
**Why it matters:** If directions are highly aligned across layers, removal is easier.
If they cluster, you may need layer-group-specific directions.

### Residual Stream Decomposition
**File:** `residual_stream.py`
**Purpose:** Breaks down refusal into Attention vs MLP contributions
**Output:** Per-layer Attention/MLP contribution to refusal direction
**Why it matters:** Helps decide whether to target attention heads, MLPs, or both

### Riemannian Manifold Geometry
**File:** `riemannian_manifold.py` (673 lines)
**Purpose:** Analyzes the weight manifold geometry around refusal directions
**Output:** Curvature, geodesics, tangent space analysis
**Why it matters:** Research-grade; helps understand the geometric structure of alignment

### Whitened SVD
**File:** `whitened_svd.py`
**Purpose:** Covariance-normalized SVD extraction
**How:** Whitens the activation covariance before computing refusal directions,
separating true refusal signal from natural activation variance
**Output:** Cleaner refusal directions with less noise
**Why it matters:** Produces more precise directions, especially for noisy activations

## Probing & Classification

### Activation Probing
**File:** `activation_probing.py`
**Purpose:** Post-excision probing to verify refusal signal is truly gone
**Output:** Residual refusal signal strength per layer
**Why it matters:** Verification that abliteration was complete

### Probing Classifiers
**File:** `probing_classifiers.py`
**Purpose:** Trains linear classifiers to detect refusal in hidden states
**Output:** Classification accuracy per layer (should drop to ~50% after abliteration)
**Why it matters:** Quantitative measure of refusal removal completeness

### Activation Patching
**File:** `activation_patching.py`
**Purpose:** Interchange interventions — swap activations between harmful/harmless runs
**Output:** Which components are sufficient (not just necessary) for refusal
**Why it matters:** Complementary to causal tracing; together they give full picture

## Transfer & Robustness

### Cross-Model Transfer
**File:** `cross_model_transfer.py`
**Purpose:** Tests if refusal directions from one model work on another
**Output:** Transfer success rate between model pairs
**Why it matters:** If directions transfer, you can skip PROBE stage on similar models

### Defense Robustness
**File:** `defense_robustness.py`
**Purpose:** Evaluates how robust the model's refusal defenses are
**Output:** Robustness score, entanglement mapping
**Why it matters:** Higher robustness = need more aggressive method

### Spectral Certification
**File:** `spectral_certification.py`
**Purpose:** Certifies completeness of refusal direction removal
**Output:** Spectral gap analysis, completeness score
**Why it matters:** Formal verification that all major refusal components are addressed

## Advanced / Research

### SAE-based Abliteration
**File:** `sae_abliteration.py` (762 lines)
**Purpose:** Uses Sparse Autoencoder features to decompose refusal at feature level
**Output:** Refusal-specific SAE features, targeted removal
**Why it matters:** Most fine-grained approach; can target individual refusal "concepts"

### Wasserstein Optimal Extraction
**File:** `wasserstein_optimal.py`
**Purpose:** Optimal transport-based direction extraction
**Output:** Wasserstein-optimal refusal directions
**Why it matters:** Theoretically optimal direction extraction under distributional assumptions

### Bayesian Kernel Projection
**File:** `bayesian_kernel_projection.py`
**Purpose:** Bayesian approach to refusal direction projection
**Output:** Posterior distribution over refusal directions
**Why it matters:** Quantifies uncertainty in direction estimation

### Conditional Abliteration
**File:** `conditional_abliteration.py`
**Purpose:** Domain-specific conditional removal (remove refusal for topic X but keep for Y)
**Output:** Per-domain refusal directions
**Why it matters:** Selective uncensoring — remove only specific refusal categories

### Steering Vectors
**File:** `steering_vectors.py`
**Purpose:** Generate inference-time steering vectors (reversible alternative)
**Output:** Steering vector files that can be applied/removed at inference
**Why it matters:** Non-destructive alternative to permanent weight modification

### Tuned Lens
**File:** `tuned_lens.py`
**Purpose:** Trained linear probes per layer (more accurate than raw logit lens)
**Output:** Layer-by-layer refusal representation with trained projections
**Why it matters:** More accurate than logit lens, especially for deeper models

### Multi-Token Position Analysis
**File:** `multi_token_position.py`
**Purpose:** Analyzes refusal signal at multiple token positions (not just last)
**Output:** Position-dependent refusal direction maps
**Why it matters:** Some models encode refusal at the system prompt position, not the query

### Sparse Surgery
**File:** `sparse_surgery.py`
**Purpose:** Row-level sparse weight surgery instead of full matrix projection
**Output:** Targeted weight modifications at the row level
**Why it matters:** More surgical than full-matrix projection, less collateral damage
