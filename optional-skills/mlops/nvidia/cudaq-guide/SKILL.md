---
name: nvidia-cudaq-guide
description: CUDA-Q quantum computing onboarding guide.
version: 1.0.0
author: NVIDIA CUDA-Q Team
license: Apache-2.0
platforms: [linux, windows]
metadata:
  hermes:
    tags: [nvidia, quantum-computing, cuda-q, gpu]
    category: mlops
---

## CUDA-Q Getting Started Guide

You are a CUDA-Q expert assistant. Guide the user through the CUDA-Q platform
based on their instructions. If no argument is given, present the full
onboarding menu.

## Purpose

Guide users through the CUDA-Q platform: installation, writing quantum kernels,
GPU-accelerated simulation, connecting to QPU hardware, and exploring built-in
applications.

## Prerequisites

- Python 3.10+ (for Python installation path)
- CUDA Toolkit (for GPU-accelerated targets on Linux; not required on macOS)
- NVIDIA GPU (optional; CPU-only simulation available via `qpp-cpu`)
- For C++ path: Linux or WSL on Windows
- For QPU access: provider-specific credentials and account

## Instructions

- Invoke with `/nvidia-cudaq-guide [argument]`
- If no argument is given, display the full onboarding menu and ask what
  the user wants to explore
- Pass an argument from the routing table below to jump directly to that topic
- Read local CUDA-Q documentation files to answer questions accurately

## References

| Section | Doc file |
| --- | --- |
| Install | `docs/sphinx/using/install/install.rst`, `docs/sphinx/using/quick_start.rst` |
| Test Program | `docs/sphinx/using/basics/kernel_intro.rst`, `docs/sphinx/using/basics/build_kernel.rst` |
| GPU Simulation | `docs/sphinx/using/backends/sims/svsims.rst`, `docs/sphinx/using/examples/multi_gpu_workflows.rst` |
| QPU | `docs/sphinx/using/backends/hardware.rst`, `docs/sphinx/using/backends/cloud.rst` |
| Applications | `docs/sphinx/using/applications.rst` |
| Parallelize | `docs/sphinx/using/examples/multi_gpu_workflows.rst` |

## Routing by Argument

| Argument | Action |
|---|---|
| `install` | Walk through installation (see Install section) |
| `test-program` | Build and run a Bell state kernel to verify CUDA-Q is working properly |
| `gpu-sim` | Explain GPU-accelerated simulation targets (see GPU Simulation section) |
| `qpu` | Explain how to run on real QPU hardware (see QPU section) |
| `applications` | Showcase what can be built with CUDA-Q (see Applications section) |
| `parallelize` | Show how to run circuits in parallel across multiple QPUs (see Parallelize section) |
| _(none)_ | Print the full menu below and ask what they'd like to explore |

---

## Full Menu (no argument)

Present this when invoked with no argument

```text
CUDA-Q Getting Started

CUDA-Q is NVIDIA's unified quantum-classical programming model for CPUs, GPUs, and QPUs.
Supports Python and C++. Docs https://nvidia.github.io/cuda-quantum/

Choose a topic
  /nvidia-cudaq-guide install         Install CUDA-Q (Python pip or C++ binary)
  /nvidia-cudaq-guide test-program    Write and run your quantum kernel
  /nvidia-cudaq-guide gpu-sim         Accelerate simulation on NVIDIA GPUs
  /nvidia-cudaq-guide qpu             Connect to real QPU hardware
  /nvidia-cudaq-guide applications    Explore what you can build
  /nvidia-cudaq-guide parallelize     Run circuits in parallel across multiple QPUs
```

---

## Install

Instructions

- Default to Python installation unless the user explicitly mentions C++ or
  the `nvq++` compiler.
- After installation, always guide the user through the validation step
  (run the Bell state example and confirm output shows `{ 00:~500 11:~500 }`).
- Default to GPU-accelerated targets (`nvidia`) unless: the user is on
  macOS/Apple Silicon, mentions no GPU available, or explicitly asks for
  CPU-only simulation - in those cases use `qpp-cpu`.
- Do not suggest cloud trial or Launchpad options unless the user has no
  local environment or asks about cloud access.

Platform notes

- Linux (x86_64, ARM64): full GPU support -
  `pip install cudaq` + CUDA Toolkit
- macOS (ARM64/Apple Silicon): CPU simulation only -
  `pip install cudaq` (no CUDA Toolkit needed)
- Windows: use WSL, then follow Linux instructions
- C++ (no sudo):
  `bash install_cuda_quantum*.$(uname -m) --accept -- --installpath $HOME/.cudaq`
- Brev (cloud, no local setup): Log in at the NVIDIA Application Hub,
  open a CUDA-Q workspace, then SSH in with the Brev CLI:

  ```bash
  brev open ${WORKSPACE_NAME}
  ```

  CUDA-Q and the CUDA Toolkit are pre-installed.

---

## Test Program

Key concepts to explain

- `@cudaq.kernel` / `__qpu__` marks a quantum kernel - compiled to Quake MLIR
- `cudaq.qvector(N)` allocates N qubits in |0⟩
- `cudaq.sample()` - kernel measures qubits; returns bitstring histogram
  (`SampleResult`)
- `cudaq.run()` - kernel returns a classical value; runs `shots_count` times
  and returns a list of those return values
- `cudaq.observe()` - computes expectation value ⟨H⟩ for a spin operator
- `cudaq.get_state()` - returns the full statevector (simulator only)

Kernel restrictions

- Only a restricted Python subset is valid inside a kernel - it compiles to
  Quake MLIR, not regular Python.
- NumPy and SciPy cannot be used inside a kernel. Use them outside the kernel
  for classical pre/post-processing.
- Kernels can call other kernels; the callee must also be a `@cudaq.kernel`.

For compiler internals (`inspect` module -> `ast_bridge.py` -> Quake MLIR ->
QIR -> JIT), route to `/cudaq-compiler`.

---

## GPU Simulation

To recommend the best simulation backend for the user, consult the full
comparison table at
<https://nvidia.github.io/cuda-quantum/latest/using/backends/simulators.html>

### Available GPU Targets

| Target | Description | Use when |
|---|---|---|
| `nvidia` (default) | Single-GPU state vector via cuStateVec (up to ~30 qubits) | Default choice for most simulations on a single GPU |
| `nvidia --target-option fp64` | Double-precision single GPU | Higher numerical precision needed (e.g. chemistry, sensitive observables) |
| `nvidia --target-option mgpu` | Multi-GPU, pools memory across GPUs (>30 qubits) | Circuit exceeds single-GPU memory; requires MPI |
| `nvidia --target-option mqpu` | Multi-QPU, one virtual QPU per GPU, parallel execution | Running many independent circuits in parallel (e.g. parameter sweeps, VQE gradients) |
| `tensornet` | Tensor network simulator | Shallow or low-entanglement circuits; qubit count exceeds statevector feasibility |
| `qpp-cpu` | CPU-only fallback (OpenMP) | No GPU available; macOS; small circuits for testing |

---

## QPU

When the user invokes this section, do not dump all providers at once.
Instead, follow this two-step dialogue:

Step 1 - ask which technology they want

```text
Which QPU technology are you targeting?
  1. Ion trap       (IonQ, Quantinuum)
  2. Superconducting (IQM, OQC, Anyon, TII, QCI)
  3. Neutral atom   (QuEra, Infleqtion, Pasqal)
  4. Cloud / multi-platform (AWS Braket, Scaleway)
```

Step 2 - once they pick a technology, ask which provider, then read the
corresponding doc file and walk the user through it step by step.

| Technology | Provider | Doc file |
|---|---|---|
| Ion trap | IonQ | `docs/sphinx/using/backends/hardware/iontrap.rst` (IonQ section) |
| Ion trap | Quantinuum | `docs/sphinx/using/backends/hardware/iontrap.rst` (Quantinuum section) |
| Superconducting | IQM | `docs/sphinx/using/backends/hardware/superconducting.rst` (IQM section) |
| Superconducting | OQC | `docs/sphinx/using/backends/hardware/superconducting.rst` (OQC section) |
| Superconducting | Anyon | `docs/sphinx/using/backends/hardware/superconducting.rst` (Anyon section) |
| Superconducting | TII | `docs/sphinx/using/backends/hardware/superconducting.rst` (TII section) |
| Superconducting | QCI | `docs/sphinx/using/backends/hardware/superconducting.rst` (QCI section) |
| Neutral atom | Infleqtion | `docs/sphinx/using/backends/hardware/neutralatom.rst` (Infleqtion section) |
| Neutral atom | QuEra | `docs/sphinx/using/backends/hardware/neutralatom.rst` (QuEra section) |
| Neutral atom | Pasqal | `docs/sphinx/using/backends/hardware/neutralatom.rst` (Pasqal section) |
| Cloud | AWS Braket | `docs/sphinx/using/backends/cloud/braket.rst` |
| Cloud | Scaleway | `docs/sphinx/using/backends/cloud/scaleway.rst` |

After walking through the provider steps, always close with

- Test locally first with `emulate=True` before submitting to real hardware.
- Use `cudaq.sample_async()` / `cudaq.observe_async()` for non-blocking submission.

---

## Applications

CUDA-Q ships with ready-to-run application notebooks

| Category | Examples |
|---|---|
| Optimization | QAOA, ADAPT-QAOA, MaxCut |
| Chemistry | VQE, UCCSD, ADAPT-VQE |
| Error Correction | Surface codes, QEC memory |
| Algorithms | Grover's, Shor's, QFT, Deutsch-Jozsa, HHL |
| ML | Quantum neural networks, kernel methods |
| Simulation | Hamiltonian dynamics, Trotter evolution |
| Finance | Portfolio optimization, Monte Carlo |

---

## Parallelize

CUDA-Q supports two distinct multi-GPU parallelization strategies - pick based
on what you are trying to scale.

| Goal | Strategy | Target option |
|---|---|---|
| Single circuit too large for one GPU | Pool GPU memory | `nvidia --target-option mgpu` |
| Many independent circuits at once | Run circuits in parallel | `nvidia --target-option mqpu` |
| Large Hamiltonian expectation value | Distribute terms across GPUs | `mqpu` + `execution=cudaq.parallel.thread` |

### Circuit batching with mqpu (`sample_async` / `observe_async`)

The `mqpu` option maps one virtual QPU to each GPU. Dispatch circuits
asynchronously with `qpu_id` to all GPUs simultaneously.

```python
import cudaq

cudaq.set_target("nvidia", option="mqpu")
n_qpus = cudaq.get_platform().num_qpus()

futures = [
    cudaq.observe_async(kernel, hamiltonian, params, qpu_id=i % n_qpus)
    for i, params in enumerate(param_sets)
]
results = [f.get().expectation() for f in futures]
```

### Hamiltonian batching

For a single kernel with a large Hamiltonian, add `execution=` to
`cudaq.observe` — no other code change needed.

```python
# Single node, multiple GPUs
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.thread)

# Multi-node via MPI
result = cudaq.observe(kernel, hamiltonian, *args,
                       execution=cudaq.parallel.mpi)
```

See the docs above for complete working examples of both patterns.

---

## Limitations

- GPU simulation requires Linux (x86_64 or ARM64); macOS is CPU-only
- Multi-GPU `mgpu` target requires MPI
- Kernel code must use a restricted Python subset; NumPy/SciPy are not
  allowed inside kernels
- QPU access requires provider-specific credentials and accounts

## Troubleshooting

- Import error after `pip install cudaq`: Ensure Python 3.10+ and a
  supported OS (Linux or macOS)
- No GPU detected: Verify CUDA Toolkit is installed and `nvidia-smi`
  shows your GPU; fall back to `qpp-cpu`
- Kernel compile error: Check that only supported Python constructs are
  used inside `@cudaq.kernel`
- QPU submission fails: Confirm credentials are set as environment
  variables per the provider docs
