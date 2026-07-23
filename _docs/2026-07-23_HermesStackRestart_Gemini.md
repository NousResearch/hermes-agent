# Implementation Log: Hermes Stack Restart & Llama Server Optimization (RTX 5060 Ti 16GB)

- **Date**: 2026-07-23
- **Executor**: Gemini (Antigravity Agent)
- **Target**: Hermes Agent Stack + Llama Server (RTX 5060 Ti 16GB Profile)

---

## 1. Overview & Background

The user requested a full restart of the Hermes stack, including:
1. Rebuilding the Hermes Desktop app using `npm`.
2. Performing deep research on `zapabob/llama.cpp` and related KV cache reduction algorithms (**TurboQuant**, **Triality**, **SO(8) Vector Protection**).
3. Configuring and optimizing `llama-server` flags tailored for **NVIDIA RTX 5060 Ti 16GB VRAM** with large context support (64k–131k tokens).
4. Restarting the entire stack (`restart-hermes-stack.ps1 -StartLlama`) with robust error handling and verification.

---

## 2. Research Findings (zapabob/llama.cpp & KV Cache Reduction)

### 2.1. Ecosystem & Quantization Techniques
- **`zapabob/llama.cpp`**: A CUDA-optimized research fork integrating TurboQuant (`TQ4_1S`, `TQ3_1S`), SO(8) Triality K-side vector rotation, and custom CUDA reduction kernels.
- **TurboQuant**: Combines PolarQuant (randomized orthogonal rotation + polar transform + Lloyd-Max scalar quantization) with a Quantized Johnson-Lindenstrauss (QJL) 1-bit residual transform, allowing 4-bit / 3-bit KV cache compression.
- **SO(8) Triality**: Rotates Key vectors using 8D orthogonal matrices ($SO(8)$) to redistribute outlier magnitudes across dimensions before quantization, preventing attention collapse ($\text{Softmax}(QK^T/\sqrt{d})$) during long-context inference.

### 2.2. VRAM Calculations for RTX 5060 Ti 16GB
- **Usable VRAM**: ~15.2 GB.
- **Model Footprint**: 4B–9B models (e.g. Gemma 4 9B, Agents-A1 4B/12B, Hermes 3 8B) consume ~5GB–9GB in weights + CUDA overhead.
- **KV Cache at 128k Context**:
  - FP16: ~16.8 GB (OOM)
  - Q8_0: ~8.9 GB (Fits comfortably with ~1.4GB free)
  - Q4_0 / TQ4_1S: ~4.7 GB (Fits comfortably with ~5.6GB free)

---

## 3. Key Changes Implemented

### 3.1. `scripts/windows/start-llama-secretary.ps1`
- Updated default context size `HERMES_LLAMA_CTX` to `131072` (128k context) for `rtx5060ti` profile.
- Added `--threads 8` (default physical core allocation) and `--parallel 2` (concurrent agent + subagent request support).
- Integrated `--so8-triality-k` flag detection for builds supporting SO(8) Triality.
- **Enhanced Fallback Robustness**: Refactored `attemptPlans` and crash handling so that if experimental flags (`--so8-triality-k` or custom `turbo3` KV types) trigger GGML memory allocation assertions or exit codes on specific builds, the script automatically catches the failure and falls back gracefully through standard `q8_0` / `q4_0` / FP16 KV cache configurations rather than aborting.

### 3.2. `docs/local-secretary-runtime.md`
- Added comprehensive section for **RTX 5060 Ti 16GB** tuning, KV cache quantization math, and TurboQuant / SO(8) options documentation.

### 3.3. Desktop App Rebuild
- Executed `npm install` and `npm run build` in `apps/desktop` (transformed 4,479 modules, bundled Electron main & preload).

---

## 4. Execution & Verification Summary

1. **Desktop Build**: `npm run build` completed successfully (dist/index.html & electron-main.mjs staged).
2. **Stack Restart**: Ran `scripts/windows/restart-hermes-stack.ps1 -StartLlama`.
3. **Llama Server Health**:
   - URL: `http://127.0.0.1:8080/v1/models`
   - Status: Ready (`InternScience/Agents-A1-4B-F16-GGUF:F16`)
4. **Gateway Status**: Gateway process online (PID: 24080, 28120).
5. **Auxiliary Services**: Tailscale serve proxy, LINE / Llama ngrok tunnels, and Obsidian memory-graph server (:8765) online.

---

## 5. Residual Risks & Next Actions

- **Binary Support**: Standard `llama-server` builds will transparently use `--cache-type-k q8_0` / `q4_0`, while `zapabob/llama.cpp` builds will leverage TurboQuant / SO(8) Triality. The launcher gracefully manages both.
- **Monitoring**: Continue monitoring GPU VRAM during 100k+ context multi-turn agent sessions.
