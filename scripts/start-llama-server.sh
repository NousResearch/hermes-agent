#!/usr/bin/env bash
# start-llama-server.sh — turnkey llama-server startup that lines up with
# Hermes' built-in `llama-cpp` provider.
#
# Boots llama.cpp's llama-server on the same host:port that the
# `plugins/model-providers/llama-cpp/` profile defaults to
# (http://127.0.0.1:8088/v1). After this script is running,
#
#   hermes chat --provider llama-cpp --model <whatever>
#
# works with no further config. To override the port/host, set PORT / HOST
# below AND export LLAMA_CPP_BASE_URL=http://${HOST}:${PORT}/v1 so Hermes
# follows.
#
# Usage:
#   start-llama-server.sh <gguf-path>
#   start-llama-server.sh ~/models/some-model.gguf
#
# Or with explicit overrides:
#   PORT=9000 CTX=32768 THREADS=8 start-llama-server.sh <gguf>
#   N_GPU_LAYERS=0 start-llama-server.sh <gguf>   # force CPU-only, skip auto-detect
#
# Requires: llama-server binary on PATH (or LLAMA_SERVER env var pointing to it)
#           See https://github.com/ggml-org/llama.cpp/releases for prebuilt binaries.

set -e

GGUF="${1:-}"
PORT="${PORT:-8088}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"
HOST="${HOST:-127.0.0.1}"
# N_GPU_LAYERS is intentionally left unset here (not defaulted to 0) so the
# auto-detect block below can tell "user didn't set it" apart from
# "user explicitly asked for CPU-only with N_GPU_LAYERS=0". See
# detect_llama_gpu_backend() / detect_system_gpu() for how the guess is made.
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"

if [ -z "$GGUF" ]; then
  cat << 'USAGE'
Usage: start-llama-server.sh <path-to-gguf>

Environment overrides:
  PORT=8088              Server port (matches Hermes llama-cpp provider default)
  CTX=16384              Context window
  THREADS=12             CPU threads
  N_GPU_LAYERS           Unset = auto-detect (offload all layers if a GPU +
                         GPU-enabled llama-server build are both found, else
                         CPU-only). Set explicitly to override: 0=CPU only,
                         -1=all layers on GPU, positive int=N layers.
  LLAMA_SERVER=llama-server   Path to llama-server binary

After boot:
  hermes chat --provider llama-cpp --model <model-id-from-/v1/models>
USAGE
  exit 1
fi

[ ! -f "$GGUF" ] && { echo "ERROR: GGUF file not found: $GGUF"; exit 2; }

if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
  cat << 'EOF'
ERROR: llama-server not found on PATH.

Install:
  Linux x86_64: https://github.com/ggml-org/llama.cpp/releases
  macOS:        brew install llama.cpp
  Pip:          pip install llama-cpp-python[server]
                # then: python -m llama_cpp.server --model <gguf> --port 8088

Or set LLAMA_SERVER=/path/to/llama-server explicitly.
EOF
  exit 3
fi

# --- GPU auto-detection ------------------------------------------------
#
# Two independent questions, both required for GPU offload to actually work:
#   1. Does this machine have a GPU at all?
#   2. Was THIS llama-server binary built with a GPU backend?
#
# The second question matters more than it looks. The official llama.cpp
# release binaries for Linux ship several separate builds (cpu, vulkan,
# rocm, sycl) — there is no prebuilt CUDA build for Linux at all, only for
# Windows. So it's entirely possible to have an NVIDIA GPU, a `llama-server`
# binary on PATH, and `-ngl 999` set, and still be running 100% on CPU with
# zero warning, because that particular binary was compiled CPU-only. This
# is not a hypothetical: it's what happened during this script's own
# development on Linux/NVIDIA, silently, for a while. Vulkan is the fix on
# Linux + NVIDIA/AMD/Intel — it works with the vendor's normal display
# driver, no CUDA toolkit install needed, and there IS a prebuilt Linux
# Vulkan release.
#
# GPU backends load as shared libraries alongside the llama-server binary
# (libggml-cuda*, libggml-vulkan*, libggml-metal*, libggml-hip*,
# libggml-sycl*). Their presence is a much cheaper and more reliable check
# than trying to parse startup log output, and doesn't require spinning up
# a throwaway server process just to inspect it.

detect_llama_gpu_backend() {
  local server_bin server_dir
  server_bin="$(command -v "$LLAMA_SERVER" 2>/dev/null || echo "$LLAMA_SERVER")"
  server_dir="$(dirname "$server_bin")"
  if ls "$server_dir"/libggml-cuda* >/dev/null 2>&1; then echo "cuda"; return; fi
  if ls "$server_dir"/libggml-vulkan* >/dev/null 2>&1; then echo "vulkan"; return; fi
  if ls "$server_dir"/libggml-metal* >/dev/null 2>&1; then echo "metal"; return; fi
  if ls "$server_dir"/libggml-hip* >/dev/null 2>&1; then echo "hip"; return; fi
  if ls "$server_dir"/libggml-sycl* >/dev/null 2>&1; then echo "sycl"; return; fi
  echo "none"
}

detect_system_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia"; return
  fi
  if [ "$(uname)" = "Darwin" ]; then
    echo "apple"; return   # Metal is always available on macOS
  fi
  if command -v rocm-smi >/dev/null 2>&1 && rocm-smi >/dev/null 2>&1; then
    echo "amd"; return
  fi
  if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | grep -qiE 'vga|3d controller'; then
    echo "present"; return   # a GPU exists but vendor tooling isn't on PATH to name it
  fi
  echo "none"
}

GPU_BACKEND="$(detect_llama_gpu_backend)"
SYSTEM_GPU="$(detect_system_gpu)"

if [ -z "${N_GPU_LAYERS+x}" ]; then
  # Not explicitly set by the caller — auto-detect.
  if [ "$GPU_BACKEND" != "none" ] && [ "$SYSTEM_GPU" != "none" ]; then
    N_GPU_LAYERS=999
    echo "  auto-detect: GPU ($SYSTEM_GPU) + $GPU_BACKEND-enabled llama-server -> offloading all layers"
  else
    N_GPU_LAYERS=0
    if [ "$SYSTEM_GPU" != "none" ] && [ "$GPU_BACKEND" = "none" ]; then
      cat >&2 << EOF
  WARNING: a GPU was detected ($SYSTEM_GPU) but this llama-server binary has
           no GPU backend compiled in (no libggml-cuda/vulkan/metal/hip/sycl
           found next to it at: $(dirname "$(command -v "$LLAMA_SERVER")")).
           Falling back to CPU-only.
           Fix: grab a GPU-enabled build from
           https://github.com/ggml-org/llama.cpp/releases
           On Linux + NVIDIA/AMD/Intel, the *-vulkan-* build is the easy
           path — no CUDA toolkit needed, works with your normal display
           driver. (There is no official prebuilt CUDA build for Linux;
           CUDA requires building llama.cpp from source there.)
EOF
    fi
  fi
else
  # Caller set N_GPU_LAYERS explicitly — respect it, but still warn loudly
  # if it can't possibly do anything (the exact silent-CPU-fallback trap
  # described above).
  if [ "$N_GPU_LAYERS" != "0" ] && [ "$GPU_BACKEND" = "none" ]; then
    cat >&2 << EOF
  WARNING: N_GPU_LAYERS=$N_GPU_LAYERS is set, but this llama-server binary has
           no GPU backend compiled in — it will silently run on CPU
           regardless of this setting. See the auto-detect note above for
           how to get a GPU-enabled build.
EOF
  fi
fi

echo "=== llama-server boot ==="
echo "  model:       $GGUF"
echo "  endpoint:    http://${HOST}:${PORT}/v1"
echo "  context:     $CTX"
echo "  threads:     $THREADS"
echo "  gpu layers:  $N_GPU_LAYERS"
echo ""

# Hermes alignment hint
EXPECTED_BASE_URL="http://127.0.0.1:8088/v1"
ACTUAL_BASE_URL="http://${HOST}:${PORT}/v1"
if [ "$ACTUAL_BASE_URL" != "$EXPECTED_BASE_URL" ]; then
  echo "  NOTE: Hermes' llama-cpp provider defaults to ${EXPECTED_BASE_URL}."
  echo "        Override with: export LLAMA_CPP_BASE_URL=${ACTUAL_BASE_URL}"
  echo ""
else
  echo "  Hermes: ready — run 'hermes chat --provider llama-cpp'"
  echo ""
fi

# --jinja loads the model's chat template, which is required for any GGUF
# that needs Jinja-rendered tool-call messages.
ARGS=(
  -m "$GGUF"
  --host "$HOST"
  --port "$PORT"
  --ctx-size "$CTX"
  --threads "$THREADS"
  --jinja
  --n-gpu-layers "$N_GPU_LAYERS"
)

exec "$LLAMA_SERVER" "${ARGS[@]}"
