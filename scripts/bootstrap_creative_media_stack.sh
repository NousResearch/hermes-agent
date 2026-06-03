#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the advanced creative media stack used by Paimon/Zeus-style agents.
# This script installs/validates lightweight dependencies. Heavy creative tools
# are opt-in because they can be large and host-specific.

ROOT="${HERMES_CREATIVE_TOOLS_ROOT:-$HOME/.hermes/tools/creative-media}"
VENV="${HERMES_CREATIVE_MEDIA_VENV:-$HOME/.hermes/tool-venvs/creative-media}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
PYTHON_BIN="${PYTHON_BIN:-$VENV/bin/python}"
INSTALL_HEAVY="${INSTALL_HEAVY_CREATIVE_TOOLS:-0}"

mkdir -p "$ROOT" "$ROOT/bin" "$(dirname "$VENV")"

log() { printf '\n[creative-media] %s\n' "$*"; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

log "checking base commands"
for cmd in "$PYTHON_BOOTSTRAP" curl tar; do
  if ! need_cmd "$cmd"; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  log "creating isolated Python venv at $VENV"
  "$PYTHON_BOOTSTRAP" -m venv "$VENV"
fi

log "installing Python media packages into isolated venv"
"$PYTHON_BIN" -m pip install --upgrade pip >/tmp/hermes-creative-pip.log 2>&1 || {
    cat /tmp/hermes-creative-pip.log >&2
    exit 1
  }
"$PYTHON_BIN" -m pip install --upgrade \
  moviepy opencv-python pillow imageio imageio-ffmpeg numpy soundfile onnxruntime \
  faster-whisper openai-whisper elevenlabs >/tmp/hermes-creative-pip.log 2>&1 || {
    cat /tmp/hermes-creative-pip.log >&2
    exit 1
  }

log "checking FFmpeg"
if ! need_cmd ffmpeg || ! need_cmd ffprobe; then
  cat >&2 <<'MSG'
FFmpeg/ffprobe are not on PATH.
Install with your OS package manager or provide a static build, then re-run.
Example Ubuntu: sudo apt-get update && sudo apt-get install -y ffmpeg
MSG
else
  ffmpeg -version | head -1
fi

log "checking ImageMagick"
if need_cmd magick; then
  magick -version | head -1
elif need_cmd convert; then
  convert -version | head -1
else
  echo "ImageMagick not found; install it if image cleanup/compositing workflows need it." >&2
fi

log "checking hosted creative CLIs"
if need_cmd higgsfield; then
  higgsfield --version 2>/dev/null || higgsfield version 2>/dev/null || true
else
  echo "Higgsfield CLI not found. Install/auth separately when Higgsfield skills are used." >&2
fi
if need_cmd mmx; then
  mmx --version 2>/dev/null || true
else
  echo "MiniMax mmx CLI not found. Install/auth separately when mmx-cli skill is used." >&2
fi

if [[ "$INSTALL_HEAVY" == "1" ]]; then
  log "heavy installs requested"
  mkdir -p "$ROOT/src"
  if ! [[ -d "$ROOT/src/ViMax/.git" ]]; then
    git clone https://github.com/HKUDS/ViMax.git "$ROOT/src/ViMax"
  fi
  echo "ViMax cloned at $ROOT/src/ViMax"
  echo "Install HyperFrames using its upstream instructions for the target host/runtime."
else
  log "skipping heavy tools (set INSTALL_HEAVY_CREATIVE_TOOLS=1 to clone ViMax/large deps)"
fi

log "verifying Python imports"
"$PYTHON_BIN" - <<'PY'
import importlib.util
mods = [
    'moviepy', 'cv2', 'PIL', 'imageio', 'imageio_ffmpeg', 'numpy',
    'soundfile', 'onnxruntime', 'faster_whisper', 'whisper', 'elevenlabs'
]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing imports: ' + ', '.join(missing))
print('creative media Python imports OK')
PY

log "done"
