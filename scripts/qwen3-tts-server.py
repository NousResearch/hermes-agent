r"""
DEPRECATED: This script is superseded by the agent-meow voice gateway
(``python -m agent_meow.hermes_voice_gateway``) as of Plan 005 Task 4.
It remains as a migration shim for operators who have not yet switched
to the agent-meow-owned voice gateway. Do not extend or build on this
script — new voice logic belongs in agent-meow.

Qwen3-TTS Bridge Server for Hermes
===================================
Lightweight HTTP server that wraps Qwen3-TTS-0.6B for Hermes command provider.
Uses the agent-meow venv (which already has qwen-tts installed).

Usage:
    cd agent-meow
    .venv/Scripts/python.exe ../hermes-agent/scripts/qwen3-tts-server.py --port 17494

Hermes config (data/config.yaml):
    tts:
      provider: qwen3-tts
      providers:
        qwen3-tts:
          type: command
          command: >-
            curl -s -X POST http://host.docker.internal:17494/tts
            -H "Content-Type: application/json"
            -d '{"text_file": "{input_path}", "output_path": "{output_path}"}'
          output_format: wav
          timeout: 120
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Lazy model loading ──────────────────────────────────────────────
_model = None
_model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_device = "cpu"


def get_model():
    global _model
    if _model is None:
        print(f"[qwen3-tts-server] Loading {_model_name} on {_device}...", flush=True)

        # Explicitly register the qwen3_tts model type with HuggingFace
        # Transformers BEFORE any from_pretrained call. The qwen_tts
        # package only registers inside Qwen3TTSModel.from_pretrained(),
        # but the internal AutoModel.from_pretrained() call needs the
        # registration to already be in place.
        from transformers import AutoConfig, AutoModel, AutoProcessor
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

        tokenizer = Qwen3TTSTokenizer.from_pretrained(_model_name)
        _model = Qwen3TTSModel.from_pretrained(_model_name, device_map=_device)
        _model.tokenizer = tokenizer
        print(f"[qwen3-tts-server] Model loaded.", flush=True)
    return _model


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": _model_name})


@app.route("/tts", methods=["POST"])
def tts():
    """Accept text (direct) or text_file path, generate TTS audio.

    Returns WAV bytes directly in the response body (Content-Type: audio/wav).
    Also supports a JSON mode: add ?format=json or Accept: application/json
    to get a JSON response with base64-encoded audio.
    """
    data = request.get_json(force=True)
    text = data.get("text")
    text_file = data.get("text_file")

    if text:
        pass  # use text directly
    elif text_file:
        try:
            text = Path(text_file).read_text(encoding="utf-8").strip()
        except (OSError, FileNotFoundError):
            return jsonify({"error": f"Cannot read text_file: {text_file}"}), 400
    else:
        return jsonify({"error": "Missing 'text' or 'text_file'"}), 400

    if not text:
        return jsonify({"error": "Empty text"}), 400

    print(
        f"[qwen3-tts-server] Generating TTS for {len(text)} chars",
        flush=True,
    )
    t0 = time.time()

    try:
        model = get_model()
        speaker = data.get("speaker", "Vivian")
        language = data.get("language", "English")
        wavs, sample_rate = model.generate_custom_voice(
            text, speaker=speaker, language=language
        )
        wav = wavs[0] if isinstance(wavs, list) else wavs

        # Write WAV to in-memory buffer
        import io as _io

        buf = _io.BytesIO()
        sf.write(buf, wav, sample_rate, format="WAV")
        wav_bytes = buf.getvalue()

        elapsed = time.time() - t0
        size_kb = len(wav_bytes) / 1024
        print(
            f"[qwen3-tts-server] Done: {size_kb:.0f} KB in {elapsed:.1f}s", flush=True
        )

        # If output_path is provided and writable, also save to disk
        output_path = data.get("output_path")
        if output_path:
            try:
                Path(output_path).write_bytes(wav_bytes)
            except OSError:
                pass  # cross-platform path — ignore

        # Check if client wants JSON (base64) or raw WAV
        wants_json = request.args.get(
            "format"
        ) == "json" or "application/json" in request.headers.get("Accept", "")
        if wants_json:
            import base64

            return jsonify({
                "success": True,
                "size_kb": round(size_kb, 1),
                "elapsed_s": round(elapsed, 1),
                "sample_rate": sample_rate,
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            })
        else:
            return wav_bytes, 200, {"Content-Type": "audio/wav"}

    except Exception as e:
        print(f"[qwen3-tts-server] Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Bridge Server")
    parser.add_argument(
        "--port", type=int, default=17494, help="Listen port (default: 17494)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", help="Model name"
    )
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    _model_name = args.model
    _device = args.device

    print(f"[qwen3-tts-server] Starting on port {args.port}...", flush=True)
    app.run(host="0.0.0.0", port=args.port, debug=False)
