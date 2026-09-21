# kenseivoice — local CPU TTS for the mesh bots

Per-persona voices via two local engines, selected as `tts.voice: "<engine>:<voice>"`:

- `kokoro:<voice>` — Kokoro-82M ONNX (Apache 2.0). Natural, lightweight, British voices `bf_*`/`bm_*`. Runs on `onnxruntime`, no Torch.
- `pocket:<voice>` — Kyutai Pocket TTS (MIT). Streams mid-utterance; needs Torch.

A bare voice id (no prefix) defaults to Kokoro. Set `tts.provider: kenseivoice` and `tts.voice` per profile.

## Install (gateway venv)

```bash
GW=~/repos/KenseiAgent/.venv/bin/python
# CPU-only Torch FIRST so pocket-tts doesn't pull CUDA wheels:
uv pip install --python $GW "torch==2.12.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python $GW numpy soundfile kokoro-onnx pocket-tts
```

## Models

Kept OUT of the repo. Default location `~/.hermes/plugins/kenseivoice/models/`
(override with `KENSEIVOICE_MODELS_DIR`). Pocket auto-downloads from HF on first use.
Kokoro needs two files there:

```
kokoro-v1.0.onnx   # github.com/thewh1teagle/kokoro-onnx releases, model-files-v1.0
voices-v1.0.bin
```

## Notes

- Pinned to 4 threads (`OMP_NUM_THREADS`) — Pocket is fastest there on ARM Neoverse; more threads regress on small models.
- `voice_compatible = True` so the gateway ffmpeg-converts output to Opus for voice delivery (ffmpeg required).
- Engines load lazily and stay process-cached; a gateway only pulls in the engine its persona uses.
