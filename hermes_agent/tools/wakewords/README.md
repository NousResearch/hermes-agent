# Bundled wake-word models

`hey_hermes.tflite` — the on-device "Hey Hermes" hotword model. This is the
default detector for the wake word feature (see
`website/docs/user-guide/features/wake-word.md`); no training or setup is
required to say "hey hermes".

- **Engine:** [pyopen-wakeword](https://github.com/rhasspy/pyopen-wakeword)
  (rhasspy's maintained fork of openWakeWord; Apache-2.0). Runs TFLite via a
  bundled `tensorflowlite_c` library — no onnx, no runtime download.
- **Provenance:** trained with the openWakeWord training pipeline (synthetic
  TTS-generated speech), which produces the `.tflite` artifact. Redistribution
  is permitted under the openWakeWord license.
- **Label:** the model registers as `hey_hermes` (matches the filename).
- **Runtime:** the `pyopen-wakeword` wheel includes the shared
  melspectrogram and embedding models. Starting this engine requires no
  model download. Identical model files alone do not establish identical
  scores across inference engines or platforms.

To use a different phrase, point `wake_word.openwakeword.model` at an
absolute path to a compatible `.tflite` model. Hermes does not download
models by name, and this engine does not load `.onnx` files. See the
wake-word docs for the training guide and platform limits.
