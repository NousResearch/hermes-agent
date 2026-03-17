# NeuTTS CLI

Small standalone CLI for installing, checking, and running [NeuTTS](https://github.com/neuphonic/neutts) locally.

This scaffold is designed to be a good fit for a future Hermes optional skill:

- predictable commands
- machine-friendly output for inspection
- local voice profile management
- direct local synthesis

## Commands

```bash
neutts install --all
neutts doctor
neutts list-models
neutts add-voice demo --ref-audio ./samples/jo.wav --ref-text-file ./samples/jo.txt
neutts list-voices
neutts synth --voice demo --text Hello from NeuTTS --out ./out.wav
neutts synth --voice demo --text Quick smoke test
```

## Install the bundled scaffold

```bash
cd optional-skills/mlops/models/neutts/assets/neutts-cli
python -m pip install -e .
```

## Add the bundled sample profile

This skill bundles an upstream NeuTTS sample reference in `samples/`.

```bash
cd optional-skills/mlops/models/neutts/assets/neutts-cli
PYTHONPATH=src python -m neutts_cli.cli add-voice jo-demo \
  --ref-audio ./samples/jo.wav \
  --ref-text-file ./samples/jo.txt \
  --language en
```

Then inspect it with:

```bash
PYTHONPATH=src python -m neutts_cli.cli list-voices
```

## Notes

- `install` installs the upstream `neutts` package into the current Python environment.
- `list-voices` shows local voice profiles created with `add-voice`.
- `synth` uses NeuTTS reference cloning. A voice profile is just a saved reference audio/text pair.
- `synth` accepts quoted or unquoted text and defaults to `./out.wav` when `--out` is omitted.
- GGUF / `llama-cpp-python` acceleration can vary by platform, so the CLI prints follow-up guidance instead of forcing one build recipe.
