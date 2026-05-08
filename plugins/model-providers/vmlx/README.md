# vMLX provider — local Apple Silicon inference

`vmlx` is a Hermes model-provider plugin that runs MLX-format LLMs on Apple
Silicon and serves them through an OpenAI-compatible chat-completions endpoint.
It is the recommended backend for fully airgapped Hermes operation on macOS.

> **Status:** v0.1.0 — initial release. The public surface (provider names,
> base URLs, aliases) may evolve in response to PR-review feedback before
> the plugin graduates to 1.0. See [`CHANGELOG.md`](./CHANGELOG.md).

## Why this exists

The bundled [`custom`](../custom/) profile already covers generic local
OpenAI-compatible servers (Ollama, vLLM, llama.cpp). `vmlx` is a sibling
that adds:

- **Sane localhost defaults** — `base_url` is pre-filled, no config needed.
- **Apple Silicon platform gate** — `ImportError` on non-Darwin so the
  plugin is invisible to Linux/Windows contributors.
- **Primary/janitor split out of the box** — two profiles registered side
  by side so auxiliary work (compression, memory writes, summarization,
  skill curation) can run on a separate `vmlx serve` instance and stop
  consuming the primary model's context window.

## What gets registered

The plugin's `__init__.py` calls `register_provider()` twice:

| Profile name   | Default base URL              | Suggested role          |
|----------------|-------------------------------|-------------------------|
| `vmlx`         | `http://localhost:8000/v1`    | Primary agent loop      |
| `vmlx-janitor` | `http://localhost:8001/v1`    | Auxiliary / aux model   |

Aliases: `mlx`, `mlx-server`, `apple-mlx`, `vmlx-primary` resolve to `vmlx`;
`vmlx-aux`, `mlx-janitor` resolve to `vmlx-janitor`.

## Hardware requirements

| Model class | Quantization | Recommended unified memory |
|-------------|--------------|----------------------------|
| 3–8 B       | 4-bit        | 16 GB                      |
| 13–14 B     | 4-bit        | 24 GB                      |
| 30–34 B     | 4-bit        | 36 GB                      |
| 70 B        | 4-bit        | 64 GB                      |

Apple Silicon (M1 / M2 / M3 / M4 / M5 family) only. Intel Macs are not
supported by MLX itself.

## Installation

```bash
pip install vmlx
```

Discovery is automatic — `providers/__init__.py._discover_providers()` picks
up the directory the next time `get_provider_profile()` is called.

## Model acquisition

Suggestions; any MLX-quantized chat model in the same size class works.

```bash
mkdir -p ~/models
huggingface-cli download mlx-community/Qwen2.5-32B-Instruct-4bit \
    --local-dir ~/models/primary
huggingface-cli download mlx-community/gemma-3-4b-it-4bit \
    --local-dir ~/models/janitor
```

## Serving both models

```bash
vmlx serve --model ~/models/primary --port 8000 --ctx-size 65536
vmlx serve --model ~/models/janitor --port 8001 --ctx-size 16384
```

For permanent setup with `launchd`, see the [macOS airgap guide](../../../website/docs/guides/macos-airgap.md).

## Hermes config

Minimal — picks up the primary profile's defaults:

```yaml
model:
  provider: vmlx
  name: primary
  context_length: 65536
  temperature: 0.2

fallback_providers: []
```

`fallback_providers: []` is the explicit airgap declaration: Hermes will
never reach for a cloud provider on inference failure.

For the primary/janitor split, point auxiliary tasks at the `vmlx-janitor`
profile via your Hermes version's auxiliary-routing config (see the
provider-runtime developer guide for resolution precedence). A common
pattern:

```yaml
auxiliary_routes:
  compression:    { provider: vmlx-janitor, name: janitor }
  memory_write:   { provider: vmlx-janitor, name: janitor }
  skill_curation: { provider: vmlx-janitor, name: janitor }
  context_summary:{ provider: vmlx-janitor, name: janitor }
```

If your Hermes release does not support per-task `provider` overrides yet,
serve both models on `:8000` and use `default_aux_model` instead.

## Verification

```bash
hermes doctor
```

Expected (relevant lines — the `/models` probe is automatic for any
`api_key`/empty-`env_vars` profile):

```
[ok]   provider 'vmlx' registered
[ok]   provider 'vmlx-janitor' registered
[ok]   http://localhost:8000/v1/models reachable
[ok]   http://localhost:8001/v1/models reachable
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `vmlx` not in `hermes doctor` output | Plugin import failed (likely non-Darwin) | This plugin is Apple Silicon only; check `python -c "import platform; print(platform.system())"` |
| `connection refused` on /models probe | `vmlx serve` not listening on the expected port | `lsof -nP -iTCP:8000 -sTCP:LISTEN`; restart the server |
| `context length exceeded` mid-loop | `--ctx-size` lower than `model.context_length` | Restart server with matching `--ctx-size` |
| Janitor crashes on first request | OOM (model + primary both loaded) | Smaller janitor model or 4-bit quantization |
| Override the bundled defaults | Plugin lives at `<repo>/plugins/model-providers/vmlx/` (bundled) | Drop a same-named directory under `$HERMES_HOME/plugins/model-providers/vmlx/` — user plugins win because `register_provider()` is last-writer-wins |

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](../../../CONTRIBUTING.md). The
plugin's macOS gate lives in `__init__.py` (`ImportError` on non-Darwin), so
non-Mac contributors can hack on Hermes without it loading.
