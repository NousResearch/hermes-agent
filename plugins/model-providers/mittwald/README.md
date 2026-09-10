# mittwald AI Hosting

Open-weight models served from German data centres over an OpenAI-compatible API (vLLM).
Base URL `https://llm.aihosting.mittwald.de/v1`.

```bash
hermes chat --provider mittwald --model Qwen3.6-35B-A3B-FP8
```

## API key

The key is created per project in mStudio under **AI-Hosting** — it is *not* one of the
profile API tokens under user settings, and those tokens do not work here. Put it in
`~/.hermes/.env` as `MITTWALD_LLM_API_KEY` (`MITTWALD_AI_API_KEY` is also accepted).
`MITTWALD_BASE_URL` overrides the endpoint.

See [Access and usage](https://developer.mittwald.de/docs/v2/platform/aihosting/access-and-usage/access/).

## Models

`/v1/models` lists every hosted model — chat, speech, embedding, rerank and OCR — in one
array. The profile filters that down to the chat models for `hermes model` and `/model`:

| Model | Context | Notes |
|-------|---------|-------|
| `Qwen3.6-35B-A3B-FP8` | 256,000 | Default pick — tools + vision |
| `Qwen3.8-27B-NVFP4` | 256,000 | Tools + vision |
| `Qwen3.5-122B-A10B-FP8` | 245,760 | Largest, tools + vision |
| `Ministral-3-14B-Instruct-2512` | 262,144 | Tools + vision |
| `gpt-oss-120b` | 131,072 | Tools + reasoning |
| `Qwen3.5-0.8B` | 262,144 | Small/cheap; the auxiliary-task default |

Context windows come from the catalog's `max_input_tokens`, so each model gets its own —
`DEFAULT_CONTEXT_LENGTHS` in `agent/model_metadata.py` only covers the offline case.
Model IDs are case-sensitive and go out verbatim.

`reasoning_effort` is passed at the top level. `none` genuinely disables Qwen's thinking
(the reply carries no `reasoning_content`), so the vLLM-specific
`chat_template_kwargs.enable_thinking` escape hatch is not used.

## Other endpoints on the same key

| Endpoint | Model | Wired into Hermes as |
|----------|-------|----------------------|
| `/v1/audio/transcriptions` | `whisper-large-v3-turbo` | built-in STT provider `mittwald` |
| `/v1/audio/speech` | `Qwen3-TTS-12Hz-1.7B-CustomVoice` | built-in TTS provider `mittwald` |
| `/v1/embeddings` | `Qwen3-Embedding-8B` (4096 dims) | mem0 OSS embedder `mittwald` |
| `/v1/rerank` | `Qwen3-VL-Reranker-2B` | not wired — Hermes has no rerank consumer |
| `/v1/chat/completions` | `GLM-OCR` | not wired — documents go through the vision models |

The rerank endpoint works (vLLM's score API) but is undocumented on mittwald's endpoint
list. Both unwired models are reachable with the same key if you call them directly.

## Limits worth knowing

- STT uploads: 25 MB (Hermes' own cap) and 10 minutes of audio (mittwald's).
- TTS `voice` is mandatory server-side; `language` takes a word form (`German`), not an
  ISO code — Hermes maps the common codes for you.
