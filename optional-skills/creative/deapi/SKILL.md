---
name: deapi
description: Generate and process media through the deAPI cloud API — text-to-image (FLUX.2 Klein and other open-source models), image editing, text-to-speech with voice cloning and voice design, audio/video transcription (YouTube, X, Twitch links or local files), OCR, background removal, image upscaling, music generation, text-to-video, image animation and text embeddings, plus a prompt booster. Use when the user asks to generate an image, speech, music or video, clone or design a voice, transcribe audio or a video link, extract text from an image, remove a background, upscale an image, animate a photo, enhance a generation prompt, create embeddings, or check their deAPI balance.
version: 1.1.0
author: deapi-ai
license: MIT
platforms: [linux, macos, windows]
compatibility: Requires Python 3.9+ and internet access to api.deapi.ai
metadata:
  author: deapi-ai
  version: "1.1"
  hermes:
    tags: [media, image-generation, tts, voice-clone, transcription, ocr, video, music, embeddings]
    category: creative
    requires_toolsets: [terminal]
required_environment_variables:
  - name: DEAPI_API_KEY
    prompt: "Enter your deAPI API key (dpn-sk-...)"
    help: "Create one at https://app.deapi.ai — free $5 credit, no card required"
    required_for: "All deAPI requests"
---

# deAPI — media generation and processing

deAPI (https://deapi.ai) serves open-source AI models over a simple REST API:
image generation and editing, speech synthesis, transcription, OCR, background
removal, upscaling, music, video and embeddings. It hosts **no LLMs** — never
route chat/completion requests here.

## When to Use

Any request to *produce or process media* cheaply via an API: "generate an
image of…", "transcribe this YouTube video", "read the text in this image",
"turn this text into speech", "make a short music track", "remove the
background", "upscale this photo", "embed these sentences".

## Setup

One environment variable is required: `DEAPI_API_KEY` (format `dpn-sk-...`,
created at https://app.deapi.ai). In Hermes it is collected on first use and
injected into the terminal sandbox automatically. Elsewhere, export it before
running the helper.

## Quick Reference

All tasks go through one helper: `scripts/deapi.py` (Python 3.9+, standard
library only — no pip installs). Run it from this skill's directory, or call
it by absolute path. It submits the job, polls until done, prints text results
to stdout and downloads file results (prints the saved path).

| Task | Command |
|---|---|
| List live models | `python3 scripts/deapi.py models --type image` |
| Generate image | `python3 scripts/deapi.py image --prompt "a red fox, studio light" --output fox.png` |
| Edit image | `python3 scripts/deapi.py edit --prompt "make it snowy" --image photo.jpg` |
| Text-to-speech | `python3 scripts/deapi.py tts --text "Hello world" --output hello.mp3` |
| Voice clone | `python3 scripts/deapi.py tts --text "Hi" --clone-audio sample.mp3 --model <voice-clone-capable slug>` |
| Voice design | `python3 scripts/deapi.py tts --text "Hi" --instruct "warm British male narrator" --model <voice-design slug>` |
| Transcribe URL | `python3 scripts/deapi.py stt --url "https://youtube.com/watch?v=..."` |
| Transcribe file | `python3 scripts/deapi.py stt --file meeting.mp3 --timestamps` |
| OCR | `python3 scripts/deapi.py ocr --image scan.png` |
| Remove background | `python3 scripts/deapi.py rmbg --image product.jpg` |
| Upscale | `python3 scripts/deapi.py upscale --image small.png` |
| Music | `python3 scripts/deapi.py music --caption "lofi hip hop, calm" --duration 30` |
| Text-to-video | `python3 scripts/deapi.py video --prompt "waves at sunset"` |
| Animate image | `python3 scripts/deapi.py animate --prompt "gentle camera pan" --image photo.png` |
| Boost a prompt | `python3 scripts/deapi.py boost --prompt "a red fox" --type image` |
| Embeddings | `python3 scripts/deapi.py embed --input "some text"` |
| Balance | `python3 scripts/deapi.py balance` |

## Procedure

1. **Models are discovered live, never hardcoded.** When `--model` is omitted
   the helper fetches `GET /api/v2/models` and picks a current model for the
   task (for images it prefers FLUX.2 Klein when available). To let the user
   choose, run `models --type <task>` first and show them the list.
2. Run the task command. Generation is asynchronous — the helper polls
   `GET /api/v2/jobs/{request_id}` (default every 5 s, up to 15 min; tune with
   `DEAPI_POLL_INTERVAL` / `DEAPI_POLL_TIMEOUT` env vars).
3. Text results (transcripts, OCR) print to stdout; binary results download to
   `--output` or a name derived from the result URL. Download happens
   immediately because result URLs expire after ~24 h.
4. For parameter details and voice lists see `references/api-reference.md` and
   `references/models.md`.

## Pitfalls

- deAPI has **no chat/LLM models**; `oai.deapi.ai/v1/chat/completions` is
  intentionally unsupported.
- Model slugs change as models are added/retired — trust `models` output, not
  memory or docs examples.
- Upload limits: audio ≤ 20 MB, video ≤ 50 MB (transcription), images ≤ 10 MB
  (OCR). For bigger media, prefer `--url` sources.
- Some image models fix their parameters (e.g. FLUX.2 Klein: 4 steps, no
  guidance/negative prompt). The helper applies the model's own defaults; a
  422 response explains any remaining conflict — surface it to the user.
- Every generation costs credits. If a job fails with 401/403, check the key
  and balance (`balance` command) before retrying.

## Verification

`python3 scripts/deapi.py balance` — confirms key + connectivity in one call.
`python3 scripts/deapi.py models` — confirms the live model list is reachable.
