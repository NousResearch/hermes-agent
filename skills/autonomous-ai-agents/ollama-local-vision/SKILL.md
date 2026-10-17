---
name: ollama-local-vision
description: Use when wiring Hermes vision to a local model via Ollama instead of the managed llama.cpp runtime.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [ollama, vision, local-models, vision-projector, config]
    related_skills: [hermes-agent]
---

# Local vision via Ollama

## Overview

Ollama's OpenAI-compatible endpoint can serve a vision model that Hermes reaches as an auxiliary
provider, with no patched local runtime involved. Ollama manages the vision projector itself, so a
GGUF already on disk can be imported together with its `mmproj` and served directly.

That matters because the managed llama.cpp path wires the projector through generated
`presets.ini` — built by `presets.py` from `catalog.json` plus `sideload_assets.json`. Those are
local edits, and `hermes update` stashes local edits without restoring them, so vision breaks
again after every update. Pointing `auxiliary.vision` at Ollama takes the projector plumbing out
of the vision path entirely.

## When to Use

- Hermes vision must run on a local or offline model through Ollama's OpenAI-compatible endpoint.
- A GGUF plus its `mmproj` projector are already on disk and should be served as-is, without
  re-downloading a registry model.
- Vision must stop depending on patched local-runtime files that `hermes update` reverts.
- Debugging a local vision backend that rejects images, or returns empty answers.

Don't use for: text-only local models (there is no projector to wire), or cloud vision providers
(they need no local endpoint and no projector).

## 1. Install without admin, on a chosen drive

Download only the official installer: `https://ollama.com/download/OllamaSetup.exe` (it redirects
to the official GitHub release). Confirm the byte count matches the advertised `Content-Length`
before running anything.

```bash
curl -L -m 900 --retry 2 -o OllamaSetup.exe https://ollama.com/download/OllamaSetup.exe
# long downloads DO hit curl's -m ceiling mid-file — resume instead of restarting:
curl -L -C - -m 580 -o OllamaSetup.exe https://ollama.com/download/OllamaSetup.exe
```

On Windows the installer is Inno Setup, so silent flags work and `/DIR` picks the location:

```powershell
$env:OLLAMA_MODELS = 'D:\Ollama\models'      # the model store, not the binary, is the big one
$env:OLLAMA_NUM_PARALLEL = 1
$env:OLLAMA_MAX_LOADED_MODELS = 1            # one resident model on a small card
.\OllamaSetup.exe /DIR='D:\Ollama' /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG=install.log
setx OLLAMA_MODELS 'D:\Ollama\models'        # persist for later launches
setx OLLAMA_NUM_PARALLEL 1
setx OLLAMA_MAX_LOADED_MODELS 1
```

Set the environment variables **before** launching the installer: the tray app inherits its
environment from the process that spawned it, and `setx` alone never reaches an already-running
child.

On macOS/Linux install through the package manager or `curl -fsSL https://ollama.com/install.sh | sh`
and export the same three variables from the shell profile.

## 2. Import a sideloaded GGUF + its projector

Two `FROM` lines; the second is the projector.

```
FROM /path/to/model.gguf
FROM /path/to/assets/model.mmproj-Q8_0.gguf
PARAMETER num_ctx 4096
```

```bash
ollama create <name>:<tag> -f Modelfile
```

`ollama create` may print a bare spinner while hashing gigabytes — do not kill it.

**Pick the right projector file first.** Read the GGUF metadata: the model's
`general.architecture` is the architecture Ollama must know, and a valid projector carries
`general.type=mmproj`, `general.architecture=clip`, `clip.has_vision_encoder=1` — exactly what
Ollama's projector detector keys on. A projector without a vision encoder is silently treated as
model weights, and image input fails much later.

**Verify the import, not just the exit code:** `ollama show <name>` must list `vision` under
Capabilities and print a Projector block, and the store manifest must carry a layer of type
`application/vnd.ollama.image.projector`.

## 3. Point Hermes auxiliary vision at it

`base_url` bypasses provider resolution for that task (a per-task direct endpoint override), so the
endpoint is all that matters:

```bash
hermes config set auxiliary.vision.base_url http://127.0.0.1:11434/v1
hermes config set auxiliary.vision.api_key ollama
hermes config set auxiliary.vision.model <name>:<tag>
hermes config set auxiliary.vision.timeout 300
hermes config set auxiliary.vision.provider custom
```

Use the actual `hermes` binary — a stale copy earlier in `PATH` can win and silently edit a
different install. Raise the timeout: the auxiliary vision default (30–120 s) is shorter than a
cold local model load on a laptop GPU, and the failure then reads as a provider outage rather than
a timeout.

A running session picked the change up with no gateway restart; if a call still lands on the old
backend, restart the gateway before believing anything else.

## 3b. Kill the thinking budget (do this before debugging anything else)

Reasoning-capable local models (Qwen3.x and friends) spend 100+ tokens of *hidden* thinking on
every call. With a modest `max_tokens` the visible answer never starts and the reply is
`content: ""` with `finish_reason: length` and `completion_tokens == max_tokens`. That looks like
a broken model or a broken pipeline. It is neither — it is the token budget.

On Ollama's OpenAI-compatible endpoint exactly one lever works:

```bash
hermes config set auxiliary.vision.reasoning_effort none
```

Measured on a 4B Q6_K VL model on an 8 GB laptop GPU, through Hermes's own vision path, same image:

| setting | latency | tokens |
|---|---|---|
| `reasoning_effort: none` | **1.8 s** | 1463 (image ≈1400) |
| default (thinking on) | **29.5 s** | 2834 |

Two decoys, both verified inert here: `chat_template_kwargs: {enable_thinking: false}` (the vLLM
recipe — 113 → 112 tokens, i.e. ignored) and `"think": false` (works only on the native
`/api/chat`, ignored by `/v1/chat/completions`).

Verify the lever took effect from Ollama's own log: every request logs
`stop processing: n_tokens = N` with its duration, so a short question on a fixed image should land
just above the image's own token count. Never assume a config field is forwarded — measure it.

## 3c. The Settings UI can silently erase a direct endpoint

Picking `auto` (or any other provider) for an auxiliary task in **Settings → Models** rewrites that
block and clears `base_url` / `model` with no warning, and the local endpoint wiring just
disappears. Symptom: vision suddenly answers from a cloud model, or fails. If the user has been in
that UI, re-read the block (`hermes config get auxiliary.vision`) before debugging the model, and
note the config file's mtime to date the change.

## 4. Prove it, and prove WHO served it

Never claim vision works because a text reply came back. Send a real image, ask for known ground
truth, and use a DIFFERENT image each time.

Attribute the backend the only way that is not guesswork:

```bash
ollama stop <name>                # unload the model, then record nvidia-smi
# ... now run the Hermes vision call, e.g. the vision_analyze tool ...
ollama ps                         # must show the model + a FRESH 'UNTIL'
```

Also confirm the managed runtime did NOT serve it (no `llama-server` process), and corroborate
from Ollama's log — a `POST "/v1/chat/completions"` from `127.0.0.1`.

## Common Pitfalls

- **A proxy that answers DNS with fake IPs breaks `ollama pull`.** Clash-style fake-IP mode returns
  reserved addresses (198.18.x.x) for every proxied domain, and Ollama refuses to follow a blob
  redirect to an address it judges non-public:
  `Error: redirect target not allowed: <host> resolves to non-public 198.18.0.105`.
  The manifest fetch succeeds (it is a plain API call) while every blob download fails, and the
  half-written file sits at full preallocated size — so *always hash it* before believing it is
  complete. Workaround without touching the user's proxy: curl the registry directly, since curl
  delegates DNS to the proxy, then assemble the store by hand:

  ```bash
  curl -sL https://registry.ollama.ai/v2/library/<model>/manifests/<tag> -o manifest.json
  # per layer: curl -sL -o "blobs/sha256-<hex>" "https://registry.ollama.ai/v2/library/<model>/blobs/sha256:<hex>"
  # also fetch manifest['config']['digest']; then save manifest.json as
  #   $OLLAMA_MODELS/manifests/registry.ollama.ai/library/<model>/<tag>
  sha256sum blobs/sha256-<hex>   # must equal the digest, or the pull is not done
  ```

  Do not silently skip the proxy config change — offer it (whitelist the storage host in
  `dns.fake-ip-filter`) versus the curl assembly, and say plainly which one you used.
- **The same image is cached.** llama.cpp caches image/prompt work, so re-sending one image reads
  ~5x faster and will convince you a slow path is fast. Benchmark with content-different images only.
- A model can be imported, report `vision`, and still answer *empty* on an ambiguous prompt — that
  is the prompt, not the pipeline. Re-ask with a concrete question before debugging config.
- Ollama ignores unknown request fields (`chat_template_kwargs` is harmless), but keep bodies minimal.
- `ollama --version` can print "could not connect to a running Ollama instance" while the tray app
  is still coming up; it is meaningless if `/api/version` answers.
- Importing is not "downloading": the projector comes from disk, so an offline local model needs no
  registry tag. Registry pulls crawl through a slow proxy — budget tens of minutes and run them in
  the background.
- Reference numbers (8 GB laptop GPU, 4B VL Q6_K at `num_ctx 4096`): 3.9 GB resident, 100% GPU,
  cold 12.8 s (load 3.5 + image 5.5 + generate 3.8), warm ~11 s on a fresh 1600 px screenshot. A
  1600 px long side works; keep to one image per request.

## Verification Checklist

- [ ] `ollama show <name>` lists `vision` under Capabilities and prints a Projector block
- [ ] The store manifest carries an `application/vnd.ollama.image.projector` layer
- [ ] `hermes config get auxiliary.vision` shows the local `base_url`, the model, and a raised `timeout`
- [ ] `reasoning_effort: none` is set, and Ollama's log shows the call is small (`stop processing: n_tokens` just above the image's own count)
- [ ] A real image with known ground truth is answered correctly, on an image never sent before
- [ ] `ollama ps` shows the model with a fresh `UNTIL` after the call, and no `llama-server` served it
