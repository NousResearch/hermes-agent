# Paimon Creative Stack Audit

Audit source: Paimon on `openclaw-miami` (`100.122.118.25`), physical design/media Hermes agent managed by Zeus.

## Why Paimon performs better for video/image work

Paimon has a specialized creative stack beyond generic image/video generation:

1. **Production-oriented skills** for reels, UGC, deterministic text overlays, FFmpeg QA, HyperFrames, ViMax, and provider validation.
2. **Local media toolchain** with FFmpeg, ImageMagick, HyperFrames, Higgsfield CLI, and ViMax.
3. **Python media libraries** installed in the Hermes venv: MoviePy, OpenCV, Pillow, imageio, torch, faster-whisper, onnxruntime, soundfile, Whisper, ElevenLabs.
4. **Hardware profile** suitable for creative/video experiments: physical workstation class machine with GPU-oriented workflows.
5. **Workflow discipline** encoded in skills: storyboard before execution, deterministic overlays instead of model-rendered text, frame/audio QA, and provider/API validation before assuming capability.

## Imported capability map

| Capability | Skill(s) | Main use |
|---|---|---|
| UGC reel planning/execution | `app-feature-ugc-reel` | Plan shots, copy, voice, app/agent inserts, final vertical reels. |
| AI video pipeline | `vimax-video`, `vimax` | Story/script to generated video with ViMax. |
| HTML motion graphics | `hyperframes-render` | Render animated HTML/CSS/JS scenes to MP4/WebM. |
| Exact text overlays | `reel-text-overlay`, `ffmpeg-media` | Caption systems, kinetic text, safe margins, final compositing. |
| Media editing/QA | `ffmpeg-media` | Trim, concat, mux, audio tails, probes, overlays, exports. |
| Background/image cleanup | `image-background-cleanup` | Remove text/logos/backgrounds or recreate clean imagery. |
| Video intelligence | `video-intel-pipeline` | Download/transcribe/analyze video reliably. |
| API capability validation | `creative-media-api-validation` | Confirm image/video/avatar/lipsync APIs are executable, not only documented. |
| Higgsfield generation | `higgsfield-cli`, `higgsfield-*` | Image/video/Soul/product/marketplace generation. |
| MiniMax generation | `mmx-cli` | MiniMax Token Plan CLI workflows. |

## Operational notes

- Exact text should be rendered locally with Pillow/FFmpeg/HyperFrames, not delegated to generative video models.
- Lipsync/avatar workflows should be validated via actual provider API or CLI call before promising availability.
- Heavy third-party repositories and local binaries remain machine-local; the repo carries skills, manifests, docs, and bootstrap instructions.
- Commercial inherited agents should receive the reusable creative stack, but GPU-heavy workflows may be enabled only on hosts that can support them.

## Source-control decision

Committed to repo:

- Skills under `skills/media`, `skills/creative`, and `skills/higgsfield`.
- `docs/creative-media-stack/*` audit and manifest.
- `scripts/bootstrap_creative_media_stack.sh` reproducible bootstrap.

Not committed:

- `/tools/ViMax` checkout.
- `/tools/hyperframes` checkout.
- `/tools/ffmpeg` static binary directory.
- Node `node_modules`, Python venvs, OAuth caches, generated media outputs, and provider credentials.
