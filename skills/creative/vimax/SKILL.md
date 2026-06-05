---
version: 0.1.0
name: vimax
description: |
  Generate full videos from an idea or a script using the ViMax
  pipeline (HKUDS/ViMax) — multi-agent screenwriting + image +
  video generation. Use when the user asks to "make a video from
  an idea", "turn this script into a video", "ViMax", "idea to
  video", "script to video", or wants an automated end-to-end
  short-film/clip pipeline (not single image/clip — for those use
  higgsfield-generate).
---

# ViMax — idea/script → video

ViMax lives at `~/tools/ViMax`. API keys are NOT in
its config files — they are injected from Infisical at runtime by the
`vimax-run` launcher. Never hardcode keys into `configs/*.yaml`.

## Run it

Always go through Infisical so the keys are present:

```bash
vimax-run idea2video     # idea  -> video
vimax-run script2video   # script -> video
```

`vimax-run` renders `configs/<profile>.yaml` from the `*_minimax.yaml`
base, overriding only the `api_key` fields with the Infisical-injected
env vars, then runs `main_<profile>.py`.

## Required secrets (in Infisical, project=prod)

| Env var | Used for |
|---|---|
| `MINIMAX_API_KEY` | chat model (MiniMax M2.7, OpenAI-compatible) |
| `VIMAX_IMAGE_API_KEY` | image generator (Nanobanana / Google) |
| `VIMAX_VIDEO_API_KEY` | video generator (Veo / Google) |

If `vimax-run` exits with `missing secrets`, tell the user exactly which
of the above to add in Infisical (http://100.68.195.19:8080) — do not try
to invent or hardcode keys.

## Customizing the idea/script

`main_idea2video.py` and `main_script2video.py` hold the `idea`,
`style`, and `user_requirement` near the top. To run a custom request,
edit those values (or copy the script) before invoking `vimax-run`.
Output lands under `ViMax/.working_dir/<profile>/`.

## Notes

- ViMax is a heavy multi-step pipeline (image+video gen, rate-limited
  per the config's `max_requests_per_*`). Expect minutes per run.
- If it fails on import/deps (not keys), report the traceback — ViMax
  has its own Python deps separate from hermes-agent.
