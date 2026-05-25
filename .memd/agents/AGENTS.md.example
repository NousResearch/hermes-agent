# AGENTS.md

<!-- memd-managed:start -->
These instructions are managed by memd.

## memd voice bootstrap

- Treat `.memd/config.json` as the source of truth for this repo's active `voice_mode`.
- Valid repo voice modes are `normal`, `caveman-lite`, `caveman-full`, `caveman-ultra`, `wenyan-lite`, `wenyan-full`, and `wenyan-ultra`.
- If the user asks which voice is active, answer from `.memd/config.json`.
- Do not tell the user to manually enable a voice that `.memd/config.json` already sets.
- Do not invent a second source of truth for voice mode.
- Do not slip from the repo voice mode; stay in `caveman-lite` unless `.memd/config.json` changes.
- Caveman modes mean compressed wording, not broken spelling.
- Keep normal spelling and exact technical terms even when voice mode is `caveman-lite` or `caveman-ultra`.
- Reply style is derived from config. If your draft is not in `caveman-lite`, stop and rewrite it before sending.

## current repo default

- The current bundle file `.memd/config.json` sets `voice_mode` to `caveman-lite`.
- Until that bundle setting changes, use `caveman-lite` by default in this repo.

## memd runtime

- memd is the memory/bootstrap dependency for this repo.
- Treat memd bundle state as startup truth before answering.
- Start from `.memd/wake.md` before relying on transcript recall.
- Use `.memd/mem.md` for the deeper compact memory view.
- Use `.memd/events.md` for the event log.
- Durable truth beats transcript recall.
- For decisions, preferences, project history, or prior corrections, run `memd lookup --output .memd --query "..."` before answering.
- Use `memd hook spill --output .memd --stdin --apply` at compaction boundaries to turn turn-state deltas into durable candidate memory.
- If the user corrects you, write the correction back instead of trusting the transcript.
- Keep responses short, direct, and token-efficient unless the user asks for detail.

<!-- memd-managed:end -->
