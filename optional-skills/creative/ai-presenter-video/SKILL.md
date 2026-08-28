---
name: ai-presenter-video
description: "Make a verified AI presenter video from script + image."
version: 1.0.0
author: cclank (https://github.com/cclank/lanshu-create-ai-presenter-video), ported by Hermes Agent
license: MIT
platforms: [linux, macos]
required_commands: [ffmpeg, ffprobe, python3]
metadata:
  hermes:
    tags: [video, presenter, avatar, lipsync, tts, captions, creative]
    category: creative
    homepage: https://github.com/cclank/lanshu-create-ai-presenter-video
    related_skills: [hyperframes, kanban-video-orchestrator, comfyui]
---

# AI Presenter Video

Turn a topic (or finished script) plus ONE authorized adult presenter image
into a complete, publish-ready presenter-led video: locked narration, avatar
generation with lip-sync QA, captions, deterministic editing, loudness-normalized
master/share encodes, and machine + visual acceptance reports.

Use this skill for new presenter videos AND for continuing, revising,
captioning, lip-sync-repairing, or re-exporting an existing presenter-video
job. The workflow is provider-neutral: pick generation capabilities from what
is actually available in the session (FAL video/image models via
`image_generate` and the video-gen plugin, TTS via `text_to_speech`, ASR via
the whisper/STT tooling, ffmpeg for everything deterministic).

> Ported from cclank/lanshu-create-ai-presenter-video (MIT). Upstream body
> kept substantively verbatim in `references/`; Hermes adaptations live in
> this hub file. Scripts are deterministic (no network, no credentials).

## Hermes adaptations (read first)

- **Skill dir resolution** — upstream hardcoded `~/.codex/skills/...`. In
  Hermes resolve it once per session:

  ```bash
  SKILL_DIR="$(dirname "$(find ~/.hermes/skills ~/.hermes/hermes-agent/optional-skills -path '*/ai-presenter-video/SKILL.md' 2>/dev/null | head -1)")"
  [ -f "$SKILL_DIR/SKILL.md" ] || echo "skill dir not found — locate ai-presenter-video/SKILL.md manually and set SKILL_DIR to its directory"
  ```

  Shell variables do not persist between tool calls — re-paste the resolution
  line (or the expanded path) in each terminal call that uses it.
- **Capability mapping** — where the references say "a voice generation
  capability", use `text_to_speech` (OpenAI/Edge/ElevenLabs per user config);
  "presenter/avatar generation" → FAL image-to-video families (Kling, Wan,
  MiniMax H3 etc.) through the configured video tooling, or an avatar/lipsync
  endpoint the user has access to; "word-timestamp ASR" → whisper via the STT
  tooling or `faster-whisper` in a venv; "deterministic compositor" → ffmpeg
  filtergraphs, or the `hyperframes` skill when installed (the editing
  reference has a HyperFrames section that maps directly onto it).
- **Visual QA** — do the "normal-speed visual review" steps with
  `vision_analyze` on the generated contact sheet plus sampled frames
  (identity, mouth timing, hands, blinking, continuity). Numeric checks come
  from the scripts' ffprobe output.
- **Paid-generation consent** — remote avatar/TTS generation is billable.
  Follow the upstream operating rules: before the first paid call state the
  uploaded assets, requested seconds, known cost, pilot size, and retry
  ceiling, and get the user's explicit go-ahead. Never upload the presenter
  image to a remote provider before `remote_upload_approved` is true in
  `job.json`.
- **Consent flags live under `input`** — `rights_confirmed`,
  `adult_presenter_confirmed`, `remote_upload_approved`, and
  `voice_clone_approved` sit inside the `input` object of `job.json` (init
  flags set them; hand-editing must target `input.*`, not the job root).
  `manual_input_review.*` sits at the root. `preflight.py` distinguishes
  `errors` (block everything) from `remote_blockers` (block only remote
  generation) — local script/audio work may proceed while remote is blocked.

## Workflow

1. **Start or resume a job.** New job:

   ```bash
   python3 "$SKILL_DIR/scripts/init_job.py" \
     --job-dir ~/Videos/my-presenter-video \
     --presenter-image /path/to/presenter.png \
     --topic "explain context engineering in one minute" \
     --duration 60 --aspect 9:16 \
     --rights-confirmed --adult-presenter-confirmed
   ```

   Use `--script` for an existing script file; other flags: `--voice-sample`,
   `--supporting-media`, `--width`, `--height`, `--fps`, `--watermark`,
   `--cta`. For an existing job, read `job.json` + QA reports and resume from
   the earliest unfinished state — never regenerate accepted work.

2. **Manual input review.** Actually look at the presenter image
   (`vision_analyze`) and listen to any voice sample; record findings by
   setting the `manual_input_review` booleans in `job.json`, e.g.:

   ```bash
   python3 - <<'PY'
   import json
   p = "~/Videos/my-presenter-video/job.json"  # expand ~ or use an absolute path
   import os; p = os.path.expanduser(p)
   j = json.load(open(p))
   j["manual_input_review"].update(image_viewed=True, single_clear_face=True,
                                   image_has_no_unwanted_text=True)
   json.dump(j, open(p, "w"), indent=2)
   PY
   ```

   Then gate:

   ```bash
   python3 "$SKILL_DIR/scripts/preflight.py" ~/Videos/my-presenter-video/job.json
   ```

   Proceed only when `ok: true`; do remote generation only when
   `remote_ready: true`. Note: preflight also updates `job.json` in place
   (records the report path) — re-read it after running rather than editing
   a stale copy.

3. **Lock content and audio** — read `references/generation.md`. Script →
   full narration via `text_to_speech` → ASR-verify the narration against the
   script → record real durations. The locked audio is the master clock for
   everything downstream.

4. **Plan and generate the presenter** — read `references/generation.md`.
   Short low-cost pilot first; full run only after the pilot passes identity
   and mouth-timing review.

5. **Edit** — read `references/editing.md`. Deterministic timeline driven by
   the locked audio; captions and keyword callouts only after audio and media
   are final.

6. **Verify and deliver** — read `references/qa-recovery.md`, render, then:

   ```bash
   bash "$SKILL_DIR/scripts/finalize_delivery.sh" \
     ~/Videos/my-presenter-video/renders/rendered.mp4 \
     ~/Videos/my-presenter-video/outputs my-video
   ```

   The finalizer preserves aspect ratio, runs two-pass loudness normalization
   (program ≈ −16 LUFS), produces master + share encodes, decode-verifies
   both, writes a delivery report JSON, and emits a nine-frame contact sheet.
   Inspect the contact sheet with `vision_analyze` before claiming completion.

## Operating rules (non-negotiable)

- Confirm image rights, adult status, remote-upload approval, and
  voice-cloning authorization before the relevant remote action.
- Never infer or clone a real person's voice from an image; use an authorized
  sample or a stock TTS voice.
- Lock the complete narration before presenter generation, caption timing, or
  final scene boundaries.
- Mute video sources in the final composition; only the approved narration
  and intentional mix tracks carry audio.
- Preserve provider request bodies and task IDs (minus credentials/expiring
  URLs). Poll interrupted work before resubmitting — avoid double billing.
- Stop after three rejected paid candidates and summarize the failure mode.
- Do not claim completion until the final files fully decode and the contact
  sheet or full playback has been reviewed.

## Defaults for minimal input

9:16, 1080×1920, 30fps; topic-derived videos target 45–75s; stock voice when
no authorized sample; presenter-led layout with hook → 2–4 beats → close;
no music/CTA unless requested; language inferred from the request.

## Reference routing

- `references/generation.md` — intake, content, voice, capability selection,
  presenter prompts, paid generation, provider changes.
- `references/editing.md` — timeline contract, openings/closes, captions,
  keyword-callout presets, HyperFrames composition, exports.
- `references/qa-recovery.md` — technical acceptance, visual acceptance, and
  recovery for lip-sync/identity/hands/exposure/freeze/caption/audio faults.

## Pitfalls

- `preflight.py` requires ffprobe; on a bare box install ffmpeg first.
- The consent booleans set by init flags land under `input.*`; editing them
  at the job-json root silently does nothing (preflight keeps blocking).
- `finalize_delivery.sh` needs bash + jq + awk and a fully decodable input —
  a truncated render fails the decode check by design, not by accident.
- Long avatar clips drift: prefer one continuous presenter source sliced on
  the audio timeline over many regenerated chapter clips (identity drift
  across regenerations is the #1 visual-QA failure).
- FAL i2v endpoints cap duration (typically 5–15s); plan chapter-level
  presenter segments accordingly and reuse the pilot's seed/params for
  consistency where the endpoint supports it.

## Verification

Validated hands-on (Aug 2026): `init_job.py` → `job.json` with correct state
machine; `preflight.py` correctly blocked on unreviewed inputs, flipped to
`ok: true` after review booleans, and kept `remote_ready: false` until
`input.remote_upload_approved`; `finalize_delivery.sh` on a synthetic 5s
1080×1920 render produced decode-verified master (631kbit/s) + share encodes,
delivery-report JSON, and a 9-frame contact sheet, exit 0.
