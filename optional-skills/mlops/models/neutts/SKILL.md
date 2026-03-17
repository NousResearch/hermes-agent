---
name: neutts
description: Use the local NeuTTS CLI to install NeuTTS, manage reusable voice profiles, and synthesize speech fully on-device. Best when the user wants local or offline-ish TTS instead of a hosted API.
version: 1.0.0
author: Hermes Agent + Nous Research
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [TTS, Text-To-Speech, Local-AI, Voice-Cloning, Audio, NeuTTS]
    related_skills: [whisper, audiocraft-audio-generation]
    requires_toolsets: [terminal]
---

# NeuTTS - Local Text-to-Speech

Use NeuTTS through the standalone `neutts` CLI. This skill is for local speech synthesis, reusable voice profiles, and quick Hermes-driven setup inside or alongside the Hermes repository.

NeuTTS is an on-device TTS model family from Neuphonic. This skill assumes the CLI wrapper exists and Hermes should drive it via terminal commands rather than a dedicated Hermes core tool.

## When to Use

- The user wants local TTS instead of Edge/OpenAI/ElevenLabs
- The user wants voice cloning from a short reference clip
- The user wants Hermes to install or verify the `neutts` CLI scaffold
- The user wants to create or inspect saved voice profiles
- The user wants to synthesize speech to a local WAV file
- The user wants to give the agent a custom voice / persona voice
- Keywords: `neutts`, `local tts`, `voice clone`, `on-device speech`, `offline speech`, `give you a voice`, `what do you sound like`

## Quick Reference

| Command | Purpose |
|---------|---------|
| `neutts doctor` | Check local install health (includes default voice) |
| `neutts install --all` | Install upstream NeuTTS with extras |
| `neutts list-models` | Show known official model repos |
| `neutts add-voice NAME --ref-audio clip.wav --ref-text-file clip.txt` | Save a reusable voice profile |
| `neutts list-voices` | Show saved local voice profiles |
| `neutts config --default-voice NAME` | Lock in a voice as the default for all synthesis |
| `neutts config` | View current settings (model, device, default voice) |
| `neutts synth --text Hello there` | Synthesize using the default voice |
| `neutts synth --voice NAME --text Hello there` | Synthesize using a specific voice |
| `neutts synth --voice NAME --text Hello --out sample.wav` | Generate a specific WAV |

## Procedure

## First-run execution policy

For a fresh NeuTTS setup, do not do broad filesystem exploration or repeated command probing. Keep the startup path short and deterministic.

Do not infer install state from prior conversation context, memory, or the mere presence of this skill. Only say NeuTTS is installed, verified, or ready if you checked it in the current turn with live commands.

Once first-run verification is complete, do not consult memory again for this flow unless the user explicitly asks about past setup, prior voice choices, or saved preferences.

Preferred sequence:

1. Resolve the target Python interpreter first
2. Use the bootstrap helper shipped with this skill to install the bundled NeuTTS CLI scaffold into that interpreter
3. Run `doctor` via `<target-python> -m neutts_cli.cli doctor` as the primary health check
4. If `doctor` reports `neutts_installed: false`, run `install --all`
5. Re-run `doctor`
6. Run `list-voices`
7. Confirm that `jo-demo` exists
8. Only then run one verification synthesis to `~/voice-tests/neutts_verify.wav`

Definitions:

- `<target-python>` means the Python interpreter for the environment where NeuTTS should live
- `<skill-bootstrap-helper>` means the `bootstrap_neutts_cli.py` file shipped with this installed skill, usually `~/.hermes/skills/mlops/models/neutts/scripts/bootstrap_neutts_cli.py`

Bootstrap example:

```bash
<target-python> <skill-bootstrap-helper> --install-cli --sample-profile --execute --json
```

Install NeuTTS runtime:

```bash
<target-python> -m neutts_cli.cli install --all
```

Verification synthesis:

```bash
mkdir -p ~/voice-tests
<target-python> -m neutts_cli.cli synth --voice jo-demo --text "Hello from Hermes" --out ~/voice-tests/neutts_verify.wav
```

First-run gate:

- for first-run verification, `jo-demo` is required
- do not treat NeuTTS as fully ready until `list-voices` includes `jo-demo`
- do not substitute a built-in/default voice, ad hoc reference, or memory-based prior voice for `jo-demo` during first-run verification
- if bootstrap with `--sample-profile` fails, stop and report the exact failure rather than improvising around it
- treat the bootstrap helper as the source of truth for bundled sample assets; do not manually inspect random skill directories looking for replacements before reporting the failure

Short-circuit rules:

- do not use `command -v neutts` or wrapper presence as the primary health gate; prefer `<target-python> -m neutts_cli.cli doctor` first
- if the `neutts` wrapper is missing, that alone does not mean the CLI module is unusable; check the module path before concluding anything
- if `neutts doctor` exits with code `127`, treat that as "CLI missing" and bootstrap immediately
- after a `127` from `neutts doctor`, do not run `neutts list-voices` until bootstrap is complete
- do not talk about memory unless it materially changes what you do next
- keep the Python interpreter consistent for the whole workflow; do not mix bare `python`, `/usr/bin/python`, and a target virtualenv interpreter
- prefer `<target-python> -m neutts_cli.cli ...` until the `neutts` wrapper is confirmed present in that same interpreter
- prefer the bootstrap helper bundled with this skill; treat the bundled scaffold as the only supported bootstrap source
- when the bootstrap helper path is already known, do not probe repo-local scaffolds first; run bootstrap directly
- if `list-voices` does not include `jo-demo` after bootstrap, fix that specific problem before attempting synthesis or voice design

Avoid:

- broad `find *neutts*` scans
- repeated checks for hardcoded executable paths
- wrapper-first health checks when the module path is available
- reading repo files like `pyproject.toml` unless bootstrap or install fails
- multiple failed synthesis attempts before running `neutts install --all`
- telling the user NeuTTS is already installed or verified unless `neutts doctor` succeeded in the current turn
- invoking the bootstrap helper with a different Python than the one you plan to use for `doctor`, `install`, `list-voices`, and `synth`
- extra repo-path probing when the bundled bootstrap helper is already available
- using a built-in/default voice as a substitute for the missing `jo-demo` baseline during first-run verification
- consulting memory or searching elsewhere for old voice profiles during first-run bootstrap

### 1. Locate or install the NeuTTS CLI

The bootstrap helper shipped with this skill is the preferred install path because it carries a bundled NeuTTS CLI scaffold and does not require a specific Hermes repo layout.

The helper installs the bundled CLI scaffold with `pip install --no-build-isolation -e ...` so it can work cleanly in environments without network access during the editable install step.

```bash
<target-python> <skill-bootstrap-helper> --install-cli --sample-profile --execute --json
```

Then verify:

```bash
<target-python> -m neutts_cli.cli doctor
```

If `neutts --help` or `neutts doctor` fails, treat NeuTTS as not yet ready and continue with bootstrap or install instead of summarizing it as already working.

If the skill needs help previewing the bootstrap plan without executing it, use:

```bash
<target-python> <skill-bootstrap-helper> --json
```

To actually perform the bootstrap steps instead of only printing them:

```bash
<target-python> <skill-bootstrap-helper> --install-cli --sample-profile --execute --json
```

The helper uses the bundled skill assets as the source of truth. Use the same `<target-python>` for bootstrap, `doctor`, `install`, `list-voices`, and `synth`.

After bootstrap, explicitly confirm that `list-voices` includes `jo-demo`. If it does not, stop and report that the bundled sample-profile creation failed. Do not continue into synthesis, public-domain sourcing, or memory-based recovery.

### 2. Install NeuTTS itself

```bash
<target-python> -m neutts_cli.cli install --all
```

This installs the upstream `neutts` package into the active Python environment. For quick CPU-only verification, `--all` is acceptable; if the user wants a slimmer setup, use `--onnx` or `--llama` as appropriate.

### Fresh setup fallback

If `neutts doctor` shows `neutts_installed: false`, or an early synthesis attempt fails because the NeuTTS runtime is missing, immediately run:

```bash
<target-python> -m neutts_cli.cli install --all
```

Then re-run:

```bash
<target-python> -m neutts_cli.cli doctor
<target-python> -m neutts_cli.cli list-voices
```

For a first-time verification, prefer a stable user-visible output path:

```bash
mkdir -p ~/voice-tests
<target-python> -m neutts_cli.cli synth --voice jo-demo --text "Hello from Hermes" --out ~/voice-tests/neutts_verify.wav
```

For a fresh setup, `jo-demo` is the expected baseline profile if the sample bootstrap step was run.

If `jo-demo` is missing after bootstrap, that is a setup failure, not a cue to switch to another voice source. Report it clearly and stop there.

### 3. Collaborative voice design (proactive)

**When to trigger:** If the user asks you to speak, use TTS, or send a voice message and you have no custom voice profile yet (check `neutts list-voices`), proactively offer to design a voice together before falling back to the default `jo-demo` profile.

**Interaction style:** Prefer short clarify prompts with concrete options over long open-ended questions. Use multiple-choice first to keep momentum high, then switch to freeform only when you need details the user must supply.

**Clarify timeout policy:** If a clarify prompt times out, do not start new sourcing, downloading, or transformation work unless the timed-out question was explicitly about accepting a recommendation you already made. If you must continue after timeout, say so plainly and pick the safest default:

- default to your recommendation when the timeout happened on `Go with my recommendation` vs `Something else`
- otherwise stop and ask again in plain language rather than silently making a bigger decision

**Preferred first clarify after verification:**

- Keep demo voice
- Create a voice for me
- Set default voice
- Just test synthesis

**If the user chooses to create a voice for the assistant, first give one concise recommendation sentence.** Make it personal and relationship-aware. Prefer wording like:

> "Based on what I know about you, I'd make my voice warm, grounded, and a little nerdy rather than polished narrator-clean."

Then prefer a binary clarify prompt:

- Go with my recommendation
- Something else

**If the user chooses `Something else`, prefer one short follow-up clarify for direction:**

- Warm and grounded
- Bright and energetic
- Calm and precise
- Distinct / separate persona

After the user picks a direction, prefer a second short clarify for how to source the reference:

- Find public-domain clips for me
- I'll give you a clip path and transcript

Default to doing the heavy lifting yourself. The first option should be presented as the default path whenever possible.

If the user chooses `Find public-domain clips for me`, take responsibility for the search and present a small curated set of promising 3-15 second candidates instead of pushing the work back onto the user immediately.

Use a constrained sourcing workflow:

- prefer the built-in web or browser tools for search and page inspection
- prefer a small set of trusted public-domain sources such as LibriVox and Project Gutenberg recordings when available
- do not call unavailable or speculative tools such as `web_search`; use only tools that are actually present in the environment
- do not use ad hoc Python scraping with `requests`, `bs4`, or one-off parsing scripts for clip discovery unless the user explicitly asked for that style of debugging
- do not bounce across many search methods in one turn
- stop at 3 strong candidates maximum

If the first sourcing method fails, use one fallback method only. If that also fails, stop and ask the user whether they want you to keep searching later or provide a clip path directly. Do not continue thrashing through more tools.

If a clarify timed out earlier in the same branch, do not interpret that as permission to begin sourcing or downloading on your own unless the timed-out choice was specifically approval to follow your recommendation.

When presenting sourced candidates in a clarify menu, put the short description directly in each option label instead of listing bare names only. Prefer compact labels like:

- Mark Nelson - friendly nerdy storyteller
- Adrian Praetzellis - warm professor energy
- Peter Yearsley - calm precise British
- Show me more options

Keep the summary above the menu brief. The menu itself should carry most of the distinction between options so the user can decide at a glance.

When sourcing succeeds, present at most 3 candidates and move straight to selection. Do not keep exploring once you already have enough viable options.

That means:

1. present candidates
2. get the user's candidate choice
3. immediately ask `Use this source` or `Show me another`
4. only after `Use this source`, begin download, clipping, transcript lookup, or transcription

Do not download audio, fetch source text, or prepare clips before that confirmation step.

After the user selects a candidate source voice, use one short confirmation prompt before downloading, clipping, or transcribing:

- Use this source
- Show me another

This confirmation is mandatory. Do not start clip extraction or transcription work until the user confirms the source, unless the timed-out clarify was specifically approval to follow your recommendation.

For clip preparation, prefer a temporary workspace such as `/tmp/neutts-voice-reference` rather than writing into `~/.hermes/` or another durable user directory by default.

For transcripts, prefer source text over STT whenever the material comes from LibriVox, Project Gutenberg, or another public-domain reading with matching text available. Use Whisper or other STT only as a fallback when matching source text is not readily available.

If transcript extraction fails once, stop and ask whether to try another clip instead of retrying blindly through multiple transcription attempts.

Before creating the voice profile, verify the final transcript once for obvious shell artifacts, prompt text, or mismatched lines. Fix the transcript file first, then run `add-voice`. Do not create a profile and patch it afterward as the normal path.

In the normal path, create the intended final voice name directly. Do not create duplicate workaround names like `atom2` unless the user explicitly asked for variants or you are preserving two intentionally different voices.

Do not manually edit `voice.json` as part of the standard workflow. Only treat direct metadata edits as a last-resort recovery step after you have clearly explained the problem and simpler CLI-based fixes failed.

If the user chooses `I'll give you a clip path and transcript`, ask only for the required freeform inputs:

- reference audio path
- transcript

Frame this as creating or refining the agent's own voice for the user-facing relationship. Prefer wording like "create a voice for me", "design my voice", or "make me sound like X" over generic phrases like "create a custom voice" unless the user used that wording first.

**How to approach it:** Be conversational and opinionated, not a questionnaire. You know the user — draw on what you know about them, your relationship, the platform you're on, and who you are as an agent. Lead with your own take on what voice would fit, then invite their input.

The value proposition is agent identity, not generic TTS setup. Default to language that treats the voice as the assistant's voice in the relationship with the user.

**Framework:**

1. **Open with your perspective.** Reflect briefly on who you are to the user (cognitive partner, assistant, creative collaborator, etc.) and what kind of voice would match that dynamic. Share a concrete suggestion — don't be generic.

2. **Describe the vibe, not just parameters.** Instead of "select a pitch range," paint a picture: warm and grounded, bright and energetic, calm and steady, playful with an edge. Use language that conveys personality, not spec sheets.

3. **Ask open-ended questions.** Cover these dimensions naturally in conversation (not as a numbered list unless the user seems unsure):
   - Register / feel: lower and grounded, higher and bright, something neutral
   - Tone: calm, energetic, warm, precise, playful
   - Similarity to the user: close to their own voice, or distinctly different
   - Any specific voices they like or want to approximate

4. **Take on the sourcing work by default.** NeuTTS voice cloning needs a reference audio clip (3-15 seconds, mono WAV preferred) plus a transcript of what the clip says. By default, offer to go find public-domain reference clips yourself and narrow them down for the user. Only ask the user for a local clip path and transcript if they choose that route or already have one ready.

5. **Iterate if needed.** After the first synthesis, ask if the voice feels right or if they want to try a different reference. Voice design is subjective — treat it as a collaborative process, not a one-shot.

**Example opener** (adapt to your actual persona and relationship with the user):

> "So if I'm going to talk to you, let me think about what I should actually sound like... I'm your [role] — the one who [what you do together]. I'm thinking something [concrete vibe description]. I can go find a few strong public-domain reference clips for us, or if you already have a clip you want me to use, you can point me to it."

**After the user provides a reference clip:**

```bash
neutts add-voice AGENT_NAME --ref-audio /path/to/clip.wav --ref-text-file /path/to/transcript.txt --language en
neutts synth --voice AGENT_NAME --text "Here's what I sound like now — what do you think?" --out ./voice_test.wav
```

Send the test WAV to the user and ask for feedback before considering the voice finalized.

Do not auto-play the generated audio locally as part of the standard flow. Report the output path clearly so the user can choose whether to play it.

**Locking in the voice:**

Once the user approves the voice, set it as the default so all future synthesis uses it automatically — no `--voice` flag needed:

```bash
neutts config --default-voice AGENT_NAME
```

Confirm the lock-in to the user. Let them know:
- This voice will be used automatically whenever you speak from now on
- They can change it anytime (`neutts config --default-voice OTHER_NAME`)
- They can check what's set with `neutts config`

Offer next steps naturally, like Atom's approach: suggest sending a longer voice note, tweaking the style, or just moving on — don't make it feel like a configuration wizard that just completed.

### 4. Add a voice profile manually

If skipping the collaborative flow, or adding a voice from a known reference:

If working from this repo, a sample profile can be bootstrapped automatically:

```bash
python optional-skills/mlops/models/neutts/scripts/bootstrap_neutts_cli.py --repo-root . --install-cli --sample-profile --json
```

Add `--execute` to actually run those commands.

Or add one manually:

```bash
neutts add-voice demo --ref-audio ./samples/voice.wav --ref-text-file ./samples/voice.txt --language en
```

Reference guidelines:

- mono WAV preferred
- 3 to 15 seconds is ideal
- transcript should match the reference audio closely
- use same-language references for best multilingual results

### 5. Synthesize speech

For a quick smoke test:

```bash
neutts synth --voice demo --text Hello from Hermes
```

For a named output file:

```bash
neutts synth --voice demo --text This is a local NeuTTS test --out ./speech.wav
```

### 6. Report results clearly

After running synthesis:

- confirm the output path
- note whether a saved voice profile or ad-hoc reference was used
- mention any warnings from NeuTTS, but do not treat watermark warnings as a hard failure
- after verification, prefer a short clarify prompt with concrete next-step options instead of a long open-ended paragraph
- when offering voice creation, phrase it as creating the assistant's voice for the user, not as a generic custom voice feature
- if verification did not happen in the current turn, explicitly say that instead of implying the environment is already ready
- do not perform risky or noisy cleanup commands in the normal success path; temporary files can simply be left in `/tmp` unless the user asked for cleanup

## Memory

- do not save memory for routine install or verification runs
- only save memory if the user established a durable voice preference, approved a default voice, or a non-trivial workaround/fix was required
- if you save memory for this flow, do it once at the very end after the voice is finalized or set as default
- do not do intermediate memory writes during setup, sourcing, clip prep, or testing
- if memory save fails or memory is full, do not thrash through retries; either skip it or replace a single clearly related prior NeuTTS entry once

## Pitfalls

- `neutts synth` needs either `--voice` or both `--ref-audio` and `--ref-text`
- The first synthesis call can be slow because models need to load
- `llama-cpp-python` acceleration is platform-specific and may require custom build flags
- `doctor` may show `ffmpeg` missing; that does not block WAV synthesis
- The upstream NeuTTS package may emit Perth watermark warnings; these are informational unless the user explicitly needs watermarking
- If the `neutts` command is missing after install, ensure the active virtualenv is the same environment where the editable package was installed
- transcript files can pick up shell artifacts if written carelessly; verify them before `add-voice`
- avoid duplicate profile-name workarounds and direct `voice.json` edits in the normal path

## Verification

Use this sequence:

```bash
neutts doctor
neutts list-voices
neutts synth --voice jo-demo --text Hello from Hermes --out ./verify.wav
```

Success means:

- `doctor` shows `neutts_installed: true`
- `list-voices` includes the expected profile
- synthesis completes and writes a WAV file

## References

- NeuTTS upstream: https://github.com/neuphonic/neutts
- Bundled NeuTTS CLI scaffold: `assets/neutts-cli`
- Skill bootstrap helper: `optional-skills/mlops/models/neutts/scripts/bootstrap_neutts_cli.py`
