---
name: image-prompt-factory
description: Grounded, validated prompt packs for image generation.
version: 1.0.0
author: SmokeDev (TheSmokeDev)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [image-generation, prompt-engineering, creative, grounding, validation]
    category: creative
    related_skills: [comfyui, codex, meme-generation, pixel-art]
---

# Image Prompt Factory Skill

Turn a visual brief into a validated, dual-variant image prompt pack grounded
in a pinned, checksum-verified corpus of worked image prompts. The skill
produces prompt packs and validates them deterministically; it does not render
images — hand the pack to any backend (the `image_gen` plugin family, the
`comfyui` or `codex` skills, or an external tool). The mechanism is the same
grounding engine already published in the Archon workflow marketplace as
image-node-factory.

## When to Use

- The user wants ad creative, hero images, social visuals, posters, or product
  shots and cares about prompt quality, not just "an image".
- A batch of on-brief concept variations is needed (1-8 concepts, one brief).
- Prompts must be defensible: citations to real corpus exemplars, or an honest
  self-authored declaration — never an invented source.
- A downstream pipeline supplies the real subject (person, mascot, product)
  and the pack must not invent one.

Do NOT use this for: a quick one-off image with no craft requirements (call an
image backend directly), editing an existing image, or video.

## Prerequisites

- Python 3.11+ on PATH (stdlib only — no extra packages).
- One-time online priming of the corpus (~1 MB from the pinned upstream
  `freestylefly/awesome-gpt-image-2`, MIT). Everything after that is offline.
- No API keys. Rendering, if wanted, uses whatever backend the install
  already has.

## How to Run

Run the two helper scripts with the `terminal` tool from the skill directory;
write the JSON artifacts with `write_file`; read exemplars and templates with
`read_file`. One-time setup:

```bash
python scripts/style_corpus.py prime
```

Then per brief: intake → select → ground → write pack → validate → hand off.
Work in any scratch directory; the three artifact files are `brief.json`,
`selection.json` (plus generated `grounding.local.json`), and
`prompt-pack.json`.

## Quick Reference

| Step | Action | Command / artifact |
|---|---|---|
| 0 | Prime corpus (once, online) | `python scripts/style_corpus.py prime` |
| 1 | Parse the brief | write `brief.json` |
| 2 | Browse taxonomy | `python scripts/style_corpus.py stats` / `template <id>` |
| 3 | Pick template + tags | write `selection.json` |
| 4 | Ground the selection (offline) | `python scripts/style_corpus.py ground --selection selection.json` |
| 5 | Write the pack (both variants) | write `prompt-pack.json` per `references/prompt-schema.md` |
| 6 | Validate | `python scripts/pack_validate.py validate --workdir .` |
| 7 | Hand off to a renderer | `image_gen` backends, `comfyui`, `codex`, or external |

## Procedure

### 1. Intake

Parse the operator request into `brief.json` (schema in
`references/prompt-schema.md`): the request, `render_mode` (`baked` default
for short English copy, `overlay` for exact fonts / long / non-Latin text),
`aspect`, `count` (1-8), `subject_mode` (`generic` or `placeholder`), and
optional `exact_text`.

### 2. Select

Browse the corpus taxonomy with `terminal`:
`python scripts/style_corpus.py stats` lists categories;
`python scripts/style_corpus.py template <template_id>` prints a template
body. Choose one template plus style/scene tags that fit the brief and write
`selection.json`. Optionally name promising `example_case_ids` as anchors.

### 3. Ground

```bash
python scripts/style_corpus.py ground --selection selection.json
```

This resolves the selection against the verified local corpus and writes
`grounding.local.json` — the ONLY authoritative source of exemplar cases.
Zero matches is an honest `grounded: false`, not an error. A cold or corrupt
cache exits 1: re-run `prime`. Retrieval is a deterministic taxonomy filter
(category, style, scene, template example cases) — no embeddings, no LLM.

### 4. Write the pack

Read the grounded exemplars with `read_file` for structure and quality, then
write fresh wording — never paste exemplar text. For each concept emit BOTH
variants (`baked_prompt` and `overlay_prompt` + `copy`) using the 13-block
scaffold in `references/prompt-schema.md`. Provenance: stamp engine, pin,
sha256, license, and resolved case ids if and only if the grounding says
`grounded: true`; otherwise omit them all and set `self_authored: true`.
When `subject_mode` is `placeholder`, every prompt's `Subject:` field starts
with the literal `[SUBJECT SUPPLIED AT RENDER TIME]` and no invented traits
appear anywhere.

### 5. Validate

```bash
python scripts/pack_validate.py validate --workdir .
```

Exit 1 blocks the pack. The validator re-checks the physical artifacts and
reports every violation at once: hollow citations (a stamped engine on an
ungrounded pack), cited ids the grounding never resolved, provenance
mismatches, more than 8 concepts, empty variants, a missing placeholder
sentinel, missing copy objects, and absolute local paths in pack text.

### 6. Hand off

Present the operator the pack summary (concepts, grounded or self-authored,
cited ids) and render on request with whatever backend is installed. The
baked variant is the post-ready single file; the overlay variant pairs the
text-free image with the `copy` object for later compositing.

## Pitfalls

- **Hollow citations.** A reference that is a pointer (URL, uninstalled file)
  silently grounds nothing while still being cited. Only
  `grounding.local.json` counts as read; if a case id is not in it, it does
  not exist for this run.
- **Never paste exemplar text into a pack.** The corpus is third-party MIT
  material to learn structure from; `*.local.json` files never ship, and
  their text never enters an artifact.
- **The pack invents a subject it was told not to.** An appended identity
  loses to the pack's own `Subject:` line — that is why `placeholder` mode
  demands the literal sentinel and forbids trait words entirely.
- **Skipping validation because the pack "looks right".** The validator
  exists because an instruction to a model is a suggestion; run it every
  time.
- **Vector search over this corpus.** A majority of prompts are CJK; an
  English embedder ranks them quietly wrong. The deterministic taxonomy
  filter is intentional — do not "improve" it with embeddings.
- **Baking long or non-English copy.** Spelling degrades; switch to the
  overlay variant.

## Verification

Skill standards and validator behavior are covered by the shipped test file:

```bash
scripts/run_tests.sh tests/skills/test_image_prompt_factory_skill.py -q
```

A live end-to-end check: `prime`, then run steps 1-5 with a toy brief and
confirm the validator exits 0 on the pack and exits 1 when you delete the
`self_authored` key from an ungrounded pack.
