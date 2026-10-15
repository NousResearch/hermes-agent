# ASCII Art Pipeline Foundation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn the current `ascii-art-pipeline` into a publishable foundation: a standalone GitHub-ready package with flagship showcase examples, plus a clean in-repo Hermes skill suitable for upstreaming to `NousResearch/hermes-agent`.

**Architecture:** Use a dual-track structure. Track A is a standalone repo focused on reusable tooling, presets, diagnostics, examples, and gallery output. Track B is a bundled Hermes skill that teaches the workflow, references the canonical presets, and ships concise supporting scripts/references into the Hermes repo.

**Tech Stack:** Python, ffmpeg, chafa, ascii-image-converter, Pillow, HTML preview pages, pytest, GitHub CLI/git.

---

## Core publishing strategy

### Track A — standalone public repo
Target repo root:
- `/Users/johann/projects/ascii-art-pipeline/`

Purpose:
- reusable CLI/tooling
- reproducible presets
- flagship examples
- packaging/readme/gallery
- easier to demo publicly and iterate independently

### Track B — Hermes built-in skill
Target Hermes repo paths:
- `/Users/johann/.hermes/hermes-agent/skills/creative/ascii-art-pipeline/SKILL.md`
- `/Users/johann/.hermes/hermes-agent/skills/creative/ascii-art-pipeline/references/*.md`
- `/Users/johann/.hermes/hermes-agent/skills/creative/ascii-art-pipeline/scripts/*.py`

Purpose:
- concise agent-usable operating manual
- references to the packaged tool’s canonical presets and workflows
- documentation page generation via Hermes docs tooling

### Why dual-track
- A public repo wants strong product framing and examples.
- Hermes wants a skill that is compact, operational, and easy to load into context.
- Keeping them aligned but distinct prevents the skill from becoming a bloated product README.

---

## Product definition

The published ASCII foundation should do five things well:

1. **Convert images to high-quality ASCII** with named presets.
2. **Convert video to animated ASCII/eikon outputs** with reproducible pipelines.
3. **Render high-resolution previews** so people can inspect quality visually, not just in raw text.
4. **Diagnose quality** using glyph metrics and visual heuristics.
5. **Showcase excellence** through a small set of elite examples that prove density, fidelity, and style.

Canonical presets to freeze:
- `stroke-clarity`
- `d30-dense`
- `braille-detail`
- `eikon-motion`

---

## Deliverables

### Standalone repo deliverables
- `README.md` with strong visual framing and exact commands
- `pyproject.toml` or equivalent packaging metadata
- `ascii_pipeline/` package or `scripts/` CLI entrypoints
- `examples/` tree with 3-5 flagship examples
- `gallery/` or `site/` static HTML comparison page
- `diagnostics/diagnose_glyph_quality.py`
- `render/render_preview.py`
- `presets/` definitions for named modes

### Hermes repo deliverables
- new bundled skill under `skills/creative/ascii-art-pipeline/`
- supporting `references/` and `scripts/`
- generated docs via:
  - `/Users/johann/.hermes/hermes-agent/website/scripts/generate-skill-docs.py`
- tests covering skill presence/basic integrity if needed

### Publication deliverables
- GitHub-ready standalone repo with screenshots/GIFs/examples
- Hermes PR-ready skill patch set
- release checklist for both destinations

---

## Flagship showcase set

These are not filler examples. They are the proof-of-quality set.

1. **Portrait fidelity demo**
   - target: clear eyes, mouth, hair contours
   - modes: `stroke-clarity`, `d30-dense`, `braille-detail`

2. **Dense cybernetic field demo**
   - target: abstract monochrome or glitch composition with intentional density
   - mode: `d30-dense`

3. **Landscape / architecture demo**
   - target: layered depth, readable silhouette, tonal separation
   - modes: `stroke-clarity`, `braille-detail`

4. **Animated eikon / avatar demo**
   - target: motion richness, glyph stability, instant recognizability
   - mode: `eikon-motion`

5. **Mode comparison board**
   - target: same source rendered via all major presets side-by-side
   - output: PNG montage + local HTML viewer

Acceptance bar for showcase examples:
- recognizable at a glance
- dense without becoming garbled
- strong side-by-side difference between presets
- good at terminal viewing size, not just image zoom

---

## Repo structure proposal

### Standalone repo
```text
/Users/johann/projects/ascii-art-pipeline/
├── README.md
├── pyproject.toml
├── ascii_pipeline/
│   ├── __init__.py
│   ├── cli.py
│   ├── presets.py
│   ├── image_modes.py
│   ├── video_modes.py
│   ├── eikon.py
│   ├── preview.py
│   └── diagnostics.py
├── scripts/
│   ├── render-example.py
│   ├── render-gallery.py
│   └── benchmark-presets.py
├── examples/
│   ├── portrait/
│   ├── dense-field/
│   ├── landscape/
│   ├── eikon/
│   └── comparison/
├── assets/
│   └── sources/
├── gallery/
│   └── index.html
└── tests/
    ├── test_presets.py
    ├── test_diagnostics.py
    └── test_cli.py
```

### Hermes skill
```text
/Users/johann/.hermes/hermes-agent/skills/creative/ascii-art-pipeline/
├── SKILL.md
├── references/
│   ├── glyph-quality-principles.md
│   ├── chafa-resolution-tuning.md
│   ├── herm-tui-eikon-downsampling.md
│   ├── known-good-eikons.md
│   └── publication-checklist.md
└── scripts/
    └── diagnose-glyph-quality.py
```

---

## Execution plan

### Phase 1: Freeze the canonical foundation

### Task 1: Inventory current pipeline assets and decide what is canonical
**Objective:** Separate reusable core from session-specific experiments.

**Files:**
- Review: `/Users/johann/.hermes/profiles/museah/skills/creative/ascii-art-pipeline/SKILL.md`
- Review: `/Users/johann/projects/herm/scripts/generate_eikon_scratch.py`
- Review: current local comparison/render helpers used during eikon work
- Output doc: `/Users/johann/projects/ascii-art-pipeline/docs/canonical-foundation.md`

**Steps:**
1. List all pipeline components currently in use.
2. Mark each one as `core`, `experimental`, or `project-specific`.
3. Freeze the preset vocabulary and quality rubric.
4. Document the canonical choices.

**Verification:**
- clear list exists for presets, tools, scripts, and examples
- no ambiguous “maybe use X or Y” at the foundation level

### Task 2: Define the preset contract
**Objective:** Make each preset deterministic and nameable.

**Files:**
- Create: `/Users/johann/projects/ascii-art-pipeline/ascii_pipeline/presets.py`
- Test: `/Users/johann/projects/ascii-art-pipeline/tests/test_presets.py`

**Preset contract fields:**
- source scaling strategy
- converter backend
- charset/palette
- target dimensions
- optional pre-processing
- preview renderer settings
- quality thresholds

**Verification:**
- one data structure can describe `stroke-clarity`, `d30-dense`, `braille-detail`, `eikon-motion`

### Task 3: Freeze the quality rubric
**Objective:** Turn taste into testable heuristics.

**Files:**
- Create: `/Users/johann/projects/ascii-art-pipeline/ascii_pipeline/diagnostics.py`
- Create: `/Users/johann/projects/ascii-art-pipeline/tests/test_diagnostics.py`
- Mirror into Hermes references: `.../skills/creative/ascii-art-pipeline/references/glyph-quality-principles.md`

**Metrics:**
- unique glyph count
- heavy-glyph ratio
- low-noise ratio
- fill ratio
- frame-to-frame diff for motion presets
- dimension validation

**Verification:**
- diagnostics run on example outputs and return structured JSON or text summary

---

## Phase 2: Build the standalone public package

### Task 4: Bootstrap the standalone repo
**Objective:** Create the publishable project shell.

**Files:**
- Create: `/Users/johann/projects/ascii-art-pipeline/README.md`
- Create: `/Users/johann/projects/ascii-art-pipeline/pyproject.toml`
- Create package skeleton shown above

**Verification:**
- repo installs locally
- basic CLI entrypoint resolves

### Task 5: Implement image conversion entrypoints
**Objective:** Expose a stable CLI for still images.

**Files:**
- Create: `/Users/johann/projects/ascii-art-pipeline/ascii_pipeline/cli.py`
- Create: `/Users/johann/projects/ascii-art-pipeline/ascii_pipeline/image_modes.py`
- Test: `/Users/johann/projects/ascii-art-pipeline/tests/test_cli.py`

**Commands to support:**
- `ascii-pipeline render-image --preset stroke-clarity ...`
- `ascii-pipeline render-image --preset d30-dense ...`
- `ascii-pipeline render-image --preset braille-detail ...`

**Verification:**
- each command produces text output and optional saved preview assets

### Task 6: Implement video/eikon entrypoints
**Objective:** Expose a stable CLI for motion workflows.

**Files:**
- Create: `/Users/johann/projects/ascii-art-pipeline/ascii_pipeline/video_modes.py`
- Create: `/Users/johann/projects/ascii-art-pipeline/ascii_pipeline/eikon.py`

**Commands to support:**
- `ascii-pipeline render-video ...`
- `ascii-pipeline build-eikon ...`
- `ascii-pipeline render-preview ...`

**Verification:**
- can reproduce one known-good eikon workflow from a source clip

### Task 7: Implement preview rendering
**Objective:** Make quality legible to humans.

**Files:**
- Create: `/Users/johann/projects/ascii-art-pipeline/ascii_pipeline/preview.py`
- Create: `/Users/johann/projects/ascii-art-pipeline/scripts/render-gallery.py`

**Outputs:**
- hi-res PNG from text art
- animated GIF or sampled frame sheet from eikons
- side-by-side comparison boards

**Verification:**
- examples can be judged from rendered images without opening raw text

---

## Phase 3: Build the elite showcase set

### Task 8: Generate the portrait fidelity example
**Objective:** Produce the strongest human-recognition example.

**Files:**
- Output: `/Users/johann/projects/ascii-art-pipeline/examples/portrait/`

**Verification:**
- includes source, command, outputs, metrics, preview PNG

### Task 9: Generate the dense field example
**Objective:** Prove that density can be intentional rather than garbled.

**Files:**
- Output: `/Users/johann/projects/ascii-art-pipeline/examples/dense-field/`

**Verification:**
- example reads as deliberate structured density, not accidental noise

### Task 10: Generate the landscape/architecture example
**Objective:** Prove depth-layer readability.

**Files:**
- Output: `/Users/johann/projects/ascii-art-pipeline/examples/landscape/`

### Task 11: Generate the animated eikon example
**Objective:** Prove motion richness and avatar legibility.

**Files:**
- Output: `/Users/johann/projects/ascii-art-pipeline/examples/eikon/`

### Task 12: Generate the mode-comparison board
**Objective:** Create the showcase’s most persuasive single artifact.

**Files:**
- Output: `/Users/johann/projects/ascii-art-pipeline/examples/comparison/`
- Output: `/Users/johann/projects/ascii-art-pipeline/gallery/index.html`

**Verification for Tasks 8-12:**
- each example includes command history, metrics, text output, and preview image(s)

---

## Phase 4: Author the Hermes-publishable skill

### Task 13: Draft the bundled Hermes skill
**Objective:** Convert the operational knowledge into a clean in-repo skill.

**Files:**
- Create: `/Users/johann/.hermes/hermes-agent/skills/creative/ascii-art-pipeline/SKILL.md`

**Requirements:**
- frontmatter matches Hermes conventions
- concise but complete
- points to references/scripts rather than bloating `SKILL.md`
- focuses on triggers, presets, diagnostics, and workflows

**Verification:**
- follows guidance from `hermes-agent-skill-authoring`
- valid frontmatter

### Task 14: Add supporting references and scripts
**Objective:** Keep the skill compact while preserving depth.

**Files:**
- Create under `.../references/`
- Create under `.../scripts/`

**Minimum support files:**
- `glyph-quality-principles.md`
- `chafa-resolution-tuning.md`
- `herm-tui-eikon-downsampling.md`
- `publication-checklist.md`
- `diagnose-glyph-quality.py`

### Task 15: Generate Hermes docs pages
**Objective:** Ensure the skill becomes visible in Hermes docs.

**Files / commands:**
- Run: `python website/scripts/generate-skill-docs.py`
- Review generated docs under:
  - `/Users/johann/.hermes/hermes-agent/website/docs/user-guide/skills/bundled/creative/`
  - `/Users/johann/.hermes/hermes-agent/website/docs/reference/skills-catalog.md`

**Verification:**
- generated bundled skill page exists
- catalog updates include the skill

---

## Phase 5: QA, GitHub readiness, and upstreaming

### Task 16: Test the standalone repo locally
**Objective:** Make the public repo actually runnable.

**Commands:**
- `python -m pytest /Users/johann/projects/ascii-art-pipeline/tests -q`
- exercise each CLI preset manually

### Task 17: Test the Hermes skill repo changes
**Objective:** Make sure the Hermes contribution is clean.

**Commands:**
- `cd /Users/johann/.hermes/hermes-agent && python -m pytest tests/skills -q`
- `cd /Users/johann/.hermes/hermes-agent && python website/scripts/generate-skill-docs.py`
- if broader verification is needed: `cd /Users/johann/.hermes/hermes-agent && scripts/run_tests.sh`

### Task 18: Prepare public-facing README and gallery copy
**Objective:** Make the package legible to outsiders immediately.

**Files:**
- `/Users/johann/projects/ascii-art-pipeline/README.md`
- `/Users/johann/projects/ascii-art-pipeline/gallery/index.html`

**README must show:**
- what the tool is
- why it is different
- exact install/use commands
- best examples first
- side-by-side preset comparisons

### Task 19: Create GitHub publication branches/PRs
**Objective:** Separate product publishing from Hermes upstreaming.

**Track A:** standalone public repo
- create repo
- push initial package + examples

**Track B:** Hermes repo
- branch from current Hermes checkout
- commit new skill + refs/scripts + generated docs
- open PR toward Hermes Agent

---

## Success criteria

### Public repo success
- someone can clone it and generate one great example quickly
- examples visibly prove fidelity and density
- presets are named, deterministic, and documented

### Hermes skill success
- skill loads cleanly and teaches the workflow well
- docs page is generated automatically
- references/scripts are sufficient without bloating the main skill

### Strategic success
- this becomes the canonical base for future ASCII work
- future eikon/splash/showcase efforts build on this foundation, not ad hoc experiments

---

## Immediate recommendation

Start with this order:
1. freeze presets + diagnostics
2. bootstrap standalone repo
3. build the 5 flagship examples
4. author the Hermes bundled skill
5. generate docs + prepare GitHub pushes
