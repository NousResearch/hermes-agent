# Kanban Setup — Project Bootstrap & Profile Configuration

Once the brief is locked and the team is designed, the next step is producing
the actual `setup.sh` that creates the project workspace, configures Hermes
profiles, and fires the initial kanban task.

This file documents the patterns. The companion script
`scripts/bootstrap_pipeline.py` automates most of it from a structured input
JSON.

> **Credit:** the single-project-workspace layout, profile-config patching
> approach, SOUL.md-per-profile convention, and `--workspace dir:<path>` rule
> are adapted from alt-glitch's original multi-agent video pipeline:
> [NousResearch/kanban-video-pipeline](https://github.com/NousResearch/kanban-video-pipeline).
> This skill generalizes those patterns across video styles and replaces the
> string-replacement config patcher with a PyYAML-based one.

## Project workspace structure

Every video project gets one workspace under `~/projects/video-pipeline/<slug>/`:

```
~/projects/video-pipeline/<slug>/
├── brief.md                       ← the contract; all tasks reference
├── TEAM.md                        ← team composition + task graph (director reads this)
├── taste/
│   ├── brand-guide.md             ← color, typography, motion rules
│   ├── emotional-dna.md           ← what the piece should FEEL like
│   └── style-frames/              ← optional: visual references
├── audio/
│   ├── track.mp3                  ← provided music (if any)
│   ├── voiceover/                 ← per-line TTS clips
│   └── sfx/                       ← sound effects
├── assets/
│   ├── logos/
│   ├── fonts/
│   └── existing-footage/          ← reusable provided clips
├── scenes/
│   ├── scene-01/
│   │   ├── VISUAL_SPEC.md         ← cinematographer's per-scene spec
│   │   ├── render.py              ← renderer's code (or sketch.html, etc.)
│   │   ├── checkpoints/           ← preview frames for QA
│   │   └── clip.mp4               ← the deliverable for this scene
│   ├── scene-02/...
│   └── ...
├── checkpoints/                   ← global review frames
├── tools/                         ← optional project-local helpers
└── output/
    ├── final.mp4                  ← stitched + audio
    ├── final-noaudio.mp4
    ├── final-9x16.mp4             ← optional: vertical alternate
    └── captions.srt               ← optional: subtitle file
```

**The slug** is derived from the brief title: lowercase, hyphen-separated.
Example: `q3-product-teaser`, `ascii-mood-loop`, `interview-cut-2026-q1`.

## The setup.sh script

The setup script does six things in order:

1. **Create workspace tree** — all directories above
2. **Create profiles** — `hermes profile create <name> --clone`
3. **Configure profiles** — patch each profile's
   `~/.hermes/profiles/<name>/config.yaml` to set toolsets, always_load skills,
   and `cwd`
4. **Write SOUL.md per profile** — the personality + role definition
5. **Copy any provided assets + write `brief.md`, `TEAM.md`, and `taste/`**
6. **Fire the initial kanban task** — `hermes kanban create` assigned to the director

See `assets/setup.sh.tmpl` for the skeleton.

### Profile creation pattern

```bash
hermes profile create director --clone 2>/dev/null || true
```

The `--clone` flag clones from the active profile (preserving model, base
config). The `|| true` makes the script idempotent — re-running won't error if
the profile already exists.

### Profile config patching

Each profile has a YAML config at `~/.hermes/profiles/<name>/config.yaml`. The
setup script edits exactly two keys:

1. `toolsets:` — replace the default with the role's required toolsets
2. `skills.always_load:` — list the role's must-load skills (may be empty)

**Do NOT** modify `approvals.mode` (controls user-confirmation of tool calls
— a security setting that must stay as the user configured it). **Do NOT**
modify `terminal.cwd` — the kanban dispatcher overrides cwd per-task via
`--workspace dir:<path>`, so the profile's cwd is irrelevant to the kanban
work and changing it could break the user's interactive use of the profile.

Use **PyYAML**, not string replacement, so the patch is robust against
default-config schema drift:

```bash
configure_profile() {
    local profile="$1"
    local toolsets_json="$2"     # JSON array, e.g. '["kanban","terminal","file"]'
    local skills_json="$3"       # JSON array, e.g. '["kanban-worker","ascii-video"]'
    python3 - "$profile" "$toolsets_json" "$skills_json" <<'PY'
import json, os, sys, yaml
profile, ts_json, sk_json = sys.argv[1:4]
p = os.path.expanduser(f"~/.hermes/profiles/{profile}/config.yaml")
with open(p) as f:
    cfg = yaml.safe_load(f) or {}
cfg["toolsets"] = json.loads(ts_json)
cfg.setdefault("skills", {})["always_load"] = json.loads(sk_json)
with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}
```

PyYAML must be installed in the user's Python (it ships with most Hermes
installs). If absent: `pip install pyyaml`.

The setup script should also **validate** the patch by re-reading the file
and comparing — see `assets/setup.sh.tmpl` for the validation pattern.

### SOUL.md per profile

Each profile gets a `SOUL.md` at `~/.hermes/profiles/<name>/SOUL.md` that
defines its role, voice, and rules. See `assets/soul.md.tmpl` for the
template. Customize per role and per project.

The director's SOUL.md should be the most opinionated — its voice flavors
the entire production. **Critical content for the director's SOUL.md:**

- **Anti-temptation rules:** "Do not execute the work yourself. For every
  concrete task, create a kanban task and assign it. Decompose, route, comment,
  approve — that's the whole job." (The `kanban-orchestrator` skill provides
  the deeper playbook; load it.)
- **Decomposition steps:** Read `brief.md`, `TEAM.md`, `taste/`. Use the team
  graph in `TEAM.md` to fan out tasks.
- **The workspace_path rule** (see below).

Other profiles' SOUL.md is briefer; mostly mechanical: who you are, what you
read, what you produce, what skills/tools to use, where to write outputs.
Most non-director profiles should `always_load: kanban-worker` for the
deeper-than-baseline kanban guidance.

### Initial kanban task

The final action of setup.sh is firing the kanban:

```bash
hermes kanban create "Direct production of <video title>" \
    --assignee director \
    --workspace dir:"$HOME/projects/video-pipeline/${PROJECT_SLUG}" \
    --tenant ${PROJECT_SLUG} \
    --priority 2 \
    --max-runtime 4h \
    --body "$(cat <<EOF
Read brief.md, TEAM.md, and taste/.
Decompose into the team graph defined in TEAM.md.
All child tasks MUST use:
  workspace_kind="dir"
  workspace_path="$HOME/projects/video-pipeline/${PROJECT_SLUG}"
  tenant="${PROJECT_SLUG}"
EOF
)"
```

The `--workspace dir:<path>` flag is **critical** — it tells the kanban that
all child tasks share this workspace. Skipping or using `worktree` will
isolate profiles and break artifact sharing.

## The TEAM.md file

Alongside `brief.md`, write a `TEAM.md` that the director reads. It documents
the team composition + task graph the orchestrator should follow. This
removes ambiguity and prevents the director from inventing extra steps.

Example structure (for an ASCII video with a music supervisor and editor):

```markdown
# Team & Task Graph — <video title>

## Team

- `director` (this profile) — vision, decomposition, approval
- `cinematographer` — visual spec, quality review (loads `ascii-video`)
- `renderer-ascii` — ASCII scenes (loads `ascii-video`)
- `music-supervisor` — track analysis (loads `songsee`)
- `voice-talent` — narration (uses ElevenLabs API)
- `audio-mixer` — final mix (ffmpeg)
- `editor` — assembly (ffmpeg)
- `reviewer` — final QA gate

## Task Graph

T0: this task — decompose
 │
 ├── T1: cinematographer  "Design visual language"            (parent: T0)
 │    │
 │    ├── T2a: renderer-ascii   "Scene 1 — title card"        (parent: T1)
 │    ├── T2b: renderer-ascii   "Scene 2 — main beat"         (parent: T1)
 │    ├── T2c: renderer-ascii   "Scene 3 — outro"             (parent: T1)
 │
 ├── T3: music-supervisor "Analyze track + emit beats.json"   (parent: T0)
 │
 ├── T4: voice-talent     "Generate narration"                (parent: T0)
 │
 ├── T5: audio-mixer      "Mix VO + bg music"                 (parents: T3, T4)
 │
 ├── T6: editor           "Assemble cut + mux audio"          (parents: T2*, T5)
 │
 └── T7: reviewer         "Final QA"                          (parent: T6)
```

The director turns this into actual `kanban_create` calls.

## API-key prerequisites check

Before firing the kanban, verify required keys are available. Check both
`~/.hermes/.env` and macOS Keychain (if on macOS):

```bash
check_key() {
    local var="$1"
    local kc_account="$2"
    local kc_service="$3"
    if grep -q "^${var}=" ~/.hermes/.env 2>/dev/null && \
       [ -n "$(grep "^${var}=" ~/.hermes/.env | cut -d= -f2-)" ]; then
        return 0
    fi
    if command -v security >/dev/null 2>&1 && \
       security find-generic-password -a "${kc_account}" -s "${kc_service}" -w >/dev/null 2>&1; then
        return 0
    fi
    echo "ERROR: ${var} not set in ~/.hermes/.env or Keychain (${kc_account}/${kc_service})"
    return 1
}

check_key ELEVENLABS_API_KEY hermes ELEVENLABS_API_KEY || exit 1
check_key OPENROUTER_API_KEY hermes OPENROUTER_API_KEY || exit 1
# ...
```

If a key is missing, the script aborts with a clear message rather than
firing a kanban that will hit credential errors mid-execution.

## Critical rules

1. **`workspace_kind="dir"` + `workspace_path="<absolute>"` on every kanban_create.** Otherwise profiles can't share artifacts.

2. **Tenant every task.** `--tenant <project-slug>` keeps the dashboard scoped
   and prevents cross-pollination with other ongoing kanbans.

3. **Idempotency keys.** For tasks that should not duplicate on re-run (e.g.,
   setup creating profiles), use the `idempotency_key` argument or check
   existence first.

4. **`max_runtime_seconds` per task.** Renderers that get stuck eat compute.
   Standard defaults:
   - Renderer task: 1800s (30min)
   - Editor task: 600s (10min)
   - Voice-talent task: 300s (5min)
   - Image-generator task: 600s (10min)
   - Image-to-video-generator task: 900s (15min)

5. **Heartbeats for long renders.** Tasks expected to run >5min should emit
   `kanban_heartbeat` periodically with progress. Renderers should report
   frame counts; the editor should report assembly progress.

6. **The `audio/` and `taste/` dirs are populated BEFORE firing the kanban.**
   Don't ask the director's pipeline to source these — copy at setup time.

7. **`brief.md` is read-only after setup.** If the brief changes during
   execution, that's a significant pivot — re-fire the kanban rather than edit
   live.
