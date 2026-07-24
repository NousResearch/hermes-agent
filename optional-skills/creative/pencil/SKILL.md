---
name: pencil
description: "Create, edit, and export .pen design files via the CLI."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: creative
    tags: [Pencil, Design, UI, Figma, Design-to-Code, Canvas, pencil.dev]
    related_skills: [excalidraw, claude-design, popular-web-designs]
prerequisites:
  commands: [pencil, node]
---

# Pencil Skill

Optional skill — **not active until installed**:

```bash
hermes skills install official/creative/pencil
```

After install, scripts live under `~/.hermes/skills/creative/pencil/`. In prose
below, `{SKILL_DIR}` means that directory (or the repo path
`optional-skills/creative/pencil/` before install).

Drive [Pencil](https://pencil.dev) — an IDE-first design canvas that stores
designs as version-controlled `.pen` files — from Hermes through its CLI. You
can call Pencil's design tools directly (deterministic, no extra API key) or
hand Pencil a natural-language prompt and let its built-in agent generate a
design. This skill does **not** add a dependency to Hermes: the `pencil` binary
is a lazy runtime dependency you install only if you use this skill.

> **Discover the CLI at runtime — don't trust hardcoded specifics.** Pencil's
> commands, tools, and `batch_design` schema change between releases. Before
> acting, learn the *current* surface from the CLI itself:
> `pencil --help`, `pencil interactive --help` (prints the live tool reference),
> and `get_editor_state({ include_schema: true })` for the document schema. This
> skill is a map, not a spec; the CLI's own `--help` is the source of truth.
> (Guidance shaped with the Pencil team.)

## When to Use

- The user wants to create or edit a `.pen` design (screens, components,
  design systems) that lives in their repo.
- The user wants a design rendered to an image (PNG/JPEG/WEBP/PDF).
- The user wants to read an existing `.pen` file's structure, variables, or
  components to keep design and code in sync.

## When NOT to Use

- Quick throwaway diagrams (arch/flow/sequence) → use the `excalidraw` skill.
- Pure HTML/CSS mockups with no design file → use `popular-web-designs` /
  `claude-design`.
- The user is editing a Figma `.fig` file → that's a different product
  (OpenPencil); this skill targets `pencil.dev` `.pen` files.

## Prerequisites

- **Node.js** and the Pencil CLI: `npm install -g @pencil.dev/cli`. Docs say
  Node 18+, but newer CLI builds need **Node 22+** — upgrade if you hit
  `ERR_REQUIRE_ESM`. No sudo? `npm install --prefix ~/.local`.
- **Auth is required for BOTH modes.** `pencil login` (stores
  `~/.pencil/session-cli.json`) or set `PENCIL_CLI_KEY`. Pencil has no offline
  mode — even headless `pencil interactive` refuses to start unauthenticated.
  Verify with `pencil status`.
- **Prompt-driven generation (Mode C)** additionally needs an agent key, e.g.
  `ANTHROPIC_API_KEY` (or `PENCIL_AGENT_API_KEY`).
- Run the preflight check first via the `terminal` tool:
  `python {SKILL_DIR}/scripts/pencil_doctor.py`

## How to Run

Three modes. **Always run the doctor first** — it reads `pencil --help` and
tells you which paths this build exposes:

```bash
python {SKILL_DIR}/scripts/pencil_doctor.py
```

### Mode A — Headless MCP via `pencil start` (preferred, when available)

The cleanest path for an agent without a REPL loop: `pencil` starts a headless
session that Hermes talks to over MCP using Pencil's real tools — no stdin
wrapper. This is **emerging** in the Pencil CLI, so use it only when the doctor
reports `start` (or `pencil --help` lists it). **Confirm the exact flags and
transport with `pencil start --help`** before wiring — the command shape is
still settling.

Typical wiring (verify against `pencil start --help`): register it as a Hermes
MCP server so the session's lifecycle is managed for you, e.g.

```bash
hermes mcp add pencil --command pencil --args start --output design.pen
```

or, if your build serves MCP over a local port, start it as a background
process (`terminal` with `background=true`) and `hermes mcp add pencil --url
<printed-url>`. Either way, start a new Hermes session so the Pencil MCP tools
load, then call them directly. If `pencil start` is absent, use Mode B.

### Mode B — REPL wrapper via `pencil interactive` (fallback)

Where `start` isn't available yet, Hermes drives the `pencil interactive` REPL
non-interactively through `scripts/pencil_repl.py` via the `terminal` tool.
Learn the current tools first with `pencil interactive --help`, then read the
document schema:

```bash
python {SKILL_DIR}/scripts/pencil_repl.py --out design.pen \
  --cmd 'get_editor_state({ include_schema: true })'
```

Then issue tool calls (`batch_get`, `batch_design`, `get_screenshot`, …). The
wrapper appends `save()` (headless) and `exit()` automatically:

```bash
python {SKILL_DIR}/scripts/pencil_repl.py --out design.pen \
  --cmd 'batch_design({ operations: "hero=I(document,{type:\"frame\",name:\"Hero\",x:0,y:0,width:1440,height:900,fill:\"#0A0A0A\"})" })' \
  --cmd 'get_screenshot({ nodeId: "hero" })'
```

Connect to a running Pencil desktop app instead (changes apply live):

```bash
python {SKILL_DIR}/scripts/pencil_repl.py --app desktop --in design.pen \
  --cmd 'batch_get({ patterns: [{ reusable: true }] })'
```

See `references/mcp-tools.md` for a tool-surface primer (verify against
`pencil interactive --help`).

### Mode C — Prompt-driven generation (delegates to Pencil's agent)

```bash
# New design from a prompt
pencil --out landing.pen --prompt "Create a SaaS landing page with hero, features, pricing" --agent claude

# Modify an existing design
pencil --in landing.pen --out landing-v2.pen --prompt "Add a dark footer with social links"

# Attach reference images (repeatable)
pencil --out ui.pen --prompt "Match this style" -f ./ref.png

# Export to an image
pencil --in landing.pen --export landing.png --export-scale 2 --export-type png
```

## Quick Reference

| Goal | Command |
| --- | --- |
| Preflight check | `python .../scripts/pencil_doctor.py` |
| Discover CLI + tools | `pencil --help` · `pencil interactive --help` |
| Headless MCP (when available) | `hermes mcp add pencil --command pencil --args start --output design.pen` |
| Read document schema + DSL | REPL: `get_editor_state({ include_schema: true })` |
| List design-system components | REPL: `batch_get({ patterns: [{ reusable: true }] })` |
| Mutate the design | REPL: `batch_design({ operations: "..." })` |
| Screenshot a node | REPL: `get_screenshot({ nodeId: "..." })` |
| Generate from a prompt | `pencil --out x.pen --prompt "..." --agent claude` |
| Export an image | `pencil --in x.pen --export x.png --export-type png` |
| List models | `pencil --list-models` |

## Procedure

1. **Preflight.** Run `pencil_doctor.py`. If `pencil` is missing, tell the user
   to `npm install -g @pencil.dev/cli`; if unauthenticated, `pencil login`. Note
   which path the doctor reports (`start` → Mode A, else `interactive` → Mode B).
2. **Discover.** Read `pencil --help` and `pencil interactive --help` (or
   `pencil start --help`) so you use this build's actual commands and tools.
3. **Pick the file.** New design → choose an output path ending in `.pen`.
   Editing → pass the existing file as `--in`.
4. **Inspect first.** Call `get_editor_state({ include_schema: true })` and,
   when editing, `batch_get` to learn the current node tree and the exact
   `batch_design` DSL for this Pencil version.
5. **Make changes.** Issue `batch_design` operations (Mode A/B) or a prompt
   (Mode C). Keep `batch_get` `readDepth` ≤ 3 to avoid flooding context.
6. **Verify** (see below), then **commit** the `.pen` file with the related
   code change so design and implementation move together in Git.

## Pitfalls

- **Auth is mandatory.** There is no offline path; both modes refuse to run
  without a `pencil login` session or `PENCIL_CLI_KEY`.
- **`batch_design` binding names are per-call.** A name you bind (e.g.
  `hero=I(...)`) only exists within that single `batch_design` call. To touch
  the node in a later call, reference it by its real node id (from the call's
  output or `batch_get`) — do **not** reuse the binding name across calls.
  Combine related inserts/edits into one `batch_design` where possible.
- **Pencil must be running for `--app` mode.** App mode connects over a local
  socket to the desktop app / extension. If it isn't running, use headless
  mode (`--out`, no `--app`).
- **Don't guess the `batch_design` DSL.** Operation letters/args can change
  between versions — read them from `get_editor_state`, not from memory.
- **Multi-line stdin: use `--cmds-file`, not inline escaping.** `batch_design`
  operations are strings of JS-object literals; piping them through a shell
  `printf`/`--cmd '...'` with escaped quotes is fragile. For anything
  non-trivial, write the REPL calls to a file and pass `--cmds-file`.
- **Mode C costs tokens and needs an agent key.** It runs Pencil's own LLM
  agent. For precise, free, deterministic edits prefer Mode A/B.
- **`.pen` is the source of truth, in your repo.** Treat it like code: review
  the diff, commit it alongside the implementation.

## Verification

- After a mutation, run `get_screenshot({ nodeId })`, save the PNG, and read it
  back with Hermes's `vision_analyze` / `read_file` to confirm the result.
- Re-run `batch_get` / `snapshot_layout` to confirm the node tree and bounds.
- Confirm the `.pen` file was written (headless `save()` succeeded) before
  committing.
