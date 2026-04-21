# Port Notes — baoyu-comic

Ported from [JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills) v1.56.1.

## Changes from upstream

### SKILL.md adaptations

| Change | Upstream | Hermes |
|--------|----------|--------|
| Metadata namespace | `openclaw` | `hermes` (with `tags` + `homepage`) |
| Trigger | Slash commands / CLI flags | Natural language skill matching |
| User config | EXTEND.md file (project/user/XDG paths) | Removed — not part of Hermes infra |
| User prompts | `AskUserQuestion` (batched) | `clarify` tool (one question at a time) |
| Image generation | baoyu-imagine (Bun/TypeScript) | `image_generate` tool |
| Platform support | Linux/macOS/Windows/WSL/PowerShell | Linux/macOS only |
| File operations | Generic instructions | Hermes file tools (`write_file`, `read_file`) |
| Runtime abstraction | `${BUN_X}` resolution | Direct `bun` invocation for `scripts/merge-to-pdf.ts` |

### Structural removals

- **`references/config/` directory** (removed entirely):
  - `first-time-setup.md` — blocking first-time setup flow for EXTEND.md
  - `preferences-schema.md` — EXTEND.md YAML schema
  - `watermark-guide.md` — watermark config (tied to EXTEND.md)
- **Workflow Step 1.1** — "Load Preferences (EXTEND.md)" section removed from `workflow.md`; steps 1.2/1.3 renumbered to 1.1/1.2.
- **Generic "User Input Tools" and "Image Generation Tools" preambles** — SKILL.md no longer lists fallback rules for multiple possible tools; it references `clarify` and `image_generate` directly.

### SKILL.md reductions

- CLI option columns (`--art`, `--tone`, `--layout`, `--aspect`, `--lang`, `--ref`, `--storyboard-only`, `--prompts-only`, `--images-only`, `--regenerate`) converted to plain-English option descriptions.
- Preset files (`presets/*.md`) and `ohmsha-guide.md`: `` `--style X` `` / `` `--art X --tone Y` `` shorthand rewritten to `art=X, tone=Y` + natural-language references.
- `partial-workflows.md`: per-skill slash command invocations rewritten as user-intent cues.
- `auto-selection.md`: priority order dropped the EXTEND.md tier.
- `analysis-framework.md`: language-priority comment updated (user option → conversation → source).

### What was preserved verbatim

- All 6 art-style definitions (`references/art-styles/`)
- All 7 tone definitions (`references/tones/`)
- All 7 layout definitions (`references/layouts/`)
- Core templates: `character-template.md`, `storyboard-template.md`, `base-prompt.md`
- Preset bodies (only the first few intro lines adapted; special rules unchanged)
- `scripts/merge-to-pdf.ts` (Bun-compatible on Linux/macOS)
- Author, version, homepage attribution

## Syncing with upstream

To pull upstream updates:

```bash
# Compare versions
curl -sL https://raw.githubusercontent.com/JimLiu/baoyu-skills/main/skills/baoyu-comic/SKILL.md | head -5
# Look for the version: line

# Diff a reference file
diff <(curl -sL https://raw.githubusercontent.com/JimLiu/baoyu-skills/main/skills/baoyu-comic/references/art-styles/manga.md) \
     references/art-styles/manga.md
```

Art-style, tone, and layout reference files can usually be overwritten directly (they're upstream-verbatim). `SKILL.md`, `references/workflow.md`, `references/partial-workflows.md`, `references/auto-selection.md`, `references/analysis-framework.md`, `references/ohmsha-guide.md`, and `references/presets/*.md` must be manually merged since they contain Hermes-specific adaptations.
