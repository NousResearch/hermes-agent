# Port Notes — baoyu-article-illustrator

Ported from [JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills) v1.57.0.

## Changes from upstream

`SKILL.md`, `references/workflow.md`, `references/usage.md`, `references/style-presets.md`, `references/styles.md`, `references/prompt-construction.md`, and `prompts/system.md` were adapted. The 23 style files and 4 palette files are verbatim copies. The `references/config/` directory was removed entirely.

### Adaptations

| Change | Upstream | Hermes |
|--------|----------|--------|
| Metadata namespace | `openclaw` | `hermes` |
| Trigger | `/baoyu-article-illustrator` slash command + CLI flags | Natural language skill matching |
| User config | EXTEND.md (project/user/XDG paths) + first-time-setup | Removed — not part of Hermes infra |
| User prompts | `AskUserQuestion` (batched, multi-question) | `clarify` tool (one question at a time) |
| Image generation | `baoyu-imagine` (Bun/TypeScript, multi-provider, accepts `--ref`, writes to local path) | `image_generate` (returns URL only; agent downloads via `terminal`/`curl`) |
| Backend selection | User picks provider via CLI flags | Not agent-selectable — `image_generate` uses the user-configured FAL model. Removed hardcoded "nano banana pro" line from `prompts/system.md`. |
| Reference images | Passed to backend via `--ref`, copied via shell | `vision_analyze` extracts a textual description (binary never touched by `write_file`/`read_file`); description is embedded in prompts. Optional `terminal cp` for a local record. |
| Platform support | Linux/macOS/Windows/WSL/PowerShell | Linux/macOS only |
| File operations | Bash commands | Hermes file tools: `write_file`/`read_file` for text, `terminal` for binaries and URL downloads, `vision_analyze` for reading images |
| Watermark | Driven by EXTEND.md `watermark.enabled` | Optional — user asks for it per-article |
| Output directory | EXTEND.md `default_output_dir` (imgs-subdir / same-dir / illustrations-subdir / independent) | Defaults based on input type; user overrides in request |

### What was preserved

- Type × Style × Palette three-dimension framework
- All style definitions (23 files, verbatim)
- All palette definitions (4 files, verbatim)
- Core reference files (workflow, prompt-construction, styles, style-presets) — adapted for Hermes tooling
- Core principles and workflow structure (analyze → confirm → outline → prompts → generate)
- Prompt-file-as-reproducibility-record discipline
- Author, version, homepage attribution

## Syncing with upstream

To pull upstream updates:

```bash
# Compare versions
curl -sL https://raw.githubusercontent.com/JimLiu/baoyu-skills/main/skills/baoyu-article-illustrator/SKILL.md | head -5
# Look for version: line

# Diff style/palette files (safe to overwrite — unchanged from upstream)
diff <(curl -sL https://raw.githubusercontent.com/JimLiu/baoyu-skills/main/skills/baoyu-article-illustrator/references/styles/blueprint.md) references/styles/blueprint.md
```

`references/styles/*` and `references/palettes/*` can be overwritten directly. `SKILL.md`, `references/workflow.md`, `references/usage.md`, `references/style-presets.md`, `references/styles.md`, `references/prompt-construction.md`, and `prompts/system.md` must be manually merged since they contain Hermes-specific adaptations (tool wiring, backend neutrality, removed EXTEND.md references).
