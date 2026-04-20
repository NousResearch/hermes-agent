# Port Notes — baoyu-article-illustrator

Ported from [JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills) v1.57.0.

## Changes from upstream

`SKILL.md`, `references/workflow.md`, `references/usage.md`, `references/style-presets.md`, `references/styles.md`, and `references/prompt-construction.md` were adapted. The 21 style files, 4 palette files, and `prompts/system.md` are verbatim copies. The `references/config/` directory was removed entirely.

### Adaptations

| Change | Upstream | Hermes |
|--------|----------|--------|
| Metadata namespace | `openclaw` | `hermes` |
| Trigger | `/baoyu-article-illustrator` slash command + CLI flags | Natural language skill matching |
| User config | EXTEND.md (project/user/XDG paths) + first-time-setup | Removed — not part of Hermes infra |
| User prompts | `AskUserQuestion` (batched, multi-question) | `clarify` tool (one question at a time) |
| Image generation | `baoyu-imagine` (Bun/TypeScript, multi-provider, accepts `--ref`) | `image_generate` tool (describes references in prompt text) |
| Platform support | Linux/macOS/Windows/WSL/PowerShell | Linux/macOS only |
| File operations | Bash commands | Hermes file tools (`write_file`, `read_file`) |
| Watermark | Driven by EXTEND.md `watermark.enabled` | Optional — user asks for it per-article |
| Output directory | EXTEND.md `default_output_dir` (imgs-subdir / same-dir / illustrations-subdir / independent) | Defaults based on input type; user overrides in request |

### What was preserved

- Type × Style × Palette three-dimension framework
- All style definitions (23 files)
- All palette definitions (4 files)
- Core reference files (workflow, prompt-construction, styles, style-presets)
- `prompts/system.md` (generation prompt template)
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

`references/styles/*`, `references/palettes/*`, and `prompts/system.md` can be overwritten directly. `SKILL.md`, `references/workflow.md`, `references/usage.md`, `references/style-presets.md`, `references/styles.md`, and `references/prompt-construction.md` must be manually merged since they contain Hermes-specific adaptations.
