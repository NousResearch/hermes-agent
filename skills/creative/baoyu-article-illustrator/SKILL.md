---
name: baoyu-article-illustrator
description: Analyzes article structure, identifies positions requiring visual aids, generates illustrations with Type × Style × Palette three-dimension approach. Use when user asks to "illustrate article", "add images", "generate images for article", or "为文章配图".
version: 1.57.0
metadata:
  openclaw:
    homepage: https://github.com/JimLiu/baoyu-skills#baoyu-article-illustrator
---

# Article Illustrator

Analyze articles, identify illustration positions, generate images with Type × Style × Palette consistency.

## User Input Tools

When this skill prompts the user, follow this tool-selection rule (priority order):

1. **Prefer built-in user-input tools** exposed by the current agent runtime — e.g., `AskUserQuestion`, `request_user_input`, `clarify`, `ask_user`, or any equivalent.
2. **Fallback**: if no such tool exists, emit a numbered plain-text message and ask the user to reply with the chosen number/answer for each question.
3. **Batching**: if the tool supports multiple questions per call, combine all applicable questions into a single call; if only single-question, ask them one at a time in priority order.

Concrete `AskUserQuestion` references below are examples — substitute the local equivalent in other runtimes.

## Image Generation Tools

When this skill needs to render an image:

- **Use whatever image-generation tool or skill is available** in the current runtime — e.g., Codex `imagegen`, Hermes `image_generate`, `baoyu-imagine`, or any equivalent the user has installed.
- **If multiple are available**, ask the user **once** at the start which to use (batch with any other initial questions).
- **If none are available**, tell the user and ask how to proceed.

**Prompt file requirement (hard)**: write each image's full, final prompt to a standalone file under `prompts/` (naming: `NN-{type}-[slug].md`) BEFORE invoking any backend. The backend receives the prompt file (or its content); the file is the reproducibility record and lets you switch backends without regenerating prompts.

Concrete tool names (`imagegen`, `image_generate`, `baoyu-imagine`) above are examples — substitute the local equivalents under the same rule.

## Reference Images

Users may supply reference images via `--ref <files...>` or by providing file paths / pasting images in conversation. Refs guide style, palette, composition, or subject for specific illustrations.

Full detection, storage, and processing rules are in [references/workflow.md](references/workflow.md) (Step 1.0 saves to `references/NN-ref-{slug}.{ext}`; Step 5.3 processes per-illustration usage `direct | style | palette`). When the chosen backend supports batch input, `direct`-usage entries in each prompt file's `references:` frontmatter should be propagated into its batch payload so backends can pass them through (e.g. `baoyu-imagine` accepts `ref` per task).

## Three Dimensions

| Dimension | Controls | Examples |
|-----------|----------|----------|
| **Type** | Information structure | infographic, scene, flowchart, comparison, framework, timeline |
| **Style** | Rendering approach | notion, warm, minimal, blueprint, watercolor, elegant |
| **Palette** | Color scheme (optional) | macaron, warm, neon — overrides style's default colors |

Combine freely: `--type infographic --style vector-illustration --palette macaron`

Or use presets: `--preset edu-visual` → type + style + palette in one flag. See [Style Presets](references/style-presets.md).

## Types

| Type | Best For |
|------|----------|
| `infographic` | Data, metrics, technical |
| `scene` | Narratives, emotional |
| `flowchart` | Processes, workflows |
| `comparison` | Side-by-side, options |
| `framework` | Models, architecture |
| `timeline` | History, evolution |

## Styles

See [references/styles.md](references/styles.md) for Core Styles, full gallery, and Type × Style compatibility.

## Workflow

```
- [ ] Step 1: Pre-check (EXTEND.md, references, config)
- [ ] Step 2: Analyze content
- [ ] Step 3: Confirm settings (AskUserQuestion)
- [ ] Step 4: Generate outline
- [ ] Step 5: Generate images
- [ ] Step 6: Finalize
```

### Step 1: Pre-check

**1.5 Load Preferences (EXTEND.md) ⛔ BLOCKING**

Check EXTEND.md in priority order — the first one found wins:

| Priority | Path | Scope |
|----------|------|-------|
| 1 | `.baoyu-skills/baoyu-article-illustrator/EXTEND.md` | Project |
| 2 | `${XDG_CONFIG_HOME:-$HOME/.config}/baoyu-skills/baoyu-article-illustrator/EXTEND.md` | XDG |
| 3 | `$HOME/.baoyu-skills/baoyu-article-illustrator/EXTEND.md` | User home |

| Result | Action |
|--------|--------|
| Found | Read, parse, display summary |
| Not found | ⛔ Run [first-time-setup](references/config/first-time-setup.md) |

Full procedures: [references/workflow.md](references/workflow.md#step-1-pre-check)

### Step 2: Analyze

| Analysis | Output |
|----------|--------|
| Content type | Technical / Tutorial / Methodology / Narrative |
| Purpose | information / visualization / imagination |
| Core arguments | 2-5 main points |
| Positions | Where illustrations add value |

**CRITICAL**: Metaphors → visualize underlying concept, NOT literal image.

Full procedures: [references/workflow.md](references/workflow.md#step-2-setup--analyze)

### Step 3: Confirm Settings ⚠️

**ONE AskUserQuestion, max 4 Qs. Q1-Q2 REQUIRED. Q3 required unless preset chosen.**

| Q | Options |
|---|---------|
| **Q1: Preset or Type** | [Recommended preset], [alt preset], or manual: infographic, scene, flowchart, comparison, framework, timeline, mixed |
| **Q2: Density** | minimal (1-2), balanced (3-5), per-section (Recommended), rich (6+) |
| **Q3: Style** | [Recommended], minimal-flat, sci-fi, hand-drawn, editorial, scene, poster, Other — **skip if preset chosen** |
| Q4: Palette | Default (style colors), macaron, warm, neon — **skip if preset includes palette or preferred_palette set** |
| Q5: Language | When article language ≠ EXTEND.md setting |

Full procedures: [references/workflow.md](references/workflow.md#step-3-confirm-settings-)

### Step 4: Generate Outline

Save `outline.md` with frontmatter (type, density, style, palette, image_count) and entries:

```yaml
## Illustration 1
**Position**: [section/paragraph]
**Purpose**: [why]
**Visual Content**: [what]
**Filename**: 01-infographic-concept-name.png
```

Full template: [references/workflow.md](references/workflow.md#step-4-generate-outline)

### Step 5: Generate Images

⛔ **BLOCKING: Prompt files MUST be saved before ANY image generation.** This is a hard requirement regardless of which backend is chosen — the prompt file is the reproducibility record.

1. For each illustration, create a prompt file per [references/prompt-construction.md](references/prompt-construction.md)
2. Save to `prompts/NN-{type}-{slug}.md` with YAML frontmatter
3. Prompts **MUST** use type-specific templates with structured sections (ZONES / LABELS / COLORS / STYLE / ASPECT)
4. LABELS **MUST** include article-specific data: actual numbers, terms, metrics, quotes
5. **DO NOT** pass ad-hoc inline prompts to `--prompt` without saving prompt files first
6. Select the backend via the `## Image Generation Tools` rule at the top: use whatever is available; if multiple, ask the user once. Do this once per session before any generation.
7. **Execution strategy**: When multiple illustrations have saved prompt files and the task is now plain generation, prefer the chosen backend's batch interface (if it offers one) over spawning subagents. Use subagents only when each image still needs separate prompt iteration or creative exploration. If the backend has no batch interface, generate sequentially.
8. Process references (`direct`/`style`/`palette`) per prompt frontmatter
9. Apply watermark if EXTEND.md enabled
10. Generate from saved prompt files; retry once on failure

Full procedures: [references/workflow.md](references/workflow.md#step-5-generate-images)

### Step 6: Finalize

Insert `![description]({relative-path}/NN-{type}-{slug}.png)` after paragraphs. Path computed relative to article file based on output directory setting.

```
Article Illustration Complete!
Article: [path] | Type: [type] | Density: [level] | Style: [style] | Palette: [palette or default]
Images: X/N generated
```

## Output Directory

Output directory is determined by `default_output_dir` in EXTEND.md (set during first-time setup):

| `default_output_dir` | Output Path | Markdown Insert Path |
|----------------------|-------------|----------------------|
| `imgs-subdir` (default) | `{article-dir}/imgs/` | `imgs/NN-{type}-{slug}.png` |
| `same-dir` | `{article-dir}/` | `NN-{type}-{slug}.png` |
| `illustrations-subdir` | `{article-dir}/illustrations/` | `illustrations/NN-{type}-{slug}.png` |
| `independent` | `illustrations/{topic-slug}/` | `illustrations/{topic-slug}/NN-{type}-{slug}.png` (relative to cwd) |

All auxiliary files (outline, prompts) are saved inside the output directory:

```
{output-dir}/
├── outline.md
├── prompts/
│   └── NN-{type}-{slug}.md
└── NN-{type}-{slug}.png
```

When input is **pasted content** (no file path), always uses `illustrations/{topic-slug}/` with `source-{slug}.{ext}` saved alongside.

**Slug**: 2-4 words, kebab-case. **Conflict**: append `-YYYYMMDD-HHMMSS`.

## Modification

| Action | Steps |
|--------|-------|
| Edit | Update prompt → Regenerate → Update reference |
| Add | Position → Prompt → Generate → Update outline → Insert |
| Delete | Delete files → Remove reference → Update outline |

## References

| File | Content |
|------|---------|
| [references/workflow.md](references/workflow.md) | Detailed procedures |
| [references/usage.md](references/usage.md) | Command syntax |
| [references/styles.md](references/styles.md) | Style gallery + Palette gallery |
| [references/style-presets.md](references/style-presets.md) | Preset shortcuts (type + style + palette) |
| [references/prompt-construction.md](references/prompt-construction.md) | Prompt templates |
| [references/config/first-time-setup.md](references/config/first-time-setup.md) | First-time setup |
