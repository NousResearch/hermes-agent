---
name: baoyu-article-illustrator
description: Analyzes article structure, identifies positions requiring visual aids, generates illustrations with Type × Style × Palette three-dimension approach. Use when user asks to "illustrate article", "add images", "generate images for article", or "为文章配图".
version: 1.57.0
author: 宝玉 (JimLiu)
license: MIT
metadata:
  hermes:
    tags: [article-illustration, creative, image-generation]
    category: creative
    homepage: https://github.com/JimLiu/baoyu-skills#baoyu-article-illustrator
---

# Article Illustrator

Adapted from [baoyu-article-illustrator](https://github.com/JimLiu/baoyu-skills) for Hermes Agent's tool ecosystem.

Analyze articles, identify illustration positions, generate images with **Type × Style × Palette** consistency.

## When to Use

Trigger this skill when the user asks to illustrate an article, add images to an article, generate illustrations for content, or uses phrases like "为文章配图", "illustrate article", or "add images". The user provides an article (file path or pasted content) and optionally specifies type, style, palette, or density.

## Three Dimensions

| Dimension | Controls | Examples |
|-----------|----------|----------|
| **Type** | Information structure | infographic, scene, flowchart, comparison, framework, timeline |
| **Style** | Rendering approach | notion, warm, minimal, blueprint, watercolor, elegant |
| **Palette** | Color scheme (optional) | macaron, warm, neon — overrides style's default colors |

Combine freely: `type=infographic, style=vector-illustration, palette=macaron`.

Or use presets: `edu-visual` → type + style + palette in one shot. See [style-presets.md](references/style-presets.md).

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

See [references/styles.md](references/styles.md) for Core Styles, the full gallery, and Type × Style compatibility.

## Output Structure

```
{output-dir}/
├── source-{slug}.{ext}    # Only for pasted content
├── outline.md
├── prompts/
│   └── NN-{type}-{slug}.md
└── NN-{type}-{slug}.png
```

**Default output directory**:

| Input | Output Directory | Markdown Insert Path |
|-------|------------------|----------------------|
| Article file path | `{article-dir}/imgs/` | `imgs/NN-{type}-{slug}.png` |
| Pasted content | `illustrations/{topic-slug}/` (cwd) | `illustrations/{topic-slug}/NN-{type}-{slug}.png` |

If the user asks for a different layout (e.g., images alongside the article, or a `illustrations/` subdirectory), honor that.

**Slug**: 2-4 words, kebab-case. **Conflict**: append `-YYYYMMDD-HHMMSS`.

## Core Principles

- **Visualize concepts, not metaphors** — if the article uses a metaphor (e.g., "电锯切西瓜"), illustrate the underlying concept, not the literal image.
- **Labels use article data** — actual numbers, terms, and quotes from the article, not generic placeholders.
- **Prompt files are reproducibility records** — every illustration must have a saved prompt file under `prompts/` before any image is generated.
- **Strip secrets** — scan source content for API keys, tokens, or credentials before writing anything to disk.

## Workflow

```
- [ ] Step 1: Detect reference images (if provided)
- [ ] Step 2: Analyze content
- [ ] Step 3: Confirm settings (clarify tool, one question at a time)
- [ ] Step 4: Generate outline
- [ ] Step 5: Generate prompts
- [ ] Step 6: Generate images (image_generate)
- [ ] Step 7: Finalize
```

### Step 1: Detect Reference Images

If the user supplies reference images (paths pasted inline, attachments, or a list of files):

1. Copy each reference to `{output-dir}/references/NN-ref-{slug}.{ext}` using `write_file`.
2. Create a sidecar `NN-ref-{slug}.md` describing the reference.
3. If the user described a reference but can't provide a file path, extract style/palette verbally and record under `references/extracted-style.md` — do NOT add a `references:` field to prompt frontmatter in that case.

Full procedures: [references/workflow.md](references/workflow.md#step-1-detect-reference-images).

### Step 2: Analyze

| Analysis | Output |
|----------|--------|
| Content type | Technical / Tutorial / Methodology / Narrative |
| Purpose | information / visualization / imagination |
| Core arguments | 2-5 main points |
| Positions | Where illustrations add value |

Read source (file path → `read_file`, or pasted text) and write the analysis to `{output-dir}/analysis.md` using `write_file`.

Full procedures: [references/workflow.md](references/workflow.md#step-2-analyze).

### Step 3: Confirm Settings

Use the `clarify` tool. Since `clarify` handles one question at a time, ask the most important question first. Skip any question whose answer is already present in the user's request.

| Order | Question | Options |
|-------|----------|---------|
| Q1 | **Preset or Type** | [Recommended preset], [alt preset], or manual: infographic, scene, flowchart, comparison, framework, timeline, mixed |
| Q2 | **Density** | minimal (1-2), balanced (3-5), per-section (Recommended), rich (6+) |
| Q3 | **Style** *(skip if preset chosen in Q1)* | [Recommended], minimal-flat, sci-fi, hand-drawn, editorial, scene, poster |
| Q4 | **Palette** *(optional)* | Default (style colors), macaron, warm, neon |
| Q5 | **Language** *(only if article language is ambiguous)* | article language / user language |

Don't ask more than 2-3 `clarify` questions in a row. If the user already specified these in their request, skip entirely.

Full procedures: [references/workflow.md](references/workflow.md#step-3-confirm-settings).

### Step 4: Generate Outline → `outline.md`

Save `{output-dir}/outline.md` using `write_file` with frontmatter (type, density, style, palette, image_count) and one entry per illustration:

```yaml
## Illustration 1
**Position**: [section/paragraph]
**Purpose**: [why]
**Visual Content**: [what to show]
**Filename**: 01-infographic-concept-name.png
```

Full template: [references/workflow.md](references/workflow.md#step-4-generate-outline).

### Step 5: Generate Prompts

**BLOCKING**: Every illustration must have a saved prompt file before any image is generated — the prompt file is the reproducibility record.

For each illustration:

1. Create a prompt file per [references/prompt-construction.md](references/prompt-construction.md).
2. Save to `{output-dir}/prompts/NN-{type}-{slug}.md` using `write_file` with YAML frontmatter.
3. Prompts MUST use type-specific templates with structured sections (ZONES / LABELS / COLORS / STYLE / ASPECT).
4. LABELS MUST include article-specific data: actual numbers, terms, metrics, quotes.
5. Process references (`direct`/`style`/`palette`) per prompt frontmatter — for `direct` usage, embed a textual description of the reference in the prompt (since `image_generate` doesn't take reference-image inputs).

### Step 6: Generate Images

Use the `image_generate` tool with the assembled prompt from each prompt file.

- Map aspect ratio to `image_generate` format: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`. For custom ratios, pick the closest named aspect.
- Generate sequentially through the outline. On failure, auto-retry once.
- Save each image to `{output-dir}/NN-{type}-{slug}.png`.

### Step 7: Finalize

Insert `![description]({relative-path}/NN-{type}-{slug}.png)` after the corresponding paragraph. Alt text: concise description in the article's language.

Report:

```
Article Illustration Complete!
Article: [path] | Type: [type] | Density: [level] | Style: [style] | Palette: [palette or default]
Images: X/N generated
```

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
| [references/usage.md](references/usage.md) | Invocation examples |
| [references/styles.md](references/styles.md) | Style gallery + Palette gallery |
| [references/style-presets.md](references/style-presets.md) | Preset shortcuts (type + style + palette) |
| [references/prompt-construction.md](references/prompt-construction.md) | Prompt templates |

## Pitfalls

1. **Data integrity is paramount** — never summarize, paraphrase, or alter source statistics. "73% increase" stays "73% increase".
2. **Strip secrets** — scan source content for API keys, tokens, or credentials before including in any output file.
3. **Don't illustrate metaphors literally** — visualize the underlying concept.
4. **Prompt files are mandatory** — no image generation without a saved prompt file. The file is what lets you regenerate or switch backends later.
5. **`image_generate` aspect ratios** — the tool supports `landscape`, `portrait`, and `square`. Custom ratios map to the nearest option.
