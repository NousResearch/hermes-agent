---
name: baoyu-article-illustrator
description: "Article illustrations: type × style × palette consistency."
version: 1.57.0
author: 宝玉 (JimLiu)
license: MIT
platforms: [linux, macos, windows]
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

If the user supplies reference images (paths pasted inline, attachments, or a URL):

1. For each reference, call `vision_analyze` with the path/URL and a question asking for style, palette, composition, and subject. Record the returned description in `{output-dir}/references/NN-ref-{slug}.md` via `write_file`.
2. **Do not** try to copy the binary via `write_file` / `read_file` — those are text-only. If you want a local copy for the record, use `terminal` (`cp "$src" "{output-dir}/references/NN-ref-{slug}.{ext}"`). The skill itself never needs to read the binary; it works off the vision description.
3. Since `image_generate` doesn't take image inputs, the vision description is what gets embedded in prompts during Step 5.

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

For each prompt file:

1. Call `image_generate(prompt=..., aspect_ratio=...)`. `image_generate` returns a JSON result containing an image URL; it does NOT write to disk and does NOT accept an output path.
2. Map the prompt's `ASPECT` to `image_generate`'s enum: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`. Custom ratios → nearest named aspect.
3. Download the returned URL to `{output-dir}/NN-{type}-{slug}.png` via `terminal` (e.g. `curl -sSL -o "{output-dir}/NN-{type}-{slug}.png" "{url}"`).
4. On generation failure, auto-retry once.

Note: the underlying image-generation backend is user-configured (default: FAL FLUX 2 Klein 9B) and is NOT agent-selectable via `image_generate`. Do not write model names into prompts expecting them to route.

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
6. **`image_generate` returns a URL, not a local file** — always download via `terminal` (`curl`) before inserting local image paths into the article.
7. **No backend selection from the agent** — `image_generate` uses whatever model the user configured (default: FAL FLUX 2 Klein 9B). Don't write `"use <model> to generate this"` into prompts expecting it to route.
