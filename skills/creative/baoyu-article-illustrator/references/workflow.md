# Detailed Workflow Procedures

## Step 1: Detect Reference Images

If the user provides reference images (local path or URL), the goal is to produce **textual descriptions** that can be embedded in prompts — `image_generate` doesn't accept reference-image inputs, and Hermes' text file tools can't read or write binaries.

**Tool rules**:

| Task | Tool | Notes |
|------|------|-------|
| Analyze a reference image | `vision_analyze` | Accepts URL or local path. Ask for style, palette, composition, subject. |
| Write the text description | `write_file` | Sidecar `.md` files only — never try to `write_file` a PNG/JPG. |
| (Optional) Keep a local copy of the binary | `terminal` | `cp "$src" "{output-dir}/references/NN-ref-{slug}.{ext}"` — purely for the record; the skill itself doesn't read the binary. |

| Input Type | Action |
|------------|--------|
| Image file path provided | `vision_analyze` → write sidecar `.md`. Optional `terminal cp` for a local record. |
| Image URL provided | `vision_analyze` with the URL → write sidecar `.md`. |
| Image in conversation (no path, no URL) | Ask via `clarify` for a path or URL, or for a verbal description. |
| User can't provide either | Extract style/palette verbally from the user → write `references/extracted-style.md`. Do NOT add `references:` to prompt frontmatter. |

**Procedure** (when a path/URL is available):

1. Call `vision_analyze(image_url=..., question="Describe the style, color palette (with hex approximations), composition, and subject so this can be used as a style/palette reference for another illustration.")`.
2. Write `{output-dir}/references/NN-ref-{slug}.md` via `write_file` with the description.
3. (Optional) Run `terminal` with `cp` (or `curl -sSL -o ...` for URLs) to keep a local binary copy. Not required by the skill.
4. Mark the reference in the outline with usage `direct` / `style` / `palette`. In Step 5.1 the description gets appended to the prompt body.

**Sidecar File Format**:
```yaml
---
ref_id: NN
source: "<original path or URL>"
local_copy: "NN-ref-{slug}.png"   # omit if no copy made
usage_hint: style                 # direct | style | palette
---
[vision_analyze description — colors, style, composition, subject]
```

---

## Step 2: Analyze

### 2.1 Determine Output Directory

| Input | Output Directory | Source-save path |
|-------|------------------|------------------|
| Article file path | `{article-dir}/imgs/` (default) | — (read article via `read_file`) |
| Pasted content | `illustrations/{topic-slug}/` (cwd) | `source-{slug}.{ext}` (save via `write_file`) |

If the user explicitly asked for a different layout (e.g., images in the article's folder, or an `illustrations/` subdirectory), honor that.

### 2.2 Analyze Content

| Analysis | Description |
|----------|-------------|
| Content type | Technical / Tutorial / Methodology / Narrative |
| Illustration purpose | information / visualization / imagination |
| Core arguments | 2-5 main points to visualize |
| Visual opportunities | Positions where illustrations add value |
| Recommended type | Based on content signals and purpose |
| Recommended density | Based on length and complexity |

Save analysis to `{output-dir}/analysis.md` using `write_file`.

### 2.3 Extract Core Arguments

- Main thesis
- Key concepts reader needs
- Comparisons/contrasts
- Framework/model proposed

**CRITICAL**: If the article uses metaphors (e.g., "电锯切西瓜"), do NOT illustrate literally. Visualize the **underlying concept**.

### 2.4 Identify Positions

**Illustrate**:
- Core arguments (REQUIRED)
- Abstract concepts
- Data comparisons
- Processes, workflows

**Do NOT Illustrate**:
- Metaphors literally
- Decorative scenes
- Generic illustrations

### 2.5 Plan Reference Image Usage (if analyzed in Step 1)

For each reference image (use the `vision_analyze` description from Step 1):

| Analysis | Description |
|----------|-------------|
| Visual characteristics | Style, colors, composition |
| Content/subject | What the reference depicts |
| Suitable positions | Which sections match this reference |
| Style match | Which illustration types/styles align |
| Usage recommendation | `direct` / `style` / `palette` |

| Usage | When to Use | How it's applied in Step 5.1 |
|-------|-------------|------------------------------|
| `direct` | Reference matches desired output closely | Paste the description (composition + subject + style + palette) into the prompt body |
| `style` | Extract visual style characteristics only | Append style traits to prompt body |
| `palette` | Extract color scheme only | Append extracted hex colors to prompt body |

Note: `image_generate` does not accept reference-image inputs under any usage type. Everything is mediated through the `vision_analyze` description.

---

## Step 3: Confirm Settings

Use the `clarify` tool. Since `clarify` handles one question at a time, ask the most important question first. Skip any question the user already answered in their request.

### Q1: Preset or Type (highest priority)

Based on Step 2 content analysis, recommend a preset first (sets both type & style). Look up [style-presets.md](style-presets.md) "Content Type → Preset Recommendations" table.

- [Recommended preset] — [brief: type + style + why]
- [Alternative preset] — [brief]
- Or choose type manually: infographic / scene / flowchart / comparison / framework / timeline / mixed

**If user picks a preset → skip Q3** (type & style both resolved).
**If user picks a type → Q3 is required.**

### Q2: Density

- minimal (1-2) — Core concepts only
- balanced (3-5) — Major sections
- per-section — At least 1 per section/chapter (Recommended)
- rich (6+) — Comprehensive coverage

### Q3: Style (skip if preset chosen in Q1)

Present Core Styles first:

- [Best compatible core style] (Recommended)
- [Other compatible core style 1]
- [Other compatible core style 2]
- Other (see full Style Gallery)

**Core Styles** (simplified selection):

| Core Style | Maps To | Best For |
|------------|---------|----------|
| `minimal-flat` | notion | General, knowledge sharing, SaaS |
| `sci-fi` | blueprint | AI, frontier tech, system design |
| `hand-drawn` | sketch/warm | Relaxed, reflective, casual |
| `editorial` | editorial | Processes, data, journalism |
| `scene` | warm/watercolor | Narratives, emotional, lifestyle |
| `poster` | screen-print | Opinion, editorial, cultural, cinematic |

Style selection based on Type × Style compatibility matrix ([styles.md](styles.md)).
**In Step 5**, read `styles/<style>.md` for visual elements and rendering rules.

### Q4: Palette (optional)

If the preset did not specify a palette, offer:

- Default (use style's built-in colors) (Recommended)
- `macaron` — soft pastel blocks on warm cream
- `warm` — warm earth tones, no cool colors
- `neon` — vibrant neon on dark backgrounds

**Skip if**: preset already resolved palette, or user specified a palette in the request.

See Palette Gallery in [styles.md](styles.md#palette-gallery) and full specs in `palettes/<palette>.md`.

### Q5: Image Text Language (only when ambiguous)

If the article language is different from the user's conversational language, ask which to use:
- Article language (match article content) (Recommended)
- User's conversational language

**Skip if**: languages match, or the user already specified in the request.

### Display Reference Usage (if references saved in Step 1)

When presenting the outline preview to the user, show reference assignments:

```
Reference Images:
| Ref | Filename | Recommended Usage |
|-----|----------|-------------------|
| 01 | 01-ref-diagram.png | direct → Illustration 1, 3 |
| 02 | 02-ref-chart.png | palette → Illustration 2 |
```

---

## Step 4: Generate Outline

Save as `{output-dir}/outline.md` using `write_file`:

```yaml
---
type: infographic
density: balanced
style: blueprint
image_count: 4
references:                    # Only if references provided
  - ref_id: 01
    filename: 01-ref-diagram.png
    description: "Technical diagram showing system architecture"
  - ref_id: 02
    filename: 02-ref-chart.png
    description: "Color chart with brand palette"
---

## Illustration 1

**Position**: [section] / [paragraph]
**Purpose**: [why this helps]
**Visual Content**: [what to show]
**Type Application**: [how type applies]
**References**: [01]                    # Optional: list ref_ids used
**Reference Usage**: direct             # direct | style | palette
**Filename**: 01-infographic-concept-name.png

## Illustration 2
...
```

**Backup rule**: If `outline.md` exists, rename to `outline-backup-YYYYMMDD-HHMMSS.md` before writing.

**Requirements**:
- Each position justified by content needs
- Type applied consistently
- Style reflected in descriptions
- Count matches density
- References assigned based on Step 2.5 analysis

---

## Step 5: Generate Prompts

**BLOCKING**: Every illustration must have a saved prompt file before any image is generated.

For each illustration in the outline:

1. **Create prompt file**: `{output-dir}/prompts/NN-{type}-{slug}.md` via `write_file`
2. **Include YAML frontmatter**:
   ```yaml
   ---
   illustration_id: 01
   type: infographic
   style: custom-flat-vector
   ---
   ```
3. **Load style specs**: Read `styles/<style>.md` (via `read_file`) for visual elements, style rules, and rendering instructions
4. **Load palette specs** (if palette specified): Read `palettes/<palette>.md` for colors and background. Palette colors **replace** the style's default Color Palette. If no palette specified, use the style's built-in colors.
5. **Follow type-specific template** from [prompt-construction.md](prompt-construction.md), using rendering from style + colors from palette (or style default)
6. **Prompt quality requirements** (all REQUIRED):
   - `Layout`: Describe overall composition (grid / radial / hierarchical / left-right / top-down)
   - `ZONES`: Describe each visual area with specific content, not vague descriptions
   - `LABELS`: Use **actual numbers, terms, metrics, quotes from the article** — NOT generic placeholders
   - `COLORS`: Specify hex codes from palette (or style default) with semantic meaning
   - `STYLE`: Describe line treatment, texture, mood, character rendering per style rules
   - `ASPECT`: Specify ratio (e.g., `16:9`)
7. **Apply defaults**: composition requirements, character rendering, text guidelines
8. **Backup rule**: If a prompt file exists, rename to `prompts/NN-{type}-{slug}-backup-YYYYMMDD-HHMMSS.md`

**CRITICAL - References in Frontmatter**:
- Only add `references` field if a sidecar `.md` description exists in `{output-dir}/references/`
- If style/palette was extracted verbally (no description file), append info to prompt BODY only
- Before writing frontmatter, confirm the sidecar exists (try `read_file` on the `.md`)

### 5.1 Process References (if analyzed in Step 1)

Read the `vision_analyze` description from the sidecar `references/NN-ref-{slug}.md` (via `read_file`) and embed it in the prompt body. `image_generate` never receives the binary.

| Usage | Action |
|-------|--------|
| `direct` | Paste the full reference description (composition, subject, style, palette) into the prompt body |
| `style` | Append only the style traits: "Style: clean lines, gradient backgrounds..." |
| `palette` | Append only the hex colors: "Colors: #E8756D coral, #7ECFC0 mint..." |

---

## Step 6: Generate Images

`image_generate` returns a JSON blob with a URL (`{"success": true, "image": "<url>"}`). It does NOT save a local file, does NOT accept an output path, and does NOT let the agent pick a backend/model. Treat the URL as a temporary artifact and download it explicitly.

For each prompt file:

1. Read the prompt file (via `read_file`) and extract the assembled prompt
2. Map the prompt's `ASPECT` to `image_generate`'s enum: `16:9` → `landscape`, `9:16` → `portrait`, `1:1` → `square`. Custom ratios → nearest named aspect.
3. Call `image_generate(prompt=<assembled>, aspect_ratio=<enum>)` and extract the `image` URL from the returned JSON.
4. **Backup rule**: If `{output-dir}/NN-{type}-{slug}.png` already exists, rename it via `terminal` (`mv "{output-dir}/NN-{type}-{slug}.png" "{output-dir}/NN-{type}-{slug}-backup-YYYYMMDD-HHMMSS.png"`) before writing.
5. Download the URL via `terminal`:
   ```bash
   curl -sSL -o "{output-dir}/NN-{type}-{slug}.png" "{image_url}"
   ```
   If `curl` is unavailable, fall back to `wget -qO "{output-dir}/NN-{type}-{slug}.png" "{image_url}"`.
6. Verify the file exists and has non-zero size (`terminal`: `test -s "{path}" && echo ok`).
7. On generation failure, retry `image_generate` once. On download failure, retry `curl` once with a longer timeout. Then log and continue.
8. After each generation, report "Generated X/N".

---

## Step 7: Finalize

### 7.1 Update Article

Insert after the corresponding paragraph, using the path relative to the article file:

| Input | Insert Path |
|-------|-------------|
| Article file path (default `imgs-subdir`) | `![description](imgs/NN-{type}-{slug}.png)` |
| Article file path (images alongside) | `![description](NN-{type}-{slug}.png)` |
| Article file path (`illustrations/` subdirectory) | `![description](illustrations/NN-{type}-{slug}.png)` |
| Pasted content | `![description](illustrations/{topic-slug}/NN-{type}-{slug}.png)` (relative to cwd) |

Alt text: concise description in the article's language.

### 7.2 Output Summary

```
Article Illustration Complete!

Article: [path]
Type: [type] | Density: [level] | Style: [style]
Location: [directory]
Images: X/N generated

Positions:
- 01-xxx.png → After "[Section]"
- 02-yyy.png → After "[Section]"

[If failures]
Failed:
- NN-zzz.png: [reason]
```
