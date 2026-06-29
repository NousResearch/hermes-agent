---
name: 'step-02w-nb-compose-prompt'
description: 'Translate page specification into an effective AI image generation prompt for Nano Banana'

# File References
nextStepFile: './step-03-review-integrate.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-visual.md'
---

# Step 2W: Compose Nano Banana Prompt

## STEP GOAL:

Translate a page specification into an effective AI image generation prompt that balances creative exploration with spec adherence.

**Reference:** Load `../data/guides/NANO-BANANA-PROMPT-GUIDE.md` for compression strategy and examples.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Freya, a creative and thoughtful UX designer collaborating with the user
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design expertise and systematic thinking, user brings product vision and domain knowledge
- ✅ Maintain creative and thoughtful tone throughout

### Step-Specific Rules:

- 🎯 Focus on composing an effective generation prompt from page spec data
- 🚫 FORBIDDEN to generate without user confirming creative direction
- 💬 Approach: Extract, present, let user override, compose, generate, iterate
- 📋 Track all generations in agent experience file

## EXECUTION PROTOCOLS:

- 🎯 Extract image descriptions, gather creative direction, compose prompt, generate
- 💾 Log all generations to agent experience file
- 📖 Reference NANO-BANANA-PROMPT-GUIDE.md for compression strategy
- 🚫 FORBIDDEN to skip creative direction step

## CONTEXT BOUNDARIES:

- Available context: Page specification, visual direction, design tokens
- Focus: Prompt composition and image generation
- Limits: Follow prompt guide constraints (8192 char limit)
- Dependencies: Page specification must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 0. Start Generation Log

Create an agent experience file to track this visual generation session.

**File location:** `{output_folder}/_progress/agent-experiences/`
**Naming:** `{date}-visual-{page-name}.md`
**Example:** `2026-02-19-visual-1.1-hem.md`

```markdown
# Visual Generation: {page-name}

## Inputs
- **Page spec:** {path to page spec}
- **Visual direction:** {path or "not available"}
- **Design tokens:** {path or "not available"}

## Image Descriptions Extracted
{filled in step B}

## Creative Direction
{filled in step C -- user overrides recorded here}

## Generation Log
{each generation appended here in step H}
```

**If a previous generation log exists** for this page, read it for context — previous creative direction, successful prompts, and lessons learned.

### A. Load Inputs

1. Read the **page specification** from `{output_folder}/C-UX-Scenarios/` (or equivalent)
2. Read the **visual direction** from `{output_folder}/A-Product-Brief/` (if available)
3. Read the **design system tokens** from `{output_folder}/D-Design-System/` (if available)

### B. Extract Image Descriptions from Spec

Scan the page specification for all objects that contain image descriptions in their **Content** fields. These are natural prompt seeds.

**Look for patterns like:**
```markdown
**Content:**
- **Image:** [description of what the image shows]
```

**Collect each as a prompt seed:**

```
For each image object found:
  - Object ID: {id}
  - Section: {section name}
  - Image description: {the Content > Image text}
  - Alt text: {the primary language alt text}
```

**Present to user:**

```
I found {N} image descriptions in the spec:

1. [{section}] {object_id}
   "{image description}"

2. [{section}] {object_id}
   "{image description}"

...
```

### C. User Creative Direction

Present the extracted image descriptions and ask for overrides or enhancements.

```
Would you like to adjust any of these image descriptions before generation?

You can:
- Override: Replace the description entirely ("make it a dramatic sunset")
- Enhance: Add to the description ("...with warm golden light")
- Accept: Use as-is from the spec

Which images would you like to adjust?
```

Record any overrides. The final image description = spec description + user modifications.

### D. Choose Generation Scope

```
What would you like to generate?

[P] Full page mockup -- All sections in one image (layout overview)
[S] Section focus -- One section at high detail (e.g., just the hero)
[I] Image asset -- A single image described in the spec (e.g., hero photo, season card)
[W] Wireframe -- Clean digital wireframe from spec (recommended first step)
```

**Scope determines prompt strategy:**

| Scope | Prompt content | Best for |
|-------|---------------|----------|
| Full page | All sections compressed, layout focus | Understanding overall flow |
| Section focus | One section expanded, full detail | Detailed design of key areas |
| Image asset | Single image description + style context | Generating actual visual assets |
| Wireframe | Layout structure only, grayscale boxes | Layout validation, pipeline step 1 |

**Recommended pipeline for full-page mockups:**

If the user selects [P], recommend the **two-step wireframe pipeline** (see `NANO-BANANA-PROMPT-GUIDE.md`):
1. First generate a clean wireframe [W] from the spec
2. Then transform the wireframe into a polished mockup using edit mode

**Set expectations with user:** NB mockups are for layout exploration and mood visualization only. All text will be garbled, logos will be approximate, and some wireframe labels may leak through. For production-quality output, use the approved layout as reference for HTML/CSS prototypes or Figma.

### E. Reference Images (Optional)

```
Do you have reference images for visual conditioning? (up to 3)

These help Nano Banana understand the visual context:
- Workshop photos (actual facility)
- Brand logo
- Sketches or wireframes
- Mood board images
- Competitor screenshots

Provide file paths, or skip:
```

Map provided paths to `input_image_path_1`, `input_image_path_2`, `input_image_path_3`.

**Slot priority for edit mode:**
- **Slot 1 = layout source** -- the image whose structure you want to preserve (wireframe, sketch, or previous mockup)
- **Slot 2-3 = style references** -- photos, logos, or mood images that influence visual treatment
- In edit mode, slot 1 controls layout; in generate mode, all slots influence style/subject equally

**Auto-detect sketches:** Check `{output_folder}/C-UX-Scenarios/[scenario]/[page]/Sketches/` for hand-drawn wireframes. If found, offer to use as reference.

### F. Choose Creative Mode

```
How much creative freedom should the AI have?

[F] Faithful -- Clean UI mockup, close to spec layout and content
[E] Expressive -- Follows structure, takes creative liberties with visual treatment
[V] Vision -- Artistic concept, captures mood and brand essence
```

### G. Compose Prompt

Follow the compression strategy from `NANO-BANANA-PROMPT-GUIDE.md`.

**Assembly order:**

1. **Creative mode preamble** -- sets the generation style
2. **Page/section context** -- what this is, who it's for
3. **Layout structure** -- sections top-to-bottom (for full page/section scope)
4. **Image descriptions** -- with user overrides applied (for image asset scope)
5. **Design tokens** -- colors, fonts, key sizes
6. **Key content** -- headlines and CTA labels (primary language only)
7. **Brand atmosphere** -- mood words from visual direction

**Compose system_instruction** (max 512 chars):
- Brand voice + style direction
- Technical constraints (viewport, style)

**Set parameters:**

| Parameter | Value |
|-----------|-------|
| `aspect_ratio` | Full page scroll: `9:16`, Desktop viewport: `16:9`, Tablet: `3:4`, Image asset: per spec. **CRITICAL in edit mode:** always pin this or model may change it and lose content |
| `model_tier` | `pro` for first generation and wireframes, `flash` for quick iterations |
| `mode` | `generate` for new images/wireframes, `edit` for wireframe->mockup or refinement |
| `negative_prompt` | Generate: "lorem ipsum, placeholder, watermark". Edit from wireframe: "wireframe style, gray boxes, placeholder text, section labels" |
| `output_path` | `{output_folder}/D-Design-System/01-Visual-Design/design-concepts/` |

**Verify:** Total prompt must be under 8192 characters. If over:
1. Drop section descriptions (keep names only)
2. Drop secondary content (keep headlines, drop body text)
3. Drop footer details
4. Prioritize above-the-fold content

### H. Generate

Call `mcp__nanobanana__generate_image` with the assembled prompt, system instruction, parameters, and reference images.

Present the result to the user.

**Log to agent experience file:**

```markdown
### Generation {N} -- {timestamp}

**Scope:** {Full page / Section focus / Image asset}
**Creative mode:** {Faithful / Expressive / Vision}
**Aspect ratio:** {ratio}
**Model tier:** {flash / pro}
**Reference images:** {paths or "none"}

**System instruction:**
{the composed system instruction}

**Prompt:**
{the composed prompt}

**Output:** {path to generated image}
**User feedback:** {filled after review}
```

Update the agent experience file.

### I. Iterate

```
How does this look?

[A] Accept -- Save and proceed to review
[R] Refine -- Adjust the prompt and regenerate
[E] Edit -- Send this image back with targeted changes (edit mode)
[M] Mode change -- Try a different creative mode (F/E/V)
[S] Scope change -- Switch scope (full page / section / image asset / wireframe)
[N] New direction -- Start over with different creative direction
```

**On [R] Refine:** Ask what to change, update prompt, regenerate from scratch.
**On [E] Edit:** Use the generated image as `input_image_path_1` in edit mode with targeted instructions. Follow these rules:
- **Always pin `aspect_ratio`** to match the current image -- omitting it causes content loss
- **Be specific:** "Add a blue navigation bar with links Hem, Nyheter, Om oss, Hitta hit to the header" works better than "improve the header"
- **One change at a time:** targeted edits succeed; broad "make it better" instructions cause section loss
- **Adding works, removing doesn't:** edit mode handles adding new visible elements well, but struggles to remove or restructure existing elements

**On [M] Mode change:** Recompose with new mode preamble, regenerate.
**On [S] Scope change:** Return to step D, recompose prompt for new scope.
**On [N] New direction:** Return to step C for new creative overrides.

### Batch Mode: Multi-Page Generation

For projects with many similar pages (e.g., 11 vehicle type pages, 6 service pages, 4 seasonal articles), batch mode generates visuals across a page sequence.

**When to Use Batch Mode:**
- **Same layout, different content** -- Vehicle types, service pages, article pages
- **Shared design system** -- All pages use the same colors, fonts, component patterns
- **Image asset sequences** -- Hero images for a set of similar pages

**When NOT to Use Batch Mode:**
- Pages with significantly different layouts
- First-time visual exploration (establish template first)
- Pages where creative direction varies significantly

### J. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Review & Integrate | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#j-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has accepted a generated visual and selected an option from the menu will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Generation log created in agent experiences
- Image descriptions extracted from spec
- User creative direction captured
- Prompt composed within 8192 char limit
- Image generated and presented
- Generation logged to agent experience file
- User accepted or iterated to satisfaction

### ❌ SYSTEM FAILURE:

- Generating without user creative direction
- Exceeding prompt character limit
- Not logging generations to agent experience file
- Not presenting iteration options
- Skipping reference image auto-detection

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
