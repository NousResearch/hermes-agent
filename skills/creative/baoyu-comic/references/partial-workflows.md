# Partial Workflows

Options to run specific parts of the workflow.

## Options Summary

| Option | Steps Executed | Output |
|--------|----------------|--------|
| `--storyboard-only` | 1-3 | `storyboard.md` + `characters/` |
| `--prompts-only` | 1-5 | + `prompts/*.md` |
| `--images-only` | 7-9 | + images + PDF |
| `--regenerate N` | 7 (partial) | Specific page(s) + PDF |

---

## Using `--storyboard-only`

Generate storyboard and characters without prompts or images:

```bash
/baoyu-comic content.md --storyboard-only
```

**Workflow**: Steps 1-3 only (stop after storyboard + characters)

**Output**:
- `analysis.md`
- `storyboard.md`
- `characters/characters.md`

**Use case**: Review and edit the storyboard before generating images. Useful for:
- Getting feedback on the narrative structure
- Making manual adjustments to panel layouts
- Defining custom characters

---

## Using `--prompts-only`

Generate storyboard, characters, and prompts without images:

```bash
/baoyu-comic content.md --prompts-only
```

**Workflow**: Steps 1-5 (generate prompts, skip images)

**Output**:
- `analysis.md`
- `storyboard.md`
- `characters/characters.md`
- `prompts/*.md`

**Use case**: Review and edit prompts before image generation. Useful for:
- Fine-tuning image generation prompts
- Ensuring visual consistency before committing to generation
- Making style adjustments at the prompt level

---

## Using `--images-only`

Generate images from existing prompts (starts at Step 7):

```bash
/baoyu-comic comic/topic-slug/ --images-only
```

**Workflow**: Skip to Step 7, then 8-9

**Prerequisites** (must exist in directory):
- `prompts/` directory with page prompt files
- `storyboard.md` with style information
- `characters/characters.md` with character definitions

**Output**:
- `characters/characters.png` (if not exists)
- `NN-{cover|page}-[slug].png` images
- `{topic-slug}.pdf`

**Use case**: Re-generate images after editing prompts. Useful for:
- Recovering from failed image generation
- Trying different image generation settings
- Regenerating after manual prompt edits

---

## Using `--regenerate`

Regenerate specific pages only:

```bash
# Single page
/baoyu-comic comic/topic-slug/ --regenerate 3

# Multiple pages
/baoyu-comic comic/topic-slug/ --regenerate 2,5,8

# Cover page
/baoyu-comic comic/topic-slug/ --regenerate 0
```

**Workflow**:
1. Read existing prompts for specified pages
2. Regenerate images only for those pages
3. Regenerate PDF

**Prerequisites** (must exist):
- `prompts/NN-{cover|page}-[slug].md` for specified pages
- `characters/characters.png` (for reference)

**Output**:
- Regenerated `NN-{cover|page}-[slug].png` for specified pages
- Updated `{topic-slug}.pdf`

**Use case**: Fix specific pages without regenerating entire comic. Useful for:
- Fixing a single problematic page
- Iterating on specific visuals
- Regenerating pages after prompt edits

**Page numbering**:
- `0` = Cover page
- `1-N` = Content pages
