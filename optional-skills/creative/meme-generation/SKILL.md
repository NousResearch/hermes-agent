---
name: meme-generation
description: Generate real meme images by picking a template and overlaying text with Pillow. Produces actual .png meme files.
version: 2.0.0
author: adanaleycio
license: MIT
metadata:
  hermes:
    tags: [creative, memes, humor, images]
    related_skills: [ascii-art, generative-widgets]
    category: creative
---

# Meme Generation

Generate actual meme images from a topic. Picks a template, writes captions, and renders a real .png file with text overlay.

## When to Use

- User asks you to make or generate a meme
- User wants a meme about a specific topic, situation, or frustration
- User says "meme this" or similar

## Available Templates

The script supports **any of the ~100 popular imgflip templates** by name or ID, plus 10 curated templates with hand-tuned text positioning.

### Curated Templates (custom text placement)

| ID | Name | Fields | Best for |
|----|------|--------|----------|
| `this-is-fine` | This is Fine | top, bottom | chaos, denial |
| `drake` | Drake Hotline Bling | reject, approve | rejecting/preferring |
| `distracted-boyfriend` | Distracted Boyfriend | distraction, current, person | temptation, shifting priorities |
| `two-buttons` | Two Buttons | left, right, person | impossible choice |
| `expanding-brain` | Expanding Brain | 4 levels | escalating irony |
| `change-my-mind` | Change My Mind | statement | hot takes |
| `woman-yelling-at-cat` | Woman Yelling at Cat | woman, cat | arguments |
| `one-does-not-simply` | One Does Not Simply | top, bottom | deceptively hard things |
| `grus-plan` | Gru's Plan | step1-3, realization | plans that backfire |
| `batman-slapping-robin` | Batman Slapping Robin | robin, batman | shutting down bad ideas |

### Dynamic Templates (from imgflip API)

Any template not in the curated list can be used by name or imgflip ID. These get smart default text positioning (top/bottom for 2-field, evenly spaced for 3+). Search with:
```bash
python "$SKILL_DIR/scripts/generate_meme.py" --search "disaster"
```

## Procedure

### Mode 1: Classic Template (default)

1. Read the user's topic and identify the core dynamic (chaos, dilemma, preference, irony, etc.)
2. Pick the template that best matches. Use the "Best for" column, or search with `--search`.
3. Write short captions for each field (8-12 words max per field, shorter is better).
4. Find the skill's script directory:
   ```
   SKILL_DIR=$(dirname "$(find ~/.hermes/skills -path '*/meme-generation/SKILL.md' 2>/dev/null | head -1)")
   ```
5. Run the generator:
   ```bash
   python "$SKILL_DIR/scripts/generate_meme.py" <template_id> /tmp/meme.png "caption 1" "caption 2" ...
   ```
6. Return the image with `MEDIA:/tmp/meme.png`

### Mode 2: Custom AI Image (when image_generate is available)

Use this when no classic template fits, or when the user wants something original.

1. Write the captions first.
2. Use `image_generate` to create a scene that matches the meme concept. Do NOT include any text in the image prompt — text will be added by the script. Describe only the visual scene.
3. Find the generated image path from the image_generate result URL. Download it to a local path if needed.
4. Run the script with `--image` to overlay text, choosing a mode:
   - **Overlay** (text directly on image, white with black outline):
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png /tmp/meme.png "top text" "bottom text"
     ```
   - **Bars** (black bars above/below with white text — cleaner, always readable):
     ```bash
     python "$SKILL_DIR/scripts/generate_meme.py" --image /path/to/scene.png --bars /tmp/meme.png "top text" "bottom text"
     ```
   Use `--bars` when the image is busy/detailed and text would be hard to read on top of it.
5. **Verify with vision** (if `vision_analyze` is available): Check the result looks good:
   ```
   vision_analyze(image_url="/tmp/meme.png", question="Is the text legible and well-positioned? Does the meme work visually?")
   ```
   If the vision model flags issues (text hard to read, bad placement, etc.), try the other mode (switch between overlay and bars) or regenerate the scene.
6. Return the image with `MEDIA:/tmp/meme.png`

## Examples

**"debugging production at 2 AM":**
```bash
python generate_meme.py this-is-fine /tmp/meme.png "SERVERS ARE ON FIRE" "This is fine"
```

**"choosing between sleep and one more episode":**
```bash
python generate_meme.py drake /tmp/meme.png "Getting 8 hours of sleep" "One more episode at 3 AM"
```

**"the stages of a Monday morning":**
```bash
python generate_meme.py expanding-brain /tmp/meme.png "Setting an alarm" "Setting 5 alarms" "Sleeping through all alarms" "Working from bed"
```

## Listing Templates

To see all available templates:
```bash
python generate_meme.py --list
```

## Pitfalls

- Keep captions SHORT. Memes with long text look terrible.
- Match the number of text arguments to the template's field count.
- Pick the template that fits the joke structure, not just the topic.
- Do not generate hateful, abusive, or personally targeted content.
- The script caches template images in `scripts/.cache/` after first download.

## Verification

The output is correct if:
- A .png file was created at the output path
- Text is legible (white with black outline) on the template
- The joke lands — caption matches the template's intended structure
- File can be delivered via MEDIA: path
