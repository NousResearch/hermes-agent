---
name: pattern-learning
description: Learn from conversation patterns and create skills automatically
---

# Pattern Learning

Automatically analyze conversation logs and create skills from repeated patterns.

## When to Use

- After completing a complex task that involved multiple iterations
- When you notice you've solved a similar problem before
- When the user corrects you on something you should remember

## How It Works

1. **Detect Patterns**: Monitor for repeated error-fix cycles or successful workflows
2. **Extract Knowledge**: Pull out the key insight, command, or approach
3. **Create Skill**: Save as a reusable skill with trigger conditions

## Example

If you notice you've fixed FFmpeg black screen issues 3 times by checking aspect ratios:

```yaml
# Auto-creates: ~/.hermes/skills/ffmpeg-aspect-ratio-fix
---
name: ffmpeg-aspect-ratio-fix
description: Fix FFmpeg black screen by ensuring correct aspect ratio
---
# FFmpeg Aspect Ratio Fix

When video output has black bars or gray screen:

1. Check source dimensions: `ffprobe -v error -select_streams v:0 -show_entries stream=width,height`
2. Scale to target: `ffmpeg -i input.mp4 -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1" output.mp4`
3. Never use non-integer pixel values in crop filter
```

## Automatic Triggers

- Error occurs 3+ times → suggest creating a fix skill
- Complex workflow completed successfully → suggest saving as skill
- User provides correction → update relevant skill or create new one

## Integration

Works with:
- `session_search` to find historical patterns
- `skill_manage` to create/update skills
- Memory system to track pattern frequency
