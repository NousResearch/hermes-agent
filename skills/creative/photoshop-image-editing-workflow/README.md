# Photoshop Image Editing Workflow for Hermes

This folder adds a Hermes built-in creative skill for Photoshop-assisted image editing.

The skill teaches Hermes how to handle image-editing requests where the output should be measured against an original design and then opened or reviewed in Adobe Photoshop.

## What problem does this solve?

Image-editing tasks often fail when an agent only says "make it similar" and relies on visual guessing. Small differences in typography, font weight, line spacing, text placement, or anti-aliasing can make a generated image feel wrong.

This skill makes the workflow more systematic:

1. Preserve the original image.
2. Measure the original design before editing.
3. Extract typography and layout criteria.
4. For simple edits, generate or edit a candidate image.
5. For complex images, decompose the design into background, main object, cards, icons, CTA, proof assets, and editable text layers.
6. Compare candidates against the source with a QA sheet and report.
7. Open the final candidate in Windows Photoshop from WSL.
8. Record the production result in Obsidian or another project log.

## Key idea

Do not edit by eyeballing alone.

Use the source image as the design specification. Measure the original first, then edit.

For example, if an original image has two Korean text lines where the second line is visually heavier than the first, the agent should not render both replacement lines with the same weight. It should measure each line separately and preserve the hierarchy:

- Line 1: lighter bold text
- Line 2: heavier emphasis text

## Example use cases

Use this skill when a user asks Hermes to:

- edit an image with Photoshop
- replace text in an existing marketing/PDP image
- match a font or font weight from a reference image
- compare an edited image against the original
- open a generated image in Windows Photoshop for final review
- create a QA report for an image-editing task
- document a production image workflow in Obsidian

## Typical environment

The workflow is designed for Hermes running in WSL while Adobe Photoshop runs on Windows.

Typical paths:

```text
WSL project path:      /mnt/c/Users/<WindowsUser>/Documents/...
Windows Photoshop exe: C:\Program Files\Adobe\Adobe Photoshop 2025\Photoshop.exe
Windows font folder:   /mnt/c/Windows/Fonts
```

Hermes can convert file paths with `wslpath`, open the image via PowerShell, and verify Photoshop with `Get-Process Photoshop`.

## Workflow summary

### Simple edit path

```text
source image
  ↓
measure typography/layout/color
  ↓
create edited candidate
  ↓
generate QA sheet + QA report
  ↓
open final candidate in Photoshop
  ↓
verify file/process state
  ↓
record result in Obsidian/project log
```

### Complex composite path

```text
benchmark/source image
  ↓
Director: define scope and complexity level
  ↓
Reference Analyst + Layout Architect + Typography Agent
  ↓
Asset Decomposer: split background, main object, icons, cards, CTA, proof assets, text
  ↓
Prompt Engineer + Image Generation Worker: generate only independent visual assets
  ↓
Photoshop Compositor: assemble named PSD layers
  ↓
QA Agent: inspect generated assets, final composite, mobile readability, and claim risk
  ↓
Ops Logger: save prompts, files, QA report, and production log
```

For complex work, the key rule is: GPT Image models create visual components, Photoshop owns exact Korean text, numbers, CTA, cards, proof assets, and final alignment.

## Files

- `SKILL.md` — the actual Hermes skill loaded by the agent.
- `README.md` — this public-facing explanation for GitHub readers.

## Why this belongs in Hermes

Hermes can use local tools, file operations, image QA artifacts, Windows process control, and project notes together. This makes it useful as a production operator for design edits, while Photoshop remains the final visual editing and review surface.

The goal is not to replace designers or Photoshop. The goal is to make AI-assisted image editing repeatable, measurable, and easier to review.
