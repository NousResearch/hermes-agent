---
name: photoshop-image-editing-workflow
description: Edit images with Photoshop through a Windows handoff.
version: 1.1.0
author: Manseong Lee (@aiebrain), Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    category: creative
    tags: [photoshop, image-editing, design-qa, wsl, windows, typography]
    related_skills: [obsidian]
---

# Photoshop Image Editing Workflow Skill

Use this optional skill for measured image edits that require a final Adobe
Photoshop review. Hermes prepares and verifies the asset; Photoshop remains the
manual design surface. It does not build Photoshop plug-ins, UXP extensions, or
custom JSX automation.

## When to Use

Use it for reference-led text replacement, typography or composition matching,
PDP/landing assets, and image work that needs a QA sheet plus a Photoshop manual
pass. Use a dedicated software-development workflow instead for plug-in work.

## Prerequisites

This skill runs only from **WSL on Linux** with all of the following available:

- Windows is mounted at `/mnt/c` and `powershell.exe` is reachable from WSL.
- Adobe Photoshop is installed on Windows. The default path is
  `C:/Program Files/Adobe/Adobe Photoshop 2025/Photoshop.exe`; pass a different
  path to the helper when necessary.
- The source and final image exist on a Windows-mounted path. Do not overwrite
  the source image.
- `terminal`, `search_files`, `vision_analyze`, `read_file`, and `write_file`
  are available to Hermes. `image_generate` is optional for candidate creation.

Before editing, use `terminal` to run the packaged prerequisite check:

```text
python scripts/open_in_photoshop.py --check --input <source-image>
```

A non-zero result is a clean prerequisite failure, not a reason to claim that
Photoshop was opened. It reports whether WSL, PowerShell, Photoshop, and the
input asset are available.

## How to Run

1. Use `search_files` to locate source assets and choose a project-local output
   folder such as `assets/source`, `assets/edited`, `assets/qa`, and
   `assets/reports`.
2. Use `vision_analyze` to inspect the source and define measurable criteria:
   bounding boxes, colors, line spacing, font hierarchy, and text density.
3. Create versioned candidates. Use `image_generate` only for visual elements;
   retain exact text, prices, CTA copy, and alignment as editable Photoshop work.
4. Use `vision_analyze` to compare the candidate against the source. Write a
   QA sheet and report with `write_file` before the Photoshop handoff.
5. Run the helper through `terminal`:

```text
python scripts/open_in_photoshop.py --input <final-image>
```

The helper validates prerequisites, converts a `/mnt/<drive>/...` path to its
Windows form, starts Photoshop, and returns a JSON result. Do not report a
successful handoff unless its result says `"ok": true`.

## Quick Reference

| Task | Hermes surface | Expected evidence |
|---|---|---|
| Discover source/font files | `search_files` | source and candidate paths |
| Inspect reference/candidate | `vision_analyze` | measured visual criteria |
| Write QA report/log | `write_file`, `read_file` | report read-back |
| Check/open Photoshop | `terminal` + helper script | JSON `ok: true` |
| Verify final asset | `read_file` / `vision_analyze` | file exists and visual QA |

## Procedure

### 1. Preserve and measure

Keep the original untouched. Copy it into the project structure and record the
source path. For typography, measure each line separately: width, height,
baseline, fill color, line spacing, and approximate filled-pixel density. If a
second line is visually heavier, preserve that hierarchy rather than applying a
single font weight everywhere.

### 2. Choose the edit path

For a simple edit, replace or cover the target region, render text to the
measured constraints, and export a lossless versioned PNG. For a complex
composition, use the element-decomposition guide in
`references/complex-image-element-decomposition.md`: generate background and
visual objects independently, then build text, cards, prices, proof assets, and
alignment as named Photoshop layers.

### 3. Create QA artifacts

For non-trivial work, create both:

```text
qa/<source-stem>_qa_sheet.png
reports/<source-stem>_qa_report.md
```

The report records source/candidate paths, the user request, measurements,
applied font/position criteria, a pass-or-revise decision, and caveats. Compare
original and candidate at the same crop and zoom before handoff.

### 4. Handoff and manual pass

Run the helper only after QA passes. In Photoshop, compare source and candidate
at equal zoom; check hierarchy, optical centering, baseline, edges, and
background patching. If manual edits produce another export, return it to the
same QA loop before calling it final.

### 5. Log durable work

For assets that matter beyond this chat, use the `obsidian` skill to write a
project note with the source, final asset, QA paths, measured criteria, handoff
result, and remaining manual checks. Use `read_file` to read the note back.

## Pitfalls

- A successful process launch alone is not a verified Photoshop handoff; use the
  helper's JSON result.
- Do not assume two reference text lines share the same visual weight.
- Do not let an image model generate exact Korean copy, prices, or CTA text when
  Photoshop text layers can preserve control.
- Do not substitute a standalone Linux image editor when Photoshop is required.
  Report the prerequisite failure clearly instead.
- Do not overwrite the source or call an unverified manual export final.

## Verification

1. Run `python scripts/open_in_photoshop.py --check --input <final-image>` with
   `terminal`; it must return `"ok": true` before handoff.
2. Read the QA report with `read_file`; confirm it names the source, candidate,
   measurements, decision, and caveats.
3. Inspect the final image with `vision_analyze` against the source.
4. If a project log was requested, read it back with `read_file` before
   reporting completion.
