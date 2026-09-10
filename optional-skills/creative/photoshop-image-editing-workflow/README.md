# Photoshop Image Editing Workflow

An optional Hermes creative skill for WSL users who need a measured image-editing
workflow and a final Windows Adobe Photoshop handoff.

It is not enabled by default because it requires WSL, Windows PowerShell, and a
local Photoshop installation. The skill preserves the original asset, measures
reference criteria, creates QA artifacts, and only opens Photoshop after an
explicit prerequisite check succeeds.

## Included workflow

1. Locate and preserve source assets with Hermes file tools.
2. Measure typography, spacing, color, and composition from the reference.
3. Generate or edit versioned candidates.
4. Compare the original and candidate with a visual QA sheet and report.
5. Validate the WSL/PowerShell/Photoshop handoff, then open the final asset.
6. Log durable production work with the built-in `obsidian` skill.

## Prerequisites

- Hermes running in WSL.
- `powershell.exe` callable from WSL.
- Adobe Photoshop installed on Windows.
- The asset to open stored beneath `/mnt/<drive>/`.

The script `scripts/open_in_photoshop.py` returns structured JSON and exits
non-zero when a prerequisite is unavailable. That is an intentional, clear
failure mode; it never reports a completed Photoshop handoff without launching
it successfully.

## Files

- `SKILL.md` — operational instructions.
- `scripts/open_in_photoshop.py` — prerequisite check and Windows handoff.
- `references/complex-image-element-decomposition.md` — complex-composite
  guidance.
