# Bayville / Sergio PDF export lesson

## Durable lesson
For premium brand-book-style artifacts, standardize on a two-step path:
1. compose the visual source artifact as HTML
2. export the final PDF via Playwright Chromium

This avoids overfitting the skill to a brittle renderer choice like WeasyPrint on macOS lanes.

## What to say when blocked
If the agent completes the HTML/composed artifact but cannot produce the PDF, report it as:
- content/artifact composition: complete
- final PDF truth artifact: blocked by runtime/tooling

Do **not** claim the PDF exists unless a non-empty file was actually written.

## Why this matters
This session showed a useful distinction:
- the agent's content and artifact routing can be correct
- while the local PDF toolchain is incomplete

The workflow should preserve that distinction rather than collapsing it into a vague failure.

## Standard prompting lesson
If the user is testing another agent's routing, keep the prompt natural and human. Explicitly naming `coach-brand-ingestion` is fine, but avoid converting the request into a long operator SOP. The evaluation target is whether the downstream agent routes into the brand-ingestion workflow and produces the right artifact class.
