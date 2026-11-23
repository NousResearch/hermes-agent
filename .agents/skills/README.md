# Agent Skills Reference

Reference list of skills available to `agy` (Antigravity CLI) when working on this project. This directory is documentation only — no skills are installed or duplicated here.

## Engineering & Debugging
- **`systematic-debugging`** — 4-phase root-cause debugging workflow. Use before proposing or applying any fix when behavior does not match expectations (reproduce/inspect real DOM or logs first, never guess).
- **`ponytail`** — Minimal-diff engineering discipline ("the smallest complete change that fully solves the task"). Use when implementing fixes, refactoring, or avoiding over-engineering and unnecessary abstractions.
- **`grilling`** — Relentless design and plan stress-testing. Use before starting non-trivial architectural changes to surface edge cases and assumptions early.

## UI & Design
- **`frontend-design`** — Intentional visual design and UI craft guidance. Use when designing or refining visual hierarchy, typography, and surface contrast.
- **`popular-web-designs`** — Reference catalog of 54 real-world design systems as HTML/CSS. Use as a visual reference when comparing layout or component patterns.

## Git, Review & Concise Communication
- **`caveman-commit`** — Conventional Commits messages compressed to pure intent. Use when drafting concise, high-signal commit messages.
- **`caveman-review`** — Compressed code review format (one line per finding: location, problem, fix). Use when reviewing diffs or PRs.
- **`caveman`** — Terse technical communication mode (substance kept, filler removed). Use only when explicitly requested.

## Session Handoff (User-Invoked Only)
- **`handoff`** — Compacts the current session into a handoff document for another agent. Configured with `disable-model-invocation: true`, so it is invoked explicitly by the user (`/handoff`) rather than auto-activated by the model.
