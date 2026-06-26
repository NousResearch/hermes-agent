# James-shared coach brand ingestion crawler import

On 2026-06-24, James shared a zipped skill artifact over HTTP with:
- index: `http://100.98.218.49:8794/coach-brand-ingestion-skill-crawler-2026-06-24/`
- skill file: `http://100.98.218.49:8794/coach-brand-ingestion-skill-crawler-2026-06-24/coach-brand-ingestion.skill`
- sha256: `5ab0ab8d93cbb30e95ab4bb4da6b236ee88a60e90a332663dcd9c3c7dbaa4e3d`

Important lessons from the import/use case:
- The shared `.skill` artifact was a ZIP archive containing `SKILL.md`, `references/`, and `scripts/`.
- This skill is the right class-level workflow for ingesting a club/coach brand from a website and producing a brand-design document.
- When a user asks for a test prompt for another agent, do not replace this skill with a long operator-authored procedural prompt. Use a short natural prompt and rely on the skill to carry the hidden workflow.
- The natural test shape is whether the receiving agent understands that a club website should be treated as brand source truth and routes into crawl -> defaults packet -> polished brand doc.

Suggested natural prompt shape:
> Please ingest the Bayville Golf Club brand from this website and create a clean, high-quality brand design doc:
> https://bayvillegolfclub.com/
>
> I want a human to be able to open it and quickly understand the logo/mark system, colors, typography direction, visual style, and how the brand should show up in documents.
>
> Please make it polished and nicely formatted. PDF is fine if that works better than Google Docs.
