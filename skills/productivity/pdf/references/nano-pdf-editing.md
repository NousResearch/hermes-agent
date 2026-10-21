# Natural-language PDF editing with nano-pdf

Use `nano-pdf` through the `terminal` tool when the user wants generative edits to an existing PDF page. It regenerates a page image and rebuilds a text layer; it is not a deterministic text replacement. For local structural PDF work and OCR, use the parent `pdf` skill and `references/ocr-extraction.md`.

## Prerequisites

- A separately installed `nano-pdf` CLI and its documented system dependencies. Follow the [tool author's setup](https://github.com/gavrielc/Nano-PDF); do not modify the Hermes-managed Python environment with raw `pip` or `uv pip`.
- A paid Gemini API key with billing enabled, supplied through the user's existing credential mechanism. Do not read, print, or copy the key into prompts, logs, or skill files.
- Explicit approval for the paid call and for sending the target page image and edit prompt to Google's Gemini API. Style-reference images and extracted document text are also transmitted when requested. An instruction to edit a local PDF does not by itself authorize this upload.

The contract below is verified against [nano-pdf 0.2.1](https://pypi.org/project/nano-pdf/0.2.1/) and the tool author's source. Inspect the installed version and its help before using a different release; do not guess its page numbering by making an edit.

## Procedure

1. Inspect the original document locally through the parent PDF helpers. Confirm the physical page position, page count, requested change, and any difference between printed page labels and physical page numbers.
2. Keep the original unchanged and choose a separate, unused output path. Confirm the upload and paid-call scope before executing.
3. Run the edit on the confirmed physical page. In 0.2.1, `edit` page numbers and style-reference page numbers are **1-based**, in the range `1..page_count`. Never retry a wrong-page edit with `±1`; stop, recheck the mapping locally, and obtain approval for any further paid call.
4. Inspect the full command result for per-page errors. A partial success can still produce an output PDF and a final "Done!" message.
5. Verify the output locally as described below before reporting completion.

```bash
# Only after approval; page 1 is the first physical page.
# --disable-google-search disables search grounding, not the Gemini upload.
nano-pdf edit deck.pdf 1 "Change the title to 'Q3 Results' and fix the typo in the subtitle" --output reviewed-deck.pdf --disable-google-search
```

## Data and effect boundaries

- Google Search grounding is enabled by default in 0.2.1. Use `--disable-google-search` unless search grounding is explicitly requested; the edit still sends page content to Gemini.
- `edit` does not include full-document text unless `--use-context` is enabled. `add` enables it by default; `--no-use-context` disables that extra text. These options do not suppress target or reference image uploads.
- `add <file.pdf> 0 <prompt>` inserts before the first page. Its `after_page` value is an insertion position in `0..page_count`, not evidence that `edit` accepts zero-based page numbers.
- Generative image editing and OCR rehydration can change text, layout, or searchable content outside the requested edit. Prefer deterministic local PDF operations when those meet the task.

## Verification

Render the original and output target pages locally with the parent PDF helpers and inspect them with `vision_analyze`. Confirm the intended page, requested text, layout, and unintended changes. Extract text with the parent helpers and compare important names, dates, numbers, and searchable content; review neighboring or unchanged pages as appropriate. Confirm the page count and investigate every reported page error. File existence, nonzero size, opening successfully, or "Done!" alone does not establish a correct edit.

If verification fails, preserve the original and report the discrepancy. Do not automatically repeat the upload, shift the page number, or overwrite the original.
