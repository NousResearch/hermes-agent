# Glued Media Directive Extraction Design

## Problem

`MEDIA_TAG_CLEANUP_RE` requires punctuation, whitespace, or end-of-string
after a known file extension. A directly appended `[[as_document]]`
directive starts with `[`, so valid output such as
`MEDIA:/tmp/report.xlsx[[as_document]]` is not extracted for delivery.

## Design

Extend the existing post-extension lookahead to accept `[`. Keep the shared
regex, extraction pipeline, path validation, and directive stripping unchanged.
This makes the existing all-or-nothing `[[as_document]]` behavior work
without adding a second parser or special-case rewrite.

## Verification

Add one regression test to the existing media-extraction suite. It will pass a
glued directive to `BasePlatformAdapter.extract_media()` and assert:

- the media path is extracted once;
- `[[as_document]]` and the `MEDIA:` tag are absent from cleaned text;
- surrounding user-visible text remains.

Run the focused test through `scripts/run_tests.sh`.

## Non-goals

No new directive syntax, per-file document mode, extension changes, or changes
to protected code/JSON spans.
