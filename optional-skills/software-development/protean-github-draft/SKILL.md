---
name: protean-github-draft
description: Use when a GitHub draft needs review before posting.
version: 1.0.0
author: Hermes Agent contributors
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [GitHub, Draft, Review, HTML, Markdown, SOP]
---

# GitHub Draft Review HTML

Render every GitHub post draft as a self-contained GitHub-dark HTML file before posting it. The HTML is the review artifact. The Markdown source remains the posting source.

## Trust boundary and safety

Draft Markdown, repository/tab/title/badge/state labels, and owner attribution are untrusted input. Normal CLI use places all of these values into the local review artifact, so the renderer escapes chrome values, treats raw Markdown HTML as text, removes event-handler attributes, and neutralizes unsafe URL schemes. The only script in the artifact is the renderer's fixed copy-button code; draft content cannot add scripts or executable attributes. Review generated HTML as untrusted content and do not weaken these protections for compatibility.

## Use this skill when

- A GitHub issue, pull request, review, reply, or release note needs human review.
- The text will be posted by `gh` or the GitHub API.
- Formatting, code fences, tables, links, or quoted evidence must be checked before posting.

## Prerequisites

- Python 3.9 or newer and a working `python3` (use `python` on systems where that is the configured command).
- The `markdown` and `pygments` Python packages. Install them in the interpreter that runs the pipeline:

  ```bash
  python3 -m pip install markdown pygments
  ```

- GitHub CLI (`gh`) for live repository/object inspection and posting, authenticated for the target repository. Git is optional unless a quoted code fence is verified with a `REF:PATH` source.
- Network access is required only for live GitHub inspection, posting, and read-back. Rendering and all local gates work offline.

## GitHub object scope

This skill covers issue bodies, pull-request bodies, issue/PR comments, pull-request reviews, and release notes. It does not authorize merging, closing, labeling, assigning, editing repository settings, or changing permissions. Treat each target as a separate object and verify its repository, number or ID, and target commit where applicable.

## Required procedure

1. Pull the live repository, issue, or PR state before drafting.
2. Write the exact postable text to a Markdown source file.
3. Run `scripts/draft_pipeline.py` once for that source.
4. Open the generated HTML file and review the rendered result.
5. If the text changes, rerun the pipeline with the previous draft as its baseline.
6. Post only after explicit human approval.
7. Read the live GitHub body, review body, comment, or release-note body back after posting. Compare it with the approved Markdown source. Only a final newline difference is acceptable.

If `gh` is missing, unauthenticated, cannot resolve the repository/object, or fails because of network, API, or rate-limit errors, stop before posting. Report the object state as unknown, preserve the draft locally, and do not claim that live state, CI, review, or posting was verified. Retry only after the failure is understood or the operator explicitly chooses another authenticated path.

## Build command

Run from the skill directory or pass paths to the scripts and files:

```bash
python3 scripts/draft_pipeline.py \
  --src <draft.md> \
  --slug <slug> \
  --tab "pull request draft" \
  --title "<title>" \
  --repo "owner/repository" \
  --drafts-root <drafts-root> \
  --batch "YYYY-MM-DD - HHhMMm" \
  --owners <posting-account>
```

Use `--tab "issue draft"` for issue text. Use `--verify-against <file>` for every quoted code or source block. Use `--evidence <file>` for quoted command output. Use `--diff-base <previous.md>` when revising a draft. The `--repo` value is the exact `OWNER/REPOSITORY` target; do not rely on a default when preparing a reusable or external draft.

## Runtime dependencies

The renderer requires the Python packages `markdown` and `pygments`. The standard library is otherwise sufficient. Install them in the interpreter used for the pipeline:

```bash
python3 -m pip install markdown pygments
```

If either package is unavailable, rendering stops with a clear dependency error and the pipeline keeps any existing HTML artifact unchanged. Do not use a hand-edited fallback HTML file.

## Portable installation and cross-platform use

Copy this directory into the host's optional-skills directory, or keep it anywhere and invoke the scripts by path. No profile, repository checkout, shell plugin, or service is required. Use forward-slash paths in examples, but pass native paths when invoking the scripts. On Windows, use `py -3` in place of `python3` and PowerShell equivalents for shell continuation. The pipeline uses Python standard-library path handling and creates the dated output directory automatically.

- `scripts/draft_pipeline.py` and `scripts/render_draft_html.py` support backtick and tilde fenced blocks; Python evidence fences may use `python` or `py`. Keep fences unindented in postable drafts.
- Run `python3 tests/test_renderer_safety.py` after installing the documented renderer dependencies. It covers normal Markdown, documented CLI-style values, raw HTML, event handlers, and unsafe URLs.

## Gates

The pipeline must pass all gates:

- prose and identifier checks
- verbatim evidence checks
- final oversight line
- GitHub-dark rendering
- source-to-HTML text fidelity
- required palette and dark color scheme
- self-contained HTML with no external assets
- visible change banner on revisions

A non-zero result blocks posting. Do not bypass a failed gate by hand-editing the HTML.

## Public-safety rules

- common private paths, local endpoints, secret assignments, and private-key markers are rejected by the automated scan
- manual review remains required for internal names, hostnames, connection strings, and other sensitive text that cannot be detected reliably by a generic pattern
- Keep the poster account in the HTML chrome, not in the post body unless GitHub requires it.
- Do not claim CI, review, merge, release, or install results without live evidence.
- Add this exact final line to every post:

> Human approval is required before posting.

## Output layout

Create one dated batch directory under the chosen drafts root:

```text
<drafts-root>/YYYY-MM-DD - HHhMMm/<slug>.html
```

Keep the Markdown source beside the working evidence, not as the review artifact. Do not deliver `.prev.html` files for review.

## Posting read-back

For a PR body:

```bash
gh pr view <number> --repo <owner/repository> --json body --jq .body
```

For a comment:

```bash
gh api repos/<owner>/<repository>/issues/comments/<comment-id> --jq .body
```

For other supported objects, use the corresponding `gh` read command or GitHub API endpoint and read back the exact body after writing. Never treat a successful local pipeline run as evidence that a GitHub write succeeded. Record the live URL, object ID, target head SHA, and any difference.

## What the HTML contains

The renderer creates one local file with:

- GitHub-dark repository and tab chrome
- `DRAFT - NOT POSTED` badge
- rendered Markdown headings, lists, tables, links, quotes, and code fences
- inline CSS and embedded syntax highlighting
- revision change banner when a baseline exists

The page must work offline. It must not fetch a stylesheet, script, font, or image.
