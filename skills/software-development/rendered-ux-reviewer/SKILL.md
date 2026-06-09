---
name: rendered-ux-reviewer
description: "Review rendered user-facing artifacts: UX, product copy, page comprehension."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ux-review, product-review, rendered-preview, visual-qa, delegation]
    related_skills: [requesting-code-review, github-pr-workflow, web-ui-deployment-access]
---

# Rendered UX/Product Reviewer

## Overview

Use an independent reviewer persona to evaluate rendered, user-facing artifacts before they ship. This catches product comprehension and visual-quality failures that code diffs, unit tests, text-presence checks, and green CI often miss.

**Core principle:** user-facing copy/layout PRs must not auto-merge from CI alone. They need Ryan review or this rendered reviewer gate.

## When to Use

Use this skill for changes that affect what a user sees or understands, especially:

- landing pages, marketing pages, docs homepages, examples pages, and onboarding pages
- product copy, empty states, tooltips, pricing/explainer copy, or feature descriptions
- bookmarklets, snippets, forms, interaction flows, and demo flows
- generated visual artifacts, screenshots, slides, reports, and preview renders
- private preview URLs from Tailscale, Cloudflare Pages, local servers, or static HTML files

Skip only when the change is purely internal code, non-user-facing config, or Ryan explicitly says to skip rendered review.

## Required Inputs

Collect enough rendered context before invoking the reviewer:

1. **Rendered artifact:** preview URL, screenshot(s), rendered HTML, static export path, or browser snapshot.
2. **Product intent:** what the page/artifact is supposed to communicate or let users do.
3. **Target audience:** who the artifact is for and what they likely know already.
4. **Changed scope:** issue/PR acceptance criteria, changed files, and any known non-goals.
5. **Artifact-specific risks:** e.g. long URLs, mobile layout, screenshots, code snippets, CTA clarity, examples quality.

For page-level copy/layout review, provide full-page context. Do not ask the reviewer to judge a page from isolated changed lines unless the page is unavailable; if unavailable, fail closed or clearly mark the verdict as incomplete.

## Invocation Pattern

Use `delegate_task` directly so the reviewer has a fresh context. When delegation is unavailable, use a fresh agent/session with the same prompt; do not ask the implementer to self-review. Give the reviewer rendered context and the checklist below.

```python
delegate_task(
    goal="""You are an independent rendered UX/product reviewer. Review the artifact as a real user would see it, not as code.

Return a compact structured verdict. Be strict: green CI, passing tests, or text appearing in the DOM is not enough if the rendered product is confusing, misleading, awkward, or visually broken.

CHECK:
- Page-level comprehension: can the target user understand what this is, build the right mental model, and know what to do next?
- Scope truthfulness: does the artifact advertise only behavior that is live/implemented, not proposed?
- Copy quality: concise, concrete, non-generic, not over-explained, not missing key context.
- Layout/rendering: awkward wraps, long URLs, overflowing code/snippets, mobile/desktop obvious issues.
- Token/rendering bugs: inline tokens accidentally rendered as full-line blocks; markdown/code formatting mistakes.
- Example quality: examples are recognizable/high-signal for the target user, not arbitrary filler.
- Interaction clarity: CTAs, forms, bookmarklets, demos, or next steps are understandable and believable.

FAIL-CLOSED RULES:
- No full-page/rendered context for a page-level review -> verdict must be FAIL with an `incomplete_context` blocker.
- Misleading live/proposed behavior -> verdict must be FAIL.
- Major comprehension or layout issue a user would notice -> verdict must be FAIL.

Return ONLY this Markdown:
## Verdict
PASS or FAIL

## Blockers
- ... or "None"

## Suggestions
- ... or "None"

## Summary
One concise paragraph.""",
    context="""
    PRODUCT INTENT:
    [what the artifact/page should communicate]

    TARGET AUDIENCE:
    [who this is for]

    ACCEPTANCE CRITERIA / CHANGED SCOPE:
    [issue/PR requirements and non-goals]

    RENDERED CONTEXT:
    [preview URL, screenshots, browser snapshot, or rendered HTML]

    KNOWN RISKS:
    [long URLs, token formatting, examples, mobile, live-vs-proposed claims, etc.]
    """,
    toolsets=["browser", "web", "vision"]
)
```

If using a static screenshot or local artifact instead of a URL, include `vision_analyze` output or the artifact path in the context and enable the relevant toolsets.

## How to Use the Result

- **PASS:** mention the rendered reviewer gate in the PR/issue verification notes.
- **FAIL:** fix blockers before merging. Re-run the reviewer after fixes.
- **Incomplete context:** gather the missing rendered context or ask Ryan for review; do not treat this as approval.
- **Suggestions only:** decide pragmatically. Non-blocking polish can be deferred, but do not hide it.

## Common Pitfalls

- Reviewing only the diff for a page-level change. Full-page context is required.
- Treating DOM text-presence checks as proof that the copy/layout works.
- Letting generic examples through when Ryan-owned or high-signal examples are expected.
- Shipping text that describes future/planned behavior as if it is live.
- Ignoring awkward wrapping of URLs, tokens, badges, or snippets because tests passed.
- Replacing Ryan's taste/product review. This reviewer is a safety net, not final taste authority.

## Verification Checklist

Before merging a user-facing copy/layout PR, verify:

- [ ] Rendered artifact was inspected via URL, screenshot, browser snapshot, or rendered HTML.
- [ ] The reviewer received product intent, target audience, changed scope, and risks.
- [ ] Page-level reviews included full-page context.
- [ ] The verdict was PASS, or Ryan explicitly accepted remaining blockers.
- [ ] PR/issue notes record the rendered-review result.
