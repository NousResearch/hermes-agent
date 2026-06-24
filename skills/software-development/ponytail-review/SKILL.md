---
name: ponytail-review
description: Review a diff only for removable complexity.
version: 1.0.0
author: SeoYeonKim (@westkite1201), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [code-review, simplification, yagni, dependencies, refactor]
    related_skills: [ponytail, simplify-code, requesting-code-review]
---

# Ponytail Review Skill

Ponytail Review performs a narrow, read-only review of a concrete diff for
unnecessary complexity. It reports what can safely be deleted, reused, or
replaced, but it is not a correctness, security, or performance review.

## When to Use

Use this skill when the user asks for an over-engineering-only review, a YAGNI
review, what can be deleted from a diff, or whether a change is overbuilt. It
is especially useful when a diff adds dependencies, wrappers, abstractions,
configuration, factories, custom parsers, or duplicate helpers.

Do not use it as the only pre-merge review for risky changes. Pair it with the
normal code-review or security-review workflow when those concerns matter.

## Prerequisites

- A concrete diff, commit, pull request patch, or file scope.
- The repository's base branch and local instructions.
- Access to nearby code so `reuse:` findings can name a real existing helper.
- Use `terminal` to obtain the diff, `search_files` to find callers/helpers,
  and `read_file` to inspect evidence.

This workflow is report-only. Editing, committing, pushing, replying to a pull
request, or resolving a review thread requires a separate explicit request.

## How to Run

Choose the smallest diff source that answers the request. For example, use
`terminal` to inspect uncommitted work, staged work, a branch against its base,
or one named file:

```bash
git diff
git diff --staged
git diff origin/main...HEAD
git diff -- path/to/file
```

If the diff spans unrelated subsystems, review each subsystem separately so
every finding remains evidence-backed.

## Quick Reference

Use one tag per finding:

| Tag | Use for | Required replacement |
|---|---|---|
| `delete:` | Dead or speculative code. | Nothing. |
| `reuse:` | Code duplicated from this repository. | Name the existing helper. |
| `stdlib:` | Hand-rolled standard-library behavior. | Name the built-in. |
| `native:` | Custom code replacing platform behavior. | Name the native feature. |
| `yagni:` | Flexibility with no concrete second use. | Inline or defer it. |
| `shrink:` | Equivalent behavior in clearer, fewer code. | Show the smaller form. |

Never flag these merely because they add lines:

- trust-boundary validation, auth, permissions, or secret handling
- data-loss prevention and corrupt-state recovery
- migrations or compatibility paths for existing users/data
- accessibility basics
- a focused behavior test for non-trivial logic
- production observability required by operators

## Procedure

1. **Set scope.** Identify the exact diff and base. Completion: no file outside
   the requested scope can produce a finding.
2. **Read intent.** Inspect the request, changed behavior, and nearby callers.
   Completion: each candidate can be judged without guessing its purpose.
3. **Search before labeling.** Find existing helpers before `reuse:` and confirm
   caller counts before `yagni:`. Completion: every claim names evidence.
4. **Apply the safety boundary.** Exclude security, data protection,
   compatibility, accessibility, necessary observability, and focused tests.
   Completion: no finding weakens one of those invariants.
5. **Write only actionable findings.** Use one line per item:

   ```text
   <file>:L<line>: <tag> <what to cut>. <replacement>.
   ```

   Completion: every line has a location, tag, cut, and replacement.
6. **Separate other defects.** If a correctness, security, or performance issue
   is noticed, label it `normal-review:` so it is not mistaken for the
   Ponytail score. Completion: scope boundaries are explicit.
7. **Estimate impact.** End with a conservative line/dependency reduction:

   ```text
   net: roughly -<N> lines possible, -<M> deps possible.
   ```

   If there is nothing meaningful to cut, return exactly the useful outcome:

   ```text
   No over-engineering findings in this scope. Run normal verification before shipping.
   ```

Do not apply any suggested change unless the user explicitly asks for a
follow-up implementation.

## Pitfalls

1. **Vague advice.** A question like "could this be simpler?" is not a finding;
   cite the line and the concrete replacement.
2. **Deleting verification.** A focused behavior check is usually the minimum,
   not bloat.
3. **Unverified reuse.** Search and name the existing helper before using the
   `reuse:` tag.
4. **Style disguised as simplification.** Naming and formatting preferences do
   not count unless they remove real complexity.
5. **Scope drift.** Keep correctness and security observations under
   `normal-review:` and recommend the appropriate follow-up review.
6. **Applying fixes during review.** The default deliverable is a report, not a
   modified working tree.

## Verification

- [ ] The review used a concrete, user-requested diff scope.
- [ ] Existing helpers and callers were searched before making claims.
- [ ] Findings cover removable complexity only.
- [ ] Safety, compatibility, accessibility, observability, and tests remain.
- [ ] Each finding has one location, one tag, and one replacement.
- [ ] Non-Ponytail defects are labeled `normal-review:`.
- [ ] The result ends with conservative net impact or the no-findings message.
- [ ] No files or external review state were changed without explicit approval.

## Attribution

Adapted from [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)
under the MIT License.

```text
MIT License

Copyright (c) 2026 DietrichGebert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
