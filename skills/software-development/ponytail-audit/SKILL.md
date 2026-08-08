---
name: ponytail-audit
description: Audit a codebase for removable complexity.
version: 1.0.0
author: SeoYeonKim (@westkite1201), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [audit, code-review, simplification, yagni, dependencies, refactor]
    related_skills: [ponytail, ponytail-review, simplify-code]
---

# Ponytail Audit Skill

Ponytail Audit extends the over-engineering review from one diff to a whole
repository or named subsystem. It produces a ranked, read-only report of likely
deletions and simplifications; it does not apply them by default.

## When to Use

Use this skill when the user asks for a repository-wide or subsystem-wide
over-engineering audit, a ranked list of bloat, what can be deleted, or where
dependencies, abstractions, wrappers, and configuration exceed current needs.

Do not use it as a correctness or security audit. If the request names a
subsystem, keep the audit there instead of expanding to the whole repository.

## Prerequisites

- A local repository and a user-approved repository or subsystem scope.
- Repository instructions, branch/status context, and the relevant base branch.
- Access to manifests, callers, configuration, and extension-point docs needed
  to distinguish dead flexibility from public API.
- Use `terminal` for version-control context, `search_files` for candidates and
  callers, and `read_file` for evidence.

This workflow is report-only. Editing files, removing dependencies, committing,
pushing, or changing external review state requires a follow-up request.

## How to Run

Start with the narrowest context that matches the request. Use `terminal` to
inspect repository status and branch scope before searching implementation:

```bash
git status --short --branch
git diff --stat origin/main...HEAD
```

Then inspect likely complexity centers without dumping the entire tree:

- dependency manifests
- directories or symbols named helpers, services, managers, adapters,
  providers, factories, interfaces, or registries
- configuration values and feature flags
- wrappers that only delegate
- files, components, or classes with one caller
- hand-rolled parsing, validation, formatting, retry, debounce, caching, or
  date logic

## Quick Reference

Rank findings by likely removable maintenance burden, not by cosmetic line
count. Use these tags:

| Tag | Candidate | Replacement |
|---|---|---|
| `delete:` | Dead code, stale flags, unused config. | Nothing. |
| `reuse:` | Duplicate repository behavior. | Existing named helper. |
| `stdlib:` | Hand-rolled library behavior. | Named standard-library API. |
| `native:` | Custom platform/framework behavior. | Named native feature. |
| `yagni:` | One-use flexibility. | Inline now; extract on a real second use. |
| `shrink:` | Equivalent but verbose code. | Clear smaller form. |

Use `verify:` whenever a candidate might be public API, a plugin or extension
surface, a compatibility contract, or an externally consumed configuration.
Do not present uncertain removal as safe.

Never recommend removing these merely to reduce size:

- trust-boundary validation, auth, permissions, or secret handling
- data-loss prevention and corrupt-state recovery
- migrations and backward compatibility for existing users/data
- accessibility basics
- focused tests for non-trivial behavior
- production observability that operators rely on
- extension points with real external consumers

## Procedure

1. **Confirm scope.** Record repository status, branch/base, and the exact
   subsystem requested. Completion: the audit boundary is explicit.
2. **Map candidates.** Inspect manifests, abstraction-heavy directories,
   wrappers, configuration, flags, and hand-rolled utilities. Completion: the
   candidate list covers the requested scope without a blind full-tree dump.
3. **Trace use.** Search callers, imports, configuration reads, documentation,
   and plugin/extension contracts. Completion: every finding distinguishes
   unused code from an external surface.
4. **Protect invariants.** Exclude security, data protection, compatibility,
   accessibility, necessary observability, focused tests, and proven extension
   consumers. Completion: no recommendation weakens those boundaries.
5. **Rank by impact.** Put the largest likely maintenance/dependency cut first.
   Completion: ordering reflects value, not arbitrary discovery order.
6. **Write the report.** Use this form:

   ```text
   1. <tag> <what to cut>. <replacement>. [path:line or path + symbol]
   2. <tag> <what to cut>. <replacement>. [path:line or path + symbol]
   net: roughly -<N> lines, -<M> deps possible.
   ```

   Add a `verify:` clause before uncertain public or extension-surface cuts.
   If nothing meaningful is removable, report:

   ```text
   No meaningful over-engineering findings in this scoped audit.
   ```
7. **Separate other defects.** Label incidental correctness, security, or
   performance concerns `normal-review:` and recommend the appropriate review.
   Completion: the Ponytail ranking remains complexity-only.

Do not implement the findings unless the user explicitly asks for a separate
cleanup pass.

## Pitfalls

1. **Auditing beyond the request.** A subsystem request is not permission for a
   repository-wide cleanup.
2. **Fake precision.** Estimate conservatively and use ranges when the change
   has not been implemented.
3. **Ignoring external consumers.** Public APIs, plugin hooks, CLI contracts,
   and compatibility settings need `verify:` unless non-use is proven.
4. **Calling safety code bloat.** Security, data safety, accessibility,
   migration, observability, and focused tests are protected boundaries.
5. **Ranking by line count alone.** Prefer maintenance and dependency impact.
6. **Applying fixes during the audit.** The default output is a report.

## Verification

- [ ] Repository status, branch/base, and requested scope were confirmed.
- [ ] Candidates were traced to callers and configuration reads.
- [ ] Existing helpers were verified before `reuse:` findings.
- [ ] Public APIs and extension surfaces use `verify:` when uncertain.
- [ ] Safety, compatibility, accessibility, observability, and tests remain.
- [ ] Findings are ranked by likely impact and cite evidence.
- [ ] Estimates are conservative rather than exact guesses.
- [ ] No implementation or external mutation occurred without explicit approval.

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
