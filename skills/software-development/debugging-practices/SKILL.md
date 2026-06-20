---
name: debugging-practices
description: "Field-proven, stepwise debugging for software, content, UI/UX, and workflow failures—code, docs, standards compliance, and accessibility. Based on real AI coding session failures (GEPA-evolved)."
version: 1.0.0
author: Hermes Agent (GEPA-evolved from real Claude Code session traces)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [debugging, troubleshooting, accessibility, ui, content, standards, multi-domain]
    related_skills: [systematic-debugging, test-driven-development, requesting-code-review]
---

# Debugging Practices (Field-Proven, Multi-Domain)

## Overview

This skill delivers precise, actionable debugging steps for failures in software, content, documentation, UI/UX, and workflow. It addresses real-world, high-impact bug categories surfaced in AI coding sessions: logic, configuration, environment, specification drift, data/resource limits, content/policy errors, standards (accessibility, style guide) breaches, and UI/UX defects. Steps are designed for thorough root cause isolation and durable, verifiable fixes—including for micro bugs and compliance issues often missed by code-only approaches.

---

## 1. Reproduce and Gather Evidence

- **Replicate the precise failure:**  
  - Use _exact_ environment, inputs, config, file versions, and workflow steps.  
  - For UI/content bugs, use the same browser, device, zoom, and accessibility settings as affected users.
- **Collect artifacts:**  
  - Error/output logs, screenshots, terminal output, failing test cases, content diffs, and links.
  - Save both current and previous versions for side-by-side comparison.
  - For documentation/UI bugs, save broken screenshots, copy exact text, and note missing/incorrect elements.
- **Document observed vs. expected outcome:**  
  - Write down the _exact_ failure (e.g., "Last updated link is broken on updates.html", "Table header missing scope attribute", "SQL keywords lowercased against style guide").

---

## 2. Isolate and Categorize the Bug

### a. Find the Source

- **Code/Config/Workflow:**  
  - Use `git bisect`, file diffs, rollback, or minimal working examples to locate the change that introduced the bug.
  - For process/workflow: Audit each step; compare actual vs. required outputs (e.g., agent modifying read-only input).
- **Content/Documentation/UI:**  
  - Use diff tools to spot changes in text, markup, alt-text, table structure, etc.
  - For visual/UI bugs, compare screenshots or run visual regression tools (Percy, Storybook, jest-image-snapshot).

### b. Categorize (with concrete, real-case examples)

#### Logic or Implementation Bugs

- **Validate all invariants:**  
  - Confirm permissions, data shape, unique keys.  
  - Regression example: Fixing a broken template literal in a deduplication key.
- **Log/print all intermediates:**  
  - Don’t assume correctness—trace transformations.
- **Boundary tests:**  
  - Check for null, overflow, truncation (e.g., lost table rows, prompt truncation).

#### Configuration/Environment Bugs

- **Audit all config/env variables and dependencies:**  
  - Confirm deployment targets, dependency versions, secret keys.
  - Re-run provisioning after correcting syntax or ID misconfigurations.
- **Check system/OS-level constraints:**  
  - Permissions, code signing, DB state (e.g., macOS TCC, Entra ID, etc.).

#### Data & Resource Limit Bugs

- **Explicitly check for silent truncation/missing data:**  
  - Compare input vs. output for dropped or incomplete sections.

#### Specification, Standards, or Policy Drift

- **Compare implementation/output _directly_ with requirements:**  
  - Use design docs, style guides, accessibility standards (WCAG, ARIA), and assignment specs.
  - Common misses:  
    - Missing aria/scope attributes in tables.
    - Incorrect SQL casing.
    - Broken policy on read-only files (e.g., agent rewriting source).
    - Non-compliant markdown code block rendering.
- **Update or clarify ambiguous specs:**  
  - If requirements are unclear or have changed, note and resolve the ambiguity.

#### Content/Documentation Clarity & Accuracy

- **Run link and asset checkers:**  
  - Use tools (e.g., broken-link-checker) to find dead URLs, missing/broken images, outdated references.
- **Check for completeness and accuracy:**  
  - Confirm all required sections, tables, assignment titles, and naming conventions.
- **Audit documentation examples and CLI usage:**  
  - Compare with real command behavior; validate by copy-pasting into a working shell.
- **Explicitly check for micro errors:**  
  - Typos, missing words, inconsistent instructions, or unclear notes.

#### UI/UX and Accessibility Bugs

- **Automated accessibility checks:**  
  - Run tools like axe, Lighthouse, or WCAG checkers for contrast, labeling, and ARIA attributes.
- **Check for standards compliance:**  
  - Table headers must use `scope="col"`; images must have correct alt-text; links are descriptive.
- **Visual regression and manual inspection:**  
  - Automated screenshot diffing _and_ manual review in target browsers, devices, and zoom levels.
- **Enforce legibility floors and scaling:**  
  - Remove or correct small text classes that violate accessibility (e.g., remove all `text-xs`/`text-sm` classes).

---

## 3. Fix and Validate—Targeted, Durable Steps

- **Make focused, minimal changes:**  
  - Fix only the root cause (e.g., add early return for broken link, update OAuth scope format, correct config).
- **Automate fix application if repeated:**  
  - Use scripts/regex for large-scale or pattern-based cleanups (e.g., stripping text size classes, fixing all table headers).
- **Test fixes in all required environments:**  
  - Run both automated and manual checks—unit/integration tests, content linters, accessibility audits, link checkers, and visual regression suites.
- **Validate against original failure evidence:**  
  - Reproduce the scenario to verify the bug is resolved (e.g., re-run provisioning, reload UI, open docs in screen reader, paste CLI usage).
- **Regression test for edge/micro-case:**  
  - Add/expand tests covering the fixed edge case (e.g., self-referential link, fallback logic, missing scope).
- **Document fix scope and rationale:**  
  - Note what changed, why, and if further monitoring or automated protection (e.g., lint rule, CI check) is added.

---

## 4. Prevent Recurrence

- **Add automated checks for class of bug:**  
  - Lint rules for code, docs, markdown, accessibility, and naming.
  - Visual regression and accessibility checks in CI/CD.
- **Update documentation/specs:**  
  - Clarify requirements or standards; add examples reflecting common mistakes and fixes.
- **Share learnings with team:**  
  - Surface root causes and fixes in changelogs, retrospectives, or knowledge base.
- **Monitor for regressions:**  
  - Add or expand tests, monitoring, or logging to catch future instances early.

---

# Quick Debug Checklist (Summary)

- [ ] Replicate bug with original inputs, environment, and workflow.
- [ ] Gather full evidence: all artifacts, diffs, logs, screenshots.
- [ ] Pinpoint and categorize: logic, config, spec, data, content, UI/accessibility, workflow.
- [ ] Check for standards, accessibility, and style guide compliance (not just code correctness).
- [ ] Fix only the root cause; test in all relevant environments.
- [ ] Run automated and manual post-fix checks: linters, accessibility, visual regression, and user scenarios.
- [ ] Document what was fixed, why, and how it is prevented going forward.

---

# Example Real-Case Patterns (from AI coding sessions)

- **Workflow violation:**  
  Agent rewrote read-only input file; fix by reverting file, updating process guardrails.
- **Micro-content bug:**  
  Broken self-referential link; fix by adding guard clause for updates page.
- **UI/accessibility breach:**  
  Removed all small-text classes violating legibility/accessibility.
- **Syntax/logic error:**  
  Broken fallback in deduplication—corrected template literal.
- **Security/config bug:**  
  Path traversal vulnerability—fix by sanitizing input and guarding file writes.
- **Content style/standards:**  
  Missing table `scope` or SQL keyword casing—fix markup/code, add linter/check.
- **Documentation clarity:**  
  CLI usage note unclear—fix by editing example for accuracy and testability.
- **Infrastructure/config:**  
  Fix DB syntax and Entra ID config; confirm by successful rerun.

---

# Use this skill to debug and correct not only code, but also documentation, UI, accessibility, standards compliance, and workflows.  
Apply the steps above to ensure all bug classes—including micro bugs, content/presentation standards, and compliance issues—are durably resolved and prevented from recurring.

---