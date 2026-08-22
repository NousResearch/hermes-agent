---
name: anti-hallucination
description: Catch fabricated APIs, citations, configs before writing.
version: 1.0.0
author: Hermes Agent (adapted from Alex_ACT_Supervisor)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [verification, hallucination, anti-fabrication, epistemic, factual, citations]
    related_skills: [requesting-code-review, plan, test-driven-development, humanizer]
---

# Anti-Hallucination

Prevent fabricated content at the point of generation, not after. Once an invented API call, fake citation, or made-up config key is in the output, downstream review has to spot it among real content — which is far harder than not generating it in the first place.

This skill is the *pre-generation* discipline. `requesting-code-review` is the *post-generation* defense. They compose: stop fabrication first, catch what slipped through second.

## When to Use

Load this skill whenever about to generate any of:

- Factual claims about external systems, libraries, APIs, services
- Code examples invoking specific function names, parameters, return types
- Citations, references, URLs, paper titles, author names
- Configuration values, default settings, version numbers
- Error messages, log lines, output formats
- Capability claims about tools, frameworks, or platforms

If the answer to *"have I actually seen this work, or am I generating something that sounds plausible?"* is "the second one," stop. Verify or acknowledge uncertainty.

## The Core Discipline

| Stage | Question | If unsure |
|---|---|---|
| **Input-discipline** (what you're about to generate) | Am I about to write something I can't verify is real? | Stop. Verify or say "I don't know" |
| **Output-discipline** (what you're about to report) | Am I about to claim a check succeeded that I didn't actually run? | Cite what was actually checked, or hedge the claim |

The line in the middle is **between thinking and typing**.

## Procedure

### Step 1 — Recognize the signal

Before typing a specific factual claim, ask which category it falls into:

| Claim type | Verification source |
|---|---|
| API method / function exists | `read_file` or `search_files` against the library source |
| Library / package exists at a version | `web_extract` against the package registry (npmjs.com, pypi.org, crates.io, rubygems.org) |
| Config key / env var works | `search_files` in source / `web_extract` against the project docs |
| CLI flag exists | Invoke `--help` through the `terminal` tool, capture output |
| File / directory path exists | `search_files` with `target='files'` |
| Command succeeded | Capture `terminal` exit code AND the relevant output lines, cite both |
| Absence (no matches found) | Cite the search scope explicitly: `search_files` parameters used, file count scanned |
| URL / citation resolves | `web_extract` and quote the actual page; never trust training-data memory for citations |
| Doc says X | `web_extract` or `read_file` the doc AND cross-check against the implementation |

### Step 2 — Verify before generating

Run the verification *first*. Then write the claim. The order matters: if you write first and verify second, you'll find yourself defending the claim instead of revising it.

### Step 3 — When verification isn't possible, name the gap

If the verification source is unreachable (offline, private repo, paid API), do not fabricate. Use one of:

- "I don't know"
- "I haven't verified this, but my understanding is..." (clearly marked inference)
- "Based on training data, this was true as of [date]" (clearly marked recall)

### Step 4 — Cite what was actually checked

When reporting verification results, name the specific check:

- Good: "Confirmed via `read_file` on `src/parser.ts` line 42 — `parseAst()` accepts `{ strict: true }`."
- Bad: "Verified — the method accepts a strict option."

The unattributed "verified" is theatre. The attributed one is auditable.

## Input-discipline Signals

These phrases in your own internal narration mean STOP:

| Signal in your thinking | What to do |
|---|---|
| "I think there might be a `parseFile()` method..." | Run `search_files` for `parseFile` in the library |
| "One approach could be calling `foo.bar(opts)`..." | Verify the API exists with those params via `read_file` |
| "Try this workaround: set `FOO_DEBUG=1`..." | Confirm the env var is real via `web_extract` against docs or `search_files` in source |
| About to invent step-by-step instructions for an unfamiliar tool | Stop. Use `web_extract` on the actual docs first |
| User says "that doesn't exist" or "that method isn't real" | Acknowledge immediately. No defending the fabrication |

## Output-discipline Signals

| Signal in your output | What to do |
|---|---|
| "No matches found" / "Verified clean" / "Nothing returned" | Before reporting absence, name the search scope. A failed search and a clean search look identical without the scope check |
| "The doc says X" / "Per the README" / "According to spec" | Before treating doc content as ground truth, cross-check against the current source via `read_file`. Docs drift from code; cite both |
| "I checked and..." / "Verified that..." / "Confirmed..." | Name what was actually checked: file path, command, output snippet |

## Pitfalls

| Pitfall | Why it fails | Correction |
|---|---|---|
| "I'll generate the answer and check it after" | Once written confidently, downstream review struggles to flag it | Verify before generating |
| "Probably works like X" | "Probably" is the fabrication signal | Either verify, or say "I don't know" |
| Defending a fabrication when challenged | Doubles down on the original lie | Acknowledge, correct, move on |
| Generating plausible-looking citations | Citation confabulation is a well-documented LLM failure mode | Use `web_extract` on the actual URL; mark inferred refs as inferred |
| "The documentation must say..." | Inferring doc content is fabrication | Use `web_extract` or hedge the claim |
| Adding "(citation needed)" to fabricated content | The hedge labels the fabrication but doesn't remove it | Remove the fabrication; don't decorate it |
| Trusting training-data recall for version numbers, API surfaces, or package names | Training data has a cutoff; these change | Verify via `web_extract` against registry or `search_files` against source |
| "It compiles, so it's correct" | Compilation proves syntax, not semantics; LLM-generated code can compile while calling fake methods on real objects | Read the called functions via `read_file` to confirm they exist and accept the passed arguments |

## Verification

The discipline fired correctly when the final output meets all of:

- Every specific factual claim has a verification source named in-line or available on request
- Absence claims ("nothing found", "no matches") cite the search scope used
- Citations point to URLs that were actually fetched, or are marked as recalled/inferred
- The user can audit any claim by re-running the cited check
- Pushback from the user (`"that method doesn't exist"`) is acknowledged, not defended

## Why It's Separate From Other Review Skills

| Skill | Catches |
|---|---|
| **anti-hallucination** (this) | Fabricated facts, invented APIs, citation confabulation — *before they're written* |
| **requesting-code-review** | Security issues, logic errors, regressions — *after the code is written* |
| **plan** | Missing context, vague tasks, undocumented decisions — *before implementation starts* |
| **test-driven-development** | Code that compiles but doesn't behave correctly — *via failing tests* |

Each catches what the others miss. This skill prevents the inputs that the others would otherwise have to detect downstream.

## Attribution

Adapted from [Alex_ACT_Supervisor](https://github.com/fabioc-aloha/Alex_ACT_Supervisor) (Fabio Correa, MIT) where it ships as part of the ACT epistemic-integrity framework. The core discipline (input vs output, verify-before-generate, attributed verification) is preserved verbatim; Hermes-specific adaptations are limited to tool-name references (`search_files`, `read_file`, `web_extract`, `terminal`) and skill cross-references (`requesting-code-review`, `plan`, `test-driven-development`, `humanizer`).

Originally written 2026-05-29 for the Alex_ACT_Edition baseline brain; ported to Hermes Agent 2026-06-07.
