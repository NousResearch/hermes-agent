---
name: goal-prompt-pack
description: "Use when creating the strongest possible long-running goal prompt for Codex Goal Mode, Hermes workers, or another autonomous coding agent; can independently create product specs, UI specs, UI mockup prompts/images, acceptance criteria, and the final build goal without requiring the idea or Superpowers workflows."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [goal-prompt, codex, hermes, autonomous-agents, ui-mockups, prompt-engineering, app-builds]
    related_skills: [youtube-content, idea-superpowers-suite, idea-to-ui-design-brief, idea-to-implementation-doc]
---
# Goal Prompt Pack

## Overview

This skill creates a **Goal Prompt Pack**: a complete, self-contained package for launching a long-running autonomous app build in Codex Goal Mode, Hermes, Claude Code, OpenCode, or a similar coding agent.

It is standalone. It does **not** require the idea workflow, Superpowers, or an existing design doc. If those artifacts exist, use them as inputs. If they do not exist, create the needed product spec, UI spec, mockup prompts, selected mockup notes, and final goal prompt directly inside this workflow.

The goal is to make the final prompt as strong as possible by giving the build agent a bounded, visible, testable finish line.

Core method:

1. Define the app/product target.
2. Write a compact but complete product spec.
3. Write a screen-level UI spec.
4. Generate or prepare high-fidelity UI mockups from that UI spec.
5. Review the mockups and translate them back into written visual requirements.
6. Create the final autonomous goal prompt with hard requirements, non-goals, acceptance criteria, blocker rules, stop rules, verification requirements, and final report format.

## When to Use

Use this skill when the user asks for:

- a goal prompt;
- a Codex Goal Mode prompt;
- an autonomous build prompt;
- a prompt pack to create an app;
- the spec → UI mockups → goal prompt workflow;
- a polished multi-screen app prompt where UI quality matters;
- a prompt that includes UI mockups and a full build prompt;
- a reusable standalone process for creating the best possible app-building prompt;
- a Hermes worker goal prompt with visual references and acceptance criteria.

Also use it when the user gives only a rough app idea but wants to end with a build-ready goal prompt.

## When Not to Use

Do not use this skill when:

- the user only wants a quick product note;
- the task is a bug with unknown root cause;
- there is no clear finish line and the user wants open-ended exploration;
- the user wants you to immediately build the app rather than prepare the goal prompt;
- the app has no UI and a normal implementation plan is enough, unless the user explicitly asks for a goal prompt;
- the target is unsafe, credential-harvesting, deceptive, or otherwise inappropriate.

## Output Modes

Choose one mode up front.

### Standalone Mode

Use when no idea/design artifacts exist or the user wants the video-style workflow from scratch.

Produce a self-contained Goal Prompt Pack with:

- product spec;
- UI spec;
- mockup generation prompts;
- optional generated mockup image paths/URLs;
- selected mockup analysis;
- final autonomous goal prompt.

### Plug-in Mode

Use when an existing idea/design/implementation package exists.

Read the available artifacts, fill gaps, then create a Goal Prompt Pack as an additional artifact. Do not require or invoke the idea workflow unless the user explicitly wants that broader process.

### Text-only Mode

Use when image generation is unavailable or the user does not want to generate images yet.

Still produce:

- high-fidelity mockup prompts;
- a visual target description;
- placeholders such as `[ATTACH SELECTED MOCKUPS HERE]` in the final goal prompt;
- explicit instructions that the final prompt becomes stronger after mockups are generated and attached.

### Image-backed Mode

Use when image generation is available through Codex, Hermes, GPT-5.5, or another tool.

Generate or instruct generation of mockups, then analyze the mockups and incorporate them into the final goal prompt as visual targets.

## Required Inputs

Minimum input:

- app/product idea;
- target platform, or permission to recommend one;
- whether visual fidelity matters.

Strong preferred inputs:

- target users;
- core workflow;
- must-have screens;
- desired visual feel;
- stack preference;
- data/storage requirements;
- integrations;
- auth/privacy constraints;
- examples of apps/styles to emulate or avoid;
- target executor: Codex Goal Mode, Hermes, Claude Code, OpenCode, generic coding agent.

If critical inputs are missing, ask concise questions. If the user wants speed, make sensible assumptions and label them.

## Hard Gates

- Do not write the final goal prompt until the app purpose, primary user flow, and MVP scope are clear enough to state in one paragraph.
- Do not let generated mockups become the only source of truth. Convert them into written UI requirements.
- Do not claim the Goal Prompt Pack is build-ready unless it contains testable acceptance criteria.
- Do not omit blocker and stop rules for long-running agents.
- Do not ask an agent to “match exactly” without defining acceptable fidelity and when to stop.
- Do not execute the build unless the user explicitly asks to start execution after the pack is created.

## Protocol

### 1. Select target executor

Identify the intended build agent:

- Codex Goal Mode — preferred when the user wants OpenAI Codex goal behavior and strong built-in image generation.
- Hermes — useful when the goal prompt will be run by a Hermes worker or GPT-5.5-backed process.
- Generic coding agent — use neutral language and include all context inline.

If unknown, write the final prompt in Codex Goal Mode style and add a Hermes/generic variant note.

### 2. Capture or synthesize the product spec

Write a concise product spec with:

- app name / working title;
- one-line summary;
- target user;
- core problem;
- MVP user flow;
- functional requirements;
- data/persistence needs;
- platform target;
- technical constraints or recommended defaults;
- non-goals;
- open questions / assumptions.

### 3. Create the UI spec

For UI-bearing apps, write a screen-level UI spec:

- product feel;
- design principles;
- navigation model;
- screen list;
- screen-by-screen layout;
- component inventory;
- visual style: color, typography, spacing, density, motion if relevant;
- states: empty, loading, error, success, first-run, permissions;
- accessibility and responsiveness;
- copy/content notes;
- visual non-goals.

For non-UI tools, replace this with a CLI/API/output-experience spec.

### 4. Create mockup generation prompts

Create prompts for the minimum set of screens needed to define the visual target.

Default app mockup set:

1. Primary/home screen.
2. Main workflow/action screen.
3. Library/detail/settings screen, whichever is core.
4. Secondary output/result screen.

For each prompt, specify:

- platform and device/frame;
- screen purpose;
- target user;
- exact content/components to show;
- layout and hierarchy;
- visual direction;
- density/spacing;
- things to avoid;
- that it should be a realistic product UI screenshot, not marketing art.

Default prompt shape:

```text
Create a high-fidelity <platform> app UI mockup for <app name>.
Screen: <screen name>.
Target user: <target user>.
Purpose: <what this screen helps the user do>.
Layout: <specific layout/hierarchy>.
Must show: <components, labels, sample data, actions>.
Visual style: <colors, typography, density, mood>.
Avoid: generic SaaS filler, irrelevant metrics, stock photos, unreadable text, decorative-only cards, redesigned navigation, extra screens.
The result should look like a real app screenshot that a coding agent can implement.
```

### 5. Generate mockups when available

If image generation is available and the user wants actual mockups:

- Generate the selected screens.
- Save or reference image paths/URLs.
- If using Codex because of stronger image generation, instruct the user or agent to run the mockup prompts in Codex before using the final goal prompt.
- If using Hermes image generation, use available image tools and record the returned paths/URLs.

If image generation is unavailable, continue in Text-only Mode and mark image attachment as a recommended next step.

### 6. Review mockups and translate them to written requirements

For each generated or selected mockup, extract:

- screen name;
- layout structure;
- visible components;
- hierarchy;
- spacing/density;
- color/style rules;
- content examples;
- deviations from the UI spec;
- implementation notes.

If the mockups are generic, inconsistent, ugly, or contradict the spec, revise the UI spec and regenerate or update the mockup prompts before producing the final goal prompt.

### 7. Define acceptance criteria

Create acceptance criteria covering:

- functional behavior;
- screen coverage;
- visual fidelity;
- data persistence;
- platform/build requirements;
- testing/lint/typecheck/build commands;
- screenshot evidence;
- known non-goals.

Each criterion must be checkable by a future agent or human.

### 8. Write blocker and stop rules

Long-running goals need explicit failure behavior.

Include rules for:

- missing credentials;
- failed dependency install;
- simulator/browser/device failure;
- blocked external services;
- ambiguous requirements;
- impossible visual fidelity;
- repeated failing verification;
- maximum passes/turns before stopping.

### 9. Create the final goal prompt

Write the final prompt as a copy-paste-ready block for the selected executor.

For Codex, start with `/goal` when appropriate.

The final goal prompt must include:

- mission;
- source artifacts or pasted source content;
- attached mockup references;
- priority order;
- functional requirements;
- UI fidelity requirements;
- technical requirements;
- non-goals;
- verification requirements;
- blocker rule;
- stop/loop rule;
- final report format.

### 10. Produce readiness verdict

End with:

- `READY` — complete enough to run;
- `READY AFTER MOCKUPS` — text pack is ready, but actual image mockups should be generated/attached first;
- `NEEDS CLARIFICATION` — missing decisions would materially change the app;
- `NOT SUITABLE FOR GOAL MODE` — finish line is not bounded/testable or the task is exploratory/unknown.

## Artifact Contract

A complete Goal Prompt Pack should contain:

```markdown
# <App Name> Goal Prompt Pack

Verdict: READY | READY AFTER MOCKUPS | NEEDS CLARIFICATION | NOT SUITABLE FOR GOAL MODE
Target executor: Codex Goal Mode | Hermes | Generic coding agent
Mode: Standalone | Plug-in | Text-only | Image-backed

## 1. Product Spec

## 2. UI Spec

## 3. Mockup Generation Prompts

### Mockup 1 — <Screen>

### Mockup 2 — <Screen>

### Mockup 3 — <Screen>

### Mockup 4 — <Screen>

## 4. Mockup References

## 5. Mockup-to-Requirement Translation

## 6. Acceptance Criteria

## 7. Blocker and Stop Rules

## 8. Final Goal Prompt

```text
/goal
...
```

## 9. Hermes / Generic Variant Notes

## 10. Open Questions and Assumptions
```

## Final Goal Prompt Template

```text
/goal

Build <app name> as a <platform/stack> app.

You must treat the written spec and attached mockups as the build target.

Source of truth:
1. Product spec: <path, paste, or section below>
2. UI spec: <path, paste, or section below>
3. Mockups: <attach/list image paths or say ATTACH SELECTED MOCKUPS HERE>

Priority order:
1. Non-negotiable product requirements
2. Functional correctness
3. Screen-by-screen UI fidelity to the mockups
4. Tests/build verification
5. Polish

If the mockups and written spec conflict, follow the written spec for behavior and the mockups for visual layout/style. Report the conflict in the final answer.

Do not reinterpret, simplify, merge screens, redesign, create a generic version, or add unrelated features.

Functional requirements:
- <requirement>

UI fidelity requirements:
- Implement every required screen from the UI spec.
- Match the attached mockups for layout, hierarchy, component inventory, color direction, spacing, density, and content structure.
- Preserve screen-specific navigation and actions.
- Implement required empty/loading/error/success states.
- If exact pixel parity is impossible, get as close as practical and explain the deviation.

Technical requirements:
- <stack/platform>
- <data/storage>
- <auth/integrations>
- <project constraints>

Non-goals:
- Do not <non-goal>.

Acceptance criteria:
- <criterion>

Verification requirements:
- Run <command> and inspect the result.
- Run <command> and inspect the result.
- Capture screenshots for <screens>.
- Compare implemented screenshots against the mockups.
- Do another focused pass for meaningful visual or functional gaps before finalizing.

Blocker rule:
If blocked by external tooling, missing credentials, unavailable simulator/browser/device, dependency failure, unsupported API, repeated failing verification, or ambiguous requirements, stop and report:
1. exact blocker;
2. command/error/output;
3. what you tried;
4. what human action is needed;
5. whether the goal can resume afterward.

Loop/stop rule:
Continue iterating until the acceptance checklist passes or a real blocker is reached. Stop after <N> substantial passes/turns if still blocked or if remaining differences are minor pixel-level deltas that do not affect product quality. Do not continue indefinitely.

Final answer format:
1. Summary of what was built
2. Files changed/created
3. Commands run and results
4. Screenshot evidence paths
5. Screen-by-screen fidelity review
6. Acceptance checklist with PASS/PARTIAL/FAIL
7. Known deviations
8. Remaining limitations
9. Recommended next step
```

## Target Executor Notes

### Codex Goal Mode

Use Codex when the user wants the closest match to the video workflow or wants Codex's image generation and Goal Mode loop.

Recommendations:

- Generate the UI spec first as a normal prompt, not as a goal.
- Generate mockups from the UI spec next, also not as a goal.
- Review/select the mockups.
- Attach or reattach the selected mockups when launching the final `/goal`.
- Include max-pass/blocked-stop instructions.

### Hermes

Use Hermes when the prompt pack will feed a Hermes worker, GPT-5.5-backed workflow, or local orchestration.

Recommendations:

- If using Hermes image generation, include generated image paths in the pack.
- If not using image generation, produce mockup prompts and tell the user to generate/attach images before execution.
- Include verification commands and evidence requirements because Hermes final synthesis should not trust worker self-reports blindly.

### Generic Coding Agent

If the target agent does not support persistent goal mode, keep the prompt usable by instructing it to:

- create a plan;
- implement in passes;
- run verification after each pass;
- stop and report blockers;
- produce final evidence.

## Common Pitfalls

1. **Vague finish line.** “Build a polished app” is not a goal. Define screens, behaviors, states, checks, and done criteria.
2. **Mockups without written requirements.** Images help, but agents need text criteria to verify against.
3. **No stop rule.** Visual fidelity loops can run too long chasing tiny differences.
4. **Generic UI slop.** Mockup prompts must include real content, specific screens, and product-specific hierarchy.
5. **Conflicting sources.** Decide whether written spec or mockups win for each kind of conflict.
6. **No evidence requirement.** Require commands, screenshots, and PASS/PARTIAL/FAIL acceptance reporting.
7. **Premature execution.** This skill creates the prompt pack; do not build until explicitly asked.

## Verification Checklist

- [ ] Target executor is identified.
- [ ] Product spec is clear enough to summarize the app and MVP.
- [ ] UI spec covers required screens and states.
- [ ] Mockup prompts are specific and image-generation-ready.
- [ ] Generated mockups, if present, are referenced and translated into written requirements.
- [ ] Final goal prompt includes requirements, non-goals, acceptance criteria, verification, blocker rules, stop rules, and final report format.
- [ ] Readiness verdict is honest.
- [ ] The pack can stand alone without requiring the idea or Superpowers workflows.
