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
