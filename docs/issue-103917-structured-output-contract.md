# Issue #103917 — simplified structured-output contract

## Verdict

Issue #103917 is real as an API-ergonomics and reliability gap. The existing structured-output feature is useful, but its model-facing contract asks the parent model to emit an arbitrary JSON Schema object inside a `delegate_task` call. That nests a second JSON language inside the tool-call JSON and increases failure risk for weaker tool callers.

The exact provider report (five malformed `deepseek-v4-flash` calls) cannot be replayed offline without that provider/session. The mechanism is nevertheless reproducible locally: the served `delegate_task` schema exposes `tasks[].output_schema` as a free-form object, and the current sanitizer turns that node into an object with `properties: {}`. The parent therefore has to produce a nested schema payload that is both syntactically valid tool-call JSON and semantically useful.

## Remote research and duplicate check

No pull request for #103917 or for the exact phrase `simplified structured-output` was open when this investigation was run. The following work is related but not duplicated by this change:

Two live triage comments on the issue independently confirm the feature request as valid and recommend keeping raw `output_schema` while adding a simpler shorthand for common cases. That matches the implementation below.

- #81144 (merged): introduced the raw per-task `output_schema`, the child output-contract prompt, validation, and one bounded correction retry.
- #96424 (merged): made `tasks[]` the only advertised spawn shape and kept legacy top-level arguments handler-compatible but unadvertised.
- #99232 (merged): makes a schema-invalid final child result report `status: "failed"`.
- #89537 (open): forwards legacy top-level `output_schema` through `AIAgent._dispatch_delegate_task`; it is a separate forwarding gap.
- #96355 (open): reports the pre-#99232 contradiction between `status: "completed"` and `schema_valid: false`.
- #96734 (open): reports the sanitizer flattening free-form object parameters to `properties: {}`. #96765 is an open candidate for that sanitizer bug; #102804 is a separate open Bedrock compatibility candidate, and #102844 was closed as a duplicate of it. This implementation does not copy either candidate's sanitizer patch.
- #41823 (open): records weak-model failures caused by an under-constrained `delegate_task` schema. Its proposed top-level `goal` requirement conflicts with the current, intentional `tasks[]`-only advertised interface, but its reliability evidence is relevant.
- #85483 (open): adds a separate child-stop/partial-output result contract; it does not define or compile caller-declared output schemas.
- #68499 (open): proposes broader lifecycle/task-outcome consolidation across async, durable, gateway, Desktop, and TUI boundaries; it is adjacent to result status handling, not the shorthand contract.

## Graphify evidence

The repository graph was used before broad source traversal:

- `graphify explain 'delegate_task'` resolves the primary node to `tools/delegate_tool.py:348` and lists its callers/tests.
- `graphify path 'delegate_task()' '_coerce_task_schemas()'` gives the direct dispatch path: `delegate_task() --calls--> _coerce_task_schemas()`.
- `graphify path 'delegate_task()' '_validate_child_output_schema()' --undirected` identifies the shared result-validation boundary through `tools/delegate_tool_child_run.py`.
- `graphify affected 'tools/delegation_output_schema.py' --depth 2` identifies `_build_children`, `_coerce_task_schemas`, `_run_single_child`, `_validate_child_output_schema`, and the existing delegation test files.

The graph was treated as navigation evidence and checked against the current source lines before editing.

## Root cause

1. `tools/delegate_tool.py` advertises `tasks[].output_schema` as an object without a fixed property vocabulary.
2. `tools/schema_sanitizer.py::_sanitize_node()` unconditionally injects `properties: {}` into object nodes without properties. `model_tools.get_tool_definitions()` applies this sanitizer to the model-facing schema for every backend.
3. A model must author nested `properties`, nested field schemas, `items`, and `required` arrays inside another JSON object. The result is a high-syntax tool-call path even when the caller only needs fields such as `summary`, `changed_files`, and `verification`.
4. Omitting `output_schema` avoids the syntax burden but removes the machine-validated contract.

The existing child prompt, validator, one-retry bound, and result metadata are not the cause and are reused.

## Implemented solution

The advertised task item now supports an additive shorthand:

```json
{
  "goal": "Review the change",
  "output_fields": {
    "summary": "string",
    "changed_files": "string[]",
    "verification": "string",
    "blockers": "string[]"
  },
  "required_output_fields": ["summary", "verification"]
}
```

`tools/delegation_output_schema.py::compile_output_fields()` compiles this into the existing contract representation:

```json
{
  "type": "object",
  "properties": {
    "summary": {"type": "string"},
    "changed_files": {"type": "array", "items": {"type": "string"}},
    "verification": {"type": "string"},
    "blockers": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["summary", "verification"]
}
```

Supported type names are `string`, `integer`, `number`, `boolean`, `object`, and each of those names with `[]` for an array. The vocabulary is intentionally closed; nested constraints, enums, unions, and other JSON Schema features remain available through the raw `output_schema` escape hatch.

Resolution is centralized in `coerce_output_contract()` and used by the existing `_coerce_task_schemas()` seam. It rejects malformed shorthand before any child is built, rejects an empty shorthand contract instead of silently accepting any object, rejects undeclared required fields and duplicate required names, and rejects mixing `output_schema` with `output_fields`.

The rest of the path is unchanged:

1. compile the shorthand once at dispatch;
2. append the compiled schema to the child `OUTPUT CONTRACT` prompt;
3. validate the final child response with the existing validator;
4. use the existing single bounded correction retry;
5. expose the existing `schema_valid`, `schema_errors`, and `schema_retries` result metadata.

Schema-less delegations keep their existing result shape. The raw advanced form remains available and was not reimplemented or replaced.

## Why this addresses the cause

The parent now emits a flat field-to-type map and a flat required-name list instead of a nested JSON Schema AST. The model-facing `output_fields` declaration also carries an explicit `additionalProperties` value schema, so the current sanitizer preserves the meaning that arbitrary field names map to type-name strings. The fix lowers generation complexity without weakening the runtime contract or duplicating the existing validator/retry implementation.

## Verification

Focused canonical runs, all with `scripts/run_tests.sh`:

- `tests/tools/test_delegate_output_schema.py`: 36 passed.
- `tests/tools/test_delegate.py`: 81 passed.
- `tests/tools/test_schema_sanitizer.py`: 32 passed.
- TDD RED evidence: before the implementation, the new focused tests failed during collection because `compile_output_fields` did not exist; after implementation, the same focused file passed all 36 tests.
- A direct served-schema inspection confirmed `output_fields.additionalProperties == {"type": "string"}` after the normal `model_tools.get_tool_definitions()` sanitization path.
- A direct compiler probe confirmed the issue's representative fields compile to the expected object/array schema and return no error.
- The four-file delegation regression set (`test_delegate_batch_validation.py`, `test_delegate_control_actions.py`, `test_delegate_capability_inheritance.py`, and `test_delegate_toolset_scope.py`) passed 66 tests.
- `tests/test_model_tools.py` and `tests/test_model_tools_async_bridge.py` passed 40 tests.
- Ruff lint, Python bytecode compilation, and `git diff --check` passed.
- The RED check kept the new test file while temporarily restoring the three production files to `origin/main`; collection failed with `ImportError: cannot import name 'compile_output_fields'`, then the restored implementation passed all 36 tests.
- `npm run check` was attempted and exited 1 because the checkout has no installed JavaScript dependencies/type definitions (`react`, `@nanostores/react`, `vitest`, `@types/node`, and related packages). No JavaScript or TypeScript file is changed by this PR.
- `ruff format --check` exits 1 for the four files even at the unmodified `origin/main` baseline; it would reformat pre-existing file-wide style, so no unrelated formatter rewrite was included.

The first baseline run of the pre-existing output-schema file had 23 passed and 4 failures because the repository's local `.venv` did not contain the already-used `jsonschema` and `anthropic` packages. Installing those exact existing package versions into the local test environment removed that environmental blocker; no project dependency file was changed.

## Infographic artifact

The requested infographic was generated in English with the `bento-grid` layout and `technical-schematic` blueprint style, then inspected for clipping, duplicated text, and legibility. The final repair pass corrected Panel 4 to read `4) ONE CANONICAL PIPELINE`.

- Requested output: `C:\Users\Nitro\hermes-agent\penai_codex_gpt-image-2-medium_20260904_190447_021c2adb.png`
- Format: PNG, 1536 × 1024, RGB
- The generated PNG is intentionally not committed: the repository `.gitignore` policy keeps PR infographics as local artifacts rather than adding binary files to the source tree.

## Files changed

- `tools/delegation_output_schema.py`: one closed-vocabulary compiler and one shared contract resolver.
- `tools/delegate_tool_tasks.py`: route task contracts through the shared resolver and reject invalid combinations before child construction.
- `tools/delegate_tool.py`: advertise the additive shorthand and guide models toward it for flat contracts.
- `tests/tools/test_delegate_output_schema.py`: compiler, validation, served-schema, dispatch, conflict, and empty-contract regressions.
- `website/docs/reference/tools-reference.md`: public usage example for the shorthand.
- `docs/issue-103917-structured-output-contract.md`: investigation, related-work, root-cause, and verification record.
- `penai_codex_gpt-image-2-medium_20260904_190447_021c2adb.png`: locally generated and inspected infographic artifact; intentionally excluded from the commit by repository policy.

No `.env`, credential store, project dependency manifest, sanitizer implementation, or unrelated source path was modified by this change.
