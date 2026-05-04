# skill_intake

## What it is

`skill_intake` is a governed intake pipeline for candidate skills.

It supports four stages:
- `audit`
- `convert`
- `re-audit`
- `admit`

Use it to inspect third-party or legacy skill material before trusting or installing it.

## Core rule

Always use the pipeline in this order:
1. `audit`
2. `convert`
3. `re-audit`
4. `admit`

Do not convert and install directly.

## Supported inputs

- local file
- local directory
- raw URL
- GitHub file URL
- GitHub tree URL
- GitHub repo URL
- `CLAUDE.md` / project-instructions markdown
- generic workflow markdown
- OpenClaw-like package (`AGENT.md` + `scripts/` and/or `references/`)

## Modes

### audit
Evaluates:
- compatibility
- security risk
- semantic portability
- recommendation
- provenance

Example payload:
```json
{"mode":"audit","source":"./candidate-skill"}
```

### convert
Generates a Hermes draft skill.

Capabilities:
- `discarded` tracking
- `mapping` from source to generated output
- optional `out_dir`
- preservation of `references/` and `scripts/` for OpenClaw-like packages

Example payload:
```json
{"mode":"convert","source":"./CLAUDE.md","out_dir":"./out"}
```

### re-audit
Revalidates converted output and produces readiness:
- `ready`
- `review`
- `blocked`

Example payload:
```json
{"mode":"re-audit","generated_skill":{...}}
```

### admit
Applies policy gate:
- `conservative`
- `balanced`
- `permissive`

Example payload:
```json
{"mode":"admit","reaudit_result":{...},"policy":"conservative"}
```

## Current conversion output structure

Typical generated `SKILL.md` sections:
- `When to use`
- `Prerequisites`
- `Steps`
- `Commands`
- `Examples`
- `Pitfalls`
- `Verification`

## Tests

Focused tests:
```bash
uv run --extra dev python -m pytest tests/tools/test_skill_intake_tool.py -q -o 'addopts='
uv run --extra dev python -m pytest tests/tools/test_skill_intake_tool_integration.py -q -o 'addopts='
```

Regression subset:
```bash
uv run --extra dev python -m pytest tests/test_toolsets.py tests/tools/test_skill_intake_tool.py tests/tools/test_skill_intake_tool_integration.py -q -o 'addopts='
```

## Current status

V2 baseline is functionally complete.

Post-V2 work belongs in V2.1+:
- richer provenance
- larger fixtures
- eval planning
- sandbox/CI hardening
