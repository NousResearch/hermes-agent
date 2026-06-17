---
schema_version: hermes.agent_spec/v1alpha1
id: backend-engineer-draft
profile_id: backend-engineer
display_name: Backend Engineer
role_category: implementation
model:
  provider: openai-codex
  model: gpt-5.5
  fallback: []
reasoning_effort: medium
runtime:
  max_turns: 20
  preview_only: true
  safety_level: high
toolsets:
  enabled:
  - terminal
  - file
  - web
  - skills
  disabled: []
mcp: []
sandbox:
  desired: workspace-write
  backend: local
  enforcement_status: declared_only
skills:
  required: []
  recommended:
  - API contract review
  - secret scan
memory:
  policy: durable_only; no transient task progress
artifacts:
  allowed_roots:
  - tests/fixtures/agent_specs/profiles
  - .hermes/project/agent-spec-preview
  requirements:
  - preview-report
gates:
- id: read-only-preview
  owner: context-manager
  blocking: false
---
Draft typed-agent spec fixture for Backend Engineer (backend-engineer).

Source taxonomy: current draft derived read-only from /home/marcus/.hermes/profiles/backend-engineer/SOUL.md and config.yaml. It is a non-mutating Phase 1/2 preview fixture only; it is not installed into the live profile and is not runtime enforcement.
