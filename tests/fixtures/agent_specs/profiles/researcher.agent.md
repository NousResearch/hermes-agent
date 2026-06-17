---
schema_version: hermes.agent_spec/v1alpha1
id: researcher-draft
profile_id: researcher
display_name: Researcher
role_category: research
model:
  provider: openai-codex
  model: gpt-5.5
  fallback: []
reasoning_effort: medium
runtime:
  max_turns: 20
  preview_only: true
  safety_level: medium
toolsets:
  enabled:
  - terminal
  - file
  - web
  - search
  - x_search
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
  - source reliability assessment
  - source status audit
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
Draft typed-agent spec fixture for Researcher (researcher).

Source taxonomy: current draft derived read-only from /home/marcus/.hermes/profiles/researcher/SOUL.md and config.yaml. It is a non-mutating Phase 1/2 preview fixture only; it is not installed into the live profile and is not runtime enforcement.
