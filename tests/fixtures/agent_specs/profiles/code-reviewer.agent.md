---
schema_version: hermes.agent_spec/v1alpha1
id: code-reviewer-draft
profile_id: code-reviewer
display_name: Code Reviewer
role_category: verification/review
model:
  provider: openai-codex
  model: gpt-5.5
  fallback: []
reasoning_effort: medium
runtime:
  max_turns: 20
  preview_only: true
  safety_level: low
toolsets:
  enabled:
  - terminal
  - file
  - web
  - session_search
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
  - requesting-code-review
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
Draft typed-agent spec fixture for Code Reviewer (code-reviewer).

Source taxonomy: current draft derived read-only from /home/marcus/.hermes/profiles/code-reviewer/SOUL.md and config.yaml. It is a non-mutating Phase 1/2 preview fixture only; it is not installed into the live profile and is not runtime enforcement.
