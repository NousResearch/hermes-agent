---
schema_version: hermes.agent_spec/v1alpha1
id: context-preview
profile_id: context-manager
display_name: Context Preview
role_category: context
model:
  provider: openrouter
  model: anthropic/claude-sonnet-4
  fallback: [openai/gpt-4.1]
  request_overrides:
    api_key: dummy-should-not-leak
reasoning_effort: medium
runtime:
  max_turns: 20
toolsets:
  enabled: [terminal, file]
  disabled: [browser]
mcp:
  - server_id: context7
    required: false
  - server_id: github
    tool: get_issue
    required: true
sandbox:
  desired: workspace-write
  backend: local
  enforcement_status: declared_only
skills:
  required: [kanban-worker]
  recommended: [test-driven-development]
memory:
  policy: preview_only
artifacts:
  allowed_roots: [.hermes/project/agent-spec-preview]
  requirements: [cli-smoke]
gates:
  - id: security-final
    owner: security-engineer
    blocking: true
---
Full preview body.
