# Predefined task-agent configuration

Hermes exposes a small, predefined set of task-specific agent types through the
`task_agents.definitions` config section. The section is intentionally not an
open plugin registry: each entry must use one of the approved stable ids below
so the main-session and orchestrator integration can validate requests before
spawning a child agent.

Supported ids and purposes:

- `research` — collect and synthesize source-backed information before implementation or decisions.
- `implementation` — build or modify the requested artifact using repository tools and verification checks.
- `review` — evaluate completed work for correctness, regressions, security, and readiness before handoff.
- `documentation` — produce user-facing or integration documentation from implemented behavior and verified examples.

Schema summary:

- `id` (string, required): stable predefined id. Must be one of `research`,
  `implementation`, `review`, or `documentation`; duplicates are rejected.
- `enabled` (boolean, required): whether this configured type may be invoked.
- `purpose` (string, required): human-readable purpose. Purposes must be
  distinct after whitespace/case normalization.
- `invocation` (mapping, required): how the integration invokes this type.
  `entrypoint` must currently be `delegate_task`, `task_category` must match
  `id`, and `parameters.required` must include `prompt`.
- `runtime` (mapping, optional): existing model/runtime knobs the invocation
  integration may pass through or apply when spawning the agent. Supported keys:
  `model`, `provider`, `base_url`, `api_key`, `api_mode`, `reasoning_effort`,
  `max_iterations`, `timeout_seconds`, `max_summary_chars`, `enabled_toolsets`,
  `disabled_toolsets`, and `skills`.

Validated example:

```yaml
task_agents:
  definitions:
    - id: research
      enabled: true
      purpose: Collect and synthesize source-backed information before implementation or decisions.
      invocation:
        entrypoint: delegate_task
        task_category: research
        parameters:
          required: [prompt]
          optional: [context, limit, sources, deliverable]
      runtime:
        enabled_toolsets: [web, file, terminal, skills]
        skills: [deep-research]
        model: ""
        provider: ""
        base_url: ""
        api_key: ""
        api_mode: ""
        reasoning_effort: ""
        max_iterations: 50
        timeout_seconds: 0
        max_summary_chars: 24000

    - id: implementation
      enabled: true
      purpose: Build or modify the requested artifact using repository tools and verification checks.
      invocation:
        entrypoint: delegate_task
        task_category: implementation
        parameters:
          required: [prompt]
          optional: [context, workdir, tests, constraints]
      runtime:
        enabled_toolsets: [file, terminal, search, skills, todo]
        skills: [test-driven-development]
        model: ""
        provider: ""
        base_url: ""
        api_key: ""
        api_mode: ""
        reasoning_effort: ""
        max_iterations: 50
        timeout_seconds: 0
        max_summary_chars: 24000

    - id: review
      enabled: true
      purpose: Evaluate completed work for correctness, regressions, security, and readiness before handoff.
      invocation:
        entrypoint: delegate_task
        task_category: review
        parameters:
          required: [prompt]
          optional: [context, diff, tests, risk_focus]
      runtime:
        enabled_toolsets: [file, terminal, search, skills]
        skills: [github-code-review, requesting-code-review]
        model: ""
        provider: ""
        base_url: ""
        api_key: ""
        api_mode: ""
        reasoning_effort: ""
        max_iterations: 50
        timeout_seconds: 0
        max_summary_chars: 24000

    - id: documentation
      enabled: true
      purpose: Produce user-facing or integration documentation from the implemented behavior and verified examples.
      invocation:
        entrypoint: delegate_task
        task_category: documentation
        parameters:
          required: [prompt]
          optional: [context, audience, format, examples]
      runtime:
        enabled_toolsets: [file, search, skills]
        skills: []
        model: ""
        provider: ""
        base_url: ""
        api_key: ""
        api_mode: ""
        reasoning_effort: ""
        max_iterations: 50
        timeout_seconds: 0
        max_summary_chars: 24000
```

Integration notes:

- Use `hermes_cli.agent_types.load_task_agent_definitions(config)` to obtain
  validated `TaskAgentDefinition` objects keyed by id. It raises
  `TaskAgentConfigError` if the section contains errors.
- Use `hermes_cli.config.validate_config_structure(config)` for user-facing
  diagnostics; it reports malformed, duplicate, unknown, disabled-shape, and
  shared-purpose issues as `ConfigIssue` entries.
- Unknown ids are rejected even if their shape otherwise looks valid. To add a
  new approved type, update `PREDEFINED_TASK_AGENT_IDS`,
  `PREDEFINED_TASK_AGENT_DEFINITIONS`, tests, and this document together.
- Empty string model/provider values mean “inherit the parent session setting,”
  matching the existing delegation config convention. A `timeout_seconds` or
  `max_summary_chars` value of `0` preserves the existing unlimited/default
  runtime behavior where supported.
