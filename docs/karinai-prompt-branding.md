# KarinAI prompt and branding customization

Status: Accepted planning direction
Date: 2026-06-22

KarinAI is a productized fork/custom distribution of upstream Hermes Agent. Product-facing prompts should present the runtime as the KarinAI agent while preserving upstream Hermes references where they are internal, technical, or needed for sync/debugging.

## Goals

- Avoid hardcoding KarinAI identity directly throughout upstream engine files.
- Render product-facing assistant identity from KarinAI-owned prompt templates.
- Make assistant name, product name, brand/company name, runtime mode, policy mode, and workspace context configurable.
- Keep upstream sync manageable by detecting new product-facing Hermes prompt text after merges.
- Allow future variants such as KarinAI Dev, KarinAI Analyst, enterprise branding, or workspace-specific assistant names without another broad code migration.

## Non-goals

- Blind global search/replace of every `Hermes` string.
- Removing technical references to upstream Hermes internals, docs, compatibility headers, package names, or history.
- Forking the entire prompt system before a thin managed-mode injection layer has been proven insufficient.
- Letting user-editable workspace state override trusted product policy prompts.

## Naming distinction

Use these names intentionally:

```text
Product-facing assistant/runtime: KarinAI agent
Product/application name: KarinAI
Upstream engine/base: Hermes Agent
Runtime mode: managed KarinAI runtime
```

Examples:

- User-facing system prompt: “You are KarinAI, an AI assistant for the user's KarinAI workspace.”
- Runtime/event copy: “KarinAI agent is running tests.”
- Technical docs: “This fork tracks upstream Hermes Agent.”
- Compatibility headers: `X-Hermes-Session-Id` and `X-Hermes-Session-Key` may keep upstream names unless/until an adapter hides them.

## Template-driven prompt model

Product-facing prompts should live under a KarinAI-owned template directory, for example:

```text
karinai/prompts/
  system.base.md.j2
  managed-runtime.md.j2
  tool-policy.beta.md.j2
  scheduling.md.j2
  safety.md.j2
```

Template variables should be supplied by managed runtime config or backend/runtime-manager handoff, for example:

```yaml
assistant_name: KarinAI
product_name: KarinAI
runtime_name: KarinAI agent
upstream_engine_name: Hermes Agent
brand_name: KarinAI
workspace_id: <backend workspace id>
conversation_id: <backend conversation id>
policy_mode: beta
managed_runtime: true
```

The exact template engine can be simple. Jinja-style templates are readable, but a minimal Python renderer or string-template renderer is also acceptable if it avoids unnecessary dependencies.

## Prompt assembly principle

Managed runtime prompts should be assembled in layers:

1. Upstream engine/system requirements that are still needed for correctness.
2. KarinAI product identity and managed-runtime rules.
3. Tool policy and sandbox boundaries rendered by backend/runtime-manager.
4. Scheduling/cron behavior that points to backend-owned schedule intent.
5. User/session/workspace context supplied by backend, not by user-editable files.

The rendered system prompt should be stable for the life of a conversation/run where upstream prompt caching requires stability. Do not mutate prompt identity mid-conversation unless the runtime/session is intentionally restarted or a new session is created.

## Migration and audit scripts

Add scripts under `karinai/scripts/` when implementation begins:

```text
karinai/scripts/audit_prompts.py
karinai/scripts/render_prompts.py
karinai/scripts/check_branding.py
```

Suggested responsibilities:

- Find hardcoded product-facing `You are Hermes Agent`-style strings.
- Render KarinAI prompt templates from sample managed-runtime config.
- Fail if product-facing managed mode still exposes upstream assistant identity.
- Allow internal technical references to Hermes in docs, upstream package paths, compatibility headers, patch logs, and comments marked as internal.
- Produce a report after every upstream sync so new upstream prompt strings can be reviewed deliberately.

## Implementation strategy

Start with a KarinAI-managed prompt injection/rendering layer instead of editing every upstream prompt call site.

Preferred order:

1. Identify where upstream Hermes builds the API-server/run-loop system prompt.
2. Check whether managed runtime can provide an ephemeral or config-rendered system prompt without core changes.
3. Add `karinai/prompts/` templates and render them during managed runtime startup or request construction.
4. Add tests proving managed mode renders KarinAI identity.
5. Patch upstream core only when an upstream prompt path cannot be configured or wrapped cleanly.
6. If a core file is patched, document the patch in `docs/karinai-patches.md`.

## Tests to add

Product tests should verify:

- Managed mode renders assistant identity from template variables.
- Product-facing system prompt does not contain “You are Hermes Agent” in managed mode.
- Template rendering fails loudly if required variables are missing.
- Internal/technical Hermes references are allowed where explicitly expected.
- Schedule-related prompt text says schedules are backend-owned and created through schedule intent.
- Tool-policy prompt text matches the beta managed runtime policy.
- Upstream sync audit catches newly introduced product-facing upstream branding.

## Allowed and forbidden examples

Allowed internal references:

```text
This fork tracks upstream Hermes Agent.
HERMES_HOME is the upstream runtime state directory.
X-Hermes-Session-Id is a compatibility header consumed by the upstream engine.
```

Forbidden in managed product-facing prompt paths:

```text
You are Hermes Agent, an intelligent AI assistant created by Nous Research.
Use Hermes local cron to create durable schedules.
The user may configure unrestricted tools by editing HERMES_HOME.
```

Preferred managed-mode wording:

```text
You are {{ assistant_name }}, an AI assistant for {{ product_name }}.
You run inside a managed KarinAI workspace container.
Use only the tools enabled by KarinAI for this workspace.
Scheduled work is created through KarinAI backend schedule intent; do not create hidden local cron jobs.
```

## Open decisions

- Exact template renderer and file extension.
- Whether prompt variables are rendered once at container startup or per `/v1/runs` request.
- How much of upstream prompt assembly can be configured without core patches.
- Whether enterprise/white-label assistant names are supported in beta or later.
