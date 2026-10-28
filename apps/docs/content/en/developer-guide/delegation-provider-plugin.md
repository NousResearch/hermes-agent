---
title: Delegation Provider Plugins
sidebar_label: Delegation provider plugins
---

# Delegation Provider Plugins

A delegation provider decides **how each `delegate_task` task becomes a child run**: which model,
provider, endpoint and reasoning effort the subagent uses, or which Hermes profile it runs as. It can
also add per-task parameters to the `delegate_task` schema, such as an enum of profile names or of
titled model options, so the model picks a route per task.

Hermes core keeps the whole subagent lifecycle: spawning, isolation, leaf/orchestrator roles, depth
limits, concurrency, background completion units, stop fan-out, result assembly, `output_schema`
validation and progress. A provider only answers "which route does this task take".

The active provider is `delegation.backend` in `config.yaml`; a provider's own settings live under
`delegation.backends.<backend name>`. The `default` strategy is today's behaviour: every child follows
the `delegation` section (`delegation.provider`, `delegation.model`, `delegation.base_url`, ...), or
inherits the parent's route when those are empty. It is itself a plugin, the bundled
`plugins/delegation/default/` (`kind: backend`, always loaded), registered through the same
`ctx.register_delegation_provider()` seam as yours; it is also the smallest complete example.
A configured backend that is not installed or reports unavailable **fails closed**: `delegate_task`
returns a tool error naming the backend and the fix, no child starts, and its schema parameters are
absent. `/delegation` switches the backend (`agent.delegation_backends`, below).

## Skeleton

```python
# ~/.hermes/plugins/delegation-tiers/__init__.py
from agent.delegation_provider import DelegationProvider, DelegationRouteError


class Tiers(DelegationProvider):
    @property
    def name(self) -> str:
        return "tiers"                      # delegation.backend: tiers

    def task_parameters(self, config):
        tiers = self.settings(config).get("tiers") or []  # delegation.backends.tiers.tiers
        return {"tier": {
            "type": "string",
            "enum": [t["name"] for t in tiers],
            "description": "; ".join(f"{t['name']}: {t['description']}" for t in tiers),
        }}

    def description(self, config):
        return "Set `tier` per task to pick the model that fits it."

    def route_task(self, task, config):
        for t in self.settings(config).get("tiers") or []:
            if t["name"] == task.get("tier"):
                return {"provider": t["provider"], "model": t["model"], "label": t["name"]}
        if task.get("tier"):
            raise DelegationRouteError(f"Unknown tier {task['tier']!r}.")
        return dict(config)                 # no tier: the default route


def register(ctx):
    ctx.register_delegation_provider(Tiers())
```

```yaml
# plugin.yaml
name: delegation-tiers
version: 1.0.0
description: Pick a model tier per subagent.
```

```yaml
# config.yaml
delegation:
  backend: tiers
  backends:
    tiers:                                  # == the backend name
      tiers:
        - {name: quick, description: Lookups and summaries., provider: openrouter, model: google/gemini-3-flash-preview}
```

```bash
hermes plugins enable delegation-tiers
```

## The interface

`DelegationProvider` (`agent/delegation_provider.py`) subclasses the shared `ProviderBase`. Every
method receives `config`, the `delegation` section of the active profile's `config.yaml`: `dict(config)`
is the default route, and `self.settings(config)` is your backend's own block,
`delegation.backends.<name>` (empty when unset). Keep your settings there, never beside the core
`delegation` keys.

| Member | Required | Contract |
|---|---|---|
| `name` | yes | The `delegation.backend` value. `default` belongs to the bundled default strategy. |
| `route_task(task, config)` | yes | Return the route dict for one task. `task` carries `goal`, `context`, `index` and the values the model supplied for your parameters. Raise `DelegationRouteError` to refuse; its message is the tool error the model reads. |
| `task_parameters(config)` | no | `{name: json_schema}` merged into `tasks.items.properties`. Core enforces `enum` before `route_task` runs and refuses core-owned names (`goal`, `context`, `index`, `output_schema`, `images`, `group`, `role`, `toolsets`, `provider`, `model`, `base_url`, `api_key`, `api_mode`, `acp_command`, `acp_args`). |
| `description(config)` | no | Guidance appended to the `delegate_task` description. |
| `is_available()` | no | No network calls. `False` fails delegation closed while your backend is selected: `delegate_task` returns a tool error naming it and the fix (`/delegation default`, or `hermes config set delegation.backend default`); it never switches to `default` on its own. |
| `settings(config)` | inherited | `delegation.backends.<name>` as a mapping (empty when unset). Do not override. |

### Routes

A route is a dict with any of:

- `model`, `provider`, `base_url`, `api_key`, `api_mode`, `request_overrides`, `reasoning_effort`,
  `fallback_providers`: shaped exactly like the `delegation` section and resolved by the same
  credential resolver. An empty route inherits the parent's route; `dict(config)` is the default route.
  A `reasoning_effort` that is not a known level (`none`, `minimal`, `low`, `medium`, `high`,
  `xhigh`, ...) refuses the task with a tool error instead of falling back to the parent's level; the
  `delegation.reasoning_effort` value itself keeps its usual warn-and-inherit rule.
- `service_tier`: `normal`, `fast` or `ultrafast`. Core checks it against the resolved route with the
  same gates as `/fast` (and, for `ultrafast`, the signed-in Codex account's catalog). A tier the
  route does not offer refuses the task; otherwise it applies to that child only and is re-applied
  after the child is built, so a custom provider's `extra_body` cannot restore another tier.
- `profile`: the name of a Hermes profile. The child runs **as that profile**: its `config.yaml`,
  primary model and credentials (a `model.default` alias or named provider resolves exactly as at
  CLI startup), credential pool, fallback chain, reasoning (`agent.reasoning_effort` and
  `agent.reasoning_overrides`), skills, `SOUL.md` and terminal policy. Core binds that profile's
  scope (HERMES_HOME override, secret scope, multiplex semantics, terminal policy) around the child's
  construction, its credential lease and every conversation turn, the same binding a multiplexed
  gateway applies to a routed turn, so a secret missing from the target profile is never read from
  the launching profile's environment. The child's transcript stays in the launching profile's
  session database (it is the parent's work) and it keeps the parent's toolset. Other route keys are
  ignored on a profile route.
- `label`: short text identifying the route (an option title, a profile name).

Unknown keys are dropped. On a profile route, `service_tier` and `reasoning_effort` are ignored too.

A non-profile route without `fallback_providers` of its own gets `delegation.fallback_providers`; return
`"fallback_providers": []` to disable fallback for a route. A profile route keeps its profile's chain.

### Refusals are per task

A `DelegationRouteError` from `route_task`, a parameter your backend did not declare, a value outside
your parameter's `enum`, an unknown
`reasoning_effort`, a tier the route does not offer, or a credential the route cannot resolve refuses
**that task only**: no child is built for it, its result entry is
`{"status": "error", "error": "<your message>", "route": {"backend": "<name>", "refused": true}}`, and
the other tasks run. A background call lists refused entries under `refused` in its dispatch payload.
The call is a tool error only when every task is refused.

### Result `route`

Every result entry of a task routed by a non-default backend carries `route`: the backend, the label
(when set), and what the child actually ran on, read off the child when its result is assembled:

```json
"route": {"backend": "model_map", "label": "deep", "provider": "openai-codex", "model": "gpt-6-astra",
          "reasoning_effort": "high", "service_tier": "fast"}
```

`reasoning_effort` is `null` when the provider's default applies; `fallback_activated: true` is added
when a fallback provider took over. The `default` backend's children leave the result entry unchanged.

## Picker metadata and switching backends

`display_name` and `summary` (one user-facing line; never shown to the model) label your backend in
pickers such as `/delegation`. A plugin uses `ctx.delegation_backends` (same methods; `list_backends()` returns
plain dicts). It is a host remote facade, so under `plugins.isolation: host` it acts on the Hermes process, not
the plugin host. Core code calls `agent.delegation_backends` directly:

| Call | Does |
|---|---|
| `list_backends() -> list[BackendInfo]` | `default` first, then every registered backend: `name`, `label`, `description`, `available`, `reason` (the fix, when unavailable). |
| `set_global_backend(name, *, session_key="", apply_now=False)` | Write `delegation.backend` to the active profile's `config.yaml`. `apply_now=True` also rebuilds the tool list of the session `session_key` names at its next turn. Raises `PermissionError` on a package-manager-managed install, and `ValueError` with the reason for a backend that cannot run (or `apply_now` without a `session_key`); nothing is written then. |

A plugin command that edits model routes can reuse the interactive model picker under its own heading:
`hermes_cli.auth_model_picker.prompt_model_selection(model_ids, current_model="", title="Select the model for this route:")`
returns the chosen model id, or `None` when cancelled.

`session_key` is the `session_key` a plugin command handler receives. A session routes on the backend its
`delegate_task` schema (your parameters) was built from: core captures it with the session's tools and pins it
with them to the session row, so a fresh agent continuing the session (gateway, `--resume`) restores both. The
schema is part of the cached prompt, so a save, or a hand edit of `delegation.backend`, changes nothing in a running
session and applies from its next session (`/new`). `apply_now=True` (what `/delegation <backend> --now` passes)
rebuilds that session's tools at its next turn instead, and routing moves with them: one prompt-cache break by the
user's choice, the same opt-in as `/skills install --now`. Context compression rebuilds the tools anyway and picks up
the saved backend. A task carrying a parameter its session's backend does not declare (a model inventing one) is
refused alone, never dropped and run on the default route:

```text
Task 0: the delegation backend is 'profile_routes', which does not take 'option' (declared by 'model_map'), so
this task was not started. This session's delegate_task was built for another backend: run
`/delegation profile_routes --now` to rebuild this session's delegate_task for 'profile_routes', or start a new
session with /new.
```

A parameter no installed backend declares is refused the same way, with "Resend it without ..." first.

## Prompt caching

The `delegate_task` definition, including your parameters and description, is built once when a
session's tool list is assembled. `task_parameters` and `description` must be deterministic for a
given `config`: same config, same bytes. Never derive them from the clock, the network or anything
that changes during a conversation.

## Process isolation

Every argument and return value is plain JSON, so a provider runs unchanged under
`plugins.isolation: host`: Hermes holds a proxy that subclasses `DelegationProvider` and every call
runs in the plugin host process.

## Testing

Load the plugin through the real discovery path against a temporary `HERMES_HOME` whose
`config.yaml` sets `plugins.enabled`, `delegation.backend` and `delegation.backends.<name>`, then assert on behaviour: the
`delegate_task` definition from `tools.registry.registry.get_definitions({"delegate_task"})` carries
your parameter, and `tools.delegate_tool.delegate_task(...)` hands each child the route you chose.
`tests/tools/test_delegation_backends.py` in the Hermes repo is a complete example.
