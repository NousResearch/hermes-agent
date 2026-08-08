---
name: multi-role-router
description: Automatically routes messages to the appropriate worker profile using a lightweight classifier.
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [routing, multi-role, sessions, classifier, automation]
    related_skills: []
---

# multi-role-router Hook

Automatically routes each incoming message to the most appropriate worker-profile
session based on what the message is asking for. The router fires on the
`message:pre_route` event — before the agent starts processing — so the
handoff is transparent to the user.

It is **stateless from the gateway's perspective**: install it by copying the
directory into `~/.hermes/hooks/` and remove it by deleting that copy.

## How it works

1. A lightweight classifier prompt is built from the new message, the current
   role name, and the last few exchanges (from the hook's own `meta.yaml`).
2. The prompt is sent to the cheap triage auxiliary LLM (flash-lite class).
3. If the response names a different role from the current one, the hook
   returns `{"decision": "switch_session", "session_id": "<id>"}` and the
   gateway redirects the turn to that session.
4. If no prior session exists for the target role, the gateway creates one
   naturally and the hook records the new session ID for future redirects.

Short continuations (`"ok thanks"`, `"and what about X?"`) are detected by
a regex fast-path and skip the classifier entirely — they always stay in the
current session.

## Installation

```bash
# Copy the hook into ~/.hermes/hooks/
cp -r ~/.hermes/hermes-agent/optional-skills/multi-role-router ~/.hermes/hooks/

# Restart the gateway (or run hermes gateway restart)
# The hook loads automatically on startup.
```

To uninstall:

```bash
rm -rf ~/.hermes/hooks/multi-role-router
```

## Configuration

All configuration lives under `roles:` and `multi_role_router:` in
`~/.hermes/config.yaml`. Neither key is required — sensible defaults
are built in.

### Example config.yaml snippet

```yaml
# --- multi-role-router ---------------------------------------------------
# Set auto: false to disable automatic routing without uninstalling the hook.
multi_role_router:
  auto: true   # default: true

# Role definitions used by the classifier.
# Each role needs a `description` string — be specific, the classifier reads it.
# Omit this section entirely to use the built-in defaults.
roles:
  code-worker:
    description: >
      Software development, debugging, code review, writing or modifying
      source files, build systems, tests, git operations, package management.

  knowledge-worker:
    description: >
      Research, summarization, document writing, Q&A, information retrieval,
      web search, analysis of text or data, drafting prose or reports.

  ml-worker:
    description: >
      Machine learning experiments, model training, dataset preparation,
      fine-tuning, evaluation metrics, GPU/TPU job management, MLflow, W&B.

  ops-worker:
    description: >
      DevOps, infrastructure, containers, CI/CD, shell scripting, server
      management, cloud deployments, monitoring, on-call operations.

  default:
    description: >
      General-purpose tasks that don't clearly fit another role, or when
      uncertain which role applies.

# Auxiliary LLM used for classification.
# Defaults to the compression slot (cheapest available text provider).
# Override to pin a specific fast/cheap model:
auxiliary:
  triage_specifier:
    provider: openrouter
    model: google/gemini-flash-1.5-8b
    timeout: 10
```

### Custom roles

You can add as many roles as you like. Each role corresponds to a Hermes
worker profile (the profile name and the role name should match). The
`description` field is what the classifier reads — be explicit about what
kinds of tasks belong here.

## Slash commands

Once the hook is installed, you can control it with the `/role` commands
(from RFC #5143, requires hermes-agent >= the release that ships the commands):

| Command | Effect |
|---------|--------|
| `/role list` | Show all configured roles and their current session IDs |
| `/role auto on` | Enable automatic routing (default) |
| `/role auto off` | Disable routing; all messages stay in current session |
| `/role switch <name>` | Manually switch to the named role's session |

## State file

The hook maintains `~/.hermes/hooks/multi-role-router/meta.yaml` with:

- `current_role` — the role active at the last classified message
- `sessions` — map of `role_name → session_id` for quick lookup
- `history` — a rolling window of recent exchanges (last 6 turns) used
  as continuation context for the classifier

Delete this file to reset all session mappings (the hook will rebuild it
from scratch as messages arrive).

## Built-in role defaults

If no `roles:` section is present in config.yaml, these defaults apply:

| Role | What it handles |
|------|----------------|
| `code-worker` | Software development, debugging, code review, git |
| `knowledge-worker` | Research, writing, Q&A, web search |
| `ml-worker` | Model training, datasets, fine-tuning, GPU jobs |
| `ops-worker` | DevOps, infra, containers, CI/CD, shell |
| `default` | Anything else |

## Troubleshooting

**The router keeps switching when I don't want it to.**
Add more role descriptions to disambiguate, or temporarily disable with
`/role auto off` (or `multi_role_router.auto: false` in config.yaml).

**I see `[multi-role-router] No base_url configured` in the logs.**
The hook can't reach the auxiliary LLM. Add an `auxiliary.triage_specifier`
block (or any other `auxiliary.*` block) with a working `base_url` and
`api_key`, or configure a provider like OpenRouter.

**The hook is not loading.**
Check that `~/.hermes/hooks/multi-role-router/HOOK.yaml` and `handler.py`
both exist. Run `hermes gateway status` and look for the hook name in the
loaded hooks list.
