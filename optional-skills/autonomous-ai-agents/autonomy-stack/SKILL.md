---
name: autonomy-stack
description: Bootstrap a self-improving autonomy stack for Hermes by installing vetted community plugins, verifying plugin state, and leaning on Hermes's built-in skill creation and improvement loop.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Autonomy, Plugins, Self-Improvement, Telemetry, Goals, Learning]
    related_skills: [honcho, livekit-presence, screenpipe]
    category: autonomous-ai-agents
prerequisites:
  commands: [python3, git]
---

# Autonomy Stack

Use this optional skill when the goal is to make Hermes more autonomous, observable, and self-tuning without forking core.

This stack is intentionally pragmatic:

- install and refresh a curated subset of community plugins from `42-evey/hermes-plugins`
- verify what is already installed under `~/.hermes/plugins`
- rely on Hermes's **built-in** skill creation and self-improvement loop for the learning side instead of adding a second competing mechanism

## Why no separate skill-factory dependency?

Hermes already has a built-in learning loop that creates and improves skills from experience. This skill keeps the autonomy stack focused on pluginized autonomy, telemetry, and orchestration instead of pinning you to another unstable external repo.

## Install

```bash
hermes skills install official/autonomous-ai-agents/autonomy-stack
```

Locate the helper:

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/autonomy-stack/scripts/autonomy_stack.py' -print -quit)"
```

## Quick Start

See current state first:

```bash
python3 "$SCRIPT" doctor
```

Install the recommended plugin subset:

```bash
python3 "$SCRIPT" install
```

Install a custom subset instead:

```bash
python3 "$SCRIPT" install --plugins evey-autonomy,evey-telemetry,evey-status,evey-goals
```

Refresh from upstream later:

```bash
python3 "$SCRIPT" update
```

## Recommended Starter Set

The helper defaults to a conservative set:

- `evey-autonomy`
- `evey-telemetry`
- `evey-status`
- `evey-reflect`
- `evey-learner`
- `evey-goals`

That gives you autonomy, introspection, and a learning loop without dumping the entire plugin pack into `~/.hermes/plugins` immediately.

## Workflow

1. Run `doctor`.
2. Install the recommended subset.
3. Restart Hermes.
4. Inspect plugin state with:

```bash
hermes plugins list
```

5. Let Hermes's built-in skill loop handle learned workflows and refinements over time.

## Notes

- Upstream plugin repo: `https://github.com/42-evey/hermes-plugins`
- The helper keeps a canonical local cache under `~/.hermes/.integrations/autonomy-stack/hermes-plugins`
- Installed plugins are copied into `~/.hermes/plugins`

## Verification

Good output from `doctor` should show:

- whether `git` is available
- which recommended plugins are installed
- whether `evey_utils.py` is present
- a note that Hermes's skill loop is already built in
