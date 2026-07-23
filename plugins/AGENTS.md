# Hermes Plugin Development Guide

These instructions apply under `plugins/`. The root `AGENTS.md` remains
authoritative for cross-cutting policy.

## Boundary rule

Plugins must not modify core files such as `run_agent.py`, `cli.py`,
`gateway/run.py`, or `hermes_cli/main.py`. If a plugin needs something the
framework does not expose, add a generic hook, context method, or abstract
interface that can serve multiple plugins. Never hardcode one plugin into core.

Third-party product integrations do not belong in this repository. Vendor SaaS
connectors, observability backends, analytics products, and similar integrations
ship as standalone plugin repositories installed into `$HERMES_HOME/plugins/`
or via Python entry points. Existing in-tree integrations are not precedent for
adding more.

The in-tree memory-provider set is closed. New memory backends also ship as
standalone plugins; fixes to existing bundled providers remain welcome.

## General plugins

`hermes_cli/plugins.py` discovers:

1. bundled plugins under `<repo>/plugins/<name>/`;
2. user plugins under `$HERMES_HOME/plugins/<name>/`;
3. Python entry points in the `hermes_agent.plugins` group.

User plugins override bundled plugins of the same name.

A general plugin uses `plugin.yaml` and may register:

- lifecycle hooks;
- tools and toolsets;
- CLI commands;
- configuration defaults;
- setup steps or migrations.

Keep registration inside the plugin directory. If the plugin needs shared
surface, make that surface category-level and prove it with a real consumer.
Do not add speculative hooks.

## Memory providers

Memory providers implement the `MemoryProvider` abstract interface and register
through `plugins/memory/__init__.py`. Provider-specific CLI commands should be
exposed only when that provider is active so disabled providers do not clutter
help or command discovery.

Provider state must use `get_hermes_home()` and remain profile-local.

## Model providers

Model providers live under `plugins/model-providers/<name>/`. Their
`__init__.py` registers a `ProviderProfile`. Discovery is lazy and separate
from the general plugin manager:

1. bundled model providers;
2. user model providers;
3. legacy modules under `providers/`.

Registration is last-writer-wins, allowing user plugins to override bundled
profiles. The general plugin manager may record a `kind: model-provider`
manifest but must not import it a second time.

Full authoring documentation:
`website/docs/developer-guide/model-provider-plugin.md`.

## Other provider families

Context engines, image-generation providers, and similar families use an
abstract interface plus orchestrator and per-provider directories. Prefer that
pattern when three or more implementations share a category.

Reference/example plugins belong in the `hermes-example-plugins` companion
repository unless they are maintained, first-party runtime features.

## Review checklist

- No plugin-specific branch was added to core.
- State and configuration are profile-safe.
- Optional dependencies are gated and bounded.
- Tools are unavailable when prerequisites are absent.
- Setup uses existing Hermes configuration UX.
- The real discovery and registration path is tested.
- User plugins can override bundled implementations without double import.
