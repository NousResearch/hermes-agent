# Plugins Guide

This directory contains repo-shipped plugins and specialized plugin backends.
User-installed plugins live in `~/.hermes/plugins/`; project plugins can live in
`./.hermes/plugins/`; pip entry points are also supported.

## General Plugins

General plugins expose `register(ctx)` and can register:

- lifecycle hooks,
- tools,
- slash commands,
- CLI commands,
- plugin-local skills.

Use the generic plugin API. Do not modify core files for plugin-specific
behavior. If a plugin needs a new capability, add a generic hook or context
method.

Discovery records `plugin.yaml` metadata and imports plugin code only where the
plugin type requires it.

## Memory Providers

Memory-provider plugins live under `plugins/memory/<name>/` and implement
`agent.memory_provider.MemoryProvider`.

The set of in-tree memory providers is closed. New memory backends must ship as
standalone plugin repos or pip packages, installed into `~/.hermes/plugins/` or
via entry points.

Existing in-tree provider bug fixes are fine. New provider directories under
`plugins/memory/` are not.

The active provider is selected by `memory.provider` in `config.yaml`; only one
external provider can be active at a time.

## Model-provider Plugins

Model-provider plugins live under `plugins/model-providers/<name>/`.
They register `ProviderProfile` objects through `providers.register_provider()`.

Discovery is lazy and separate from the general plugin manager:

1. bundled `plugins/model-providers/<name>/`
2. user `$HERMES_HOME/plugins/model-providers/<name>/`
3. legacy `providers/<name>.py`

User providers of the same name override bundled providers.

The general plugin manager may record these manifests for introspection but must
not import them and double-register provider profiles.

## Backend Plugin Families

Context engines, image generation, video generation, web providers, platform
plugins, dashboard plugins, and observability plugins each have their own
loader/orchestrator conventions. Follow the nearest existing plugin and the
matching developer guide under `website/docs/developer-guide/`.

## Dashboard Assets

Dashboard plugins should complement, not replace, the primary PTY-backed chat
surface. Keep failures contained to the plugin panel.

## Kanban Plugin

`plugins/kanban/` contains dashboard assets and deployment helpers for the
durable multi-agent board. The dispatcher can run inside the gateway via
`kanban.dispatch_in_gateway: true` or as a standalone service.

After `kanban.failure_limit` consecutive non-success attempts on a task, the
dispatcher should auto-block it to prevent spin loops.
