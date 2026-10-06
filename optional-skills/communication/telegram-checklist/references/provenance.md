# Provenance and architecture

Original project: https://github.com/bablobanov/hermes-telegram-checklist

Imported source revision: `519dcd21a0521298c6ec742ae7b0ff8a3cfe9f85` (v1.1.1).
Human author: Ilya Balobanov (bablobanov); secondary collaborator: Hermes Agent.
Copyright (c) 2026 Ilya Balobanov. The original MIT license is preserved verbatim
in the skill's `LICENSE`; copied helper and tests retain attribution headers.

This is an optional skill with one self-contained CLI helper. It adds no core
model tool, gateway integration, plugin, production setting, or mandatory
runtime dependency. Telethon is an optional online-only dependency, installed
in a separate PM-built environment. Planning and create previews do not import
Telethon; YAML settings use Hermes' YAML module when available or PyYAML in a
standalone environment. The external-product **plugin** restriction does not
require importing this helper as a plugin into core; optional skill placement
follows `skills/AGENTS.md`. Admission remains a maintainer decision.

Adaptations from source:

- Modern English skill sections and tool-framed, skill-relative invocations.
- Non-secret allowlist/session settings read from the launch profile's
  `config.yaml`; legacy process environment inputs remain fallback-compatible.
  Legacy allowlist grants in `.env` must be migrated to config.yaml: offline
  planning deliberately does not read the secret file.
  Explicit settings take priority, including an empty allowlist. Invalid
  explicit allowlist values fail closed rather than widening grants.
- Telethon imported only for online paths; no credentials/client/session writes
  for offline planning. API credentials stay in the selected home/environment.
- Plan task validation extracted without changing the source contract;
  online dispatch uses a handler table to meet the upstream complexity bar.
- JSON and post-write verification exception boundaries retain intentional
  behavior with local code-health rationales, not a blanket lint exclusion.
- Source behavior tests ported to `tests/skills/`; added real subprocess
  profile-config/offline regressions. No live Telegram calls are used.

The helper binds `HERMES_HOME` once in its standalone process. It cannot select
profiles from a sticky default and is not safe to import into a multiprofile
server as shared module state. Always pass the resolved owning home at launch.
A user session has full account authority; the helper's allowlist restricts
this CLI's operations but cannot sandbox another program using the same file.
