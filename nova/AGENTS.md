# platform/ — Development Guide

Read this before editing anything under `nova/`, `deploy/`, or `docs/platform/`.
The authoritative rules are `docs/platform/ARCHITECTURE_BOUNDARIES.md`; this is the working
summary for people and AI assistants making changes here.

## What this area is

`nova/` is the **platform-owned** half of this repository: the business, tenant and control
layer that turns the upstream Hermes Agent runtime into a reusable, white-labelable enterprise
agent platform. Everything outside the paths above is **upstream-owned** and comes from
`NousResearch/hermes-agent`.

Product working name is **NOVA**. It is a display value in tenant configuration. It must never
appear as a module name, environment-variable prefix, class name, database column, or wire
identifier.

## The one rule that matters most

> **The dependency arrow points one way. `nova/` imports Hermes. Hermes never imports
> `platform`, and must never learn that the platform exists.**

An `if platform_config_enabled:` in an upstream file is not a small compromise — it inverts the
arrow and turns every future upstream change into a potential conflict. CI enforces this.

## Before you edit a file outside platform/

Ask, in this order:

1. **Is there an extension point?** Profiles, distributions, skins, locale overlays, `SOUL.md`,
   toolsets, `mcp.json`, plugins, the dispatcher's `spawn_fn` hook, `plugins/dashboard_auth/`,
   `plugins/observability/`. `ARCHITECTURE_BOUNDARIES.md` §3 is the full table.
   **Almost always the answer is yes and you should stop here.**
2. **If genuinely not:** make the smallest additive hook, and add a line to
   `docs/platform/CORE_PATCHES.md` saying which file, why, and which seam was missing.
3. **Never:** rename an upstream module, replace a `HERMES_*` variable, or edit
   `hermes_cli/kanban_db.py` or `kanban_db_dispatch.py`.

## Never rename these

`hermes_cli`, `hermes_state*`, `agent/`, `cli.py`, `gateway/`, `tools/`, `tests/`; the 225
`HERMES_*` environment variables; `~/.hermes`; `hermes-*` toolset names; `X-Hermes-*` headers;
`hermes://`; the `hermes_agent.plugins` entry-point group; the `hermes` console script; the PyPI
name `hermes-agent`; upstream URLs.

**And above all:** `Hermes-4-405B`, `NousResearch/Hermes-3-Llama-3.1-70B` and friends are *large
language models*, not this runtime. Renaming them breaks model resolution and fails as a provider
404, not a test. `scripts/check_protected_identifiers.py` excludes `[Hh]ermes-[0-9]` absolutely.
Never add an exception.

When a new name is needed, **alias**: `NOVA_FOO` resolving first and falling back to `HERMES_FOO`;
`$NOVA_HOME` → `~/.nova` → `$HERMES_HOME` → `~/.hermes`. Both always work.

## Branding is compiled, never looked up

One source — the tenant's `identity.yaml` — projected by `nova/identity/` into a skin file, a
locale overlay, a `SOUL.md`, and theme tokens served over the Control API. Core reads a skin and a
locale file, which it already does for its own reasons, and never imports `platform.identity`.

If you find yourself adding a product-name string to a `.py` file, you are in the wrong layer.

## The control-plane boundary

The Hermes dashboard mounts inside the gateway process and acts by spawning `hermes` subprocesses
with inherited `os.environ` (`hermes_cli/web_server_gateway.py`). It is a privileged same-host
engineering tool and is **not** the customer surface.

- The platform dashboard talks **only** to the Platform Control API.
- The Control API reads through read-only models and writes through a fixed set of typed commands.
- **No arbitrary subprocess spawn from a web handler.**
- The Hermes dashboard and CLI stay on localhost for engineering. Do not rebuild the Hermes CLI.

## What exists today

`spec/` (AgentSpec, IdentitySpec, TenantBundle), `runtime/` (the `AgentRuntime` contract,
the registry, and the Hermes adapter), `audit/`, `apply.py`, `cli.py`. See
`docs/platform/PHASE_1.md` for what each does, what it deliberately does not do, and how
to extend it.

Dependencies are the standard library and PyYAML. **Do not add a third.** The layer must
import and test without the runtime installed; `tests/platform/test_boundaries.py`
enforces it.

## Never name the package `platform/`

A root `platform/` package shadows the stdlib `platform` module that 40 upstream files
import for OS detection, breaking `platform.system()` everywhere. That is why this
package is `nova/`. `scripts/check_protected_identifiers.py` fails if `platform/__init__.py`
reappears.

## The dashboard talks only to the Control API

`nova/control/static/app.js` may fetch `/platform/v1/*` and nothing else. It must never
learn a runtime path, profile directory or table name. `tests/platform/test_control_server.py`
parses the file and fails on any fetch that bypasses the API prefix.

API data is inserted with `textContent`, never as markup — task titles and agent names are
customer-controlled strings. The Control API is read-only: write methods are refused with
405 before any handler runs. Do not add a write route without the policy layer underneath it.

Status colours are reserved semantics, never themed. A customer's accent must not repaint
"blocked".

## Policy is a security control

`nova/policy/decide.py` is **copied verbatim into the runtime** on every apply. Three rules
follow from that and none of them are negotiable:

1. **It may import nothing but the standard library.** It runs where NOVA is not installed.
2. **Every failure path denies.** A missing document, a corrupt one, an unknown schema, an
   exception: all block. A control that fails open is not a control.
3. **There is one decision function.** The control plane explaining a decision and the
   runtime enforcing it must never be two implementations that can disagree — a test
   asserts the shipped copy is byte-identical.

Deny beats everything, including the worker baseline. When a customer's denial would break
their own agent, the compiler warns at build time; the runtime does not quietly overrule
them at execution time.

Enforcement tests load the plugin **by file path with NOVA off the import path**. Do not
"simplify" them into importing `nova` — that stops testing what actually ships.

## Never label a measurement as a limit

NOVA cannot stop a model call: the runtime's LLM-boundary hooks discard their return
values. There is therefore **no token or cost ceiling**, and none may be added.

Every control carries an enforcement class as data. Only `hard_preemptive` and
`hard_boundary` stop an agent; `soft_advisory`, `recorded_only` and `observed_only` do not
and must never be described as caps, ceilings or limits in any surface.

A control may not claim an enforcing class without a `verified_at` call site you have read
yourself. Enforcement is a claim about a *runtime*, so the register lives with the adapter,
not in `nova/policy/`. Tests reject an unbacked claim and reject any emitted runtime config
key that is not in the register.

## Tests

Platform tests live under `tests/platform/`. Do not modify existing upstream tests — 3,991 files
that are not shipped and are pure merge cost. If an upstream test blocks you, that is usually a
sign the change belongs in `nova/`.

## Definition of done for a platform change

- [ ] No upstream-owned file changed, or a `CORE_PATCHES.md` entry explains why.
- [ ] No upstream module imports `platform.*`.
- [ ] No product name hard-coded in a `.py`, `.ts`, or `.tsx` file.
- [ ] No protected identifier renamed; `scripts/check_protected_identifiers.py` passes.
- [ ] Platform tests pass (`python -m pytest tests/platform/`), and so does the upstream suite.
- [ ] Any change to model-visible state goes through `AuditLog.model_visible_change()`.
- [ ] Any new policy rule fails closed, and `decide.py` still imports only the stdlib.
- [ ] Any new limit is registered with a verified call site, and classified honestly.
- [ ] No new dependency beyond the standard library and PyYAML.
