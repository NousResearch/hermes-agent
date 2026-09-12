# RFC: Headed cloud sessions (PTY + WebVNC + CUA)

**Status:** interfaces only (this PR). No core desktop implementation.

**Origin:** [NousResearch/hermes-agent#108592](https://github.com/NousResearch/hermes-agent/issues/108592)

**Demand signal:** operators want to *watch and take over* a managed cloud computer from the app (terminal + GUI + computer use), then leave that session for the agent after the local PC is off. Browser-only noVNC (#92524) is the closest existing ask; this RFC is the broader **session** product.

## Product

One headed cloud session exposes **three surfaces on one display**:

1. **Remote terminal** — dashboard `/api/pty` (or equivalent) attached to the session VM/sandbox.
2. **WebVNC** — session-scoped viewer/control of that same display (operator watch + takeover).
3. **CUA** — existing `computer_use` / `ComputerUseBackend` aimed at that display, not the operator laptop.

Inspired by OpenClaw headed cloud sessions (Steinberger, 2026-09-10): remote terminal, WebVNC, and CUA for computer use on one fast cloud session.

This is **not** headless Cursor Cloud Agents ([#107480](https://github.com/NousResearch/hermes-agent/issues/107480)). That skill launches/manages agents via API. Complementary. Headed sessions are the display product.

## Compose, do not dump a second desktop

Do **not** mint a second X11/VNC/CUA stack in Hermes core. Bind the existing seams to one `DisplayTarget` fingerprint:

| Seam | Role |
| --- | --- |
| [#90380](https://github.com/NousResearch/hermes-agent/pull/90380) `ComputerUseBackend` | CUA provider interface |
| [#61310](https://github.com/NousResearch/hermes-agent/pull/61310) | Linux display requirement for computer use |
| [#97859](https://github.com/NousResearch/hermes-agent/pull/97859) Bot Desktop / [#17258](https://github.com/NousResearch/hermes-agent/pull/17258) optional local noVNC | WebVNC viewer, not a parallel desktop |
| [#90374](https://github.com/NousResearch/hermes-agent/pull/90374) gateway target invariant | Fail closed when CUA fingerprint ≠ session fingerprint |
| [#92524](https://github.com/NousResearch/hermes-agent/issues/92524) hosted noVNC of the **cloud browser** | Browser login/CAPTCHA handoff only (`host_kind=browser_handoff`). Rejected for headed-session CUA. |

`ComputerUseBackend.bound_display_fingerprint()` is additive (default `None`). Unbound backends cannot pass `assert_computer_use_targets_session`.

## Interfaces (this change)

* `agent.headed_session.DisplayTarget` — fingerprint + `host_kind`.
* `HeadedSessionSurfaces` — PTY + WebVNC URLs on that target.
* `HeadedCloudSessionProvider` ABC — `create_session` / `close_session`.
* `assert_computer_use_targets_session` — #90374 fail-closed; refuse #92524 for CUA.
* `agent.headed_session_registry` — name → provider map (same scope model as `terminal_env_registry`).

A concrete provider (Nous-hosted, Modal, Daytona, …) is **out of scope** until maintainer review. This RFC must not ship hermes-desktop binaries, Xvfb launchers, or a new CUA driver.

## Non-goals

* Reminting or implementing #107480 / headless cloud-agent skills.
* Merging or opening a PR on `NousResearch/hermes-agent` from this fork work until `github_writes` on origin is authorized.
* Treating Camofox+self-hosted VNC as the managed product.
