# Muncho fork-only operations

This directory contains fork-only operational helpers for Cloud Muncho. It is
not an upstream Hermes product surface and must not be proposed to
`NousResearch/hermes-agent`.

The helpers deliberately reuse existing Hermes behavior instead of widening
the core:

- `auto_sync_hardening.py` classifies superseded automation-owned sync PRs and
  deduplicates unchanged safety-gate notifications.
- `planned_gateway_restart.sh` writes the existing Hermes planned-stop marker
  before an external service manager restarts the gateway.

Cloud integration keeps the mutable wrappers outside the active release:

- `/opt/adventico-ai-platform/hermes-home/scripts/fork_upstream_auto_sync_pr_routine.py`
- `/usr/local/sbin/muncho-auto-deploy-release`

Rollout of either helper is a separate exact-action production change.
