# Dashboard Downstream Snapshot Feed

This document defines how the proving-project dashboards feed downstream operating systems.

The canonical machine-readable file is `docs/design/dashboard-downstream-snapshot-feed.json`.

## Rule

Producer projects own runtime truth. Consumer projects may summarize, rank, and route that truth, but they must not invent healthy state when a producer feed is missing, stale, unauthorized, or invalid.

## Producers

| Project | Runtime Signals | Consumers |
| --- | --- | --- |
| Khashi VC | research command, market cartography, scheduler capacity, segment intelligence, strategy readiness, system health | Hermes OS, TLC Capital Group OS |
| Media Engine | production runs, cost ledger, approvals, publishing, human-video packages, social readiness, system health | Media Business Operations, Hermes OS, TLC Capital Group OS |

## Consumers

| Consumer | Purpose |
| --- | --- |
| Media Business Operations | Rolls Media Engine production, publishing, cost, channel, and human-video state into the media operating-company dashboard. |
| Hermes OS | Central runtime command surface for deployed project health, logs, dashboard discovery, and production inspection. |
| TLC Capital Group OS | Enterprise readiness, executive blockers, business-unit rollups, and cross-project production status. |

## Open Rollout Work

- Khashi VC and Media Engine still need visible package-native dashboard shell migration.
- Production routes must be verified from the standard production map before marking downstream feeds operational.
- CI gating should start after static-adapter exceptions are removed.
