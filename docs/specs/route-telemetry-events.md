# Route Telemetry Events

Hermes Feishu routing is deliberately thin-cockpit: foreground turns may compute a routing decision, but background workers only count as accepted when a real background command is started. Route telemetry records that boundary without adding a new database.

```yaml
routing:
  feishu_auto_dispatch_enabled: true
  feishu_route_shadow_hints_enabled: true
```

## Event registry

| Event | When emitted | Required details |
| --- | --- | --- |
| `route_decision_resolved` | `gateway.route_decision.resolve_route_decision(..., telemetry_source=...)` computes an ROI decision | `decision_type`, `route_names`, `score`, `confidence`, `risk_class`, `reasons` |
| `route_selected_for_background` | Feishu guarded auto-dispatch actually starts a background task | `source`, `decision_type`, `score`, `confidence`, `routes` |
| `route_worker_outcome` | A background worker response is evaluated against route contracts | `passed`, `score`, `issues`, `worker_evaluation` |
| `worker_evaluation` | Embedded object inside `route_worker_outcome` details | `passed`, `score`, `issues`, `route_contracts` |

Telemetry is best-effort and must never block user-visible message handling. It is stored inside the existing `.usage.json` sidecar under `_route_usage`, capped per route, and summarized by `summarize_route_usage(window_days=30)` for ROI scoring.

`difficult_web_extract` is registered as a task-named `/bg` route for selector/batch/session-aware fallback extraction after ordinary `web_extract` fails. The optional Scrapling backend may satisfy this route, but `scrapling` is not a route name and the route does not replace browser tools or ordinary `web_extract`.

## Guardrails

- `feishu_auto_dispatch_enabled: false` downgrades high-ROI decisions to wrapper suggestions.
- `feishu_route_shadow_hints_enabled: false` suppresses foreground `RouteDecision` system hints.
- Risky external-write/destructive tasks require approval and must not auto-dispatch.
- Media, slash commands, internal replay events, non-Feishu platforms, and already-running sessions must not auto-dispatch.
