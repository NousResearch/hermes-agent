# Forward-compat conflict notes — overlay-residual-drift onto v0.17.0

These 14 files carry overlay-residual lines AND were heavily rewritten by upstream
between v0.16.0 and v0.17.0. Per-file: overlay-size vs v0.17-size + upstream churn.
Applying onto v0.17.0 requires per-conflict manual reconciliation (the overlay & v0.17
are divergent evolutions; mechanical keep-both produces invalid Python).

| file | overlay lines | v0.17 lines | v0.16->v0.17 churn | conflict nature |
|------|--------------|-------------|--------------------|-----------------|
| agent/anthropic_adapter.py | 2939 | 2590 | +320/-33 | large divergent rewrite |
| agent/auxiliary_client.py | 6009 | 6082 | +353/-99 | additive/structural overlap |
| agent/conversation_loop.py | 5201 | 4486 | +460/-865 | large divergent rewrite |
| agent/system_prompt.py | 456 | 534 | +133/-5 | additive/structural overlap |
| cli.py | 16250 | 14865 | +1999/-3213 | large divergent rewrite |
| gateway/platforms/api_server.py | 4462 | 4406 | +185/-36 | additive/structural overlap |
| gateway/run.py | 19992 | 17555 | +2713/-5093 | large divergent rewrite |
| hermes_cli/main.py | 16059 | 12625 | +2685/-6094 | large divergent rewrite |
| hermes_state.py | 4447 | 4941 | +741/-84 | large divergent rewrite |
| tests/agent/test_auxiliary_client.py | 3648 | 3989 | +401/-0 | large divergent rewrite |
| tests/hermes_cli/test_inventory.py | 447 | 727 | +347/-0 | large divergent rewrite |
| tools/mcp_tool.py | 3989 | 4716 | +835/-34 | large divergent rewrite |
| tools/skills_tool.py | 1562 | 1638 | +103/-10 | additive/structural overlap |
| tui_gateway/server.py | 8677 | 10940 | +2763/-223 | large divergent rewrite |

## Resolution guidance (for the operator pulling onto v0.17.0)
- These files' overlay versions are what live ./src runs daily (coherent on v0.16 base).
- gateway/run.py is the extreme case (overlay ~12800 lines vs v0.17 ~11160) — reconcile by
  taking v0.17's new background-task/reasoning blocks + re-applying the overlay's additions.
- The other 13 are smaller; most conflicts are keep-both-able with manual indent fixes.
- This PR carries the v0.16-coherent overlay versions so nothing is lost; the v0.17
  reconciliation is the documented pull-down step.