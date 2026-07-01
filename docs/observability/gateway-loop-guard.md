# Gateway agent-loop guard

Hermes gateways can receive messages from other Hermes agents over transports
such as email.  Without a substrate-level guard, runtime notices like progress,
shutdown, restart, or "received" acknowledgements can be mistaken for fresh
human instructions and bounce between agents.

The gateway loop guard runs outside the agent conversation, before dispatching
an inbound message to the agent and before sending an outbound email reply.
It has three layers:

1. Runtime quarantine state in `gateway-loop-guard-state.json` under the active
   `HERMES_HOME`.
2. Deterministic safety signals for obvious loops: agent-to-agent restart
   requests, Hermes runtime/status notices, duplicate low-novelty replies, hop
   count overflow, and too many recent pair events.
3. Optional auxiliary LLM judge (`loop_guard`) for semantic agent-agent loops
   that do not match fixed patterns.

## Email configuration

Enable per email platform profile in `config.yaml`:

```yaml
platforms:
  email:
    extra:
      loop_guard:
        enabled: true
        mode: protect        # observe | protect | strict | off
        ai_enabled: true     # optional; uses auxiliary task=loop_guard
        agent_id: alfred
        agent_identities:
          - alfred@sqmnet.es
          - selina@sqmnet.es
          - bishop@sqmnet.es
        quarantine_ttl_seconds: 7200
```

`observe` records decisions but allows traffic.  Use it during rollout.
`protect` suppresses high/critical loop decisions and can quarantine an agent
pair temporarily.  `strict` also suppresses medium-risk agent acknowledgements.

The email adapter adds these headers to outbound Hermes email:

- `X-Hermes-Origin-Agent`
- `X-Hermes-Origin-Address`
- `X-Hermes-Intent`
- `X-Hermes-Hop-Count`
- `X-Hermes-Reply-Policy`

Inbound emails with these headers are treated as agent-originated even when the
sender address is not listed explicitly in `agent_identities`.

## Self-restart guard

Terminal approval now hard-blocks gateway self-targeting commands when the
process is running inside a gateway (`_HERMES_GATEWAY=1`), even if yolo or
`approvals.mode: off` is active.  Blocked examples:

```bash
hermes gateway restart
hermes update
systemctl --user restart hermes-gateway
systemctl --user restart hermes-gateway-selina-email.service
```

Use the gateway `/restart` command or restart externally from SSH/systemd
instead.  The internal `/restart` path remains available because it does not run
through the terminal tool and its detached watcher clears `_HERMES_GATEWAY`.

## Testing

Minimal focused regression suite:

```bash
python -m unittest \
  tests.gateway.test_loop_guard \
  tests.gateway.test_email_loop_guard \
  tests.tools.test_gateway_self_restart_guard
```
