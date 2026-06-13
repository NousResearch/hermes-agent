# Hermes Outbound Metadata (`extra_body.hermes`)

Hermes can attach a documented set of orchestration hints to every outgoing
chat-completions request via `extra_body.hermes`.  A downstream
OpenAI-compatible proxy or dispatcher can read these hints and apply routing
policy — model selection, per-user quotas, cost attribution, observability
tracing — without parsing free text or spinning up one Hermes profile per
channel.

## Enabling

The feature is **opt-in and disabled by default**.  To enable, add to your
profile's `config.yaml`:

```yaml
extras:
  hermes_metadata:
    enabled: true
```

## Fields emitted

When enabled, Hermes adds the following to every `extra_body.hermes` object.
Fields whose runtime value is `None` (e.g. `chat_id` for the CLI) are omitted
so strict-validating servers (vLLM strict mode, certain proxies) remain
compatible.

| Field | Type | Example | Source |
|---|---|---|---|
| `session_id` | string | `"20260517_142530_a7f2c1b9"` | session identifier, unique per conversation |
| `gateway_platform` | string | `"matrix"`, `"discord"`, `"telegram"`, `"cli"` | active platform adapter |
| `chat_id` | string | `"!room:example.org"`, `"12345"` | `SessionSource.chat_id` — room/channel/DM identifier |
| `user_id` | string | `"@alice:example.org"`, `"67890"` | `SessionSource.user_id` — sender identifier |
| `command_origin` | string | `"user"`, `"scheduled"` | how this turn was initiated |

## Wire example

```jsonc
{
  "model": "ollama-cloud/glm-5.1",
  "messages": [...],
  "extra_body": {
    "hermes": {
      "session_id": "20260517_142530_a7f2c1b9",
      "gateway_platform": "matrix",
      "chat_id": "!IzxRltiFyqGSkTtDKP:chat.example.eu",
      "user_id": "@alice:chat.example.eu",
      "command_origin": "user"
    }
  }
}
```

## Compatibility note

Some OpenAI-compatible servers (e.g. vLLM in strict mode) reject unknown
request body fields with HTTP 422.  Because the feature defaults to `false`,
existing deployments are unaffected.  Enable only when your downstream server
(LiteLLM, a custom proxy, Helicone, Langfuse, Portkey, …) can tolerate or
actively consumes the `extra_body.hermes` object.

## Naming alignment

Field names in `extra_body.hermes` mirror `SessionSource` field names
(`chat_id`, `user_id`) and the `AIAgent.__init__` kwargs (`platform`, `chat_id`,
`user_id`).  If you are building a companion inbound-headers feature, the
natural mirror is `X-Hermes-Chat-Id` ↔ `extra_body.hermes.chat_id`, etc.
