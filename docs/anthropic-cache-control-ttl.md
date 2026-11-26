# Anthropic Cache Control TTL — Per-Lane Evaluation

## Configuration

The `prompt_caching.anthropic_ttl` key in `~/.hermes/config.yaml` controls the TTL
for Anthropic-native `cache_control` breakpoints placed on the `system` parameter
of Anthropic Messages API requests:

```yaml
prompt_caching:
  cache_ttl: "5m"          # generic TTL, all providers
  anthropic_ttl: "1h"     # optional: override for Anthropic-family only
```

- **Default**: `anthropic_ttl` falls back to `cache_ttl` when not set.
- **Values**: `"5m"` (ephemeral, 1.25× write cost) | `"1h"` (ephemeral + `ttl:"1h"`, 2× write cost).
- **"auto" is not a valid value** for `anthropic_ttl`; use `auto` on `cache_ttl` instead.

## Per-Lane Evaluation: When 1h Pays Off

Anthropic bills 1-hour cache writes at **2× the base cache-write rate** (vs. 1.25×
for the default 5-minute tier). This means:

| Lane                    | Typical inter-request pause | 1h worth it? | Reasoning                                  |
|-------------------------|-----------------------------|--------------|--------------------------------------------|
| Interactive chat        | < 1 minute                  | ❌ No        | 5m is more than enough; 1h wastes credits  |
| Subagent dispatch       | < 30 seconds                | ❌ No        | Tight loops, 5m is generous                |
| CRON watches            | 15–60 minutes               | ✅ Yes       | Every request is a cache miss at 5m        |
| Mailbox scans           | 5–30 minutes                | ✅ Yes       | Idle periods exceed 5m often               |
| Nightly batch jobs      | 24 hours                    | ❌ No        | 1h still won't span the gap                |
| Shared-session drift    | 2–10 minutes                | ⚠️ Maybe    | Marginal; profile your cache-hit rates     |

### Recommendation

- **Default to 5m** for interactive lanes (Hermes CLI, IDE plugins, web chat).
- **Opt into 1h per-lane** via `anthropic_ttl: "1h"` for automated, sparse-scheduled lanes
  (cron watches, mailbox scanners, periodic digests) where the inter-request
  pause routinely exceeds 5 minutes.

### Per-Lane Adoption Pattern

Since `--config profile` overrides enable per-session config, you can run
always-on lanes with a profile that sets `anthropic_ttl: "1h"` without
touching the interactive sessions' 5m default.

## How the Breakpoint Works

1. The agent reads `prompt_caching.anthropic_ttl` at init, defaulting to `cache_ttl`.
2. The TTL flows through `chat_completion_helpers._build_anthropic_kwargs` →
   `AnthropicTransport.build_kwargs` → `build_anthropic_kwargs`.
3. In `build_anthropic_kwargs`, when:
   - `anthropic_cache_ttl` is `"5m"` or `"1h"`, AND
   - `system` is a plain string (no existing markers from the prompt-caching layer),
   the system prompt is wrapped in a content block with `cache_control`:
   ```json
   [{"type": "text", "text": "<system prompt>", "cache_control": {"type": "ephemeral"}}]
   ```
   For 1h TTL, the marker includes `"ttl": "1h"`.

4. When `system` is already a block list with cache_control markers (placed by
   `prompt_caching.apply_anthropic_cache_control`), the string guard prevents
   double-wrapping.
