# Send-Gate Layer 3 Implementation: API Server Rejection

## Overview

Layer 3 adds HTTP-level rejection of send requests at the API server gateway. When `send_gate=disabled` on any enabled platform, the API server returns a **403 Forbidden** response before any agent execution or message delivery occurs.

This complements:
- **Layer 1** (gateway/platforms/base.py): `send()` raises `SendGateDisabledException` at runtime
- **Layer 2** (tools/send_gate_tool.py): Tool registration filtering prevents send tool availability
- **Layer 3** (here): HTTP request handler rejection at the gateway API server level

## Implementation Details

### New Files

**`gateway/send_gate_api.py`**
- `check_send_gate_enabled_for_api(config: Optional[GatewayConfig]) -> Tuple[bool, Optional[str]]`
- Checks gateway configuration for `send_gate=disabled` on any enabled platform
- Returns `(True, None)` if sends are allowed
- Returns `(False, error_message)` if sends are blocked, with an informative message explaining:
  - Which platforms have send_gate disabled
  - How to re-enable (set `platforms.<name>.extra.send_gate` to `"enabled"`)
- Fail-open: on any config errors, allows sends (avoids operational cascades)
- Ignores disabled platforms (`enabled=False`)

### Modified Files

**`gateway/platforms/api_server.py`**
- Added import: `from gateway.send_gate_api import check_send_gate_enabled_for_api`
- Added method `_check_send_gate()` to `APIServerAdapter` class:
  - Returns `None` if sends are allowed
  - Returns 403 web.Response if sends are blocked
  - Parallel to existing `_check_auth()` pattern
- Integrated check in two request handlers:
  - `async def _handle_chat_completions()`: Added send_gate check after auth, before message processing
  - `async def _handle_responses()`: Added send_gate check after auth, before message processing
- Both checks occur early to prevent unnecessary processing
- Non-send endpoints (health, models, sessions, etc.) are unaffected

**`tests/gateway/test_send_gate_api_server.py`** (new)
- 14 test cases covering:
  - API server check function behavior (11 tests in `TestSendGateAPICheck`)
  - Adapter integration (3 tests in `TestAPIServerSendGateIntegration`)
- All tests passing

## Configuration

Enable send_gate blocking via YAML config:

```yaml
platforms:
  telegram:
    enabled: true
    token: "xxx"
    extra:
      send_gate: "disabled"  # Block all sends on Telegram
  
  discord:
    enabled: true
    token: "xxx"
    # send_gate defaults to "enabled" if not specified
```

Or programmatically:

```python
config.platforms[Platform.TELEGRAM].extra["send_gate"] = "disabled"
```

## Error Response

When sends are blocked, the API server returns:

```json
HTTP/1.1 403 Forbidden
Content-Type: application/json

{
  "error": {
    "message": "Send operations are disabled via send_gate configuration. Disabled on: discord, slack. To re-enable sends, set platforms.<platform_name>.extra.send_gate to 'enabled' (or remove the 'send_gate' setting from your config) and restart the gateway.",
    "type": "invalid_request_error"
  }
}
```

## Behavior

| Scenario | Result | Behavior |
|----------|--------|----------|
| send_gate not set | Allow | Defaults to enabled |
| send_gate=enabled | Allow | Sends work normally |
| send_gate=disabled on enabled platform | Block | Return 403 on chat/responses endpoints |
| send_gate=disabled on disabled platform | Allow | Disabled platforms are ignored |
| Config load error | Allow | Fail-open to avoid operational issues |
| Any non-send endpoint | Allow | Health, models, sessions, etc. unaffected |

## Backward Compatibility

- Default behavior (send_gate=enabled) unchanged
- Non-send endpoints continue to work with send_gate=disabled
- Existing tests for Layers 1 and 2 continue to pass
- No API contract changes (only new 403 response for send_gate=disabled case)

## Testing

Run the test suite:

```bash
pytest tests/gateway/test_send_gate_api_server.py -v          # Layer 3 tests (14 cases)
pytest tests/gateway/platforms/test_send_gate.py -v           # Layer 1 tests (5 cases)
pytest tests/gateway/platforms/test_send_gate_registration.py # Layer 2 tests (20 cases)
```

All 39 send_gate tests pass.

## Design Rationale

1. **HTTP-level check**: Prevents agent execution entirely, not just message delivery
2. **Early in handler**: Check immediately after auth, before parsing message content
3. **Informative error**: Client gets specific platform names and remediation guidance
4. **Non-intrusive**: Minimal changes, consistent with existing `_check_auth()` pattern
5. **Fail-open**: Config errors don't cascade into operational blocks
6. **Platform-agnostic**: Works for any platform with send_gate setting

## Future Extensions

- Could add per-endpoint granularity (e.g., read-only vs send endpoints)
- Could add per-session send_gate overrides via headers
- Could add metrics/logging for send_gate blocks
- Could support graceful degradation (accept requests but warn) instead of 403
