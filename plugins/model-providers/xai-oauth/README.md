# xAI (Grok) OAuth Provider

This provider adds first-class support for [xAI Grok](https://grok.com) models in Hermes Agent using OAuth.

## Features

- Uses the official Responses API (`codex_responses`) for native reasoning support on Grok 4+ models.
- Automatically benefits from xAI's prompt caching via the `x-grok-conv-id` header.
- **Primary authentication method**: Seamless import from the official Grok CLI / Grok Build login (`~/.grok/auth.json`).
- Optional browser-based OAuth login to `auth.x.ai` as a fallback.

## Recommended Setup

The recommended way to authenticate is to first log in with the official Grok CLI:

```bash
grok login
```

Then in Hermes, select **"xAI (OAuth login)"** in `hermes model`. Hermes will automatically detect and import your existing Grok CLI credentials.

## Alternative: Browser Login

If you do not have the Grok CLI installed, you can authenticate directly via browser when selecting the provider in `hermes model`.

**Note**: Browser login uses xAI's public desktop OAuth client. Redirect URI restrictions may apply depending on your environment.

## Environment Variables

| Variable                        | Description                          | Priority |
|--------------------------------|--------------------------------------|----------|
| `GITHUB_PERSONAL_ACCESS_TOKEN` | (Not used)                           | -        |
| `XAI_API_KEY`                  | Fallback API key (if using `xai` provider) | Low     |

## Aliases

This provider responds to the following names:

- `xai-oauth`
- `grok-oauth`
- `xai-portal`
- `grok-login`

## Model Examples

- `grok-4`
- `grok-3`
- `grok-3-mini`

## Related Providers

- `xai` — The simpler API-key version of the xAI provider (uses `XAI_API_KEY`).

## Implementation Notes

- `api_mode`: `codex_responses`
- Prompt caching is handled automatically by the transport layer when the base URL contains `x.ai`.
- Credential import logic lives in `hermes_cli/auth.py` (`_import_grok_cli_into_hermes`).

## Contributing

This provider was contributed to make Grok a first-class citizen in Hermes Agent, with a focus on a great experience for users who already use the official Grok CLI.