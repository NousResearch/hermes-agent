# Cloudflare Workers AI

OpenAI-compatible chat completions on Cloudflare's edge.

## Setup

Select the provider in the model picker — it prompts for both values and
writes them to `~/.hermes/.env`:

```
hermes model
# → Cloudflare Workers AI
# → CLOUDFLARE_API_TOKEN (masked; Keep/Replace if already set)
# → CLOUDFLARE_ACCOUNT_ID (Keep/Replace if already set)
```

Create a token with **Account → Workers AI → Read** at
https://dash.cloudflare.com/profile/api-tokens
(Workers AI dashboard → Use REST API also has a prefilled token). Copy the
**Account ID** from the Workers AI page.

`CLOUDFLARE_API_KEY` is accepted as an alias for the token. `CLOUDFLARE_BASE_URL`
overrides the constructed endpoint if you need a proxy; replacing the Account
ID in `hermes model` clears that override.

Or set them yourself:

```
CLOUDFLARE_API_TOKEN=your-token
CLOUDFLARE_ACCOUNT_ID=your-32-char-account-id
```

Then:

```
hermes config set model.provider workers-ai
hermes config set model.default @cf/moonshotai/kimi-k2.6
```

Aliases: `cloudflare`, `cloudflare-workers-ai`, `cf-workers-ai`, `workersai`, `cloudflare-ai`.

## Endpoint

```
https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1
```

Model ids use the `@cf/...` catalog form. Prefer models marked for function
calling in the [Workers AI catalog](https://developers.cloudflare.com/workers-ai/models/)
— Hermes needs tool calling. gpt-oss on Workers AI is Responses-API-only and is
not on this chat-completions provider.

This is **not** Vercel AI Gateway (`ai-gateway`) and not Cloudflare AI Gateway's
multi-provider `compat` proxy.
