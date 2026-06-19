# Computer-Use Cache provider

Hermes can route OpenAI-compatible model calls through
[Computer-Use Cache](https://github.com/rohanarun/computer-use-cache), a local
cache proxy for repeated computer-use, browser, coding, and tool workflows.
The cache server sits between Hermes and an upstream provider such as
OpenRouter or OpenAI. Cache misses are forwarded upstream; exact repeated
requests can be served locally.

## Start the cache proxy

Run the proxy in front of any OpenAI-compatible upstream:

```bash
export UPSTREAM_BASE_URL=https://openrouter.ai/api/v1
export UPSTREAM_API_KEY=sk-or-v1-your-key

npx -y computer-use-cache start
```

By default, the proxy listens at:

```text
http://127.0.0.1:8000/v1
```

## Configure Hermes

Set the provider in `~/.hermes/config.yaml`:

```yaml
model:
  provider: computer-use-cache
  default: openai/gpt-4.1-mini
  base_url: http://127.0.0.1:8000/v1
```

Then set the API key Hermes should send to the cache proxy:

```bash
export COMPUTER_USE_CACHE_API_KEY=local-dev
```

If the cache proxy was started with `UPSTREAM_API_KEY`, the proxy uses that
server-side upstream key. If not, it forwards the incoming
`COMPUTER_USE_CACHE_API_KEY` value upstream as the bearer token.

For a one-off run, use the provider flag:

```bash
hermes --provider computer-use-cache -m openai/gpt-4.1-mini
```

## Verify cache behavior

The proxy exposes cache headers on every response:

```text
X-Computer-Use-Cache: HIT
X-Computer-Use-Cache: MISS
X-Computer-Use-Cache: BYPASS
```

Use stable model parameters for repeatable workflows: model, messages, tools,
tool choice, response format, temperature, top-p, max tokens, and seed all
participate in the cache key.

Do not send credentials, private tokens, passwords, or one-off private prompts
through a cacheable request. Disable caching for those requests with the
proxy's `cache: false` request field or `X-Cache-Bypass: true` header.
