# Per-service reference docs and key storage

After a service is configured, write a markdown reference for it. **SKILL.md gives the one path to write to**, so it is stated in a single place and never drifts; the file name is the service slug in lowercase kebab (`openai`, `airtable`, `stripe`). It becomes the single source of truth for that service, so future agents read it instead of re-fetching the docs.

## Template

```markdown
# <Service Name> — ShieldNode integration

## Routing
- **ShieldNode base URL**: https://proxy.shieldnode.app
- **Original API base URL** (set in the service config): https://api.example.com/v1
- **Auth header for proxy calls**: `X-Api-Key: $SHIELDNODE_<SERVICE>_KEY`
  > Never write the actual key value here. Reference the env variable only.
- **Effective call pattern**: `https://proxy.shieldnode.app/<endpoint-path>`
  > The configured base URL already includes `/v1`, so the proxied path omits it.

## Documentation
- Official docs: https://docs.example.com
- Reference page used to populate this file: https://docs.example.com/api/v1

## Auth method
Bearer token in the `Authorization` header, handled by ShieldNode. The client only sends `X-Api-Key`.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET    | /users      | List users |
| GET    | /users/{id} | Retrieve a user |
| POST   | /users      | Create a user |
| PATCH  | /users/{id} | Update a user |
| DELETE | /users/{id} | Delete a user |

## Push approval
- **Default window**: 30 min (override per call with `X-Approval-Duration`).
- **When triggered**: only when the key is disabled AND the user has the mobile app. Otherwise the key is always-on.
- **On `403 approval_required`**: wait `poll_interval_seconds` (default 30s), retry up to `timeout_seconds` (default 5 min). Tell the user once.
- **On `403 approval_denied`**: stop, surface "User declined access on ShieldNode mobile."
- **Suggested duration for this service**: <fill in, see the table below>

## Notes
- Upstream rate limit: 60 req/min per API key.
- Pagination: cursor-based via `?cursor=<token>`.
```

## Filling the Endpoints section

1. Ask for the docs URL: *"Could you give me the URL of the API documentation page that lists the endpoints? I'll extract them and save a reference file."*
2. Fetch it with whatever web tool you have.
3. Extract method, path, one-line description into the table.
4. Save the file and confirm with a clickable link.

If the user pastes raw doc content instead of a URL, parse that directly. If the docs are auth-walled and the fetch fails, ask them to paste the page.

**Many API doc sites are JavaScript-rendered SPAs.** A plain fetch returns an empty shell: only a `<noscript>` tag, a suspiciously short document (< 500 chars), or no endpoint paths anywhere. When you detect that, try in order:

1. **The machine-readable spec.** Try `<base-url>/openapi.json`, `/swagger.json`, `/.well-known/openapi.json`, or look in the docs page `<head>` for a `link rel="alternate" type="application/json"`.
2. **The project's public source.** If it is open source, the GitHub README usually lists endpoints and the repo often holds the OpenAPI file.
3. **Ask the user to paste the rendered content** from their browser. They have a real browser, you often do not.
4. **A headless browser** (Puppeteer, Playwright) if your runtime has one. Last resort: slow, fragile, not always available.

If all four fail, save the partial doc with `> _Endpoints to be populated, documentation site rendered client-side._` and ask the user to point you at another page.

## Suggested approval durations

| Workload | Value |
|---|---|
| Chat, completion, inference | 30 min |
| Long batch, training, video or audio generation | 2 h |
| One-shot lookups (geocoding, currency, weather, CMS reads) | 15 min |
| Unattended 24/7 cron | Use a scheduled window (see the scheduled windows section of SKILL.md), or note that the key should stay always-on |

If the user already set a per-key default in the dashboard, write that exact value and add *"(matches the dashboard default, agent does not need to send the header)"*. If they are clearly web-only with no mobile app, note that push approval falls back to a plain `403 key_disabled`.

## Storing the virtual key

The reference doc is designed to be committable, so the key value never goes in it, only the env variable name.

Convention: `SHIELDNODE_<SERVICE>_KEY` in uppercase, e.g. `SHIELDNODE_STRIPE_KEY`.

**Multiple environments of the same API** (Stripe test and live, staging and prod) expand to `SHIELDNODE_<SERVICE>_<ENV>_KEY`: `SHIELDNODE_STRIPE_TEST_KEY`, `SHIELDNODE_OPENAI_PROD_KEY`. Detect this by checking whether `SHIELDNODE_<SERVICE>_KEY` already exists in `.env`; if it does, propose the suffixed form and ask which environment this key is.

Steps:

1. **Check for `.env` at the project root.** Append the variable if it exists (never overwrite entries), otherwise create it:
   ```env
   # ShieldNode virtual keys, never commit this file.
   SHIELDNODE_STRIPE_KEY=shieldnode_...
   ```
2. **Update `.env.example`** (create if missing) with the value blanked. That file is committable and documents what collaborators need:
   ```env
   SHIELDNODE_STRIPE_KEY=
   ```
3. **Verify `.gitignore` excludes `.env`.** If not, append `.env` and `.env.local`. Do not touch other entries.
4. **Reference it by name only** in the per-service doc: `X-Api-Key: $SHIELDNODE_STRIPE_KEY`.
5. **Confirm to the user** what changed, and remind them the key is shown once at creation so they must paste it into `.env` immediately.

**Never ask the user to paste a key into the chat.** They paste it into `.env` on their machine. If they paste one anyway, redact it in later turns and tell them to rotate it in the dashboard if it may have hit any logs.
