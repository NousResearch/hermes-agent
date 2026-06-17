# Mealie API Reference

Load this reference for raw API calls, meal planning, shopping lists, or endpoints not directly covered by `scripts/mealie_api.py`.

## Configuration

Use a base URL without a trailing slash, for example `https://mealie.example.com`.

Headers:

```http
Authorization: Bearer <token>
Content-Type: application/json
```

Accepted environment variables:

- `MEALIE_BASE_URL` or `WXT_MEALIE_SERVER`
- `MEALIE_API_TOKEN` or `WXT_MEALIE_API_TOKEN`

Optional env files:

- `mini-mealie/.env.local`
- `.env.local`
- `~/.config/mealie.env`

## Common Endpoints

Profile:

- `GET /api/users/self`

Recipes:

- `GET /api/recipes?search=<query>&perPage=<n>`
- `GET /api/recipes/{slug}`
- `POST /api/recipes/create/url`
- `POST /api/recipes/create/html-or-json`
- `POST /api/recipes/test-scrape-url`

URL import body:

```json
{
  "url": "https://example.com/recipe",
  "includeTags": false,
  "includeCategories": false
}
```

HTML import body:

```json
{
  "data": "<html>...</html>",
  "url": "https://example.com/recipe",
  "includeTags": false,
  "includeCategories": false
}
```

Recipe list responses commonly include `items`, `page`, `perPage`, and `total`.

## Raw Call Rules

Prefer GET for inspection and POST/PATCH/DELETE only when the user clearly requests a change.

Before mutating endpoints:

- Confirm the target object by id or slug.
- Avoid repeated import/create calls after ambiguous failures.
- Summarize the planned mutation in the command output or user-facing response.

## Duplicate Checks

For URL duplicates, Mealie may store the original URL exactly. Fetch recent recipes with:

```text
GET /api/recipes?perPage=100&orderBy=dateUpdated&orderDirection=desc
```

Compare client-side after normalizing:

- Strip URL fragments.
- Strip common tracking parameters such as `utm_*`, `fbclid`, `gclid`, `mc_cid`, and `ref`.
- Strip `www`.
- Strip trailing slash except for the domain root.
