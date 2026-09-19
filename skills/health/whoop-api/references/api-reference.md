# Whoop API Reference

Base URL: `https://api.prod.whoop.com`

All v2 endpoints require the `/developer/` prefix: `https://api.prod.whoop.com/developer/v2/...`

## Authentication

OAuth 2.0 Authorization Code flow.

| Parameter | Value |
|---|---|
| Auth URL | `https://api.prod.whoop.com/oauth/oauth2/auth` |
| Token URL | `https://api.prod.whoop.com/oauth/oauth2/token` |
| Redirect URI | `http://localhost:8647/callback` |
| Scopes | `read:recovery`, `read:cycles`, `read:sleep`, `read:workout`, `read:body_measurement`, `read:profile`, `offline` |

### Token Exchange

POST to token URL with:
- `grant_type=authorization_code` (initial) or `grant_type=refresh_token` (refresh)
- `client_id` + `client_secret`
- `code` (initial) or `refresh_token` (refresh)
- `redirect_uri` (initial only)

Response:
```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

Access tokens expire in 3600s (1 hour). Refresh tokens are single-use — each refresh returns a new refresh token.

## Endpoints

### Cycle (Strain)

```
GET /v2/cycle
```

| Param | Type | Description |
|---|---|---|
| `start` | ISO 8601 | Start datetime (required) |
| `end` | ISO 8601 | End datetime (required) |
| `limit` | int | Max results (default 25, max 25) |
| `offset` | int | Pagination offset |

Response fields: `id`, `strain`, `recovery_score`, `kilojoule`, `average_heart_rate`, `max_heart_rate`, `start`, `end`, `updated_at`

### Recovery

```
GET /v2/recovery
```

Same pagination params as cycle.

Response fields: `id`, `score{recovery_score, timestamp}`, `heart_rate_variability{rmssd, timestamp}`, `resting_heart_rate{rate, timestamp}`, `sleep_quality{percentage, timestamp}`, `cycle_id`, `created_at`, `updated_at`

### Sleep

```
GET /v2/activity/sleep
```

Same pagination params.

Response fields: `id`, `score{stage_summary{total, light, deep, rem, wake}, duration, efficiency, latency}`, `start`, `end`, `nap`, `created_at`, `updated_at`

### Workout

```
 GET /v2/activity/workout
```

Same pagination params.

Response fields: `id`, `strain`, `average_heart_rate`, `max_heart_rate`, `kilojoule`, `start`, `end`, `updated_at`

### Body Measurement

```
GET /v2/user/measurement/body
```

No pagination params.

Response fields: `height_meter`, `weight_kilogram`

### Profile (Basic)

```
GET /v2/user/profile/basic
```

No pagination params.

Response fields: `user_id`, `first_name`, `last_name`, `email`

## Rate Limiting

- 100 requests per minute per user
- 10,000 requests per day per user
- Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- 429 Too Many Requests if either limit is exceeded
- Retry-After header on 429 responses

## Error Codes

| Code | Meaning |
|---|---|
| 400 | Bad request (invalid params) |
| 401 | Unauthorized (token expired or invalid) |
| 403 | Forbidden (insufficient scope) |
| 404 | Not found |
| 429 | Rate limited |
| 500 | Whoop server error |

## Data Freshness

Whoop syncs band data to servers every 5-15 minutes after activity ends. API returns
the most recent data the server has — not real-time.