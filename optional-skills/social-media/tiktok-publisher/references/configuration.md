# Configuration

Read this file when the publishing scripts cannot find TikTok credentials or when
you need to prepare the workspace for publishing.

## Supported config sources

The scripts support these configuration sources:

1. Environment variables:
   - `TIKTOK_API_KEY`
   - `TIKTOK_AUTHORIZATION_TOKEN`
   - optional `TIKTOK_PUBLISHER_CONFIG`
2. A workspace `config.json` file with a `tiktok` object

Environment variables take precedence over values from `config.json`.

## Expected config.json shape

```json
{
  "tiktok": {
    "api_key": "YOUR_API_KEY"
  }
}
```

The scripts also accept `authorization_token` as a fallback key name.

## Security rules

- Never commit real tokens into the repository.
- Prefer environment variables in shared environments.
- Do not print secrets in terminal output or logs.
