# Configuration

Read this file when the publishing script cannot find Instagram credentials or when
you need to prepare the workspace for publishing.

## Supported config sources

The script supports these configuration sources:

1. Command-line arguments:
   - `--api-key`
   - `--connection-id`
   - `--account-id`
2. Environment variables:
   - `INSTAGRAM_API_KEY`
   - `INSTAGRAM_AUTHORIZATION_TOKEN`
   - `INSTAGRAM_CONNECTION_ID`
   - `INSTAGRAM_ACCOUNT_ID`
   - optional `INSTAGRAM_PUBLISH_CONFIG`
3. A workspace `config.json` file with an `instagram` object

Arguments take precedence over environment variables, and environment variables
take precedence over `config.json`.

## Expected config.json shape

```json
{
  "instagram": {
    "authorization_token": "YOUR_API_KEY",
    "connection_id": "YOUR_CONNECTION_ID",
    "account_id": "YOUR_ACCOUNT_ID"
  }
}
```

The script also accepts `api_key` as a fallback key name.

## Security rules

- Never commit real tokens, connection IDs, or account IDs into the repository.
- Prefer environment variables or runtime arguments in shared environments.
- Do not echo secrets in logs or terminal output.
