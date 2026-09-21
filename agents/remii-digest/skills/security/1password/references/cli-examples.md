# op CLI examples

## Sign-in and identity

```bash
op signin
op signin --account my.1password.com
op whoami
op account list
```

## Read secrets

```bash
op read "op://app-prod/db/password"
op read "op://app-prod/npm/one-time password?attribute=otp"
```

## Inject secrets

```bash
echo "api_key: {{ op://app-prod/openai/api key }}" | op inject
op inject -i config.tpl.yml -o config.yml
```

## Run command with secrets

```bash
export DB_PASSWORD="op://app-prod/db/password"
op run -- sh -c '[ -n "$DB_PASSWORD" ] && echo "DB_PASSWORD is set"'
```
