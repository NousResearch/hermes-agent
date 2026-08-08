# 1Password CLI — Common Examples

## Read a Password

```bash
op item get "My Database" --field password
```

## Read a JSON Document

```bash
op read "op://Vault/Item/Field"
```

## List All Items in a Vault

```bash
op item list --vault "Production"
```

## Create a New Item

```bash
op item create \
  --vault "Development" \
  --category "login" \
  --title "My App" \
  --url "https://app.example.com" \
  --username "admin" \
  --password "$(openssl rand -base64 24)"
```

## Generate a Secure Password

```bash
op password generate --length 32 --symbols
```

## Use in Scripts (No Interactive Prompts)

```bash
# With service account token (no sign-in required)
export OP_SERVICE_ACCOUNT_TOKEN="ops_..."
DB_PASS=$(op item get "Production DB" --field password --reveal)
psql -h db.example.com -U admin -d mydb -c "SELECT 1"
```

## Sign Out

```bash
op signout --account <account-id>
```

## References

- Official docs: https://developer.1password.com/docs/cli/
- Service accounts: https://developer.1password.com/docs/service-accounts/
