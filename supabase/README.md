# Supabase backend

This directory is reserved for the Supabase backend that supports hosted Hermes Agent web experiences.

## What belongs here

- `migrations/*.sql` — ordered schema migrations generated with `supabase migration new ...`.
- `config.toml` — non-secret Supabase CLI project settings.
- Edge Functions, seed files, or typed schema artifacts can be added here when the product needs them.

## What must stay out of git

- `supabase/.temp/` — created by `supabase link` and local CLI commands.
- `.env` files and database passwords.
- Secret API keys (`sb_secret_...`) or legacy `service_role` keys.

## CI behavior

The `supabase-migrations` job in `.github/workflows/deploy-site.yml` is intentionally disabled until the repository variable `SUPABASE_MIGRATIONS_ENABLED` is set to `true`.

When enabled, it applies committed SQL migrations on `main` before the workflow triggers the Vercel deploy hook. This ordering prevents application code from deploying before its required database schema.

Required GitHub secrets for migration deployment:

- `SUPABASE_ACCESS_TOKEN`
- `SUPABASE_PROJECT_ID`
- `SUPABASE_DB_PASSWORD`

Optional GitHub variable:

- `SUPABASE_CLI_VERSION` — pin a Supabase CLI version; defaults to `latest`.

Vercel application runtime variables such as `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` should be configured in Vercel Project Settings, not committed here.
