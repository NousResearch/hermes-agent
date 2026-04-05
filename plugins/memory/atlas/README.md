# Atlas Memory Provider

This plugin exposes Atlas as an external Hermes memory provider.

Setup path:

1. Run `hermes memory setup`.
2. Choose `atlas`.
3. Enter:
   - Supabase project URL
   - Supabase service key
   - optional schema / embedding config

Atlas is now also available as a built-in Hermes memory plugin. This integration copy remains the source implementation and can still be linked as a user plugin override when needed.

Non-secret Atlas config is saved to `~/.hermes/atlas.json`.
Secrets are saved to `~/.hermes/.env`.

The provider keeps Hermes built-in memory active and adds Atlas as an external memory layer for:

- pre-turn context enrichment
- post-turn turn sync
- session-end closeout
