# Acta Briefs Worker

**Acta** is named after the Roman *Acta Diurna*, daily public notices. It hosts curated Hermes brief artifacts for `acta.imperatr.com`.

## Deploy checklist

1. Create R2 bucket:
   ```bash
   npx wrangler r2 bucket create acta-briefs
   ```
2. Set Worker secrets:
   ```bash
   npx wrangler secret put ACTA_UPLOAD_TOKEN -c cloudflare/acta/wrangler.toml
   npx wrangler secret put ACTA_SIGNING_SECRET -c cloudflare/acta/wrangler.toml
   ```
3. Deploy:
   ```bash
   npx wrangler deploy -c cloudflare/acta/wrangler.toml
   ```
4. Hermes needs the same upload token in `~/.hermes/.env`:
   ```bash
   ACTA_UPLOAD_TOKEN=...
   ```

The signing secret stays only in Cloudflare Worker secrets. Hermes uploads HTML; the Worker returns a signed read URL.
