# SahilBlog Draft Preview Workflow

## Local Astro preview

```bash
cd ~/repos/SahilBlog
pnpm install   # first time only
pnpm dev       # starts on http://localhost:4321
```

Open the printed localhost URL in a browser. Draft posts (`approved: false`)
are visible in dev mode via `import.meta.env.DEV` guards. Production builds
hide them.

## Review paths

- `/ai` — AI Decoding stream
- `/blog/pm` — PM Insights stream
- `/blog/builder` — Builder's Log stream
- `/` — home page with all streams

## Identifying drafts

Draft posts have `approved: false` in frontmatter. In dev mode they render
with a visual indicator. Check the frontmatter directly:

```bash
grep "approved:" ~/repos/SahilBlog/src/content/blog/<slug>.mdx
```

## Approving a draft

Flip `approved: false` to `approved: true` in the MDX frontmatter, or use
the publisher:

```bash
cd ~/repos/KenseiAgent/content_engine
set -a; . ~/.hermes/.env; set +a
PYTHONPATH=. ../.venv/bin/python -c "
from blog.blog_publisher import approve
approve('<slug>')
"
```

The publisher flips approval, runs `pnpm build`, commits, and pushes.
Do NOT auto-approve. Sahil reviews each post first.

## Back-population run

Generate remaining posts (after Sahil sign-off on the 3 validation samples):

```bash
cd ~/repos/KenseiAgent/content_engine
set -a; . ~/.hermes/.env; set +a
PYTHONPATH=. ../.venv/bin/python -m blog.backfill --stream ai
PYTHONPATH=. ../.venv/bin/python -m blog.backfill --stream pm
PYTHONPATH=. ../.venv/bin/python -m blog.backfill --stream builder
```

Check spend:

```bash
PYTHONPATH=. ../.venv/bin/python -c "
import budget, config
print(budget.status(cap_gbp=config.BACKFILL_SPEND_CAP_GBP))
"
```

Or inspect the backfill ledger directly:

```bash
cat ~/repos/KenseiAgent/content_engine/output/backfill_ledger.json
```