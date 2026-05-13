# Launch Day Runbook — DeepParser v1.0.0

This runbook walks through the full launch sequence. Do steps in order; each gate
must pass before continuing.

---

## Pre-flight (the day before)

- [ ] Confirm `beta.deepparser.ai` API key works: `python deepparser/examples/basic_parse.py sample.pdf "summary?"`
- [ ] Confirm Docker Hub account `ysh145` is logged in: `docker login`
- [ ] Confirm Fly.io CLI is installed and authenticated: `fly auth whoami`
- [ ] Confirm PyPI OIDC trusted publisher is configured at pypi.org under project `deepparser`
  - Trusted publisher settings: owner `ysh145`, repo `hermes-agent`, workflow `deepparser-release.yml`
- [ ] Run full test suite locally: `pytest tests/deepparser/ -v` — all must pass
- [ ] Run benchmark scorer and fill in the win-rate table in `RELEASE_v1.0.0.md`

---

## Step 1 — Final commit and tag

```bash
# From repo root, on main branch
git status          # confirm clean working tree

git add \
  deepparser/ \
  deepparser_api/ \
  benchmark/ \
  tests/ \
  Dockerfile.deepparser \
  fly.toml \
  pyproject.toml \
  .github/workflows/deepparser-release.yml \
  RELEASE_v1.0.0.md \
  launch/

git commit -m "feat: DeepParser SDK + API server v1.0.0 launch"

git tag v1.0.0
git push origin main
git push origin v1.0.0      # triggers the release workflow
```

---

## Step 2 — Watch CI

Open: https://github.com/ysh145/hermes-agent/actions

Jobs (in order):
1. **test** — pytest suite; ~2 min
2. **docker** — builds amd64 + arm64, pushes to `ysh145/deepparser-api:1.0.0` and `:latest`; ~8 min
3. **pypi** — OIDC publish to PyPI; ~2 min
4. **deploy** — `flyctl deploy`; ~3 min

Gate: all four green before continuing.

---

## Step 3 — Fly.io secrets (first deploy only)

If this is the first deploy, set secrets before the workflow runs (or re-trigger after setting):

```bash
fly secrets set \
  DEEPPARSER_API_KEY="dp_live_..." \
  ADMIN_PASSWORD="$(openssl rand -hex 24)" \
  --app deepparser-api
```

Save `ADMIN_PASSWORD` somewhere safe — you'll need it for `GET /admin/keys`.

---

## Step 4 — Smoke test production

```bash
export BASE_URL=https://deepparser-api.fly.dev

# Health check
curl -s $BASE_URL/health | jq .

# Register a key
curl -s -X POST $BASE_URL/keys/register \
  -H "Content-Type: application/json" \
  -d '{"email":"your@email.com","intended_use":"smoke test"}' | jq .

# Parse + ask (use the key returned above)
export DP_KEY="dp_live_..."
python deepparser/examples/basic_parse.py sample.pdf "What is this document about?"
```

Gate: all three commands succeed before posting to HN.

---

## Step 5 — PyPI smoke test

```bash
pip install deepparser==1.0.0
python -c "from deepparser import DeepParserClient; print('ok')"
```

---

## Step 6 — GitHub Release

1. Go to https://github.com/ysh145/hermes-agent/releases/new
2. Tag: `v1.0.0`
3. Title: `DeepParser SDK + API Server v1.0.0`
4. Body: paste contents of `RELEASE_v1.0.0.md`
5. Attach `dist/deepparser-1.0.0.tar.gz` if available
6. Publish

---

## Step 7 — Post to HN

Post time: **9:00 AM PT on a weekday** (Mon–Wed optimal for Show HN traction)

1. Go to https://news.ycombinator.com/submit
2. Title: `Show HN: DeepParser – parse DWG/CAD drawings and Excel-embedded PDFs without OCR`
3. URL: `https://github.com/ysh145/hermes-agent/tree/main/deepparser`
4. Body: paste from `launch/hn_post.md` (HN doesn't use body for Show HN — just URL + title)

Monitor comments for first hour. Common questions to be ready for:
- "How does it handle scanned PDFs?" — uses the DeepParser backend's OCR pipeline, not standard Tesseract
- "What's the pricing?" — point to beta.deepparser.ai
- "Why not use LlamaParse / Unstructured?" — benchmark data is your answer; DWG support is unique

---

## Step 8 — Post-launch

- [ ] Check Fly.io metrics dashboard 1 hour after HN post
- [ ] Check PyPI download stats: https://pypistats.org/packages/deepparser
- [ ] Respond to all HN comments within 24 hours
- [ ] File issues for any bugs reported
- [ ] Update `RELEASE_v1.0.0.md` benchmark table with final scores if not done pre-launch

---

## Rollback plan

If the deploy produces errors:

```bash
# Roll back Fly.io to previous version
fly releases list --app deepparser-api
fly deploy --image <previous-image> --app deepparser-api

# Yank PyPI release (last resort — breaks existing installs)
# pip install twine
# twine upload --skip-existing dist/*    # not needed for yank
# Use PyPI web UI: Manage → Release → Yank
```

---

## Secrets reference

| Secret | Where | Notes |
|---|---|---|
| `DEEPPARSER_API_KEY` | Fly.io + GitHub Actions | `dp_live_*` from beta.deepparser.ai |
| `ADMIN_PASSWORD` | Fly.io | Random 24-byte hex; guards `/admin/*` routes |
| `DOCKERHUB_USERNAME` | GitHub Actions secret | `ysh145` |
| `DOCKERHUB_TOKEN` | GitHub Actions secret | Docker Hub access token |
| `FLY_API_TOKEN` | GitHub Actions secret | From `fly tokens create deploy` |
| PyPI OIDC | pypi.org trusted publisher | No secret needed; configure at pypi.org |
