# Snapshot Backup & Credential Redaction

When backing up Hermes state snapshots to git/GitHub LFS, credential files must be stripped before committing.

## Files That Contain Secrets

Every snapshot directory may contain:

| File | Content | Action |
|------|---------|--------|
| `auth.json` | OAuth tokens, API credentials | **Replace with stub** |
| `config.yaml` | Full Hermes config with API keys, OAuth client secrets, env vars | **Replace with stub** |
| `.env` | Environment variables (API keys, tokens) | **Replace with stub** |
| `state.db` | Session/message database (SQLite) | **Safe to commit via LFS** |
| `state.db-journal` | SQLite journal | **Safe to commit via LFS** |

## Redaction Procedure

```bash
# 1. Copy snapshot to repo backup location
cp -a /root/.hermes/state-snapshots/<snapshot-name> /root/indigo-repo/backups/state-snapshots/

# 2. Redact auth.json (replace entirely)
echo '{"redacted": true, "note": "Credentials stripped for backup"}' \
  > /root/indigo-repo/backups/state-snapshots/<snapshot-name>/auth.json

# 3. Redact config.yaml (replace with stub — the live config is already at repo root)
printf '# config.yaml -- REDACTED\n# Credentials stripped for backup. Live config at repo root.\nconfig_version: REDACTED\ncredentials: REDACTED\n' \
  > /root/indigo-repo/backups/state-snapshots/<snapshot-name>/config.yaml

# 4. Redact .env
echo '# REDACTED - credentials stripped' \
  > /root/indigo-repo/backups/state-snapshots/<snapshot-name>/.env

# 5. Verify no secrets remain (text files only)
grep -rn 'sk-or\|GOCSPX\|mnfst_\|550801240087\|client_secret' \
  /root/indigo-repo/backups/state-snapshots/<snapshot-name>/ \
  --include='*.yaml' --include='*.json' --include='*.env'
# Should return empty

# 6. Ensure LFS tracking and push
git lfs track "backups/state-snapshots/**"
git add .gitattributes backups/state-snapshots/<snapshot-name>/
git commit -m "backup: state snapshot <snapshot-name>"
git push origin main
```

## Verification Checklist

- [ ] `auth.json` is a small stub
- [ ] `config.yaml` is a stub (not the multi-KB config with secrets)
- [ ] `.env` is empty or comment-only
- [ ] grep for known secret patterns returns no matches
- [ ] `state.db` and `state.db-journal` are intact
- [ ] `.gitattributes` includes `backups/state-snapshots/**` in LFS tracking
