# LAH-Core SFTP Deployment (liveaccesshub.com)

Use when deploying lah-core plugin files to production WordPress. Proven on
dual-funnel deploy (7168b468) and outcome-semantics deploy (056ba9f, 2026-08-11).
The plugin is `lah-core-router` (v1.0.3 ACTIVE) on Namecheap shared hosting
(LiteSpeed, WP 7.x / PHP 8.2). Full procedure also archived at
`/home/deploy/cloe-diagnostics/dual-funnel-observability-deploy-v1/lah-core-deploy/INSTALL-ROLLBACK.md`.

## Connection facts

- Host: `livewmgu@server313.web-hosting.com` port `21098`
- Key: `~/.ssh/lah_core_deploy_livewmgu` (dedicated, ed25519) — use
  `-i ~/.ssh/lah_core_deploy_livewmgu -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -P 21098`
- **Web root = `public_html`** — the SFTP home is `/home/livewmgu`; the plugin path is
  `public_html/wp-content/plugins/lah-core-router/includes/` (NOT `wp-content/...` from home).
- sftp has NO remote checksum command → download files back and `sha256sum` locally.

## Procedure (backup → upload .new → verify BEFORE mv → atomic mv → re-verify)

1. Extract files from the MERGE SHA (never working tree): 
   `git show <merge-sha>:wp-content/plugins/lah-core-router/includes/<file> > /tmp/staging/<file>`
   then `php -l` each + `sha256sum` (these are the expected hashes).
2. Read-only baseline: `get` the current live files, verify they match the previous
   deploy hashes (drift detection before touching anything).
3. Backup: `rename <file> <file>.bak-<YYYYMMDD>` — **if `.bak-<date>` already exists
   (repeat deploy same day), use `.bak-<date>b`, `.bak-<date>c`, ... NEVER overwrite
   an existing rollback copy.**
4. `put` staging file as `<file>.new` (both files), then `get` the `.new` files back
   and verify sha256 == expected BEFORE any rename.
5. `rename <file>.new <file>` + `chmod 644 <file>` (perms were 644).
6. `get` final files, verify sha256 == expected again.
7. Health (read-only): `curl -s -o /dev/null -w "%{http_code}" https://liveaccesshub.com/`
   → 200, and `https://liveaccesshub.com/lah-postback/affiliate` (GET, no params) → 400
   (route registered, NO conversion triggered). A 400 on the no-params probe is the
   expected/healthy state — do not misread it as failure.

## Verification script

`bash /home/deploy/cloe-diagnostics/dual-funnel-observability-deploy-v1/lah-core-deploy/verify-lah-core.sh`
checks: homepage 200, bridge plugin ACTIVE, site_status, endpoint 400 probe, file hashes.

## Rollback

Restore `<file>.bak-<date>[b]` → `<file>` via sftp rename, re-run verify script.
Never delete the rollback copies.

## Pitfalls

- SFTP batch mode (`sftp -b - <<'EOF'`) is non-interactive and fine for renames/puts;
  always do the hash-verify round trip before the atomic mv.
- Do NOT touch wp-config, .htaccess, other plugin files, or the MCP bridge — the
  bridge is read-only and must stay read-only.
- The lah-brain side auto-deploys on merge: verify `https://leanframeworklab.com/version`
  shows the exact merge SHA (may take a few minutes after merge — poll).
- Deployment is a production mutation: only run after explicit operator mandate
  (resume instructions) and display the exact commands before execution.
