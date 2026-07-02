# Upstream Sync — 2026-07-02

## Summary

Sync of fork `jsboige/hermes-agent` with `NousResearch/hermes-agent` (upstream).
**Scope: 2322 upstream commits** since last sync (17 juin 2026, commit `aa6f77596`).
**Drift at merge start:** 1 local merge commit (`4350d4e4b`, `/fresh-task`).

**Result: clean merge, single non-structural conflict (Dockerfile perms refactor).**
Image rebuilt as `hermes-agent:s6-sync-20260702`, deployed, all 12 hermes-verify checks
PASS except the known false positive on `Model` (script hard-codes the old `glm-5-turbo`
check; reality is `claude-sonnet-4-6` via claudish proxy).

## Timeline

| Event | Time (UTC) | Commit / Artifact |
|---|---|---|
| Last sync | 2026-06-17 | `aa6f77596` (base of this sync) |
| Last commit before sync | 2026-07-02 14:13 | `4350d4e4b` (`feat(claude): add /fresh-task command`) |
| Backup taken | 2026-07-02 14:10 | `hermes-20260702-141035.tar.gz` (2201 MB, full /opt/data) |
| Rollback refs created | 2026-07-02 14:13 | tag `pre-upstream-sync-20260702` + branch `backup/pre-upstream-sync-20260702` on `4350d4e4b` |
| Merge commit | 2026-07-02 13:17 | `6d822848c` (Merge upstream/main) |
| Image built | 2026-07-02 13:15 | `hermes-agent:s6-sync-20260702` |
| Container redeployed | 2026-07-02 13:15 | `de19f1d9a978` |
| Verify + boot complete | 2026-07-02 13:16 | Telegram connected, 11/12 checks PASS |

## Strategy

`take upstream entirely, then re-apply our patches` (canonical pattern from
`patches-to-reapply.md` + precedent `upstream-sync-2026-05-26.md`).

Used **merge** (never rebase — force-push forbidden on shared branch).

## Conflicts (1 total)

### Dockerfile — non-structural, fully resolved

Upstream refactored the `Permissions` section between merge-base and HEAD:

- `COPY --chown=hermes:hermes . .` → `COPY --link --chmod=a+rX,go-w . .`
  (skips the 21s amd64 / 222s arm64 `chmod -R` pass, #49113)
- Added `ENV PYTHONDONTWRITEBYTECODE=1` (related to our `.pyc` lessons — see
  `feedback_permission_error_root.md`)
- Added `COPY apps/shared/ apps/shared/` (workspace dep)
- Replaced `RUN chmod -R a+rX /opt/hermes && chown -R hermes:hermes …`
  with: install of `/opt/hermes/bin/hermes` exec shim + `.install_method` stamp
  (`/opt/hermes/.install_method`, baked next to code, not on shared `/opt/data`)

Our 4 Dockerfile hunks (A: `chmod 755 /root`, B: CRLF strip s6-rc.d, C: COPY
012/013-roosync + CRLF strip cont-init.d, D: `ENV HOME=/opt/data`).

**Two conflict hunks:**

#### Hunk 1 (lines 215-230): chmod/chown of /opt/hermes

| | Content |
|---|---|
| ours | `RUN chmod -R a+rX /opt/hermes && chown -R hermes:hermes … && chmod 755 /root` |
| upstream | `RUN mkdir -p /opt/hermes/bin && cp /opt/hermes/docker/hermes-exec-shim.sh … && printf 'docker\n' > /opt/hermes/.install_method` |

**Resolution: TAKE UPSTREAM.** Our `chmod 755 /root` is redundant with the new
`--chmod=a+rX,go-w` model + the `cd` guard in `docker/main-wrapper.sh` (our
patch F, preserved). Upstream's exec shim + install-method stamp are strictly
superior (privilege drop for `docker exec`; volume-share-safe stamp).

#### Hunk 2 (lines 309-327): env vars in runtime section

| | Content |
|---|---|
| ours | `ENV HOME=/opt/data` |
| upstream | `ENV HERMES_WRITE_SAFE_ROOT=/opt/data` + `ENV HERMES_DISABLE_LAZY_INSTALLS=1` + `ENV HERMES_LAZY_INSTALL_TARGET=/opt/data/lazy-packages` |

**Resolution: TAKE UPSTREAM.** The new env trio replaces our `ENV HOME=/opt/data`.
This is fine because our **patch F in `docker/main-wrapper.sh`** (preserved by
the merge — upstream did not touch that file) already exports `HOME=/opt/data`
at runtime via `with-contenv`, which is the correct layer for this fix (the
container-level ENV was redundant with the wrapper override).

### Files auto-merged clean

| File | Diff (auto-merge) |
|---|---|
| `.gitignore` | upstream added 4 entries (none in our hot zone) |
| `tools/environments/base.py` | upstream extended (we kept our `HERMES-PATCH-v3` sentinel — confirmed running) |
| `tools/environments/local.py` | upstream extended (same — sentinel running) |

### Files upstream did NOT touch (our patches intact)

- `docker/main-wrapper.sh` — shebang `#!/command/with-contenv sh`, `export HOME=/opt/data`, `cd /opt/data`, `.env` loading block all preserved
- `docker/cont-init.d/012-roosync-backup` — present, executable
- `docker/cont-init.d/013-roosync-restore` — present, executable
- `roosync-cluster/` — 0 lines diff vs our base
- `.claude/` — 0 lines diff vs our base
- `.gitattributes` — upstream extended with `Dockerfile`/`*.dockerfile`/`docker/entrypoint.sh`, our `*.sh text eol=lf` preserved

## Patches Audit (Checklist A-K)

| # | File | Patch | Status |
|---|---|---|---|
| A | Dockerfile | `chmod 755 /root` | **SUPERSEDED** by upstream's `--chmod=a+rX,go-w` + shim + patch F guard |
| B | Dockerfile | CRLF strip s6-rc.d | ✓ preserved (line 259) |
| C | Dockerfile | COPY 012/013-roosync + CRLF strip cont-init.d | ✓ preserved (lines 277, 280) |
| D | Dockerfile | `ENV HOME=/opt/data` | **SUPERSEDED** by upstream `HERMES_WRITE_SAFE_ROOT` + patch F wrapper export |
| E | main-wrapper.sh | `.env` loading block | ✓ preserved |
| F | main-wrapper.sh | `cd` de restauration | ✓ preserved |
| G | cont-init.d/013-roosync-restore | shim restore | ✓ ours-only, intact |
| H | cont-init.d/012-roosync-backup | auto-backup boot | ✓ ours-only, intact |
| I | roosync-cluster/ (22 files) | drift isolé | ✓ 0 lines diff |
| J | .claude/ (6 tracked) | config Hermes | ✓ 0 lines diff |
| K | .gitattributes | LF enforcement | ✓ preserved (extended upstream) |

## Deployment

- **Image:** `hermes-agent:s6-sync-20260702`
- **Container:** `de19f1d9a978` (replaced `5cd3…`)
- **Started at:** 2026-07-02 13:15 UTC
- **First check at:** 13:16:04 (kanban dispatcher embedded)
- **Telegram connected:** 13:15:58
- **MCP bridges:** 3/3 connected (roo-state-manager, sk-agent, searxng)
- **Cron jobs:** 3/3 scheduled (`hermes-cluster-tour`, mini-audit, pr-review)

## Rollback

If `s6-sync-20260702` proves unstable:

```powershell
docker stop hermes
docker rm hermes
docker run -d --name hermes --restart unless-stopped `
  -v C:\Users\jsboi\.hermes:/opt/data `
  -v C:\dev\roo-extensions\mcps\internal\servers\roo-state-manager:/opt/roo-state-manager:ro `
  --add-host=host.docker.internal:host-gateway -p 9120:9119 `
  hermes-agent:s6-20260528 gateway run
```

For full git rollback to pre-sync state:

```bash
git switch main
git reset --hard pre-upstream-sync-20260702   # returns to 4350d4e4b
```

## Side effects / observations

- **Auto-backup on next reboot** will snapshot the post-sync tree. If you need
  to preserve the pre-sync tree at /opt/data, do a manual backup NOW:
  `.\roosync-cluster\scripts\hermes-backup.ps1 -Reason "pre-s6-sync-revert"`
- The `claude-sonnet-4-6` model on the `claudish` proxy (via 192.168.0.46:3000 →
  `gc@glm-5.2`) is unchanged from pre-sync. Restore script still produces a
  compatible config.yaml.
- The previous full backup (`hermes-20260702-141035.tar.gz`, 2201 MB) remains
  in `C:\Users\jsboi\hermes-backups\` for as long as the 5-slot rotation
  holds (it was the most recent before this rebuild — keep for safety).

## Next steps

1. **Monitor** the cluster-tour cron (next fire 14:00Z) — confirms Telegram
   delivery still works after the rebuild.
2. **Push** `origin main` once the first cron cycle confirms stability.
3. **Audit redondances** — patches A and D are gone (good). Patch F remains
   structurally necessary as long as upstream keeps `ENV HOME=/root` in the
   `with-contenv` default. Watch for upstream adopting the wrapper export.