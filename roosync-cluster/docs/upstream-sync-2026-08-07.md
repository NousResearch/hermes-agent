# Upstream Sync — 2026-08-07

## Summary

Sync of fork `jsboige/hermes-agent` with `NousResearch/hermes-agent` (upstream).
**Scope: 6744 upstream commits** since last sync (2026-07-02, commit `ce9aa869f`).
**Drift at merge start:** 2 local commits (`13b1caffa` cron guard, `cdee6d5c9e` revert claudish → z.ai native).

**Result: clean merge, single non-structural conflict (`tools/environments/base.py` diagnostic logging), ZERO patches to re-apply** — upstream integrated all our `main-wrapper.sh` patches into a superior version. Image rebuilt as `hermes-agent:s6-sync-20260807`, deployed, **all 12 hermes-verify checks PASS**.

## Timeline

| Event | Time (UTC) | Commit / Artifact |
|---|---|---|
| Last sync | 2026-07-02 | `ce9aa869f` (base of this sync) |
| Last commit before sync | 2026-08-07 | `cdee6d5c9e` (revert Phase 2 v3 claudish → z.ai native) |
| Backup taken | 2026-08-07 14:02 | `hermes-20260807-140247.tar.gz` (3223.7 MB, full /opt/data) |
| Rollback refs created | 2026-08-07 | tag `pre-upstream-sync-20260807` + branch `sync/upstream-20260807` |
| Merge commit | 2026-08-07 | `f1b5a65715` (Merge upstream/main) |
| Image built | 2026-08-07 | `hermes-agent:s6-sync-20260807` (3.96 GB) |
| Container redeployed | 2026-08-07 | `3def73fee042` |
| Verify complete | 2026-08-07 | **12/12 checks PASS**, all 3 crons `ok`, Telegram connected |

## Strategy

`take upstream entirely, then re-apply our patches` (canonical pattern from
`patches-to-reapply.md` + precedent `upstream-sync-2026-07-02.md`). Used **merge**
(never rebase — force-push forbidden on shared branch).

## Conflicts (1 total)

### tools/environments/base.py — non-structural, fully resolved

Upstream removed the `init_session` exception block; our version (HEAD) kept a
diagnostic logging block (traceback + cwd + HOME) from the PermissionError /root
96h incident (`feedback_permission_error_root.md`).

**Resolution:** kept HEAD (our diagnostic block). It is additive and useful for
debugging terminal-environment failures — upstream simply never had it.

```python
# OUR VERSION (kept)
except Exception as exc:
    import traceback as _tb
    _tb_str = _tb.format_exc()
    logger.warning(
        "init_session failed (session=%s, cwd=%s, HOME=%s): %s "
        "- falling back to bash -l per command - TB: %s",
        self._session_id, self.cwd,
        self.env.get("HOME", "N/A") if self.env else "no-env",
        exc, _tb_str,
    )
    self._snapshot_ready = False
```

## Patches re-applied: ZERO

This is the cleanest sync yet — `patches-to-reapply.md` patches **all survived the
auto-merge or were made obsolete by upstream improvements**:

| Patch | Status |
|-------|--------|
| #1 Dockerfile CRLF strips + COPY 012/013 | ✅ Auto-merged intact (CRLF strips at lines 343-344, 364-365) |
| #2 `docker/main-wrapper.sh` (with-contenv + HOME) | ✅ **OBSOLETE** — upstream integrated all our changes into a superior version (conditional `with-contenv` re-exec, `HOME=/opt/data`, `.env` loading, UID guards). `patches-to-reapply.md` section #2 now historical. |
| #3 `.gitattributes` LF enforcement | ✅ Auto-merged intact |
| #4 `cont-init.d/012-roosync-backup` + `013-roosync-restore` | ✅ Our files only, untouched |
| #5 `hermes-restore-config.sh` XDG_STATE_HOME + HOME | ✅ Our file only, untouched |

## Upstream fixes relevant to our documented bugs

| Upstream commit | Our bug | Memory |
|-----------------|---------|--------|
| `fix(cron): bound TERMINAL_CWD lock acquire with timeout` (#79768) | Incident PermissionError /root 96h | `feedback_permission_error_root.md` |
| `fix(compress): preserve in-flight tool chain across context compression` (#79278) | Tool registry loss post-compaction | `feedback_nanoclaw_lessons.md` |
| `fix(cron): move watchdog state under the request lock` | Cron scheduler robustness | `feedback_hermes_cron_toolsets.md` |
| `fix(gateway): interrupt api server runs on shutdown` | Gateway 24/7 stability | — |

## Combined with claudish revert

This sync includes commit `cdee6d5c9e` (revert Phase 2 v3 claudish → z.ai native),
committed on main **before** the merge. The merge preserved it (`roosync-cluster/`
is isolated). The deployed image therefore ships both the upstream code AND the
z.ai native provider config. All 3 crons confirmed `status=ok` post-deploy (the
inbox-poll "No Anthropic credentials" error that preceded the revert is resolved).

## Rollback

```powershell
docker stop hermes
docker rm hermes
docker run -d --name hermes `
  --restart unless-stopped `
  -v C:\Users\jsboi\.hermes:/opt/data `
  -v C:\dev\roo-extensions\mcps\internal\servers\roo-state-manager:/opt/roo-state-manager:ro `
  --add-host=host.docker.internal:host-gateway `
  -p 9120:9119 `
  hermes-agent:s6-sync-20260702 gateway run   # previous image
```

Tag `pre-upstream-sync-20260807` + backup `hermes-20260807-140247.tar.gz` available
if a deeper rollback is needed.

## Notes for next sync

- **`patches-to-reapply.md` section #2 is OBSOLETE** — upstream now owns a better
  `main-wrapper.sh`. Do not re-apply our old version; document this so the next
  sync doesn't reintroduce a regression.
- The **extended-regex sed pattern** in `hermes-restore-config.sh`
  (`sed -i -E 's/.../"?(...)"?$/'`) handles both quoted and unquoted YAML values
  — robust against upstream config-format changes. Reuse for future edits.
- `docker exec` from Git Bash: **always `MSYS_NO_PATHCONV=1`**, else Linux paths
  get mangled (`/opt/data` → `C:/Program Files/Git/opt/data`).
- `hermes-backup.ps1` line 58 uses `-Encoding UTF8NoBOM` which fails under
  Windows PowerShell 5.1 (only valid in pwsh 7+). Non-fatal (backup still created),
  but leaves the container stopped after backup. Fix: run backup under pwsh 7, or
  patch the script to use `[System.IO.File]::WriteAllText()`.
