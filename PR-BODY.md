<!-- native-links:v1 -->
Related #64704 #70776 #70792 #64740

## What does this PR do?

Closes the **Weixin iLink getUploadUrl ret:-2** defect class (#64704, #70776, #70792): the iLink API's `getUploadUrl` endpoint now requires `context_token` for image/media upload authorization. Without it, the endpoint returns `ret: -2` (parameter validation error) and all outbound media (images, files, voice) fail to send.

**Fix:** `_get_upload_url()` accepts an optional `context_token` and threads it into the payload when available; the adapter passes the per-chat token from its token store. This is the fix from #64740 (author: **Auto-Fix-Bugs Bot**, cherry-picked with authorship preserved), which was closed unmerged.

## Repro receipt (sha256)

`repro_ilink.py` — sha256 `dee97a8026b553ee0708d4f266fffd2d8db101ac2d4be90c9422a1d249436fef`

```bash
# baseline (origin/main @ 9d6c5a920c7)
python repro_ilink.py
# FAIL: _get_upload_url has no context_token param   (exit 1)

# this branch
python repro_ilink.py
# PASS: _get_upload_url accepts context_token
# PASS: payload includes context_token when provided   (exit 0)
```

## How to test

```bash
pytest tests/gateway/test_weixin.py -q
# 29 passed, 1 pre-existing failure (test_qr_login_timeout_uses_monotonic_clock — verified identical on origin/main baseline)
```

## What platforms were tested?

- Windows 11 native: `29 passed`; the 1 failure is pre-existing on `origin/main` (baseline worktree verified) — zero regressions.

## Why this matters to users

Weixin users can send images/files/voice again instead of silent `ret: -2` failures — the exact report filed three times (#64704, #70776, #70792).

Closes #64704
Closes #70776
Closes #70792

- [x] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature
- [ ] Breaking change

## Checklist

- [x] Code follows repo style
- [x] Self-review complete
- [x] Repro receipt: baseline FAIL / fixed PASS (sha256 above)
- [x] Suite green: no regressions vs baseline (1 pre-existing failure identical)
- [x] `git diff --check` clean
- [x] Credit: implementation from #64740 by @Auto-Fix-Bugs Bot, cherry-picked with authorship preserved
