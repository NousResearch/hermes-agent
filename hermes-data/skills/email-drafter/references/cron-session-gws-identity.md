# GWS identity in cron / background sessions — wrong-mailbox resolution (verified 2026-09-09)

## Symptom

Gmail/Drive/Sheets API calls from `terminal` (trusted-process python) resolve to
the WRONG user's mailbox, even though `gws_resolve_account` shows the right token
exists and `has_token: true`.

Real case: a cron-run session (env `HERMES_SESSION_ID=cron_*`,
`HERMES_CRON_SESSION=1`) had `HERMES_SESSION_USER_ID=8502281203` while the job
owner id was `HERMES_CRON_JOB_OWNER_ID=ndr-7449813913`. With the stale id,
`build_service("gmail","v1", service_name="google-draas")` opened
**psingh@draas.com**: `q='subject:"JD for Sales Manager - DRA Homes"'` returned
COUNT 0. The token was fine — the identity mapping was wrong.

Root cause: `tools.gws_auth._current_telegram_id()` prefers
`gateway.session_context.get_gws_identity_env()` over `HERMES_SESSION_USER_ID`
(raw env fallback), but in a bare terminal subprocess the session context can be
absent and the raw env value wins. A stale `HERMES_SESSION_USER_ID` then maps to
a different vault user. The credential file simply isn't what you think it is.

## Diagnosis (1-liner)

```python
from tools.gws_auth import build_service
gmail = build_service("gmail", "v1", service_name="google-draas")
print(gmail.users().getProfile(userId='me').execute()['emailAddress'])
```

If it prints anything other than the expected mailbox (ndr@draas.com for
google-draas), the session identity is stale. `gws_fetch_token` alone does NOT
prove which mailbox the token belongs to — it returns token JSON with no
userinfo. Always profile-check before searching in cron/background contexts.

## Fix

Prefix the owner identity on the terminal command:

```bash
HERMES_SESSION_USER_ID=ndr-7449813913 /opt/hermes/.venv/bin/python3 /tmp/script.py
```

The owner id comes from `HERMES_CRON_JOB_OWNER_ID` in the env (do not guess;
read the env first with `env | grep -i hermes`). Verified mapping: with
`HERMES_SESSION_USER_ID=ndr-7449813913`, `build_service(..., service_name='google-draas')`
resolves to ndr@draas.com; the stale default in that session resolved to
psingh@draas.com. Apply the same prefix to all gws bridge calls
(`gws_skill_bridge.call('draft_reply_create', ...)`) in that terminal.

Note: the `gws_resolve_account` TOOL resolves identity from the session context
(trusted main process) and can disagree with a bare terminal env — treat the
tool's account list as ground truth for which accounts exist, and the profile
check as ground truth for which mailbox your terminal python is actually using.

## Re-runnable probe

`scripts/check_gmail_mailbox.py` prints: current `HERMES_SESSION_USER_ID`,
`get_gws_identity_env()`, and the resolved mailbox for `google-draas`. Run it
plain vs prefixed to confirm which id maps to which mailbox before any Gmail op
in a cron/background session.