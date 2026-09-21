# Gmail MCP Unblock Procedure

Proven fix for "Token OK, OAuth callback blocked" on headless VPS.
Token files show healthy (age 0 days) but MCP reads fail with auth-link-pending.

## Root cause

The workspace-mcp process holds stale in-memory OAuth state from running too long
(21+ hours is enough). Token files on disk are fine but the process can't use
them. Restarting the gateway without clearing OAuth state does NOT fix this —
the stale state survives.

## Symptoms

- token_health.py: all accounts healthy (age 0 days)
- Port 8000: one clean owner under active gateway
- But search_gmail_messages / list_gmail_labels for specific accounts fails
- Error: "OAuth callback blocked — auth link pending"

## Fix sequence (STANDARD — try this first)

Use this when token files are healthy (age 0 days per token_health.py):

```
sudo systemctl stop hermes-gateway.service
echo '{}' > ~/.google_workspace_mcp/credentials/oauth_states.json
# Do NOT delete <email>.json token files. They contain valid refresh tokens.
sudo systemctl start hermes-gateway.service
```

Then retrigger mailbox-digest-daily or run a probe call.

If this works (which it did on 2026-05-02 and may work again), you avoided
unnecessary re-auth. The user does not need to click through Google consent.

## Fix sequence (AGGRESSIVE — only if STANDARD failed)

Use this when the STANDARD sequence above did not restore access, OR when
token_health.py explicitly reports the token as expired:

```
sudo systemctl stop hermes-gateway.service
echo '{}' > ~/.google_workspace_mcp/credentials/oauth_states.json
rm ~/.google_workspace_mcp/credentials/<blocked-email>.json
sudo systemctl start hermes-gateway.service
```

⚠️ Deleting a token file forces a full OAuth re-consent. The user must open an
incognito window, click through Google consent screens, and the 7-day expiry
clock resets. This is the same flow as initial account setup. Only do this when
the STANDARD sequence already failed.

After deleting, trigger a probe call for that account to generate a fresh OAuth
URL. Pass the URL to the user. After consent, the token file will be recreated.

## Orphan process cleanup

Check: `ps aux | grep workspace-mcp | grep -v grep | wc -l`
Healthy: exactly 2 entries (1 uv wrapper + 1 python child).
Orphan: 3 or 4 entries. Kill the stale ones:

```
kill <orphan PID>  # the older one, not the gateway's child
```

Two types of orphans:
1. Port 8000 holder — blocks OAuth (must kill)
2. RAM-only — wastes ~130MB, doesn't break OAuth directly but signals trouble

Verify: `sudo ss -tlnp | grep 8000` should show exactly one python PID.

## Verification

After fix, the next mailbox-digest-daily run should show all Gmail accounts as
OK (not "Token OK, OAuth callback blocked").

## Lessons from 2026-05-03

The STANDARD fix (clear oauth_states only + restart) worked on 2026-05-02 but
did NOT work on 2026-05-03 after the gateway had been running 21+ hours. The
AGGRESSIVE fix (delete tokens) was needed, BUT: check token_health.py FIRST.
If it says age 0 days, the tokens are healthy. The issue is the MCP process,
not the token files. Deleting healthy tokens forces unnecessary user re-auth.

Key rule: always try STANDARD first. Only escalate to AGGRESSIVE when STANDARD
fails. And never delete tokens without first verifying token_health.py status.
