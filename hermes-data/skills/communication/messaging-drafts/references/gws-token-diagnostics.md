# GWS Token Diagnostics — Vault Token Checking & Re-Auth

## When to use

The vault socket is reachable (`/run/gws-vault/vault.sock` exists) but `build_service`/`gws_skill_bridge.call` raises "No X token for user Y." You need to determine:
- Does the user have a token under any service name?
- Has the token expired/been revoked?
- Has the vault lost ALL tokens (data loss scenario)?

## Method 2 — Check ALL service names (corrected for 3-account setup)

The `gws_vault_client` module may not have a `.py` source in all setups. If `ModuleNotFoundError` is raised, fall back to trying `build_service()` directly with each service name.

**CRITICAL:** The default `build_service('gmail', 'v1')` with no `service_name` looks for a token literally named `"google"`, which does **not exist** in the 3-account setup (`google-draas`, `google-gmail`, `google-ahfl`). It will report NO_TOKEN for every user even when valid tokens exist. Always check ALL known service names:

```bash
cat > /tmp/test_tokens.py << 'PYEOF'
import os, sys
os.environ.setdefault("HERMES_SESSION_USER_ID", "<uid>")
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
for svc in ['google-draas', 'google-gmail', 'google-ahfl', 'google']:
    try:
        svc_obj = build_service('gmail', 'v1', service_name=svc)
        profile = svc_obj.users().getProfile(userId='me').execute()
        print(f"{svc}: TOKEN_OK ({profile.get('emailAddress','?')})")
        break
    except FileNotFoundError:
        print(f"{svc}: NO_TOKEN")
    except Exception as e:
        e_msg = str(e)
        if "invalid_grant" in e_msg or "expired" in e_msg.lower():
            print(f"{svc}: TOKEN_EXPIRED_REVOKED")
        elif "No token" in e_msg:
            print(f"{svc}: NO_TOKEN")
        else:
            print(f"{svc}: {e_msg}")
PYEOF
for uid in ndr rnr sales1.blr pm2.blr vkdas; do
  echo "User $uid:"
  PYTHONPATH=/opt/hermes:$PYTHONPATH HERMES_SESSION_USER_ID=$uid /opt/hermes/.venv/bin/python3 /tmp/test_tokens.py 2>&1 | sed 's/^/  /'
done
```

## PITFALL — a terminal script silently drafts into the WRONG person's mailbox (hit 2026-09-21)

**Symptom:** Prakash asked for an email draft, it was created and verified successfully,
but he could not see it. It had landed in **ndr@draas.com**, not psingh@draas.com.

**Cause:** `build_service()` resolves the SESSION user. A `terminal()` script with no
`HERMES_SESSION_USER_ID` set has no session user to resolve, so the vault falls back to
the default/primary identity — **ndr**. Every `service_name='google-draas'` call then
authenticates as ndr, because `google-draas` is an ACCOUNT key (ndr@draas.com),
not a per-person key. `google-ahfl` = ndr@ahfl.in and `google-gmail` = nishantranka@gmail.com
are ALSO NDR's. So the service_name tells you nothing about whose mailbox you are in.

**Rule — always set `os.environ["HERMES_SESSION_USER_ID"] = "<user-slug>"` at the top of
any Gmail/Drive script you write, and ALWAYS assert the identity back before acting:**

```python
import os, sys
os.environ["HERMES_SESSION_USER_ID"] = "psingh"   # the SESSION user, not a service name
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
svc = build_service('gmail','v1', service_name='google-draas')
me = svc.users().getProfile(userId='me').execute().get('emailAddress')
assert me == 'psingh@draas.com', f"WRONG MAILBOX: {me}"     # <-- non-negotiable
```

Slugs observed: `ndr`, `psingh`, `sales1.blr`, `pm2.blr`. The same `google-draas`
service_name returns a DIFFERENT mailbox per user — verified 2026-09-21:
ndr → ndr@draas.com, psingh → psingh@draas.com, sales1.blr → sales1.blr@draas.com,
pm2.blr → pm2.blr@draas.com.

**Attribution caveat that follows from this:** a message authored as ndr appears in
threads with `From: Nishant Ranka <ndr@draas.com>`. When you reply on behalf of a
non-ndr user, create it from THEIR mailbox — otherwise the client sees NDR's address
on a mail he never sent. Recovering from the mistake: give the user that draft's
**Gmail link** (`https://mail.google.com/mail/u/0/#drafts?compose=<msg_id>`) rather
than copy-pasting a 34 MB MIME attachment between accounts, then create a fresh draft
in the correct mailbox and delete the stray one.

**Threads are per-mailbox.** `threadId` from ndr's mailbox is meaningless in psingh's —
a draft created with a foreign `threadId` will not thread. Omit `threadId` when creating
in a different mailbox, and keep `In-Reply-To`/`References` so the recipient's client
still threads it their end.

## Zero-Tokens-for-All Scenario (vault data loss)

**Pattern:** `has_token` returns `false` for EVERY user and every known service name, AND the previous cron run within 24 hours was successful.

**This is NOT "no user has authorized" — it's vault data loss.** The vault daemon may have restarted and lost its on-disk token store at `/opt/gws-vault/tokens/` on the host. 

**Fix:** Re-authorize the primary account. Generate an OAuth consent URL:

```python
from tools.gws_auth import get_auth_url
url = get_auth_url("[REDACTED-TID]")  # Nishant's telegram ID
print(url)
```

The user taps the link → signs in with ndr@draas.com → token is re-stored in the vault. All cron jobs and API operations resume working.

## Quick socket-level diagnostics

To query the vault directly without going through `build_service`:

```python
import socket, json

def vault_get_token(user_id, service):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect('/run/gws-vault/vault.sock')
    req = json.dumps({
        'op': 'get', 'user_id': user_id,
        'service': service, 'session_uid': user_id
    }) + '\n'
    s.sendall(req.encode())
    resp = s.recv(65536).decode()
    s.close()
    result = json.loads(resp)
    return result.get('ok', False)

def vault_has_token(user_id, service):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect('/run/gws-vault/vault.sock')
    req = json.dumps({
        'op': 'has_token', 'user_id': user_id,
        'service': service, 'session_uid': user_id
    }) + '\n'
    s.sendall(req.encode())
    resp = s.recv(65536).decode()
    s.close()
    result = json.loads(resp)
    return result.get('ok') and result.get('has_token')
```

## Environment setup for terminal scripts

When running token diagnostics from a `terminal()` call (not `execute_code`):

```bash
PYTHONPATH=/opt/hermes:$PYTHONPATH HERMES_SESSION_USER_ID=ndr /opt/hermes/.venv/bin/python3 /tmp/your_script.py
```

- `PYTHONPATH=/opt/hermes` is required for `from tools.gws_auth import build_service`
- `HERMES_SESSION_USER_ID` must match the user (e.g. `ndr`, not `ndr-[REDACTED-TID]`)
- Use terminal heredoc (`cat > /tmp/script.py << 'PYEOF'`) — `write_file` may block `/tmp/` paths

## Key pitfall — vault resolve vs. token store keys

The vault uses canonical user IDs (`ndr-[REDACTED-TID]`) internally but accepts multiple identity types to resolve them:
- `resolve(slug, ndr)` → `ndr-[REDACTED-TID]`
- `resolve(telegram, [REDACTED-TID])` → `ndr-[REDACTED-TID]`

The `canonical_uid()` function in `gws_auth.py` tries `slug` then `draas_user_id` for non-numeric, non-email IDs. This works correctly for "ndr". The token is stored under the canonical ID, so `build_service` resolves internally and finds the right key.
