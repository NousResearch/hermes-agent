# Credential Rotation

The order that keeps the service up while a credential changes, and the check that proves the old
one is dead. Used by a leaked-secret event and by a planned rotation with no exposure.

## The order

1. **Issue new** — create the replacement alongside the current credential. Most providers allow two
   live credentials per identity; where they do not, the rotation needs a second identity, which
   changes the plan — find out before step 2.
2. **Deploy new** — put the new credential everywhere the old one is used. Enumerate those places
   first: application config, CI secrets, developer machines, scheduled jobs, infrastructure-as-code
   state, and any partner who was given it. A place nobody listed is the outage.
3. **Verify new** — a real call succeeds with the new credential from each deployed environment, not
   only from a laptop.
4. **Revoke old** — disable the previous credential.
5. **Verify revoked** — a call with the old credential FAILS.

## What breaks if the order changes

| Change | Result |
| --- | --- |
| Revoke before deploy | a full outage for the window between them |
| Deploy before issue | every call fails with an authentication error nobody expects |
| Skip verify-new | a deployment that silently fell back to the old credential looks healthy until step 4 turns it off |
| Skip verify-revoked | if revocation did not take, the exposure that triggered the rotation is still open |
| Revoke immediately after deploy | in-flight requests and cached tokens issued under the old credential fail |

## The overlap window

Both credentials are live between steps 1 and 4. Size it by the longest thing that can still hold the
old one: token lifetime, cache TTL, an hourly scheduled job, a mobile client that updates weekly.

A **compromised** credential inverts this: the exposure outranks availability, so revoke first and
accept the outage. Say which case applies before naming a window.

## Revocation is proven by a failure, not by an exit code

A revoke command's success means the request was accepted, not that the credential stopped working.
Providers cache authorization decisions, propagate revocation asynchronously, and sometimes keep
issued tokens valid to their expiry.

The proof is a call made with the OLD credential that fails with an authentication or authorization
error. Record what was called, when, and the status returned. Not proof: a connection error (endpoint
unreachable), a 404 (wrong path), or a success (not propagated yet — wait and re-check).

Where the provider mints bearer tokens from the credential, a token minted before revocation may
outlive it. Name the longest such lifetime and treat the rotation as incomplete until it has passed.

## Per credential type

- **API key or token** — the five steps as written.
- **Certificate** — deploy the new chain to trust stores before switching the leaf; a client that does
  not trust the new issuer fails at step 3, which is where it should fail.
- **Database password** — some engines allow one password per role; rotation then needs a second role
  plus a connection-string switch, or a maintenance window.
- **Signing key** — verifiers must accept both keys before the signer switches; the old key retires
  only after nothing signed with it is still in flight.
- **OAuth client secret** — refresh tokens issued under the old secret may survive it; "we rotated the
  secret" and "old sessions are dead" are different claims.
