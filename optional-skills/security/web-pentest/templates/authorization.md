# Engagement Authorization

Fill out before any active testing. Save to `engagement/authorization.md`.

---

**Engagement ID:** <UUID or short slug>
**Operator:** <name of the person driving this Hermes session>
**Date opened:** <ISO 8601 timestamp>
**Engagement window:** <start ISO timestamp> through <end ISO timestamp>

## Target

- Primary URL(s):
  - https://...
- Primary IP(s):
  - X.X.X.X
- Hostnames covered:
  - host.example.com
  - api.host.example.com
- Networks covered (CIDR):
  - 10.0.0.0/24 (internal lab)

## Authorization Basis

(Pick one — record evidence in writing for anything but ownership.)

- [ ] Operator owns the application and infrastructure being tested.
- [ ] Written authorization from <name, role, organization, date>.
      Document stored at: <path or link to signed authorization>.
- [ ] Hermes Agent dashboard, running on this same workstation, used
      as a self-test target. Operator confirms no other user is
      connected to the dashboard instance during the engagement.

## Out of Scope (must not be tested)

- Production systems unless explicitly listed above
- Third-party APIs / SaaS the application calls into
- Other tenants if the target is multi-tenant
- Cloud metadata endpoints (169.254.169.254, etc.) unless explicitly
  included above
- Destructive payloads (DROP, DELETE, file writes outside test
  directories) without per-payload approval
- Active social engineering, phishing, physical security

## Constraints

- Rate limit: <N> req/s per host. Default 5/s (200ms gap).
- Hours: <none> | <only between HH:MM and HH:MM local>
- Notify-before for: <list of categories> e.g. "any payload that
  writes data," "any traffic that touches the auth endpoint after
  10pm local"

## Acknowledgement

By approving this engagement, the operator confirms:

1. The targets listed above are authorized for active testing by the
   listed authorization basis.
2. Testing may produce HTTP 4xx/5xx responses, log noise, alert
   notifications, and rate-limit triggers in monitoring systems.
3. The operator is responsible for any consequences of testing
   targets that are NOT correctly authorized.
4. The operator will revoke authorization (by stopping the agent) if
   the scope changes, the time window ends, or any unexpected
   off-scope behavior is observed.

**Operator signature (typed name):** ________________
**Confirmed at:** <ISO 8601 timestamp>
