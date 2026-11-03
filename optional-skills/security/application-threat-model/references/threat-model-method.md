# Application Threat Model Method

The prompts, vocabularies and test shapes the skill body points at. Open this while modeling.

## 1. Assets

An asset is something an attacker wants, not every table in the schema. Each row carries a data class
and exactly one loss; an asset with no loss is not an asset.

| Data class | Examples | Loss that usually matters |
| --- | --- | --- |
| Secret | credentials, signing keys, session tokens | disclosure, then impersonation |
| Regulated | card data, health records, national ids | disclosure, then penalty |
| Financial | balances, settlements, refunds, invoices | corruption or fraud |
| Operational | audit logs, configuration, feature flags | corruption, then repudiation |
| Availability-critical | checkout, auth, payment callback | unavailability |

The four loss words are fixed: **disclosure**, **corruption**, **unavailability**, **fraud**. If none
fits, the row is context, not an asset.

## 2. Trust boundaries

A boundary is any place the answer to "who is asserting this, and who checked?" changes. Ask all four
per boundary:

1. What crosses it — request, message, file, callback, schema?
2. What authenticates the crossing — mTLS, a signed webhook, a bearer token, a shared secret, nothing?
3. What does the receiving side assume without checking — that the caller is internal, that the id is
   the caller's own, that the amount was already validated?
4. What happens when the assumption is false?

Boundaries that recur: internet to edge, edge to service, service to service inside one network,
tenant to tenant in shared storage, application to third party, batch or cron to production data,
operator console to production.

## 3. Scenarios: the STRIDE prompts

Run all six against every boundary. One prompt, one scenario, or a recorded reason the category does
not reach this boundary.

| Category | Prompt for this boundary |
| --- | --- |
| Spoofing | Who can claim to be someone else across it, and what would they have to hold? |
| Tampering | What can be changed in transit or at rest, and what still accepts it afterwards? |
| Repudiation | Which action leaves no record that survives the actor deleting it? |
| Information disclosure | What leaks through errors, logs, timing, ids, or a response the caller should not see? |
| Denial of service | Which call is expensive, unbounded, or amplifiable, and what does its failure take down? |
| Elevation of privilege | What does a low-privilege caller reach that the boundary was meant to stop? |

Every scenario carries four fields: **entry point**, **path**, **precondition**, **impact**. A scenario
whose precondition nothing can satisfy is dropped with that sentence written down — never carried as
a maybe.

## 4. Control decisions

One decision per scenario, from a closed vocabulary, each with a named owner:

- **Mitigate** — add or strengthen a control. Requires a control and a test.
- **Transfer** — a third party or an insurer carries it. Requires the contract clause or the provider
  control that makes that true.
- **Accept** — the residual risk is tolerated. Requires who accepted it and when it is reviewed again.
- **Eliminate** — remove the feature, the data, or the path. Requires what is being removed.

## 5. Control state

Mark every control `deployed`, `planned` or `unverified`. `unverified` is the default: an architecture
diagram describing a control is not the control. Only observed configuration or a passing security
test moves a control to `deployed`.

## 6. Per-control tests

The test names the observable that fails when the control is removed. A test that passes with the
control deleted tests nothing.

| Control class | Test shape | Observable |
| --- | --- | --- |
| Authentication | call without, and with another tenant's, credential | 401/403, no record returned |
| Authorization | request another tenant's object id as an authenticated user | 403 or not-found, never the object |
| Input validation | submit the boundary-violating value the scenario needs | rejection before persistence, no partial write |
| Integrity | replay or alter a signed message | signature rejection, no state change |
| Rate limit or quota | drive the expensive call past its bound | throttle response, the dependency stays up |
| Audit | perform the action, then try to erase the record as the actor | the record survives, with actor and time |
| Secret handling | trigger the error path that formats the object holding the secret | redaction in the log, no value |

## 7. Residual risk

Close with what remains: accepted scenarios, unverified controls, and every scenario whose test does
not exist yet. The model is not a scan, a penetration test or a compliance attestation; say which of
the three the user still needs.
