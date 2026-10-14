# Penetration Test Report

**Target:** <name + URL>
**Engagement ID:** <slug>
**Engagement window:** <start> – <end>
**Operator:** <name>
**Tester:** Hermes Agent + operator
**Report generated:** <ISO 8601 timestamp>

---

## Executive Summary

<2-4 paragraph plain-language summary. Focus on:
 - What was tested
 - What was found (count by severity)
 - Most critical finding in one sentence
 - High-level remediation recommendation>

| Severity | Count |
|----------|-------|
| Critical | 0     |
| High     | 0     |
| Medium   | 0     |
| Low      | 0     |
| Info     | 0     |

---

## Engagement Scope

In-scope targets (from `engagement/scope.txt`):

- <host or CIDR>

Out of scope: see `engagement/authorization.md`.

Authorization basis: see `engagement/authorization.md`.

## Methodology

Approach was based on the Hermes `web-pentest` skill (a Hermes Agent
adaptation of the OWASP Testing Guide with elements of Shannon's
proof-based methodology). Phases performed:

- [ ] Pre-recon (source code review)
- [ ] Recon (live, read-only)
- [ ] Vulnerability analysis (one queue per OWASP class)
- [ ] Exploitation (proof-based)
- [ ] Reporting

Tools used: <nmap, whatweb, curl, Hermes browser tool, ...>.

## Findings (L3/L4 — Verified Exploitable)

> Every finding in this section has a reproducible proof-of-concept.
> L1/L2 candidates that were not promoted to confirmed exploitation
> are listed in the "Not Exploited" section.

### F-001: <Title>

- **Severity:** Critical | High | Medium | Low
- **CVSS 3.1 vector:** `CVSS:3.1/AV:N/AC:L/...`
- **CVSS 3.1 base score:** N.N
- **CWE:** CWE-XX
- **Affected endpoint(s):** `GET https://target.example/api/...`
- **Affected parameter(s):** `id`
- **Discovered:** <date>

#### Description

<What is the bug, in plain language.>

#### Proof

Request:

```http
GET /api/items?id=1%27%20OR%201=1-- HTTP/1.1
Host: target.example
Cookie: session=...
```

Response (excerpt):

```http
HTTP/1.1 200 OK
Content-Type: application/json

[{"id":1,...}, {"id":2,...}, ... <full table dumped>]
```

#### Reproduction

```bash
curl -sS 'https://target.example/api/items?id=1%27%20OR%201=1--' \
     -H 'Cookie: session=YOUR_TEST_SESSION'
```

#### Impact

<What an attacker gains. Be specific. "Could allow data extraction" is
worse than "Allowed extraction of all 4 columns from the `users` table
in our test (PoC redacted PII), and the same query shape applies to
any other parameter using the same code path.">

#### Remediation

<Specific, actionable. "Use parameterized queries" is better than
"sanitize inputs." Include code example if possible.>

#### Verification (post-fix)

To verify the fix, re-run the reproduction command. The response
should be HTTP 400, an empty result, or a result containing only the
record matching `id=1` literally.

---

(repeat per finding)

---

## Not Exploited (L1/L2 candidates)

Candidates that pattern-matched but were not promoted to L3 within
the engagement window. Listed for completeness; do NOT report these
as confirmed vulnerabilities.

| ID | Class | Endpoint | Status | Why not promoted |
|----|-------|----------|--------|------------------|
| INJ-002 | SQLi | `/api/search?q=` | L2 partial | Bypass set exhausted; appears to use parameterized binding |
| XSS-003 | reflected | `/error?msg=` | L1 identified | Could not produce executable context — output is JSON-encoded |

---

## Out-of-Scope Observations

(Findings or hints noticed but NOT tested because they were outside
scope. These are documentation, not findings. The operator decides
whether to extend scope and re-test.)

- The application sends to `https://third-party.example/...` — payload
  could trigger third-party-side bugs but third party is out of scope.

---

## Limitations

What was NOT tested, and why:

- <Class of test>: <reason>

Examples:
- DDoS / stress testing — explicitly excluded by engagement scope.
- Authenticated business-logic flows requiring billing — no test
  credit card available.
- Mobile API surfaces — out of scope.

---

## Appendices

- A: `engagement/authorization.md` — authorization on file
- B: `engagement/scope.txt` — machine-readable scope
- C: `engagement/request-log.jsonl` — every active request issued
- D: `findings/*-queue.json` — per-class candidate queues
- E: `evidence/` — raw captures (request/response pairs)

---

## Disclaimer

This report describes vulnerabilities discovered during a
time-bounded penetration test against the listed targets within the
listed scope. Absence of a finding in this report does not imply the
target is secure; only that no exploitable issue was found in scope
X within time T using methods Y.
