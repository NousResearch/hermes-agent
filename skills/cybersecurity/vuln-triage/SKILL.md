# Vulnerability Triage

Rapidly prioritises a CVE backlog by correlating CVSS severity, EPSS exploitation probability, and your asset inventory — all from free public APIs.

## When to use

- New CVE published and team needs to know if it's critical for their stack
- Weekly patch cycle: rank open CVEs by real-world risk
- Post-pentest: triage findings before writing the remediation report

## Workflow

### Single CVE

```
vuln_triage(cve_id="CVE-2024-XXXXX", assets=["Apache 2.4", "nginx 1.24", "Ubuntu 22.04"])
```

Interpret the result:
- **CRITICAL** → patch within 24 h; escalate to incident if exploits are circulating
- **HIGH** → patch within 72 h
- **MEDIUM** → next scheduled patch window (≤14 days)
- **LOW** → next patch cycle or accept risk

### Bulk triage (backlog)

For a list of CVEs, use `delegate_task` to triage in parallel:

```
delegate_task: "For each CVE in [CVE-A, CVE-B, CVE-C, ...], call vuln_triage with 
assets=[<asset list>] and return a table: CVE | CVSS | EPSS | Priority | Recommendation"
```

Collect results and sort by priority.

### After triage

- Open an IR incident for any CRITICAL findings: `ir_incident(action="create", severity="P1", ...)`
- Add CVE as evidence: `ir_incident(action="add_evidence", evidence_type="note", value="CVE-... CRITICAL — patch immediately")`
- Get full CVE context: `threat_intel(action="cve_lookup", cve_id=...)`
- Map to ATT&CK: `threat_intel(action="mitre_ttp", technique_id=...)` for any known associated TTPs

## Priority formula

| Condition | Label |
|-----------|-------|
| CVSS ≥ 9.0 OR (CVSS ≥ 7 AND EPSS ≥ 50%) | CRITICAL |
| CVSS ≥ 7.0 OR (CVSS ≥ 4 AND EPSS ≥ 30%) | HIGH |
| CVSS ≥ 4.0 | MEDIUM |
| Otherwise | LOW |

EPSS (Exploit Prediction Scoring System) is sourced from first.org and updated daily — it reflects the probability of exploitation in the wild within 30 days based on threat actor behaviour, PoC availability, and historical patterns.

## Notes

- NVD API is rate-limited at ~5 requests / 30 seconds. For bulk jobs, add a short delay between subagent calls or batch them in groups of 5.
- Asset matching is keyword-based (CPE vendor/product against your asset string). Verify matches manually for high-stakes decisions.
- EPSS scores are not available for very new CVEs (< 24 h since NVD publication). The tool will note when the score is unavailable.
