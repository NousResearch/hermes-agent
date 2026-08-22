# Multi-Source Protocol Research Pattern

## When to use

Any READ_ONLY_AUDIT or DESIGN_ONLY mission that requires understanding an **external protocol, API, standard, or system** that CLOE or the LAH Stack might integrate with. Use this before writing an architecture document or threat model.

## The pattern

Research in **four parallel lanes** using `anysearch` (or the available web search tool):

```
Lane A — Protocol Specification
  - Official protocol repository (GitHub, GitLab)
  - Protocol specification document (v1, v2, etc.)
  - Core message schemas, lifecycle, transport variants
  - IETF RFC or Internet-Draft if applicable

Lane B — Vendor / Official Documentation
  - Quickstart guides
  - SDK references
  - Facilitator or service documentation
  - Pricing, free tiers, rate limits
  - Supported networks, assets, schemes

Lane C — Security Research
  - Academic papers (arXiv, NDSS, IEEE, ACM)
  - Independent security analyses
  - Known vulnerabilities, CVEs
  - Practical attack demonstrations

Lane D — Standards & Ecosystem
  - IETF, W3C, or other standards body efforts
  - Ecosystem adoption metrics
  - Industry partnerships and integrations
  - Competing or complementary protocols
```

### Execution

Run all four searches **in parallel** (same turn, separate terminal/anysearch calls):

```bash
# Lane A
anysearch search "<protocol> specification <version>" --max_results 5

# Lane B
anysearch search "<vendor> <protocol> documentation" --max_results 5

# Lane C
anysearch search "<protocol> security vulnerability attack" --max_results 5

# Lane D
anysearch search "<protocol> standard IETF ecosystem" --max_results 5
```

### Read and triangulate

Read the top results from each lane, then distinguish:

| Classification | Meaning | Evidence |
|---------------|---------|----------|
| **Protocol fact** | Defined in the official spec, cross-referenced | Multiple sources agree, spec text confirmed |
| **Vendor-specific** | Behavior unique to one implementation | Single-vendor docs, not in spec |
| **Assumption** | Not stated anywhere but logically required | No source confirms, must be validated |
| **Research finding** | Demonstrated vulnerability or limitation | Peer-reviewed paper or reproducible PoC |

### Source register

For each source used, record:

```
| Source | URL | Retrieval Date | Claims Used |
|--------|-----|----------------|-------------|
| Protocol spec | <url> | <date> | Payment lifecycle, schemas |
| Vendor docs | <url> | <date> | Facilitator URL, pricing |
| Security paper | <url> | <date> | 5 identified attacks |
```

### Protocol lifecycle model

After collecting all sources, produce a canonical lifecycle that clearly separates what the protocol guarantees vs what must be assumed or verified independently. Use this structure:

```
Phase 0 — Discovery
Phase 1 — Initial Request
Phase 2 — Response / Challenge
Phase 3 — Authorization / Payment
Phase 4 — Verification
Phase 5 — Execution / Delivery
Phase 6 — Post-Delivery / Reconciliation
```

Label each field in the lifecycle as:
- `PROTOCOL FACT` — defined in spec
- `VENDOR BEHAVIOR` — specific to one implementation
- `ASSUMPTION` — not confirmed by any source
- `OPEN QUESTION` — needs future investigation

### Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Sole-source dependency** | Architecture relies on a single vendor's documentation as protocol truth | Cross-check against the open spec. Distinguish vendor-specific from protocol-level |
| **Security research recency** | Papers published after protocol release demonstrate vulnerabilities the spec doesn't address | Include active-disclosure status in threat model. Label mitigations as needed vs nice-to-have |
| **Marketing claims as facts** | Ecosystem statistics (volume, users) from vendor landing pages treated as architectural evidence | Note the source type. "Protocol processed 75M transactions" is adoption data, not protocol capability |
| **Spec version mismatch** | v1 and v2 specs differ in header format or transport | Specify which spec version each architectural claim is based on |
| **Over-assuming IETF standardization** | An Internet-Draft exists but has no standards-track status | Distinguish draft (-00, -01) from published RFC. Draft may expire without becoming a standard |
