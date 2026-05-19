# BTC Vol Desk Packet — Adversarial Audit Closure

**Audit timestamp:** 2026-05-18 15:08:26 PDT  
**Scope:** BTC Vol Desk investor/internal evidence packet, static site, PDF handoff packet, packet verifier, and source/quote/legal readiness gates.  
**Status:** `PASS FOR CURRENT INTERNAL SCREEN-ONLY PACKET`  
**Evidence status:** `SCREEN-ONLY · NOT EXECUTABLE`  

## Executive Result

No P0 findings remain. The two P1 issues found by the independent adversarial audit were patched, regenerated, and re-verified:

1. **External-use gate source-readiness bypass** — fixed. `external_use_gate` now requires packet `site_data.source_intake_validation.ready == true` before it can pass.
2. **Source-coverage wording overclaim** — fixed. Regenerated packet/site/PDF artifacts no longer use the old `Coverage completeness: Deribit + IBIT + CME Databento available` framing. They now separate `Current screen-source availability` from licensed/replay-ready source readiness.

The packet remains intentionally blocked for external use.

## Independent Audit Findings

### Initial independent audit

Result: two P1 findings, no P0 findings.

- P1-1: `external_use_gate` could be made to pass without licensed/replay-ready source readiness.
- P1-2: rendered investor artifacts used source-coverage language that could be read as readiness/completeness.

### Post-fix independent re-audit

Result: **pass**, no remaining P0/P1 findings.

The re-audit verified:

- Targeted packet verifier tests passed.
- Live `artifacts/institutional/investor-packet` verification reports `external_use_gate.ok = false` with blocker `licensed/replay-ready source intake validation not ready`.
- Old source overclaim phrase hits: `0` in regenerated packet/site/PDF/ZIP scans.
- New source label hits present: `Current screen-source availability`.
- Licensed readiness distinction remains visible in the packet.

## Patches Applied

### Packet verifier hardening

File: `institutional_btc_vol/packet_verify.py`

Added `site_data` source-intake validation into the external-use gate:

- Parses packet artifact `site/site-data.json`.
- Reads `source_intake_validation.ready`.
- Reports source readiness summary such as `0/6 licensed source groups covered`.
- Blocks external use when:
  - site data is missing,
  - source intake validation is missing,
  - source intake validation is not ready,
  - source readiness data is invalid.

Current verified external-use blockers:

- `packet publishability is not external-use-approved`
- `counsel-approved legal wrapper missing`
- `two-counterparty quote-verified evidence missing`
- `licensed/replay-ready source intake validation not ready`

### Source/readiness wording hardening

Files:

- `institutional_btc_vol/site_data.py`
- `institutional_btc_vol/investor_site.py`
- `institutional_btc_vol/investor_memo.py`
- `institutional_btc_vol/investor_tearsheet.py`

Changed display language from generic “coverage completeness” to:

- Label: `Current screen-source availability`
- Value when CME rows exist: `Current screen/vendor captures available: Deribit + IBIT + CME Databento`
- Value when CME rows are absent: `Current screen/vendor captures partial: CME unavailable`

This keeps current captured source availability separate from licensed historical/replay-ready readiness.

### PDF builder portability

File: `scripts/build_btc_vol_pdf_packet.py`

Removed hardcoded local checkout path and replaced it with:

```python
ROOT = Path(__file__).resolve().parents[1]
```

Regression test added in `tests/test_btc_vol_pdf_packet_builder.py`.

## Verification Proof

### Tests

Command:

```bash
scripts/run_tests.sh tests/test_btc_vol*.py
```

Result:

```text
145 passed in 1.31s
```

### Site/packet/PDF rebuild

Commands:

```bash
python -m institutional_btc_vol.cli build-site artifacts/institutional/data artifacts/institutional/site/index.html
python -m institutional_btc_vol.cli build-site artifacts/institutional/data artifacts/institutional/railway-site/public/index.html
python -m institutional_btc_vol.cli build-packet artifacts/institutional/data artifacts/institutional/investor-packet
python -m institutional_btc_vol.cli verify-packet artifacts/institutional/investor-packet
python scripts/build_btc_vol_pdf_packet.py
```

Results:

- Site build: `ok: true`
- Railway-site static copy build: `ok: true`
- Packet build: `ok: true`
- Packet verify: `ok: true`
- PDF build: `ok: true`
- PDF count: `29`
- Missing PDFs: `[]`

### Packet verifier result

Current packet verifier state:

```json
{
  "ok": true,
  "publishability": "internal-diligence-only",
  "external_use_gate": {
    "ok": false,
    "blockers": [
      "packet publishability is not external-use-approved",
      "counsel-approved legal wrapper missing",
      "two-counterparty quote-verified evidence missing",
      "licensed/replay-ready source intake validation not ready"
    ],
    "quote_verified_candidates": 0,
    "source_readiness": {
      "present": true,
      "ready": false,
      "summary": "0/6 licensed source groups covered",
      "errors": []
    }
  },
  "secret_scan": {"matches": []},
  "execution_cta_scan": {"matches": []},
  "control_language": {"ok": true, "missing": []},
  "errors": [],
  "warnings": []
}
```

### Regenerated artifact scans

- Old source overclaim phrase hits: `[]`
- Latest coverage label: `Current screen-source availability`
- Latest coverage value: `Current screen/vendor captures available: Deribit + IBIT + CME Databento`
- Source readiness: `False`, `0 / 6`
- ZIP path safety: no absolute members; no traversal members.
- ZIP `testzip`: `None`

### Combined PDF verification

Combined PDFs verified readable and containing required control language:

- `BTC_Vol_Desk_INVESTOR_DILIGENCE_PACKET.pdf` — 44 pages
- `BTC_Vol_Desk_INTERNAL_TEAM_PACKET.pdf` — 188 pages
- `BTC_Vol_Desk_FULL_PACKET.pdf` — 230 pages

Required markers found:

- `SCREEN-ONLY`
- `NOT EXECUTABLE`
- `NOT A QUOTE`

### Browser QA

Opened packet site:

`file:///Users/assistant/.hermes/hermes-agent/artifacts/institutional/investor-packet/site/index.html`

Browser console:

- JS errors: `0`
- Console messages: `0`

Visual QA screenshot:

`/Users/assistant/.hermes/cache/screenshots/browser_screenshot_6afa34c37cbd42a0a9fb70b6a6f481d6.png`

Visual result:

- `SCREEN-ONLY · NOT EXECUTABLE` labeling visible.
- Current source availability vs licensed readiness distinction visible/readable.
- No major overlaps or layout breaks detected.
- Minor caveat: page is dense; small diagnostics/hashes may require zooming.

## Integrity Handles

- PDF handoff ZIP: `artifacts/institutional/btc-vol-desk-pdf-handoff-packet.zip`
- ZIP SHA-256: `b2f0777f88d170e318144acc5a8157fc77407dcf3f782a7a617aa3832a876268`
- ZIP bytes: `905,979`
- Packet manifest path: `artifacts/institutional/investor-packet/packet_manifest.json`
- Packet manifest file SHA-256: `b1ed28cf015d94ccc324094287f333de22adfc5651d3955068d717b044068f77`
- Packet manifest internal packet SHA-256: `160be812149a84b234cff8a61db5d4660216a3bb84153aa5c4b9adf683558e3a`

## Remaining Blockers / Gates

These are intentional governance blockers, not engineering failures:

1. **External use blocked** — publishability remains `internal-diligence-only`.
2. **Legal blocked** — counsel-approved wrapper is missing; legal wrapper remains `draft-blocked`.
3. **Quote verification blocked** — zero quote-verified candidates; two-counterparty quote evidence required.
4. **Licensed source readiness blocked** — source intake validation is not ready: `0/6 licensed source groups covered`.
5. **No executable workflow** — no RFQ submission, trade execution, account opening, or client portal behavior exists.

## Final Audit Ledger

- P0 findings: `0`
- P1 findings before patch: `2`
- P1 findings after patch/re-audit: `0`
- Packet verifier: `PASS` for internal packet integrity
- External-use gate: `BLOCKED`, as intended
- PDF handoff: regenerated and verified
- Browser QA: pass with density caveat
