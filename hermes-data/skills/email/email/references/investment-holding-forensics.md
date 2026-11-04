# Investment Holding Forensics — Reconstruct a Personal Equity Position from Email + Drive

Use when the user asks "how many <company> shares do I have / what did we pay / what's my expected gain" — for listed, pre-IPO, or unlisted holdings. Exact quantity, cost basis, and acquisition date are recoverable from registrar + depository + deal-thread emails WITHOUT any broker portal login.

## Sources of truth (priority order)

1. **RTA/registrar dividend intimation** — the single most authoritative share-count document. MUFG Intime India (domain `in.mpms.mufg.com`) sends credit intimations (e.g. `nseil.dividend@…`) containing a full table: No. of Equity Shares, Dividend per share, Gross amount, TDS, Net paid, masked bank account. Count is as of the record date.
   - Gmail search: `from:<rta-domain> <company>` or `<company> dividend`.
2. **ISIN freeze / lock-in notices** (e.g. `donotreplynseipo@…`) — prove PRE-IPO status, lock-in tenure (SEBI ICDR Reg 17(1): entire pre-offer capital locked 6 months from allotment), DP-ID/Client ID (masked, matches the demat). They do NOT state share count.
3. **CDSL demat transaction alerts** (`services@cdslindia.co.in`, subject "Transactions In Your Demat Account") — exact Quantity + Debit/Credit + datetime per ISIN. Show the acquisition credit AND corporate actions (subdivision = debit old ISIN + credit new ISIN, same qty). Search: `from:services@cdslindia.co.in "COMPANY FULL NAME"`.
4. **NSDL e-voting / postal-ballot notices** (`evoting@nsdl.com`) — prove shareholding + full DP-Client ID (e.g. `1208160093908773`). PDF enclosure is the ballot notice, not a holding statement.
5. **Broker weekly account statements** (Zerodha `no-reply-account-statement@reportsmailer.zerodha.net`) — the securities PDF covers settled listed-trade movements and may NOT list unlisted/pre-IPO holdings in the NSDL demat. Not reliable for the count; use 1–4.
6. **Deal-thread emails** — the acquisition story: platforms/middlemen (e.g. Infinyte Club `deals@infinyte.club`, Beyond IRR `@beyondirr.tech`, Flameback Capital `kishan@flamebackcapital.com`), flow LOI → funding → SPA → DP letters → demat credit. The thread SUBJECT line quantity ("NSE Sale of 450 Shares") is frequently the PRE-corporate-action figure — do not trust it.

## Gmail search recipe (fast path)

- `"<TICKER>" shares OR holding OR statement` — broad
- `from:services@cdslindia.co.in "<COMPANY FULL NAME>"` — demat credits
- `"<tail or full DP-Client ID>"` — every mail touching their demat
- `"<ISIN>"` — all mentions across mails
- middlemen: `from:deals@infinyte.club` / `@beyondirr.tech` / `@flamebackcapital.com` / `intensivefiscal.com`
- `subject:(<TICKER>) older_than:1y` — mostly trade alerts/newsletters; skip if noisy

## Reading registrar emails

HTML-only with an EMPTY text/plain part — `gws_gmail_get` often returns `body: ""`. Fetch via Gmail API `format='full'` and decode the `text/html` part (strip tags, `<br>`→newline, unescape). See pitfall #8 in SKILL.md.

## SPA extraction (acquisition price)

- Download the signed **Share Purchase Agreement PDF** (final version) and `pdftotext -layout`.
- Schedule 1 money lines: `Number of Sale Shares`, `Sale Price per Sale Share`, `Purchase Consideration Amount`.
- **PITFALL — preliminary vs final SPA:** initial confirmation/LOI can state the PRE-subdivision quantity (e.g. 450 @ ₹5,650) while the final SPA is proportionately re-stated post-split (e.g. 2,250 @ ₹1,130) at the SAME total consideration (₹25,42,500). Never quote quantity/price from the thread subject or first confirmation — read the FINAL SPA Schedule 1 and cross-check against dividend intimation + demat credit.
- Statutory charges (stamp duty, DP fees, TCS) sit over the consideration; all-in cost = consideration + charges (take the total from the deal confirmation mail).

## Expected-gain math

- Listing expectation from GMP sources (ET/Moneycontrol): estimated listing price ≈ upper band + GMP ₹/share. Always state the unregulated-GMP caveat.
- Value = shares × expected listing price. Gain = value − all-in cost. Give both a GMP-based and an issue-price-based figure.
- Dividends already banked (net of TDS) are realized return on the position — mention them.

## Constraints to flag

- Pre-IPO shares: locked 6 months from IPO allotment (Reg 17(1) ICDR), ISIN frozen pre-listing (NSDL circular Aug 2023 / CDSL equivalent). Gain is PAPER until lock-in expiry unless an ICDR exemption applies.