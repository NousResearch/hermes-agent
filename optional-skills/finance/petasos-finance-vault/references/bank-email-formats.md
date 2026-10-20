# Bank email formats — ACB / HSBC / BIDV (Vietnam)

Six observed shapes, all bilingual VN/EN. Key fields below are the reliable parse targets; parse the Vietnamese block, cross-check the English block when present.

**Where the live regexes live:** `banks.yaml` — the vault-side registry (the skill's bundled copy is the example set). This file is the observed-shape REFERENCE: when a bank changes its email, fix that bank's entry in `banks.yaml`, verify with `python3 scripts/bank_adapters.py --probe <email.txt>` (per-field MATCH/MISS — a MISS line is the dead regex), and save the email under `_Ledger/fixtures/` with its expected values so `--test` catches regressions. A regex that never fires is invisible in `--dry` runs — the probe is the cheap catch.

| # | Subject (prefix) | Sender | Shape |
|---|---|---|---|
| 1 | `ACB-Dich vu bao so du tu dong` | mailalert@acb.com.vn | account no; `Số dư mới ... VND tính đến DD/MM/YYYY`; `Giao dịch mới nhất: Ghi nợ|Ghi có ±X VND`; `Nội dung giao dịch: <content>` |
| 2 | `Thông báo tài khoản: Biến động số dư` | hsbc.vietnam@notification.hsbc.com.vn | `Tài khoản 013****56700 ... biến động số dư ±X VND vào ngày YYYY-MM-DD`; `Số dư khả dụng hiện tại là X VND` |
| 3 | `[TB/Alert] ... Thẻ TD HSBC` | hsbc@notification.hsbc.com.hk | card `X1234`; `số tiền X USD|VND tại <merchant> vào ngày DD/MM/YYYY`; `Dư nợ hiện tại là X VND` (post-charge outstanding) |
| 4 | `Biên lai chuyển tiền ngoài BIDV` | bidvsmartbanking@bidv.com.vn | full table: Transaction time / Reference / Debit account / amount / Beneficiary name+account+bank |
| 5 | `BIDV - THÔNG BÁO GIAO DỊCH THẺ THÀNH CÔNG` | bidvcare@bidv.com.vn | card `4519xx1234`; Original amount; Time; `At:` merchant; approval code |
| 6 | `Biên lai thanh toán thẻ tín dụng` | bidvsmartbanking@bidv.com.vn | CC payment: Credit card number (may be UNTRACKED card); amount; Debit account — always a bill PAYMENT (checking → card), never a purchase |

## Parsing notes

- **Decimal separators**: shape 1 uses `1,234,567.00` (strip commas, drop `.00`); shape 2 uses `1.234.567` (strip dots); shapes 4–6 use `1,234,567` plain. Declare the bank's `decimal_separator` in `banks.yaml` — the engine strips that tail first, then deletes remaining separators; the wrong choice differs ×100.
- **HTML tables (shapes 4–6)**: tag-stripping yields SPACES, never `|` — keep pipe separators OPTIONAL in field regexes (`Label: ?\|? ?(...)`) and terminate lazy captures with a lookahead (`(?= Next label|\||$)`), or the regex never fires and the email is silently dropped. Years may be 2 OR 4 digits (`\d{2,4}`).
- **Direction**: shape 1 states it (`Ghi nợ`/`Ghi có` — marker vocabulary in `banks.yaml`); shape 2 encodes it in the sign (`+45.678.901` — `sign_of` rule); keep signs in the regex.
- **Balance semantics**: shape 1's stated balance is POST-transaction; shape 2's `Số dư khả dụng` is post-txn available. Use them as chain anchors.
- **Batched alerts**: ACB sometimes emits two alerts sharing one balance (offsetting ±deltas) — treat as one batch, not an unalerted txn.
- **Counterparty blindness**: shape 2 never names the counterparty — an OUT to an unknown destination must be flagged, not classified. Shape 4's beneficiary account identifies the destination (own account number = internal top-up).
- **Foreign currency**: shape 3 may charge USD; if the email also states a VND figure use it, else convert (observe your own rate) and note it.
- **Cards in receipts ≠ tracked cards**: always match the card number against Credit Cards before writing a payment row.
- **Bill payment ≠ purchase**: shape 6 money moves checking → card; log exactly ONE transfer row. The spending it settles already lives in (or arrives with) the card statement — adding an expense row too double-counts the month.
- **Counterparty names set transfer direction**: narration sender names identify which account SENT an internal transfer — set `from` to the named account instead of the assumed one; balances are unaffected, but the audit trail is.
- **Silent sources**: some banks send NO notification emails — income or debits there never arrive via the email pipeline and must be logged from user confirmation each cycle. Any bank outside the six shapes above is potentially silent: ask the user instead of assuming no transaction occurred.
- **App screenshots as evidence**: transaction-history screenshots from app-only banks close books-gaps emails never will. Narrations there are bank-internal codes — decode the readable purpose (rent, CC bill, insurance premium) and keep the raw string in the row as evidence; extract the text with whatever OCR or vision step your toolchain offers.
- **Recon before deletion**: when deleting bank emails after verification, re-parse ALL emails with the PRODUCTION registry (`banks.yaml` via the engine), not ad-hoc recon patterns — ad-hoc regexes miss HTML-table receipts and truncate decimals, producing false unmatched alarms. Match on ISO date + exact amount (fee-included variants ±100); a USD-charge email is matched by its converted-VND ledger row. Delete per thread via the mail API; state.json's seen-list already prevents re-parsing, and a list call returning `{"count":0}` confirms the inbox is empty — an empty inbox still emits one JSON line, so `wc -l` reading 1 does NOT mean mail remains. Deleting emails is safe for flagged items that never had email evidence — their evidence lives in rules.md flags and app screenshots, not the inbox.
