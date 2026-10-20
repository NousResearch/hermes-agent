# 📏 Rules — transaction classification (human-editable)

## 1. Auto-post patterns (below threshold + description match)

| Pattern (regex, matches merchant/desc) | Category | Type | FV | Notes |
|---|---|---|---|---|
| `IB LUNCH` | Food | Needs | Variable | workplace lunch |
| `COFFEE|HIGHLANDS` | Personal | Wants | Variable | coffee |
| `PETROL|XANG` | Transport | Needs | Variable | motorbike fuel |
| `GRAB|BE-` | Transport | Needs | Variable | ride-hailing |

## 2. Aggregate rules (no per-item logging)
- Credit-card statements → 1 expense row per cycle at rollup.
- Cash drawer → true-up row on physical count only.

## 3a. Known-merchant routing (≥ threshold still auto if matched)

| Pattern | Category | Type | FV |
|---|---|---|---|
| `ELECTRICITY|EVN` | Utilities | Needs | Variable |
| `TUITION` | Education | Needs | Variable |

## 3. Needs-confirm (never auto — ask the user)
- Any unknown ≥ threshold
- Own-name transfers with no precedent
- Mystery credits (IN)

## 4. Plumbing (reconciliation only, never income/expense)
- Internal transfers between own accounts, ATM withdrawals to cash, CC-bill payments.

## 5. Kind routing (data-driven tier defaults)

Tier vocabulary (parsed `kind` → default tier; consumed by the classifier —
no bank kind is ever hardcoded in Python):

- `expense` — always auto-post as an expense (category from the row)
- `expense-under-thresh` — expense only below the auto-post threshold
- `balance-alert` — direction-driven: OUT below threshold → expense; own-name → transfer ask; IN ≥ threshold → mystery-credit ask; IN below → oneoff
- `plumbing` — transfer, posted, reconcile-only, never an expense
- `transfer-receipt` — transfer needs-confirm (counterparty blind)

| Kind | Tier default | Category | Type | FV |
|---|---|---|---|---|
| `ACB` | balance-alert | Personal | Needs | Variable |
| `BIDV_CARD` | expense-under-thresh | Personal | Wants | Variable |
| `BIDV_CCPAY` | plumbing |  |  |  |
| `BIDV_XFER` | transfer-receipt |  |  |  |
| `HSBC_CC` | expense | Personal | Wants | Variable |
| `HSBC_ACC` | balance-alert | Personal | Needs | Variable |

## 6. Confirmed by user (never ask again)

| Date | Item | Meaning |
|---|---|---|

## 7. Pending confirmation (never auto-classify)

| Date | Item | Open question |
|---|---|---|
