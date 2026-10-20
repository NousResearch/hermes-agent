# Ledger — schema & build gates

## transactions.jsonl

One JSON object per line, append-only. Never rewrite posted history; corrections land as new rows or before a candidate is promoted.

- `id` — `YYYY-MM-DD.NN`, sequence within calendar date; unique across the file. Assign as max(existing suffix for that date) + 1 — suffix gaps from deleted rows are normal and IDs never get renumbered to be contiguous (a duplicate-ID collision from counting rows instead of scanning suffixes is what the gen_views uniqueness assertion guards).
- `date` ISO. `dir`: `in` | `out` | `transfer`. `kind`: `expense` | `transfer` | `salary` | `rental` | `oneoff` (irregular income — asset sales, fee income; counts toward income YTD, not monthly averages).
  Income math = `salary` + `rental` (+ `oneoff` where the context is income): the ledger has no `income` kind, so FCF/income queries must select kinds explicitly — a sum over all rows or a filter on a non-existent kind silently returns 0 or garbage.
- `amount` integer VND (parse away decimals/separators first); `currency`: `VND`.
- `desc`, `category`, `type` (Needs/Wants), `fv` (Fixed/Variable) — null on transfers.
- `via` (payer instrument) for expenses; `from`/`to` for transfers; `person`, `gross` (salary), `note` as needed.
- `status`: `candidate` → `posted` | `needs-confirm` | `aggregated`.
- `flags`: `unalerted-credit`, `own-name`, `mystery-in`, `chain-fail`, …
- `src`: provenance — `email:<KIND>` with `old_ref: mail:<thread8>` for pipeline rows; `user-confirm:<date>` for figures the user supplied directly (app screenshots from app-only banks that never email); `import:<source>` for imported rows.
- `old_ref` (migrated rows only): `ET:<YYYY-MM>:<row>` (imported Expense Tracker, monthly-restarted row numbers), `IT#N` (Bank Accounts internal transfers), `the user#L<N>` / `a household member#L<N>` (salary), `rent#N` (rental).

## balances.json

`as_of`; `accounts` map keyed `<BANK>-<number>`, `CASH-DRAWER`, or `CC-<last4>-<BANK>`, each with `balance`/`available`/`frozen` (or `limit`/`outstanding` for cards), `verified` note, and `flags` (`unalerted-credit-suspected`, `needs-count`, `identify-card`). `snapshot_totals` carries the derived rollup figures. Bank alert emails are authoritative for balances; books-implied credits stay flagged until the app confirms them.

## Gates (authoritative location: gen_views.py + closed_months.json)

- **Gate taxonomy:** CLOSED months are immutable assertions — gen_views.py fails if a frozen expense total drifts. LIVE periods (current month, YTD income) print as info only; asserting them against static numbers fails on every legitimate new transaction.
- **Automation trust:** the cron pipeline must run clean (zero chain discrepancies, correct tier for every row) before auto-post is considered trusted; the standing crons self-report gate failures to the configured notification channel, so no manual watch window is required once trusted.
- **Cadence:** weekly drawer-count + CC-statement reminders; monthly close freezes the month and snapshots ledger + review to `4-Archives/Finance/YYYY-MM/` (full-ledger copy as restore point — intentional, not a bug, when the snapshot contains all rows rather than only the month's).

Operational status lives in the crons + ops log (`3-Resources/Vault Structure.md`).
