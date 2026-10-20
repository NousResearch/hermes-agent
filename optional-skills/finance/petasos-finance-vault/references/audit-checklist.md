# Vault audit — "any finance data to update or correct?"

Every step is a read (ad-hoc-safe); only step 9 mutates. Ledger =
`transactions.jsonl`.

1. **Freshness**: `pipeline.py --dry` → expect `0 new transactions`, else
   sync first; cross-check `state.json` `last_run`.
2. **Open asks**: grep the LEDGER for `status=needs-confirm` — never trust
   the dashboard's pending section (a shadowing `gen_views.py` may build it
   from rules.md §7 prose and show 0 while asks exist).
3. **Metadata gaps**: OUT rows with a category but missing `type`/`fv` →
   relabel from the nearest sibling row with the same/like desc — rollup
   statement rows recur month-over-month with exact twins.
4. **IN rows without category**: `salary`/`rental`/`oneoff` carry none by
   design — not gaps.
5. **Duplicates**: same (date, amount, desc) OUT rows are usually genuine
   repeat small purchases (a second coffee) — flag only when re-send
   signals are present.
6. **Closes**: `closed_months.json` covers every fully-past month; the
   current month stays live.
7. **Per account `available` vs `balance`**: `<` = pending hold, `>` =
   in-flight credit — both balances-only, no ledger row; a `>` gap that
   persists across two syncs → have the owner check the bank app.
8. **Hardcoded constants in the vault's `gen_views.py`**: the ASSETS/DEBTS
dicts (mortgage balance, deposits total, apartment value, DCDS) have NO
covering gate — the arithmetic gate only checks internal consistency, so a
stale constant passes green forever. Cross-check each constant against the
reference notes (Debts Detail payment-history chain, Assets Detail tables);
loan balances must equal the note's latest `balance after` row. Fix via a
scratch script asserting a unique source anchor, then re-run gates and
confirm the NW moved by exactly the delta.
9. **Header hygiene**: `balances.json` `as_of` can lag account updates —
refresh it during a review. Reference notes (Assets/Debts Detail) that say
"live figures in balances.json" may legitimately lag — only the numbers a
note asserts as current must match.
10. **Drawer**: items in the `verified` note since the last true-up must
    reconcile with the balance (count baseline − post-count items), and the
    entry style must match the note's era.
11. **Fixes** go through the relabel CLI / scratch drawer scripts; after
    every fix run the VAULT's `gen_views.py` and require `ALL GATES PASS`.

A "review and cascade all finance data" request = this whole checklist plus
regenerating the views: full vault `gen_views.py` run (gates + fresh
Dashboards/Monthly Review/Salary History), `pipeline.py --dry`, non-posted
grep, and the constant cross-checks — report verified-green items, fixes
applied, and what is stale by design separately.
