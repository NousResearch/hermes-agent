---
name: petasos-finance-vault
description: Run an Obsidian household-finance ledger vault with gates.
version: 1.0.0
author: Nhut Duong (nhutdm), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Obsidian, Finance, Ledger, Household]
    related_skills: [obsidian]
    config:
      - key: petasos.vault_path
        description: Absolute path to the Obsidian vault holding the finance area
        default: ""
        prompt: Obsidian vault path (e.g. ~/Obsidian/MyVault)
    blueprint:
      schedule: "0 7 * * *"
      deliver: origin
      prompt: "Petasos daily sync: fetch bank alert emails, parse, chain-verify, classify per rules.md, append ledger rows, update balances, send the needs-confirm digest, regenerate views."
required_environment_variables:
  - name: AGENTMAIL_API_KEY
    prompt: AgentMail API key
    help: Get one at https://agentmail.to
    required_for: bank-alert email fetch in the daily pipeline
  - name: PETASOS_INBOX
    prompt: AgentMail inbox address receiving bank alerts
    help: The mailbox your banks send transaction alerts to
    required_for: daily pipeline email fetch
---

# Petasos Finance Vault

A household-finance vault for Obsidian built on a ledger architecture: an
append-only JSONL ledger as the single source of truth, a human-editable
classification rules file, generated dashboard views, and arithmetic
verification gates that must pass after every write. Bank alert emails are the
primary transaction evidence source.

The engine is split into a generic Python package (`petasos/`) plus thin
CLI wrappers (`scripts/`), with ALL bank/merchant intelligence in data you
own
(`banks.yaml`, `rules.md`) inside your vault. Manual ledger edits go through
a validated CLI (`python3 -m petasos.ledger`) — hand-editing the JSONL is
disallowed. Live runs are serialized by a lockfile, months freeze at close,
and every write re-runs the full gate set.

## When to Use

- Any task touching the vault named by `petasos.vault_path`: ledger appends,
  balance updates, classification rules, view regeneration, month close.
- Syncing transactions from bank notification emails into the ledger.
- Logging chat-reported items (cash spending, app-screenshot transactions,
  user corrections) or correcting/confirming existing rows.
- Reconciling, auditing accounts, or answering balance/spend questions from
  the ledger.

Don't use for: generic Obsidian mechanics (use the `obsidian` skill);
budgeting philosophy questions that never touch the ledger.

## Requirements & Install

- Python 3.10+ (stdlib) and PyYAML — the ONLY dependency:
  `pip install pyyaml`. A missing PyYAML fails LOUD by design with the
  install hint and exit code 2 (INV-2); nothing degrades silently.
- Env vars `AGENTMAIL_API_KEY` + `PETASOS_INBOX` for email fetch; vault path
  via `hermes config set skills.config.petasos.vault_path <abs path>` or
  `export PETASOS_VAULT=<abs path>`.

## Architecture — the engine/data split

The engine is generic and is NEVER user-edited:

- `petasos/` — ledger, chain checks, classifier, gates, views, close,
  scaffolder (all bank-agnostic logic).
- `scripts/` — thin CLI wrappers over the package.

All intelligence lives in DATA inside your vault, resolved through a
shadowing chain (first existing file wins):

| Data | Chain (first hit wins) | Loud when |
|---|---|---|
| Bank formats | `$PETASOS_BANKS` → `<ledger_dir>/banks.yaml` → bundled `banks.yaml` | env var names a missing file |
| Classification | `$PETASOS_RULES` → `<ledger_dir>/rules.md` → bundled `templates/rules.md` | env var names a missing file |

An env var that names a nonexistent path is a LOUD error listing every
searched path (exit 2) — never a silent fallback to examples. Run
`python3 scripts/bank_adapters.py --banks` to see which registry is active.

`rules.md` is the classification brain: auto-post patterns (§1), aggregate
rules (§2), known-merchant routing (§3a), needs-confirm rules (§3), plumbing
(§4), the §5 kind-routing table mapping each parsed bank-alert kind to a tier
default (`expense` / `expense-under-thresh` / `balance-alert` / `plumbing` /
`transfer-receipt`), and §6 confirmed-by-user items that are never re-asked.
No bank or merchant name is ever hardcoded in Python.

Privacy pillar: this repo is a public generic engine. Real account numbers,
names, IPs, card fragments, or any private deployment detail never belong in
it — bank entries in the bundled `banks.yaml` are worked examples only.

## Vault Structure

```
<Vault>/2-Areas/Finance/_Ledger/
  transactions.jsonl   # append-only; one JSON object per line (edits: DG-16 CLI only)
  balances.json        # accounts + DERIVED snapshot_totals (totals recomputed by gen_views)
  closed_months.json   # frozen month EXPENSE totals — IMMUTABLE once written
  state.json           # pipeline seen-threads (written by the pipeline only)
  rules.md             # classification brain — human-editable data
  banks.yaml           # bank adapter registry — human-editable data
  fixtures/            # captured emails + expected.yaml — adapter regression corpus
  gen_views.py         # if present, the vault's own copy shadows the bundled one
  petasos.lock         # run lock — exists only while a live pipeline run writes
2-Areas/Finance/       # generated views (Dashboards.md, Salary History.md) + reference notes
4-Archives/Finance/YYYY-MM/   # closed-month archive: full-ledger snapshot + Monthly Review
```

Views (`Dashboards.md`, `Salary History.md`) are GENERATED — never hand-edit
numbers there; edit ledger/rules and regenerate.

## Daily Operations — canonical commands

Export `PETASOS_VAULT`, then frame every invocation through the `terminal`
tool using the `${HERMES_SKILL_DIR}` token — Hermes substitutes it with the
installed skill directory's absolute path at load, so no path math is ever
needed. The `scripts/*` wrappers run from anywhere; the
`python3 -m petasos.ledger` and `python3 -m unittest` forms additionally
need `${HERMES_SKILL_DIR}` as the working directory:

```bash
export PETASOS_VAULT=<abs vault path>
python3 ${HERMES_SKILL_DIR}/scripts/pipeline.py --dry    # SAFE PREVIEW: writes NOTHING, marks nothing seen
python3 ${HERMES_SKILL_DIR}/scripts/pipeline.py          # daily sync: fetch → parse → chain → classify → append → balances → gates → digest
cd ${HERMES_SKILL_DIR} && python3 -m petasos.ledger add --ledger <L> --date YYYY-MM-DD --dir out \
    --kind expense --amount 20000 --desc 'HIGHLANDS COFFEE TWO'   # chat-reported item
cd ${HERMES_SKILL_DIR} && python3 -m petasos.ledger relabel --ledger <L> --id 2026-09-27.01 \
    --set kind=expense --set status=posted --set category=Personal \
    --set src=user-confirm:2026-10-14                             # correction / confirmation
python3 ${HERMES_SKILL_DIR}/scripts/gen_views.py         # regenerate views + run ALL gates (exit 0 iff pass)
python3 ${HERMES_SKILL_DIR}/scripts/monthly_close.py 2026-09   # freeze + archive (default arg: previous month)
python3 ${HERMES_SKILL_DIR}/scripts/bank_adapters.py --banks   # which banks.yaml is active (+ overlap warnings)
python3 ${HERMES_SKILL_DIR}/scripts/bank_adapters.py --probe <email.txt>   # per-field MATCH/MISS report on one email
python3 ${HERMES_SKILL_DIR}/scripts/bank_adapters.py --test    # fixture corpus must stay green
python3 ${HERMES_SKILL_DIR}/scripts/init_vault.py /path/to/vault   # scaffold a new vault (idempotent, never overwrites)
cd ${HERMES_SKILL_DIR} && python3 -m unittest discover tests       # skill self-test when developing the engine
```

(`<L>` is `$PETASOS_VAULT/2-Areas/Finance/_Ledger/transactions.jsonl` unless
`PETASOS_LEDGER_DIR` says otherwise.)

Exit codes (checked, not aspirational):

| Command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `pipeline.py` | ok, or fetch skipped | chain break / gates failed | config error | another run holds the lock |
| `gen_views.py` | ALL GATES PASS | gate failure | config/input fatal | — |
| `monthly_close.py` | frozen or verified | refused (red gates, open needs-confirm, freeze drift) | usage/config fatal | — |
| `-m petasos.ledger` | appended / relabeled | — | validation error (file untouched) | — |
| `bank_adapters.py --probe` | some format matched | no format matched | config error | — |

## Procedure

### 1. Bank-email sync (daily)

1. `pipeline.py` resolves the registry and rules LOUDLY before any fetch;
   a bad `banks.yaml`/`rules.md` stops the run with exit 2 before mail is
   touched.
2. Fetches bank threads from the inbox (REST; sorted by timestamp, not list
   order; HTML stripped). If the inbox is not visible to the credential
   (401/403/404/unreachable) it prints a skip notice and exits 0 with the
   ledger, `state.json`, and `balances.json` all untouched — catch-up resumes
   automatically. Threads are asked once: seen thread ids, subject keywords,
   and existing `old_ref`s all dedupe (a confirmed ask is never re-fetched).
3. Parses each thread via `banks.yaml`. A bank-subject thread that fails to
   parse surfaces in the digest as UNPARSED with a `--probe` hint — never a
   fabricated row.
4. Chain-verifies BEFORE writing: `bal(n) = bal(n-1) ∓ amount(n)` per
   account. A batched alert (an offsetting ± pair sharing one balance) is
   one alert, not a break. Any real mismatch appends NOTHING and exits 1.
5. Classifies via rules.md; pending holds (`available < balance`) go to
   balances only, never a ledger row.
6. Live write under the run lock `petasos.lock` (created atomically; a
   concurrent run exits 3 loudly): append → re-read + row-count verify →
   balances sync → `state.json` → gates subprocess → digest file. A crash
   mid-write leaves the ledger authoritative (balances self-heal on the next
   run); a hard kill may leave a stale `petasos.lock` — inspect and remove
   it before the next run.
7. The gates subprocess (`gen_views.py`) must print `ALL GATES PASS`;
   nonzero is fatal and the pipeline exits 1.

`--dry` previews the whole plan — rows that would be appended, the digest
text, unparsed threads — and writes NOTHING: no ledger rows, no balances
change, nothing marked seen in `state.json`. It is always safe to run.

### 2. Manual ledger edits — the DG-16 CLI, never a hand-edit

Hand-editing `transactions.jsonl` is DISALLOWED. The chain gate and the
`YYYY-MM-DD.NN` id-allocation rule (max suffix + 1, gaps normal) make
hand-edits hazardous, and the CLI validates everything for you:

- **Chat-reported item** (exact items are logged immediately, never left for
  a cash true-up):

  ```bash
  python3 -m petasos.ledger add --ledger <L> \
      --date 2026-09-28 --dir out --kind expense --amount 20000 \
      --desc 'HIGHLANDS COFFEE TWO'            # [--category --type --fv --via]
  ```

  `--kind` vocabulary: `expense | transfer | salary | rental | oneoff`.
  `--amount` is a plain integer in the ledger currency. The row lands as ONE
  `posted` row with `src: user-confirm:<date>`. With no `--category`, a
  precedent row matching the same rules.md pattern supplies
  category/type/fv — a known pattern never gets a new category; with no
  precedent the category stays unset (relabel it after, or pass
  `--category`).

  Before appending a chat-reported item, re-read today's rows for the same
  (date, desc, amount, via) — a hit is a decision point, not an append:
  re-send vs. a second occurrence. Second-occurrence signals: the message
  follows the user's new-item construction ("Spent X via Y for Z"), an
  earlier session already confirmed the existing row to the user, and an
  exact-phrase session-history search finds no match for the new wording.
  When appending anyway, flag it in the reply ("second X today, row `<id>` —
  say the word to undo") so a mis-read is cheap to reverse.
- **Correction / confirmation (in-place relabel)**:

  ```bash
  python3 -m petasos.ledger relabel --ledger <L> --id 2026-09-27.01 \
      --set category=Personal --set type=Wants --set src=user-confirm:2026-10-14
  ```

  Open fields: `category`, `type`, `fv`, `desc`, `ask`. Also allowed:
  `status` — but only promoted to `posted`; `kind` — but only from a
  `needs-confirm` row (the confirmation path); `src` — but only
  `user-confirm:<YYYY-MM-DD>`. Evidence fields are refused outright:
  `id, date, dir, amount, currency, old_ref, bal_after` can NEVER be
  relabeled. Invalid input exits 2 and leaves the file untouched.

- **Cash item ⇒ drawer step in `balances.json`**: `add` writes only the
  ledger row, never balances. When the vault tracks cash as an account
  (e.g. `CASH-DRAWER`), a `--via Cash` append must in the SAME step
  decrement that account's `balance` (and `available`), append the item to
  its `verified` note in the note's CURRENT entry style, and decrement EVERY stored total that
  includes the cash component by the same amount — at minimum
  `snapshot_totals.cash_total` and `liquid_total`, and where the vault's
  `snapshot_totals` carry the component view also `assets_total` and
  `net_worth` — otherwise the arithmetic/DRIFT gates fail on the next gate
  run. Do the whole drawer step as a scratch script under the scratch dir
  with pre-write assertions on the expected prior balance and the
  stored-vs-derived identities (`liquid = bank + cash`; `assets = bank +
  cash + deposits + funds + property`; `net_worth = assets − debts`) —
  mutating `balances.json` inline without asserts is how totals drift
  silently. The `verified` note has ERAS: between counts entries may carry
  a running `= N implied` figure, but a true-up supersedes that chain and
  later entries are bare `-amount` rows against the count baseline — assert
  `balance == available` plus the totals identities (always true), and
  assert an implied-figure/balance match only when the note's LAST entry
  actually ends in one; after a true-up anchor on the true-up marker and
  the count baseline instead, or the step aborts on a superseded-era
  figure. Assert the identities that actually HOLD pre-write: a vault
  may carry a non-standard stored model (e.g. `net_worth` tracking
  `assets_total` rather than assets−debts), in which case assert the
  relative move (each cash-inclusive total decreases by exactly the
  amount) and preserve the vault's model — never "correct" stored
  totals to the textbook formula mid-drawer-step. User preference:
  scratch script files, never inline python. Multiple cash items reported
  in one message: append each row via the CLI, then ONE drawer step for the
  summed total (single prior-balance assert, decrement by the sum) and one
  gate run — not one script per item.

After any manual append, run the gate set and require `ALL GATES PASS`. If
the vault carries its own `<ledger_dir>/gen_views.py`, run THAT copy, not
`scripts/gen_views.py` from the skill root — the vault copy shadows the
bundled one, is the one the cron wrapper invokes, and may implement a
different (older/different-model) gate set. The bundled v2 gates can then
legitimately fail on a vault the shadowing copy passes (extra chain gate,
stored-vs-derived DRIFT on totals the shadow never writes); a bundled-only
red gate means compare the two gate sets first — do not "fix" a ledger the
production gates pass. (The manual CLI validates rows but does not itself
run any gate set.)

### 3. Classification tiers and confirmations

rules.md is authoritative. Ops-level tiers:

- **auto-post** — below threshold + known pattern (§1), or known merchant
  (§3a, posts regardless of threshold) → posted silently.
- **aggregate** — credit-card statements → 1 row/cycle at rollup; cash →
  true-up row on drawer count only (recipe: `references/drawer-trueup.md`).
- **needs-confirm** — unknown ≥ threshold, own-name transfers without
  precedent, mystery credits → asked once in the digest.
- **plumbing** — internal transfers, ATM withdrawals, CC-bill payments →
  reconcile-only transfers, NEVER expenses.

Bank-alert kinds route through the §5 kind-routing table (tier defaults per
parsed kind — e.g. a balance-alert kind follows the direction-driven ladder:
OUT below threshold → expense; own-name → transfer ask; IN ≥ threshold →
mystery-credit ask holding kind `oneoff` until confirmed).

Needs-confirm lifecycle: the digest asks ONCE (the thread is marked seen),
the dashboard carries an "❓ Open confirmations" section until the row is
cleared, and clearing happens via relabel (`--set status=posted --set
kind=... --set category=... --set src=user-confirm:<date>`). Caveat: a
vault's own shadowing `gen_views.py` may build that dashboard section from
a manual list (e.g. rules.md §7 prose) instead of scanning the ledger for
`needs-confirm` rows — the dashboard can then show "0 items" while open
asks exist; grep `transactions.jsonl` for `status=needs-confirm` as the
source of truth, not the dashboard. When the owner
is live in chat (e.g. right after a requested sync), run the ask INLINE
instead of leaving it for the digest: grep the ledger for the merchant's
prior rows, present ONE question with the precedent category as the
recommended option, then relabel + add the §6 row + `pipeline.py --dry` in
the same turn. To make a confirmed item permanent, add it to rules.md §6 —
the classifier checks §6 first, so it is never re-asked. Scope the §6 entry
to what was actually confirmed: a merchant whose purpose varies across
history (Sept rows Shopping, today's row breakfast) gets an
amount-and-purpose-scoped entry, not a merchant-wide pattern — otherwise
the next different-purpose charge silently books under today's category.

After ANY rules.md edit, run `pipeline.py --dry` — the rules file feeds the
live classifier, and a malformed table fails parsing loudly (section + line
named) rather than silently.

### 4. Month close

```bash
python3 scripts/monthly_close.py 2026-09      # or no arg = previous month
```

1. Preconditions, enforced loudly (exit 1): ALL gates green AND zero
   open needs-confirm rows in the closing month.
2. On success it freezes the month's EXPENSE total into `closed_months.json`
   (immutable) and archives into `4-Archives/Finance/YYYY-MM/`: a
   full-ledger snapshot (intentional restore point) plus a generated
   Monthly Review. Re-running is safe: already-frozen months are verified,
   not rewritten.
3. The freeze covers expenses only (DG-10): late INCOME may append into a
   closed month and the frozen sum still re-checks green. An EXPENSE
   appended into a closed month makes the frozen-month gate FAIL loudly on
   the next `gen_views.py`/pipeline run (the pipeline exits 1 after the
   append). Fix the ledger — the archive snapshot is the restore point —
   never the freeze.
4. Close on the 2nd of the month, after day-1 stragglers (card alerts, T+1
   fund NAVs) land.

### 5. Views and the verification gates

`gen_views.py` runs the gate set, then (regardless of verdict, v1 parity)
regenerates `Dashboards.md` + `Salary History.md` and persists the DERIVED
`snapshot_totals` into `balances.json`. Exit 0 only when every gate passes.
The gates:

- ID uniqueness across the whole ledger.
- Every closed month: ledger expense sum == frozen total.
- Balances arithmetic: assets = bank + cash + deposits + funds + property;
  debts = mortgage + loans + CC outstanding; net worth = assets − debts;
  liquid = bank + cash. Stored totals disagreeing with derived values are
  DRIFT and fail the gate.
- Chain: `bal_after(n) = bal_after(n−1) ∓ amount(n)` over consecutive
  bal_after-bearing rows (an offsetting ± pair sharing one balance is one
  batched alert, not a break). Every bal_after-bearing row chains as ONE
  account — the frozen row schema has no account field (see the assumption
  note in `banks.yaml` before adding a second bal_after-capturing bank).
- The LIVE month is info-only: a new transaction never fails a gate.

`snapshot_totals` are DERIVED — `gen_views.py` recomputes them from the
account components; never hand-maintain a total while also editing its
component. Balances questions are answered from `balances.json`/views,
never from memory; alert emails are authoritative for account balances, and
`available < balance` on a checking account is a pending hold — recorded to
balances, no ledger row. The mirror also holds: `available > balance` is an
in-flight (authorized, unposted) credit — balances only, no ledger row; it
self-resolves when the posting alert lands on a later sync, and a gap
persisting across two syncs is an ask-the-owner-to-check-the-app signal.

### 6. Adding or fixing a bank (data only, no code)

1. Edit its `banks.yaml` entry in YOUR vault (copy a worked example first).
2. `python3 scripts/bank_adapters.py --probe <email.txt>` — paste a real
   email (raw HTML fine; `Subject:` as the first line); get a per-field
   MATCH/MISS report. A MISS line IS the dead regex — a regex that never
   fires is invisible in `--dry` runs, and this is the cheap catch.
3. Save that email under `fixtures/` beside your registry, add its expected
   values to `fixtures/expected.yaml` (cross-check figures against your
   bank's app), and keep `python3 scripts/bank_adapters.py --test` green.

The per-bank `decimal_separator` is chosen BEFORE thousands separators are
stripped — the wrong choice is a ×100 bug. `Ghi no` = debit (OUT),
`Ghi co` = credit (IN); the amount handling tolerates both directions.

## Adapting to Your Setup

Everything user-specific is env or config — no code edits:

| Knob | Default | Controls |
|---|---|---|
| `skills.config.petasos.vault_path` / `PETASOS_VAULT` (or `OBSIDIAN_VAULT_PATH`) | — | vault root (required) |
| `PETASOS_LEDGER_DIR` | `2-Areas/Finance/_Ledger` | ledger location inside the vault |
| `PETASOS_CURRENCY` | `VND` | ledger row currency (amounts are plain integers) |
| `PETASOS_THRESHOLD` | rules.md `threshold_auto_post`, else `500000` | auto-post ceiling (env wins) |
| `PETASOS_ACCT_NUM` + `PETASOS_ACCT_KEY` | — | primary alert account digits + its `balances.json` key (auto balance sync) |
| `PETASOS_OWN_NAMES` | — | comma-separated own-name matchers (own-name transfers → needs-confirm, never expenses) |
| `PETASOS_BANKS` / `PETASOS_RULES` | — | explicit registry/rules path (shadows the vault copy; a missing path is a LOUD error) |
| `PETASOS_API_BASE` | `https://api.agentmail.to/v0` | mailbox REST endpoint |
| `PETASOS_INBOX` + `AGENTMAIL_API_KEY` | — | the alert mailbox and its credential |

Mailbox REST operations beyond fetch (purge-all, snapshot-first) →
`references/mailbox-ops.md`; the periodic "anything to update or correct?"
vault sweep → `references/audit-checklist.md`.

## Automation

The blueprint above registers a suggested daily-sync cron (accept via
`/suggestions`). Recommended companions, created explicitly with
`cronjob`/`hermes cron create --skill petasos-finance-vault`:

- **Daily 07:00** — email sync + digest + views (the blueprint).
- **Monthly, 2nd 07:30** — month close, AFTER stragglers land on day 1.

Stagger any other writers on the same vault by ≥5 minutes: a live pipeline
run holds `petasos.lock` and a concurrent run exits 3 rather than overlapping.

## Pitfalls

- Hand-editing `transactions.jsonl` is disallowed — `add`/`relabel` via
  `python3 -m petasos.ledger` are the only sanctioned mutations. The CLI
  validates the vocabulary and id uniqueness before touching disk and
  re-reads the file after appending.
- `--dry` writes nothing and marks nothing seen; it is always safe. But it
  will not warn about a regex that never fires — `--probe` is the only
  cheap catch for a dead regex.
- A live run that dies mid-write may leave `petasos.lock` behind (the normal
  error path removes it); inspect and remove it, or every later run exits 3.
- Appending an expense into a CLOSED month poisons the frozen-month gate —
  the failure appears on the next gate run, after the append. Fix the
  ledger from the archive snapshot; never edit `closed_months.json`.
- Vietnamese bank formats differ BY BANK (comma- vs dot-thousands): pick the
  `decimal_separator` from a captured body, pin it with a fixture.
- Same-day rows sit in narrated-time order (the `.NN` id suffix preserves
  it); the account's final balance is the `bal_after` of the
  latest-timestamped row, not the last line.
- Same (date, amount) rows are a review signal, not proof of duplication —
  run the re-send vs. second-occurrence check in §2 before appending.
- `status` is a closed vocabulary (`candidate`/`posted`/`needs-confirm`/
  `aggregated`); anything else fails validation loudly.
- Never fabricate outcomes emails don't state — every figure must trace to
  evidence; when evidence is missing, ask for exactly the missing field.
- Take income/expense totals from gen_views output, never ad-hoc sums — a
  sum without the year filter counts prior years.
- Day- or method-scoped spend questions ("how much cash today") ARE safe as
  ad-hoc ledger reads: filter `transactions.jsonl` on the exact date +
  `via` (case-insensitive) and sum `dir=out` rows. Never quote a remembered
  amount without re-reading its ledger row first — recalled entries carry
  stale dates, and a chat item logged on an earlier day is not today's
  spend. A ZERO-row result needs a freshness check before reporting
  "nothing was spent": compare `transactions.jsonl`/`state.json` mtimes
  against the last scheduled sync and run `pipeline.py --dry` (writes
  nothing) — a missed or skipped sync leaves a whole day's transactions
  invisible, so a genuine zero and a stale ledger look identical.
- Vault data files (`rules.md`, `banks.yaml`, fixtures) live outside the
  agent write sandbox (`HERMES_WRITE_SAFE_ROOT` covers the host data root,
  not the vault mount) — `patch`/`write_file` are refused there; edit them
  via a scratch python script that asserts a unique anchor before replacing
  (the drawer-step pattern),  then re-verify rules edits with
  `pipeline.py --dry`.
- In cleanup audits, "dormant" vault data is not automatically removable:
  grep the vault's OWN `gen_views.py` for reads/asserts on any file before
  listing it deletable — a frozen series can be load-bearing for the gates
  (an assert on an emptied file makes every pipeline run fatal) even when
  its generator script is already gone and the value still feeds views
  and totals.

## Verification

After EVERY write batch, `python3 ${HERMES_SKILL_DIR}/scripts/gen_views.py`
must end with `ALL GATES PASS`. The daily pipeline enforces this itself
(nonzero gate exit is fatal). When developing the skill:
`cd ${HERMES_SKILL_DIR} && python3 -m unittest discover tests` must stay
fully green, and `hermes chat --toolsets skills -q "Use the
petasos-finance-vault skill to run a dry pipeline pass"` exercises the skill
end to end.
