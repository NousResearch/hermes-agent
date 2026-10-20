# Drawer True-Up — physical cash count recipe

Trigger: the user reports a physical count of the cash drawer ("I have X
left"). The count is authoritative; the gap versus books is unlogged spend
by definition. One batch converges books to the count.

## Steps

1. **Compute the gap**: `gap = CASH-DRAWER.balance` in `balances.json`
   (after any pending drawer steps for already-logged items) `− counted`.
   Counted ABOVE books is a mystery IN — never true-up upward silently;
   ask the user first (needs-confirm tier).
2. **Ask what the gap went to** — the reason picks category/type by
   precedent (market groceries / đi chợ → Food/Needs; mixed everyday
   household, not itemized → Other/Needs) and is quoted in the row desc
   and the rules.md decision log. Never invent or silently pick a reason
   for a large gap — the user vetoes silent trade-offs. If the user CAN
   itemize, log the real items instead (`add` each + one drawer step) and
   true-up only the residue.
3. **Append ONE catch-up row** via the CLI, dated the COUNT day (not the
   spend window):

   `--kind expense --amount <gap> --desc 'Unlogged Cash Spend Catch-up — <reason> since <last count date> (drawer true-up DD/MM; books <implied> - counted <counted>)' --category/--type/--fv per reason --via Cash`

   The `add` CLI has no `--src`/`--old_ref` flags and `relabel` only
   accepts `src=user-confirm:<date>` — the books-vs-count figures in the
   desc ARE the audit trail. Never hand-edit the JSONL to stamp
   `user-count:`/`count:` src/old_ref like imported rows carry.
4. **Drawer step** — scratch script with the assertion pattern of SKILL.md
   §2: decrement `CASH-DRAWER` balance+available by the gap; each
   cash-inclusive total (`cash_total`, `liquid_total`, `assets_total`,
   `net_worth`) −gap, preserving the vault's stored model; assert post-write
   drawer balance EQUALS the counted figure exactly; append to the
   verified note ("physical count DD/MM: <counted>; true-up row <id>
   -<gap> (<reason>)"). The true-up SUPERSEDES any earlier running-implied
   chain: subsequent drawer steps append bare `<id> cash <desc> -<amount>`
   entries against this count baseline and must not assert
   implied==balance from pre-count markers (note-era rule, SKILL.md §2).
5. **Log it in rules.md** decisions table, same shape as prior true-up
   entries: `| DD/MM | Drawer true-up <gap> | Count <counted> vs books
   <implied> → expense row <id> (<category>/<type>). <owner> DD/MM: gap =
   <reason>. Drawer now = count; next true-up on next count |` — edited
   per the vault write-sandbox pitfall in SKILL.md.
6. **Verify**: run the VAULT's `gen_views.py` → `ALL GATES PASS`, then
   `pipeline.py --dry` (a rules.md edit was made — the parse check is
   mandatory after every one).

## Guards

- True-up ONLY on a count (monthly / as needed); never itemize a gap the
  user already declared not itemized.
- Flag the gap's implied daily rate in the reply when it is high (gap ÷
  days since last count) — a rate near the needs-confirm threshold is a
  monthly-review signal, not just a post.
