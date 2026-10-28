# Selection policy: the math and the traps

The crown is chosen from the leaderboard's two columns only: the Hermes Index
(`index`) and the Hermes-measured average cost per task (`cost`). Nothing else is
measured, and we never run our own benchmark — the skill picks models that have
never run here.

## The rule

`argmax(index - cost / point_price)` over eligible rows: in-cap (`cost` strictly
under the cap), non-provisional, exactly mapped to a nous provider model id, not
cooling down. `point_price` is USD per Hermes Index point. `none` (or
`usd_per_point: null`) makes a point free, so the rule is plain max index: the
same rule at an infinite price. This family therefore contains the old max-index
rule as a special case; `none` is the shipped default, so upgrading never changes
an existing install's pick until the user opts in.

The shape is a choice, not a theorem. Any monotone preference over (capability,
cost) has an argmax optimum, but the shape is ours to pick. `index - cost/point_price`
is the simplest useful one: it says you pay at most `point_price` for one more
index point, and no more. Real value is probably concave in capability, so treat
the linearity as a deliberate simplification, not a measurement. The cap is a hard
box constraint, unrelated to the preference. Two numbers, everything derived.

## Why the pick is on the Pareto frontier

Strip every row that another row dominates (cheaper-or-equal cost and
higher-or-equal index). What remains is the frontier. No rational rule should
pick a dominated row, and under ignorance of how index maps to real outcomes the
frontier is the whole set of undominated choices. `--sample` prints it. The
`index - cost/point` argmax always lands on it, because off-frontier rows can
only score lower.

## The traps

- **The index is a proxy.** It is pass@1 over Nous's suites, not success on this
  user's tasks. Optimizing it is optimal only insofar as its ordering transfers.
  Do not claim more.
- **No measurable sigma.** The index is a frozen point estimate: no CI, and the
  per-model value is identical across daily ledger snapshots. A capability
  tolerance cannot be derived from history; it is a user-declared preference.
- **Ex-ante by construction.** The whole point is to pick untried models. Do not
  propose measuring on local traffic — that stops being an ex-ante selector.
- **Pure ratios degenerate.** `index/cost` and `score per dollar` always walk to
  the cheapest row (a 21-index model beats everything on ratio). The absolute
  `point_price` in the utility prevents that.

## Tuning the point price

A low `point_price` slides down the frontier to cheaper models; a high one buys
the smartest under the cap. Read the board's `$/point` column (cost/index) to see
where each row sits, then set the price at the trade the user is happy to make.
Two nearby rows can straddle the price, so choose it with the board in view.
