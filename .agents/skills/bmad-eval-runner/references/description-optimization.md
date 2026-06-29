# Description optimization: the trigger-eval loop

A skill's description is its only trigger. The router reads it, decides whether the user's request belongs to this skill, and either loads it or moves on. A description that is too narrow stays quiet when it should fire; one that is too broad fires on requests it cannot serve. This loop measures real firing against a held-out test set and improves the description until it triggers on what it should and stays silent on what it should not, without the improver ever overfitting to the cases it is being graded on.

The whole loop runs through the adapter, so "did the skill fire" means the skill-load event the runtime emits, defined in `references/platform-adapter.md`. No model name appears anywhere in this loop; the adapter forwards whatever a runtime needs.

## Step 1: generate the query set

Generate about twenty near-miss queries, roughly half that should trigger the skill and half that should not. The signal lives in the near misses, so make the should-not queries share keywords, domain, and phrasing with the should queries. A should-not query that obviously belongs to another skill teaches the description nothing, because any wording already handles it. The pairs that matter are the ones a careless reader would lump together: a request to build a workflow versus a request to debug an existing one, a request to write a brief versus a request to critique a brief someone already wrote.

Each query is a `{query, should_trigger}` record:

```json
{ "query": "help me turn my deploy script into a reusable skill", "should_trigger": true }
{ "query": "my deploy script keeps failing on the rollback step", "should_trigger": false }
```

Aim for variety in surface form (casual speech, a pasted error, a one-line ask, a paragraph of context) so the description is tested against the shapes real requests arrive in, not one tidy template.

## Step 2: stratified 60/40 split

Split the queries into a train set and a test set, 60 percent train and 40 percent test, stratified so the should and should-not ratio is preserved in both halves. Stratifying matters because an unstratified split can land most of the should-not queries in one half and leave the improver blind to the false-positive problem on train, or leave the test set unable to detect it.

The split is fixed once at the start of the loop and never reshuffled between rounds, because reshuffling would let a query that exposed a weakness in one round hide in the train set the next. The improver works only from the train set. It never sees the test queries, their labels, or the test score, which is what keeps the loop honest.

## Step 3: measure real triggering

Run every query through the adapter with the current description in place, several times per query because firing is probabilistic. The trigger rate for a query is the fraction of runs that produced the skill-load event. Turn each rate into a verdict against a threshold (a query "triggers" when its rate clears the bar, for example more than half its runs loaded the skill), then score against the labels:

- a should-trigger query that triggered is a true positive,
- a should-trigger query that stayed quiet is a false negative (the description is too narrow here),
- a should-not query that triggered is a false positive (the description is too broad here),
- a should-not query that stayed quiet is a true negative.

Score train and test separately. The train score and its per-query verdicts are what the improver sees; the test score is recorded but withheld.

## Step 4: improve from train failures, test blinded

Hand the improver the current description, the train queries with their labels, and the train verdicts, and ask for a rewritten description that fixes the train failures. False negatives mean the description needs to claim ground it is leaving uncovered; false positives mean it needs to draw a sharper boundary against the near misses it is wrongly catching. The improver works the train failures only and never sees a test query or the test score, so it cannot tune to the held-out set.

Also hand the improver the descriptions it already tried and why each fell short, so it tries something structurally different rather than nudging the same wording round after round. Without this, the loop tends to oscillate between two phrasings that each fix one failure and reintroduce the other. Feeding the history back pushes the improver toward a different cut of the boundary: reframing around intent instead of keywords, naming the adjacent skill the near misses belong to, or moving a qualifier from the trigger clause into the body.

Keep the description within whatever length and format bounds the runtime enforces (character cap, no angle brackets, and so on); a rewrite that triggers well but violates the bound is not a candidate.

## Step 5: re-measure and iterate

Apply the new description, re-measure train and test, and record both scores plus the description text for this round. Continue for up to five rounds. Stop early if train reaches a clean separation (all should fire, all should-not stay quiet) and the test score agrees, because more rounds past a clean split only invite overfitting.

## Step 6: pick the winner by test score

After the rounds finish, pick the description with the best test score, not the best train score. Train measures how well the improver fixed the failures it could see; test measures whether that fix generalizes to queries it never saw, which is the only thing that matters in production. When two rounds tie on test, prefer the one with the better train score as the tiebreaker, and failing that the shorter, sharper description.

Report the winning description, its test score, and the round-by-round trail (each description, its train score, its test score) so the choice is auditable and a human can override it. Log the trail to the run's memlog through `scripts/memlog.py` as the loop runs, one `event` entry per round capturing the description tried and the train and test scores, so a resumed or audited run reads the progression cleanly.

## Why each guard is here

| Guard | What it prevents |
|---|---|
| near-miss should-not queries | a test set so easy the description never has to draw a real boundary |
| 60/40 stratified split | a split that hides the false-positive or false-negative problem in one half |
| fixed split across rounds | a weakness escaping into the train set on a later round |
| test score blinded from improver | the improver tuning its wording to the held-out queries |
| pick by test score, not train | shipping a description that fixed the visible failures but does not generalize |
| prior attempts fed back | the loop oscillating between two phrasings instead of finding a new boundary |
