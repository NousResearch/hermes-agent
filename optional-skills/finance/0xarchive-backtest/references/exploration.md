# Exploration: "find me something"

Use `terminal` for the commands below, with the project Python environment active and the
study directory as the working directory. Use `read_file` for files; examples are not permission
to execute unreviewed code or open a sealed holdout.

An open search is where fake edges come from. It is allowed here because the study counts every try and
keeps the last months sealed; the rest is up to how the search is run.

## Declare the search before running it

In `STUDY.md`, before the first backtest:

- **The family:** one kind of idea per study (cross-sectional momentum; funding carry; a seasonal effect;
  order-flow imbalance). A new family is a new study.
- **The space:** each parameter and the values it may take, chosen for a reason written beside it.
- **The budget:** the most trials you will run, agreed with the user. 20 is plenty for one family; past
  50, even a real edge rarely survives the deflated Sharpe in 16 months of data.
- **The pick rule:** how the one candidate will be chosen (best gauntlet rung, then best deflated
  Sharpe), so the choice isn't made by looking at curves.

## Run it

1. Event studies first to choose which family is worth a study at all (they don't count as trials, so
   keep them few and list them).
2. Within the family, sweeps over the declared space: `0xbt sweep FILE ... -g ...`. Each point is a trial.
3. Gauntlet the top two or three by the pick rule.
4. Stop at the budget. If nothing reaches `holds out of sample`, the answer is "nothing found in this
   family", which is a result worth reporting.

## Commit

When one candidate reaches `passes deflated Sharpe`, propose it to the user with its card, the trial
count and the pick rule, and commit only on their go-ahead (`0xbt study commit`). Rerun its gauntlet so
the card shows the holdout. If the holdout fails, the study's answer is that candidate failing on
unseen data: report it, and don't pick the runner-up and commit again, because the holdout is spent.
