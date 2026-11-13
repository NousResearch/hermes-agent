# Sibling prompt cache: real delegate_task children

Do the subagents of one parent read each other's prompt cache, and does a child still follow
instructions that arrive only in its `context`?

- `launch.py <route> <model> <arm> <out.jsonl> <tree>`: resolves the route under the real home and
  starts `worker.py` with the credential passed by environment only. Run it with
  `PYTHONPATH=<tree>` and the tree's interpreter; `<arm>` is a label for the tree under test.
- `worker.py`: a real parent `AIAgent` in a temp `HERMES_HOME` calls the real `delegate_task` three
  times in a row: child A, child B with different context, child C with A's context. Each child
  reads a file with real tools and must end its reply with a word named only in its context.
  `SC_PARALLEL=1` sends the three as one batch; `SC_LONG=1` gives each child eight calls.

A fresh temp home has no skills, so the prompt is about half the size of a real install's.

## Reference run (first call of each child: prompt tokens / cache read)

| Route, model | Child | Before | After |
|---|---|---|---|
| Nous, gpt-5.6-luna | B, different context | 7,622 / 0 | 7,627 / 7,574 |
| | C, same context as A | 7,618 / 7,591 | 7,623 / 7,574 |
| Codex subscription, gpt-5.6-luna | B, different context | 7,631 / 0 | 7,631 / 7,424 |
| | C, same context as A | 7,627 / 0 | 7,627 / 7,424 |
| Nous, claude-fable-5.2 | B, different context | 11,904 / 6,354 | 11,909 / 11,761 |
| | C, same context as A | 11,900 / 11,795 | 11,905 / 11,761 |

Three or four runs per cell on gpt-5.6-luna, two on Claude; every run within 20 tokens of the row
shown, except that some Codex "before" calls read 2,304 from an unrelated earlier run.
Three children sent as one batch on the Codex route, eight calls each: 80.8% and 83.7% of prompt
tokens read from cache before, 91.2% and 87.5% after, with every later call reading as much as before.
