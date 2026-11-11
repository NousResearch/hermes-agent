# Child cache marker: real-loop check

Does a subagent whose task text differs from its sibling's still read the session prompt (skills index
and footer) from the provider cache on its first call, and does a sibling with identical task text keep
reading everything?

- `launch.py <arm> <out.jsonl> <rep> <tree>`: resolves the Nous route under the real home and starts
  `worker.py` with the key passed by environment only. Run it with `PYTHONPATH=<tree>` and the tree's
  interpreter.
- `worker.py`: temp `HERMES_HOME` with `isolate_host`, a repo-shaped workspace carrying an `AGENTS.md`,
  and three children built the way `tools/delegate_tool.py` builds them, run one after another: A, then B
  with different task text, then C with A's task text. Each does a small real task with real tools.
  Per-call prompt, cache-read and cache-write tokens come from the agent's own usage accounting.

A fresh temp home has no skills, so the session prompt here is small; the size of the saving on a real
library is measured separately with captured payloads (see the PR that added this directory).

## Reference run (anthropic/claude-fable-5.2 on the Nous chat route, 3 runs per arm)

First call of each child, `(prompt, cache read, cache write)`, identical across the three runs to within
10 tokens:

| child | main | this change |
|---|---|---|
| A, first in the run | 11,881 / 6,354 / 5,523 | 11,886 / 6,354 / 5,528 |
| B, different task text | 11,887 / 6,354 / 5,529 | 11,892 / 6,783 / 5,105 |
| C, same task text as A | 11,881 / 11,776 / 101 | 11,886 / 11,781 / 101 |

Every child completed its task (9 of 9 per arm), and every second call read the whole first call
(for example 11,877 of 12,353) on both arms.
