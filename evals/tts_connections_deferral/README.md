# text_to_speech / manage_connections deferral A/B

Does the model still reach `text_to_speech` and `manage_connections` when their schemas sit behind
`tool_search` instead of being sent on every request?

## Layout

- `worker.py`: one (arm, model, task, rep) cell in its own process: temp `HERMES_HOME`, `isolate_host`,
  a real `AIAgent` loop on a real model. Both tools, the hosted connector search/describe legs and the
  hosted connector call are deterministic stubs, so a cell measures tool selection and arguments.
- `run.py`: battery runner (resume-safe) and the readout (`--report`).
- `results/SUMMARY.md`: the checked-in readout; per-cell JSON and transcripts stay local.

## Run

```
git worktree add --detach <base> <base sha>      # arm "base"
git worktree add <pr> <branch with the change>   # arm "pr"
TCD_BASE_TREE=<base> TCD_PR_TREE=<pr> TCD_PYTHON=<hermes env python> TCD_RESULTS=<dir> \
  python3 run.py anthropic/claude-fable-5.2 1 --only=conn_required_recovery   # smoke one cell per arm
TCD_... python3 run.py anthropic/claude-fable-5.2 3
TCD_... python3 run.py qwen/qwen3.8-27b 3
python3 run.py --report
```

Inference defaults to the Nous Portal agent key, read fresh from `$HERMES_HOME/auth.json` for every
cell and passed as a `custom` provider; set `TCD_INFERENCE_URL` + `TCD_INFERENCE_KEY` for another endpoint.

## Sandboxed multi-arm battery (manage_connections only)

`battery.py` runs one tree under four arms that differ only in the cell's own config and one sentence:
A = `manage_connections` eager, B = deferred, C = deferred with the first description sentence reworded
to "Connect the user's app accounts and install MCP servers." (that sentence is the catalog line),
D = eager with the reworded sentence. The rewording is applied inside the worker; no default changes.

```
python3 relay.py <sock> <log.jsonl> &        # the only network exit; holds the agent key
TCD_TREE=<checkout> TCD_PYTHON=<hermes env python> TCD_RESULTS=<dir> TCD_CELLS=<scratch dir> TCD_RELAY_SOCK=<sock> \
  python3 battery.py anthropic/claude-fable-5.2 --arms=A,B,C --dry      # no model call: proves stubs + sandbox
TCD_... python3 battery.py qwen/qwen3.8-27b --arms=A,B,C --only=conn_mcp_install --reps=1   # smoke
TCD_... python3 battery.py qwen/qwen3.8-27b --arms=A,B,C,D
TCD_RESULTS=<dir> python3 battery.py --report
```

Every cell runs in bubblewrap (`bwrap` and `socat` must be installed): its own network namespace with
only loopback and the relay socket, `/home`, `/tmp`, `/run`, `/mnt` replaced by empty tmpfs, the tree and
the interpreter bound read-only, one writable dir at `/home/cell`. The operator's Hermes home, tokens
and CLI do not exist inside a cell, and `relay.jsonl` lists every outbound request the battery made.
The dry cell checks all of that plus the stubs (inline executor, registry and bridge paths) before any
model is called. Battery: four MCP-install phrasings (5 reps), four hosted-connection tasks, two
negatives, two eager-only controls (3 reps). Pass bar per model, against arm A: no fewer tasks done, no
more wrong-route cells (terminal, code or file tools used on a connection task), model calls per task
up by at most 0.2, total tokens per task not higher.

## Tasks (13, two-tree `run.py` form)

| group | tasks | oracle |
|---|---|---|
| text_to_speech, explicit | `tts_read_aloud`, `tts_voice_note` | the stub was called and the spoken text carries the content |
| text_to_speech, indirect | `tts_indirect_driving`, `tts_indirect_mp3` | same |
| manage_connections | `conn_connect_gmail`, `conn_status_notion`, `conn_mcp_install`, `conn_second_account` | a call the arm's own validators accept, with the right action and target |
| recovery | `conn_required_recovery` | a hosted Gmail call returns CONNECTION_REQUIRED; the email is sent only after `connect` |
| negatives | `tts_negative`, `conn_negative` | right answer and the stub never called |
| controls | `ctl_write_file`, `ctl_count_lines` | eager tools only |

## Discipline

- Same tasks and reps on both arms; every cell that ran is in the x/N denominator, errors and wall
  timeouts score 0. A cell with no assistant turn exits 3 and is not scored.
- A turn that ends in a plain-text question gets the scripted user reply (max 2), on both arms.
- "Reached" is read from the stub, never from the model's prose. "Valid on first call" also fails
  when the bridge refused the arguments before the stub ran.
- A model that cannot see `manage_connections` reaches for the host's `hermes` CLI, and absolute paths
  lead out of the temp home. The worker puts a refusing `hermes` shim first on PATH; reads of the
  operator's real `~/.hermes` by absolute path are still possible, so run this on a machine where that
  is acceptable and audit transcripts for `hermes ` commands after a battery.
