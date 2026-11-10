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

## Tasks (13)

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
