# prefix_rediet — live A/B for the fixed-prefix re-diet

Validates the PR that took the stock CLI prompt surface back below its September size: the five eager
`browser_vault_*` tools collapsed into one deferred `browser_vault`, the terminal / tool_search /
tool_call schemas re-dieted, and the "Async handoff" system-prompt block removed (its rule already
ships in the delegate_task description and in the background dispatch result).

Real in-process `AIAgent`s from two plain checkouts (base = main, pr = the branch), each cell in its
own subprocess with a temp `HERMES_HOME`, host side effects sandboxed by
`../core_tool_deferral/sandbox.py`.

## Tasks

| id | what it proves |
|---|---|
| `vault_saved_login` | a deferred vault is still found: list → fill → submit → 2FA `enter_code`, never typing the demo password the page shows as bait |
| `vault_first_login` | the discovery case that kept the vault tools eager: nothing saved, the model must reach `save_login` instead of asking for or typing a password |
| `term_slow_tests` | a ~70s command still gets run to completion and reported (foreground timeout or background + notify) |
| `term_dev_server` | a server is backgrounded and still serving when the run ends |
| `term_wait_then_write` | the "never background a sleep" rule survives the diet (partial credit if the file is written via a background sleep) |
| `search_cron_backup` | a deferred tool (cronjob_manage) is found with the leaner tool_search text |
| `search_recall_dose` | session_search is found and used for recall |

The browser page and the vault backend are deterministic stubs (`browser_exec` returns a scripted page,
the vault returns realistic results for both arms' tool names), so scores measure tool selection and
discovery, not browser plumbing. Terminal, files, cron and the session DB are real.

## Running

```bash
git worktree add .worktrees/main-snap origin/main
export PRD_BASE_TREE=$PWD/.worktrees/main-snap PRD_PR_TREE=$PWD
export PRD_PYTHON=<the install's committed env python>   # see hermes-agent-dev: worktree probes
export PRD_RESULTS=$HERMES_HOME/cache/scratch/prefix_rediet
python3 evals/prefix_rediet/run.py qwen/qwen3.8-27b 3 --parallel=7
python3 evals/prefix_rediet/run.py --report
```

Inference defaults to the Nous Portal agent key, read fresh from `$HERMES_HOME/auth.json` for every
cell (read-only; the grant is never refreshed or copied). Set `PRD_INFERENCE_URL` +
`PRD_INFERENCE_KEY` for any other OpenAI-compatible endpoint.

Errored cells score 0 and stay in the denominator; exit 3 (no model turn at all) is an infra failure
and is never written.
