# Platform adapter

Everything runtime-specific in the eval-runner lives here, behind one seam. The rest of the skill, the scripts, the case format, the grader, and the modes are written against this seam and stay platform-agnostic. No model name is hardcoded anywhere; a model is just a value the adapter forwards if a runtime needs one, never a list this skill maintains.

## The adapter config file

An adapter is a JSON file the scripts read. A working Claude Code adapter ships at `assets/adapter-claude-code.json`:

```json
{
  "name": "claude-code",
  "invocation": ["claude", "-p", "{prompt}", "--output-format", "stream-json",
                 "--verbose", "--dangerously-skip-permissions"],
  "auth_env": "ANTHROPIC_API_KEY",
  "transcript": { "format": "stdout-jsonl" },
  "skill_dir": ".claude/skills",
  "load_signal": { "skill_tool": "Skill", "read_tool": "Read" },
  "env_passthrough": []
}
```

| Key | Required | Meaning |
|---|---|---|
| `invocation` | yes | argv template for one non-interactive run. `{prompt}` (alias `{query}`) is replaced with the composed input, `{cwd}` with the case's clean working directory. |
| `auth_env` | no | name of the one env var the runtime reads for its credential. Forwarded from the host **only when set non-empty** — forwarding an empty string overrides the runtime's own credential fallback and breaks auth. |
| `transcript` | no | `{"format": "stdout-jsonl"}` (default; stdout captured as the JSONL transcript) or `{"format": "file", "path": "transcript.jsonl"}` (runtime writes a file in the cwd). |
| `skill_dir` | no | directory under the cwd where the runtime discovers skills. Default `.claude/skills`. Used to stage the skill under test and trigger mode's synthetic skill. |
| `load_signal` | trigger mode | which tool calls count as a skill load: `{"skill_tool": "Skill", "read_tool": "Read"}` (the defaults). See trigger detection below. |
| `env_passthrough` | no | extra host env var names to forward into the run, for runtimes that need more than the auth var. Empty unless a runtime forces it. |

### Discovery

`run_evals.py` and `run_triggers.py` locate the adapter in this order:

1. `--adapter <path>` on the command line.
2. `BMAD_EVAL_ADAPTER` env var pointing at a config file.
3. `adapter.json` or `.bmad-eval-adapter.json` beside the cases/queries file.

Nothing found means the run degrades to staging-only (cases prepared, results recorded as skipped). When the current runtime is Claude Code and no project adapter exists, pass `--adapter {skill-root}/assets/adapter-claude-code.json`.

## Invocation and isolation

The runner fills the invocation template with the input (any `state_prefix` already prepended) and the clean working directory, runs the command from that directory, and waits for completion. Before invoking, it stages into the cwd: the skill under test at `<cwd>/<skill_dir>/<skill-name>/`, and any case fixtures.

The subprocess environment is built from scratch, never inherited, so host shell config, memories, and tokens cannot bias the result. It contains exactly: `PATH`, a fresh empty `HOME` at `<case>/.home`, `CLAUDE_CONFIG_DIR` inside that HOME, the `auth_env` var when set non-empty on the host, and any `env_passthrough` keys present on the host. There is no container, no terminal emulation, and no credential file staging.

For a baseline run the runner issues the same command twice from the same input: once with the skill staged in the working directory and once with nothing staged, so the bare-model floor is measured under identical conditions. For a variant run it stages the full skill in one config and the `--variant-path` skill in the other.

## Transcript schema

The transcript tells `run_evals.py` where timing and token counts live and tells the grader how to read tool calls and the final message. The scripts read line-delimited JSON events: `assistant` events carry `message.content[]` items (a `tool_use` item has `name` and `input`; usage blocks carry token counts), and a `result` event's usage block is authoritative for totals. A runtime whose events differ needs its own accounting branch — that branch belongs here, behind the seam, not in a mode or the grader.

## Trigger detection: "did the skill load"

Trigger mode does not measure output; it measures whether the description caused the skill to fire. `run_triggers.py` stages a synthetic skill (unique name) in `skill_dir`, sends each query through the invocation command, and scans the transcript for a load. Each query runs several times because firing is probabilistic; the trigger rate is the fraction of runs that loaded the skill.

Only `tool_use` events count as a load: a `skill_tool` call whose input names the synthetic skill, or a `read_tool` call whose `file_path` falls inside the synthetic skill's directory (its SKILL.md). Whole-transcript substring matching is rejected outright, because the runtime's init event lists every discovered skill by name — a substring match would report 100% trigger rate no matter what the description says.

## Adding a runtime

Write an adapter file declaring the keys above; add `skill_dir` and `load_signal` if you want trigger mode. Add no model list and no provider branch anywhere else; if a value beyond these is needed, it belongs in the adapter, not in a script or a prompt.
