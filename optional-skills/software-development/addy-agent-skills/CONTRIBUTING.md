# Contributing to Agent Skills

Thanks for your interest in contributing! This project is a collection of production-grade engineering skills for AI coding agents.

## Adding a New Skill

1. Create a directory under `skills/` with a kebab-case name
2. Add a `SKILL.md` following the format in [docs/skill-anatomy.md](docs/skill-anatomy.md)
3. Include YAML frontmatter with `name` and `description` fields
4. Ensure the `description` starts with what the skill does (third person), then includes one or more `Use when` trigger conditions

### Skill Quality Bar

Skills should be:

- **Specific** — Actionable steps, not vague advice
- **Verifiable** — Clear exit criteria with evidence requirements
- **Battle-tested** — Based on real engineering workflows, not theoretical ideals
- **Minimal** — Only the content needed to guide the agent correctly

### Structure

Every new skill must have:

- `SKILL.md` in the skill directory
- YAML frontmatter with valid `name` and `description`

New skills should generally follow the standard anatomy:

- **Overview** — What this skill does and why it matters
- **When to Use** — Triggering conditions
- **Process** — Step-by-step workflow
- **Common Rationalizations** — Excuses agents use to skip steps, with rebuttals
- **Red Flags** — Warning signs that the skill is being applied incorrectly
- **Verification** — How to confirm the skill was applied correctly

The frontmatter fields above are required. The section anatomy is a recommended pattern: equivalent headings such as `How It Works`, `Workflow`, or `Core Process` are fine when they preserve the same intent and keep the skill easy to follow.

### What Not to Do

- Don't duplicate content between skills — reference other skills instead
- Don't add skills that are vague advice instead of actionable processes
- Don't create supporting files unless content exceeds 100 lines
- Don't create an empty `scripts/` directory just to match another skill — add `scripts/` only when the skill includes runnable helpers
- Don't put reference material inside skill directories — use `references/` instead

## Modifying Existing Skills

- Keep changes focused and minimal
- Preserve the existing structure and tone
- Test that YAML frontmatter remains valid after edits

## Testing Hooks

The session-start hook (`hooks/session-start.sh`) injects the `using-agent-skills` meta-skill into every new Claude Code session. A regression test at `hooks/session-start-test.sh` validates the hook's JSON payload — both when `jq` is available and when it isn't.

Run it before opening any PR that touches:

- `hooks/session-start.sh`
- `skills/using-agent-skills/SKILL.md` (the meta-skill content embedded by the hook)

```bash
bash hooks/session-start-test.sh
```

Expected output: `session-start JSON payload OK`. The script exits non-zero on any assertion failure.

### Reproducing the no-jq fallback

The hook gracefully degrades to an `INFO`-priority payload when `jq` isn't on `PATH`. To exercise that branch locally, strip `jq`'s directory from `PATH` for the test invocation:

```bash
JQ_DIR=$(dirname "$(command -v jq)")
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "^${JQ_DIR}$" | tr '\n' ':' | sed 's/:$//') \
  bash hooks/session-start-test.sh
```

This works cleanly when `jq` lives in its own directory (e.g. `/opt/homebrew/bin` from Homebrew, `/usr/local/bin` from a manual install). If your `jq` shares a system bin with other tools the test depends on (such as `mktemp` in `/usr/bin`), the simpler approach is to install `jq` via a separate package manager so it has its own bin directory, then re-run.

The hook's `command -v jq` check fails under the stripped `PATH`, the `INFO`-priority fallback runs, and the test asserts the `jq is required` guidance message instead of the normal payload.

## Reporting Issues

Open an issue if you find:

- A skill that gives incorrect or outdated guidance
- Missing coverage for a common engineering workflow
- Inconsistencies between skills

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
