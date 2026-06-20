# Fact Verification Guide

> How to verify facts before including them in agent output.
> Based on real failure modes observed in production use.

## Core Principle

Any output containing **verifiable facts** (dates, weather, stock prices, news,
file paths, version numbers, quantities) MUST have the corresponding tool executed
BEFORE generating the text. Unverified facts are not "decorative filler" — they
are misinformation.

## Common Failure Modes

### 1. Date/Time Fabrication
- **Symptom**: TTS voice says "Happy Friday!" when it's actually Saturday.
- **Root cause**: Agent generated text without running `date` first.
- **Fix**: Always run `date` before any output mentioning day of week, date, or time.

### 2. Command Non-Existence
- **Symptom**: Agent recommends `hermes config get <key>` → user gets `invalid choice: 'get'`
- **Root cause**: Agent assumed subcommand exists without checking `--help`.
- **Fix**: Run `command --help` before recommending any CLI command to a user.

### 3. Config Field Non-Existence
- **Symptom**: Agent claims "you can check with `hermes config <fake_subcommand>`"
- **Root cause**: Inferring config structure from memory rather than actual config.yaml.
- **Fix**: Grep config.yaml or run `hermes config show` before describing config state.

## Verification Checklist

When output includes any of the following, verify first:

| Category | Tool to Run | Example |
|----------|------------|---------|
| Date/time | `date` | "Today is..." |
| Weather | web_search | "It's sunny in..." |
| Stock price | web_search / API | "AAPL is at..." |
| File path | `ls` / `stat` | "The file at ~/.hermes/..." |
| CLI command | `command --help` | "You can use `hermes config get`" |
| Config field | `grep config.yaml` | "Your config has X set to Y" |
| Version | `command --version` | "Hermes v0.13.0 supports..." |

## TTS-Specific Guidance

Voice output has two layers:
- **Emotional layer** (greetings, tone, personality) — creative, no verification needed
- **Factual layer** (dates, weather, news, data) — must be verified before inclusion

Keep them separate: greetings handle mood, facts must be accurate.
