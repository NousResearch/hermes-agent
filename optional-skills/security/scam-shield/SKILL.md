---
name: scam-shield
description: Flag scam and phishing patterns in a message, with reasons.
version: 1.0.0
author: Ruslan (rusmoody)
license: MIT
metadata:
  hermes:
    tags: [Security, Safety, Anti-Scam]
    requires_toolsets: [terminal]
---

# Scam Shield Skill

Scans a message (and any links in it) for known scam and phishing mechanics and
returns a probabilistic risk score with the specific schemes that matched. It
describes the *scheme* in the text and suggests protective actions — it does not
accuse the sender, prove intent, or guarantee that a low score is safe.

## When to Use

Load this when the user forwards a message and asks whether it looks like a
scam, when a message requests wallet seed phrases / login codes / upfront
payments, when an unsolicited "airdrop", "prize" or "support" contact appears,
or when a link looks suspicious. Good for triaging forwarded chats before the
user acts.

## Prerequisites

None. Pure Python standard library, cross-platform, no API keys or network
access. The pattern set lives in `references/patterns.json` and can be extended
by appending entries — no code change required.

## How to Run

Invoke the scanner through the `terminal` tool from the skill directory:

```
python scripts/scan.py --text "<the message>"
python scripts/scan.py --file message.txt --json
```

Use `--json` when you need to parse the result programmatically; omit it for a
human-readable summary. Add `--lang en` for English output (default is Russian). For long messages, save the text with `write_file`
first and pass `--file` to avoid shell-quoting issues.

## Quick Reference

| Flag | Meaning |
| --- | --- |
| `--text "..."` | Scan an inline message string |
| `--file PATH` | Scan a UTF-8 file (use for long / multi-line messages) |
| `--json` | Emit structured JSON instead of the text summary |
| `--lang ru|en` | Output language for advice and findings (default: `ru`) |
| `--patterns PATH` | Use an alternate pattern set (default: `references/patterns.json`) |

Output fields (JSON): `risk_score` (0–100), `risk_band`, `confidence`,
`scheme_tags`, `reasons`, `url_findings`, `safe_actions`, `disclaimer`.

## Procedure

1. If the message is long or multi-line, `write_file` it to a temp file.
2. Run `python scripts/scan.py --file message.txt --json` via `terminal`.
3. Read `risk_band` and `confidence`. Report the score as a probability, never
   as a verdict — say "shows patterns consistent with X", not "this is a scam".
4. Relay the matched `scheme_tags` and the concrete `safe_actions` to the user.
5. If `url_findings` is non-empty, tell the user not to open the link and to
   reach the service by a bookmarked address instead.
6. Always surface the `disclaimer`. A low score is not a safety guarantee.

## Pitfalls

- Keyword/regex matching catches known mechanics; a novel or heavily obfuscated
  scam can score low. Treat a low score as "no known pattern matched", not "safe".
- Do not read intent into the score. It measures message mechanics, not the
  person sending it; benign forwards (e.g. someone quoting a scam to warn a
  friend) can trigger signals — read the context.
- The pattern set is deliberately conservative to limit false positives. Tune
  weights in `references/patterns.json` rather than hardcoding thresholds.

## Verification

```
python scripts/scan.py --text "подтверди кошелёк и введи сид фразу на https://metamask.top" --json
```

Expect a high `risk_score` with `credential_theft` + `wallet_drainer` in
`scheme_tags` and a lookalike-domain entry in `url_findings`.
