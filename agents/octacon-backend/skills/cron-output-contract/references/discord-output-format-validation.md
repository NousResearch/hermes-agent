# Discord Output Format Validation

Use this reference when Sahil reports ugly Discord cron output, Telegram formatting leakage, raw HTML dumps, process narration, or memory/prompt leakage.

## Core principle

Discord is the notification layer. HTML is the working artifact.

| Layer | Correct format | Wrong format |
|---|---|---|
| Discord visible message | Plain text / Markdown, max 5-8 useful lines | Telegram HTML, raw HTML, long prose, prompt/tool narration |
| Attached report | Rich HTML file attached with `MEDIA:/absolute/path.html` | Pasted HTML in chat |
| Zero-signal run | `[SILENT]` for LLM crons, empty stdout for `no_agent=true` | "Nothing to report" noise |

## Audit before patching

Do not fix only the one visible broken cron. Sahil expects a system-wide validation across related Discord-bound outputs.

Check every active job with `deliver` matching `discord:*`, `origin`, or `all`:

1. Cron prompt / prompt preview.
2. Latest output file under `~/.hermes/cron/output/<job_id>/`.
3. Script body for `no_agent=true` jobs under `~/.hermes/scripts/`.
4. Attached report path existence if the output includes `MEDIA:`.

## Risk patterns to scan for

| Pattern | Meaning | Typical fix |
|---|---|---|
| `<b>`, `<code>`, `<blockquote>` | Telegram-visible formatting leaked into Discord | Replace with Markdown/backticks or plain text |
| `Now I'll`, `I will write`, `Here is` | Process narration leaked | Add final-only/no-narration guardrail |
| `<memory-context>`, `Mnemosyne Context`, recalled memory blocks | Memory or hidden context leaked | Add explicit no prompt/system/memory leakage rule |
| `<!doctype html>`, `<html>`, `<style>` in visible response | Raw HTML dump | Force HTML into attached file only |
| `DD Mon`, `Tue 19 May`, `YYYY-MM-DD` in visible text | Non-UK user-facing date drift | Use `DD/MM/YY HH:MM:SS` |
| "Telegram", "topic" in Discord-bound prompt/output | Migration residue | Rewrite provider/channel wording |

## Expected Discord shape

```text
✅ Job name · DD/MM/YY HH:MM:SS
verb · count · signal

• Short bullet
• Short bullet

MEDIA:/absolute/path/report.html
```

Keep the visible message short enough to read from a phone notification. Put detail, evidence, tables, and styled sections in the HTML attachment.

## HTML artifact upgrades worth using

The article/post pattern "Using Claude Code: The Unreasonable Effectiveness of HTML" supports the existing direction: HTML is valuable as an agent artifact, not as chat markup.

Useful upgrades for Hermes cron reports:

- Shared report component system for consistent headers, badges, cards, tables.
- Severity badges: fire / watch / ok.
- Filterable tables for kanban, research, job hunt, and system reports.
- A top `Decision required` panel so action items are not buried.
- Timeline strip showing run time, delivery target, and previous status.
- Evidence/source section so the report is auditable without bloating Discord.
- Embedded JSON payload in `<script type="application/json">` for future dashboard ingestion.

## Reporting back to Sahil

Use a clean table with at least:

| Severity | Job | Mode | Current risk | Evidence | Fix needed |
|---|---|---|---|---|---|

Separate prompt issues from script formatter issues and delivery-routing issues. Do not mass-edit many crons without making the scope clear first.