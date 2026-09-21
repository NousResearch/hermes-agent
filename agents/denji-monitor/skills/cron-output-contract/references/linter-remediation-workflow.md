# Linter Remediation Workflow

When `cron-output-lint.py` returns issues, follow this structured remediation.

## Step 1 — Understand what each finding means

The linter checks prompts AND scripts (for LLM-driven and no_agent crons respectively)
against four pattern classes:

| Finding | Pattern | What it catches |
|---------|---------|-----------------|
| `telegram_html_tags` | `<b>`, `<code>`, `<blockquote` | Telegram HTML rendered as literal text on Discord |
| `raw_html_visible` | `<!doctype html`, `<html`, `<style` | Full HTML document template leaked into visible prompt/output |
| `process_narration_literal` | "Now I'll", "I will write" | LLM describing actions instead of doing them |
| `memory_leak_literal` | `<memory-context>`, "Mnemosyne Context" | Internal memory blocks exposed in user-visible output |
| `non_uk_visible_date` | `%a %d %b`, `Mon 18 May`, `2026-05-18`, `May 18, 2026` | Non-UK date format in delivered output |
| `iso_timestamp_in_report` | `2026-05-26 14:48:24 UTC` | ISO/UTC format instead of UK (`DD/MM/YY HH:MM:SS`) |
| `first_line_missing_ddmmyy_hhmmss` | First line doesn't match `^.+ · DD/MM/YY HH:MM:SS$` | Missing or wrong-format timestamp header |

## Step 2 — Determine remediation by cron mode

### LLM-driven crons (`no_agent=false`)

The linter checks the **prompt** (skill content + cron prompt text). The LLM
reads this at run time and follows its instructions. If the prompt contains
HTML template examples or Telegram tags, the LLM reproduces them in output.

**Remediation:**

1. Find the source of the HTML tag in the prompt:
   - Check the cron's loaded skills (listed in `skills[]` in jobs.json)
   - Check the cron's `prompt` field directly
   - Both may contain HTML examples in their documentation

2. For `telegram_html_tags` in prompts:
   - Replace bare `<b>text</b>` examples with `**text**` (Discord bold)
   - Replace `<code>text</code>` with `` `text` `` (Discord code)
   - Replace `<blockquote expandable>` with standard markdown or remove
   - The prompt CAN safely contain HTML when it's inside code fences
     or explicitly describing what NOT to do (e.g. "Do not use `<b>` tags")

3. For `raw_html_visible` in prompts:
   - HTML template examples that include `<!doctype html>` or `<html>` or `<style>`
     should be wrapped in markdown code fences so the LLM sees them as
     reference examples, not literal output models
   - If the tag is inside a code fence and the linter still catches it,
     the linter regex is scanning raw text, not parsing structure — that's
     a false positive; suppress by noting it in the cron prompt

### no_agent crons (`no_agent=true`)

The linter checks the **script file** content directly. The script's stdout
IS the message. If the script prints HTML tags, they appear as literal text.

**Remediation:**

1. Read the script's stdout-producing lines (`print()`, `echo`, `cat`)
2. Replace each Telegram HTML tag with its Discord-safe equivalent:

   | Telegram | Discord | Example |
   |----------|---------|---------|
   | `<b>text</b>` | `**text**` | `print(f"**{title}**")` |
   | `<code>text</code>` | `` `text` `` | `print(f"\`{task_id}\`")` |
   | `<blockquote expandable>` | remove entirely | delete the line |
   | `<i>text</i>` | `*text*` or remove | `print(f"*{note}*")` |
   | `<a href="url">text</a>` | `text — url` | `print(f"{label} — {url}")` |

3. Run the script manually to verify stdout is clean:

   ```bash
   python3 /home/kensei/.hermes/scripts/your-script.py | grep -c '<b\|<code\|<blockquote'
   # Should return 0
   ```

### Real example: denji-self-eval-reminder.py

The script printed Telegram HTML throughout:
```python
print("🔄 <b>Self-Eval Reminder</b> · " + now.strftime("%d/%m/%y · %H:%M:%S"))
print("<b>Who should eval this week</b>")
print("Use the template at <code>/home/kensei/.hermes/governance/self-eval-schema.md</code>")
```

Fixed to:
```python
print("🔄 **Self-Eval Reminder** · " + now.strftime("%d/%m/%y · %H:%M:%S"))
print("**Who should eval this week**")
print("Use the template at `/home/kensei/.hermes/governance/self-eval-schema.md`")
```

## Step 3 — Re-run linter and verify

```bash
/home/kensei/.hermes/scripts/cron-output-lint.py
```

Must return `"count": 0`. If it still has hits, go back to Step 2.

## Step 4 — Run a representative cron to verify delivery

For LLM-driven crons, trigger a manual run:
```bash
hermes cron run <job-id>
```

For no_agent crons, run the script directly and check the Discord output:
```bash
bash /home/kensei/.hermes/scripts/your-script.sh
# Or
python3 /home/kensei/.hermes/scripts/your-script.py
```

Verify:
- No raw `<b>`, `<code>`, `<blockquote>` text in Discord
- MEDIA: tag attaches a file (not shown as literal path)
- UK date format (`DD/MM/YY HH:MM:SS`)
- First line has a timestamp header

## When to ignore linter findings

- **Code fences:** HTML inside triple backticks is reference content, not
  output. The linter regex scans raw text and may flag these. If the tag
  is inside a markdown code block and exists only as documentation, the
  prompt is fine — the LLM won't reproduce it.
- **Anti-examples:** If the prompt says "Do NOT use `<b>` tags" and the
  linter flags that occurrence, it's a false positive. Wrapping the anti-
  example in backticks helps: "Do NOT use `` `<b>` `` tags"
- **`--latest` mode:** Historical output files that haven't rerun yet will
  still show old hits. Only fix these if the cron has since rerun and still
  produces bad output. Default mode (without `--latest`) only checks current
  prompts and scripts, which is the right target.