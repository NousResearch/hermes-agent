# HTML emails with facts in the body — NDR's hard preference

Source: NDR, 14 Oct 2026, on the passport-grievance emails.

> "the email body itself should have all of the facts, figures, maybe even all of the
> necessary attachments and a strong case made out plus highlighted with bullets, bold,
> maybe use HTML CSS to make it so it's easily readable. This email is pure text and it's
> not easily readable."

## Rule

Plain text is acceptable only for one-line internal notes. Any email that carries a case,
a briefing, a set of facts to forward, or an escalation must be **HTML with inline CSS**,
with all facts and figures in the body — not only in attachments.

## Shape that works

- Body font Georgia/serif ~14–15px, line-height ~1.6, colour #1f2328. Header font
  Arial/Helvetica.
- Inline CSS only (`<style>` in the head, no `<script>` — the rendering origin's CSP blocks
  inline scripts, and inline `<script>` never executes there anyway).
- Top summary box: background #eef3f8, left 4px border #1a3a5c, holds the whole ask in
  one or two lines.
- Bold section headers with a 2px bottom border in #1a3a5c.
- Bullet lists for appointments, name forms, and "already done" facts.
- A numbered table of every attached document and what each one establishes
  (`<table>` with #1a3a5c header row, 1px #dde4ea row borders, alternating #f7f9fb rows).
- Red callout (border #b3261e, background #fdf2f1) for blockers; green callout
  (border #1e7a3c, background #f0f7f2) for what is already done.

## Attachments

Attach the actual evidence files, not just a Drive link:

```
from email.mimemultipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
```

Attach with `MIMEApplication` (`_subtype='pdf'`) and
`add_header('Content-Disposition','attachment', filename=...)`. Then verify by walking the
raw MIME of the created draft — count the parts and confirm each starts with `%PDF` —
before telling NDR the draft is ready.

## Pair with

- `passports-name-reconciliation` → `passport-seva-grievance-portal.md` for the
  paste-ready browser-agent fill prompt for long government web forms.
- `references/third-party-handoff.md` (schengen-visa-file-preparation) for the pattern of
  handing a document pack to an outside party with a compliance report.

## Note

This SKILL.md is over the 100k-char limit and cannot be patched in place. Keep this rule in
the references/ directory until the umbrella is split.
