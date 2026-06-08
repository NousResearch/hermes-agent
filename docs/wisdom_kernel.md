# Hermes Wisdom Kernel

# What Wisdom Is Now

Wisdom is Hermes' local personal knowledge loop for ideas worth preserving.

It is not Notion, Obsidian, a graph database, a dashboard, or an enterprise
knowledge platform. It is a small Hermes-native loop:

```text
capture exact words -> retrieve -> review -> apply
```

Exact originals remain the source of truth. Interpretations, review status, and
application proposals are annotations.

v1 built the memory kernel: exact-original preservation, deterministic
capture/classification, SQLite storage, search, deterministic interpretation,
and internal application proposals.

v2 made Wisdom native to Hermes/Codex by exposing the kernel as first-class
model tools. You can talk naturally; `/wisdom` commands are fallback/debug
controls.

v3 added the review and quality loop. Hermes can prioritize what deserves
review, surface high-potential or unapplied ideas, suggest related captures
without embeddings, and mark captures as accepted, dismissed, applied, or
archived.

v4 improves application quality. When you ask what to do with an idea, Wisdom
generates more useful deterministic proposals: client language, investment
rules, checklists, health experiments, principles, writing seeds, and decision
rules.

Wisdom does not write to the old productivity database and does not take external actions.

## Natural Language Tools

Hermes now registers these native model tools in the default CLI and messaging toolsets, including Telegram:

```text
wisdom_status
wisdom_capture
wisdom_search
wisdom_original
wisdom_interpret
wisdom_apply
wisdom_review
wisdom_related
wisdom_accept
wisdom_dismiss
wisdom_archive
wisdom_inbox
wisdom_set_enabled
```

Expected natural-language behavior:

```text
Remember this: clients don't buy alpha, they buy peace of mind.
Save this investing thought: I confuse a good thesis with a good position size.
Podcast note: Acquired - Costco episode. Context: trust, low-price positioning.
Find that idea about peace of mind.
What have I said about position sizing?
Show me exactly what I wrote about peace of mind.
Show exact wording for #12.
What should I review?
Show related ideas.
Accept that.
Dismiss that.
Turn that into client language for x10x.
Make this an investment rule.
Make this a checklist.
Turn this into a health experiment.
What have I been thinking about investing recently?
```

The model sees tool descriptions that tell it when to use Wisdom. There is no large regex command translator in v2. The model decides when a durable operation is needed, calls the relevant Wisdom tool, and replies from the tool result.

Wisdom should not be used for ordinary informational chat such as:

```text
Explain PMS vs AIF.
Help me debug Hermes.
```

unless the user explicitly asks to remember, save, find, retrieve, review, or apply saved Wisdom material.

## Storage

Default database:

```text
~/.hermes/wisdom/wisdom.db
```

Wisdom also creates a local HMAC salt at:

```text
~/.hermes/wisdom/salt
```

The salt is used to hash session and message identifiers before storage. It is not printed or rendered.

## Exact Originals

The exact accepted original is stored in `raw_events.original_text`.

Wisdom does not normalize punctuation, whitespace, capitalization, emojis, or line breaks in that field. Cleaned text, titles, interpretations, and application proposals are annotations only.

Secret-like captures are rejected by default instead of storing a redacted value while claiming exact preservation.

## Source-Aware Capture

Wisdom preserves the exact original first. It also extracts short source/context
metadata when the wording is obvious and non-secret.

Examples:

Quick capture:

```text
Remember this: clients don't need rear-view mirrors, they need windshields.
```

Podcast note:

```text
Podcast note: Acquired - Costco episode.
Context: trust, low-price positioning
Costco's membership model creates trust because low-price positioning makes the retailer feel aligned with members.
```

Book note:

```text
Book note: The Great Mental Models.
Context: courage and uncomfortable decisions
Before building a system, ask whether it is avoiding a harder human action.
```

Investing thought:

```text
Investing thought: I confuse a good thesis with a good position size.
```

Business/client-language note:

```text
Business idea: Client reports should help decide what to do next, not only describe what happened.
```

Health observation:

```text
Health note: I make worse decisions after poor sleep but behave as if cognition is constant.
```

`Source:` and `Context:` are lightweight hints, not a rigid syntax. The exact
original is stored even when no metadata is extracted.

## Commands

```text
/wisdom status
/wisdom capture <text>
/wisdom inbox
/wisdom search <query>
/wisdom original <id>
/wisdom interpret <id>
/wisdom apply <id>
/wisdom archive <id>
/wisdom review [category|unapplied|high-potential]
/wisdom related <id>
/wisdom accept <id>
/wisdom dismiss <id>
/wisdom on
/wisdom off
/wisdom help
```

`/wisdom original <id>` returns the exact stored original for accepted non-secret captures.

Commands and native tools share the same Wisdom service layer and the same SQLite-backed kernel.

## Review and Quality Loop

Manual review is the v3 product center.

Use natural language:

```text
What should I review?
Review my recent business ideas.
What have I captured but not applied?
Show me high-potential ideas.
What should I do with my Wisdom notes?
Show related ideas for #12.
Accept #12.
Dismiss that one.
Turn #15 into an investment rule.
```

Review items include:

- capture ID
- original excerpt
- category and source type
- review status
- deterministic quality indicators
- why it may matter
- suggested next action
- related captures, when found

Review statuses are:

```text
unreviewed
reviewed
accepted
dismissed
applied
archived
```

`accepted` means the idea is worth keeping and compounding. `dismissed` means it should not keep resurfacing in normal review, but the exact original is still preserved. `applied` means internal application proposals exist. `archived` hides the capture from normal surfaces without deleting it.

Related captures use deterministic FTS/LIKE search, category/source matching, keyword overlap, and recency. v3 does not use embeddings.

## Explicit Gateway Capture

When enabled, v1 captures only explicit trigger prefixes:

```text
remember this:
remember this <space>
save this:
save this thought:
note this:
business idea:
investing thought:
health note:
life thought:
book note:
podcast idea:
```

The match is prefix-only and case-insensitive after leading whitespace. Ordinary chat and non-Wisdom slash commands are not captured by this deterministic gateway intercept.

This v1 intercept remains for low-friction explicit capture. Natural-language use happens through model tool calls, not through expanding this deterministic router.

## Configuration

Primary configuration lives in `config.yaml`:

```yaml
wisdom:
  enabled: true
  db_path: ""
  capture_mode: explicit
  max_results: 5
  interpret_timeout_seconds: 5
  interpretation:
    mode: deterministic
  application:
    mode: deterministic   # deterministic | llm
    timeout_seconds: 30
```

Environment variables are accepted as runtime/test overrides:

```text
HERMES_WISDOM_ENABLED
HERMES_WISDOM_DB_PATH
HERMES_WISDOM_CAPTURE_MODE
HERMES_WISDOM_MAX_RESULTS
HERMES_WISDOM_INTERPRET_TIMEOUT
HERMES_WISDOM_INTERPRETATION_MODE
HERMES_WISDOM_APPLICATION_MODE
HERMES_WISDOM_APPLY_TIMEOUT
```

V1 supports `off` and `explicit` capture modes. Smart capture is intentionally not implemented.

## Interpretation and Applications

Capture never calls an LLM.

`/wisdom interpret <id>` creates a deterministic interpretation when none exists. The interpretation is conservative and includes a counterpoint.

`/wisdom apply <id>` and `wisdom_apply` create internal application proposals only and mark the capture as applied. They do not create Hermes todos, reminders, files, calendar entries, Telegram messages, or old productivity DB rows.

By default, application proposals are deterministic. If you set:

```yaml
wisdom:
  application:
    mode: llm

auxiliary:
  wisdom_apply:
    provider: auto
    model: ""
```

Wisdom asks Hermes' auxiliary LLM router for proposal drafts using the
`auxiliary.wisdom_apply` slot. In a Codex-backed setup, `provider: auto` can
route through the active Codex/main-model path. The model output is accepted
only if it is valid JSON, uses allowed application types, covers the required
proposal set for the capture category, has bounded non-secret text, and creates
internal proposals only. Invalid output, provider errors, timeouts, or missing
credentials fall back to deterministic templates.

v4 deterministic application templates are domain-aware:

- business/x10x: client language, operating principles, report/process proposals
- investing: investment rules, checklists, risk heuristics, decision rules
- health: small experiments, personal rules, tracking questions, decision boundaries
- life: principles, reflection prompts, writing seeds, decision rules

Examples:

```text
Capture: Clients don't need rear-view mirrors, they need windshields.

Client language:
"This review is not just a record of what happened. Its job is to help decide what should be done next."

Principle:
Client reporting should reduce decision uncertainty, not merely describe past performance.

Task proposal:
Add a "What this means now" section to client reports.
```

```text
Capture: I confuse a good thesis with a good position size.

Investment rule:
Do not size a position based only on thesis confidence. Size it based on survivability, liquidity, downside path, and forced-exit risk.

Checklist:
1. What loss can I survive?
2. What adverse move breaks the trade?
3. What forces exit?
4. Is liquidity adequate?
5. Am I sizing by conviction or survivability?
```

For richer wording, the model can use the stored proposal and produce a natural
response. Durable writes still go through the deterministic Wisdom kernel.

## Using Wisdom in Telegram

Use natural language first:

```text
Remember this: ...
Podcast note: ...
Find that idea about ...
Show exact wording for #...
What should I review?
Turn #12 into client language for x10x.
Make #15 an investment checklist.
Accept that.
Dismiss that.
```

Use `/wisdom` commands when you want explicit control or debugging:

```text
/wisdom status
/wisdom capture <text>
/wisdom search <query>
/wisdom original <id>
/wisdom apply <id>
/wisdom review
```

Wisdom should fail open. If a normal Telegram message is not a Wisdom command or
explicit capture, Hermes continues the normal chat path.

## 7-Day Wisdom Trial

Use this to decide whether Wisdom is helping rather than becoming a notes pile:

1. Capture 20 real ideas in business, investing, health, or life.
2. Review every 2-3 days.
3. Accept or dismiss aggressively.
4. Apply 3-5 ideas into client language, rules, checklists, experiments, or decision rules.
5. Note whether Hermes uses Wisdom tools naturally from Telegram.
6. Note whether application outputs are genuinely useful or generic.
7. Keep using it only if the applied ideas change decisions or communication.

## Gateway Behavior

Wisdom is wired as a normal built-in gateway command and a small post-auth, pre-agent explicit capture intercept. v2 does not add a new gateway router; it relies on the existing AIAgent tool loop.

If Wisdom fails while handling ordinary non-command text, Hermes continues to the normal chat path. If a `/wisdom` command fails, the user receives a concise error.

The running launchd gateway must be restarted after deploying new Wisdom code before it loads the changed tool module.

Safe restart command:

```bash
launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway
```

## Known Limitations

- No smart capture.
- No auto-capture of every conversation.
- No voice transcription capture.
- No scheduled reviews.
- No embeddings or dashboard.
- No old productivity migration.
- No external task/reminder execution.
- No automatic review pings.
- No theme graph.
- No deletion workflow.
- No live model-call tests are required for the tool layer; tests prove registration, schemas, dispatcher behavior, and deterministic tool results.
- `period` and `context` tool inputs are lightweight guidance for review/reply ergonomics, not new schedulers or creative DB writers.

## Future Ideas

- Smart capture mode.
- Voice note transcript capture.
- Topic-specific capture modes.
- Weekly and monthly reviews.
- Skeptic/challenge mode.
- Theme graph and recurring patterns.
- First-class principles and checklists.
- Source-backed task/reminder execution.
- Optional import from old productivity data after source recovery.
- Apple Notes, Readwise, Notion, or Obsidian import/export.
- Semantic embeddings and dashboard/export.
