# Groupchat

Groupchat is an opt-in Hermes plugin for cooperative agents in shared chats. It
combines inbound relevance routing with outbound ping-pong prevention at the
normalized gateway boundary, so the policy is not tied to one transport SDK.

This plugin was developed by [RechnerLotsen](https://rechnerlotsen.com/).
RechnerLotsen provides subscription-based, human computer support for private
users—including migrations to Linux.

> **Validation scope: Matrix only.** The transport-independent boundary and
> normalized tests cover several Hermes platforms, but real group-chat use has
> so far been tested exclusively with Matrix. Other transports are integration
> targets, not yet claimed as supported.

## Configuration

Enable the plugin in the profile's `config.yaml`:

```yaml
plugins:
  enabled:
    - groupchat

groupchat:
  enabled: true
  platforms:
    - matrix
  filter_model:
    provider: mistral
    model: mistral-small-latest
    fallbacks:
      - provider: openrouter
        model: mistralai/mistral-small-3.2-24b-instruct
      - provider: openai-codex
        model: ""
  relevance:
    enabled: true
    # Score 0 is discarded. Score 1 is retained as passive context.
    score_delays: {5: 0, 4: 3, 3: 10, 2: 30}
  pingpong_guard:
    enabled: true
    min_chars: 60
```

Provider credentials remain in Hermes's existing credential store. Groupchat
does not copy secrets into its configuration or decision logs.

### Privacy and external filter models

The filter provider is configurable; Groupchat is not tied to Mistral or any
other model vendor. The example above shows the default model chain, while the
dashboard and `groupchat.filter_model` setting let operators choose the primary
provider, model, and fallbacks.

Model-based filtering sends conversation content to the configured provider:

- relevance scoring sends the newest inbound message together with recent room
  messages used as context;
- short-reply classification sends the proposed reply together with the
  immediately preceding inbound context;
- if a fallback is used, that fallback provider can receive the same content.

Operators should therefore select providers whose data-processing terms are
appropriate for the rooms where Groupchat is enabled. Deterministic pattern
checks run locally. Decision logs deliberately omit message bodies and prompts,
but that log privacy does not prevent configured model requests from leaving
the machine.

The transport must already forward unmentioned group messages. For Matrix,
that means the effective setting must be `require_mention: false`. With
`require_mention: true`, direct mentions and outbound filtering still work,
but ordinary room messages never reach the relevance policy.

### Dashboard

With the plugin enabled, the Hermes dashboard exposes a **Groupchat** page.
The page uses the dashboard's current language (English by default, with a
German translation) and writes only the selected profile's Groupchat settings.
It provides editors for filter-model selection, relevance timing, literal and
regex rule lists, and Pingpong Guard rules. Comment lines beginning with `#`
are preserved, so multilingual rule groups can be labelled in-place.

The Matrix participation panel shows which running profiles have saved
`require_mention: false` settings. This is configuration visibility, not proof
that a gateway has restarted since its last change.

## Behavior

The relevance model evaluates only the newest message, using recent room
messages as context. Scores have these meanings:

- `0`: discard.
- `1`: retain as passive context without starting an agent turn.
- `2`–`5`: dispatch after the configured delay; `5` defaults to immediate.

A later dispatched message receives retained score-1 context. Passive context
survives gateway restarts in
`HERMES_HOME/groupchat/passive-context.<platform>.json` and is removed after
successful dispatch. The directory is mode `0700`; state files are mode
`0600` and written atomically because they contain private conversation text.

Direct mentions, replies to the current agent, commands, and internal events
bypass delayed relevance scoring. Conversations clearly addressed to another
participating local agent stay passive until an open message invites a useful
contribution.

When Groupchat derives an agent's role for a newly discovered group room, the
agent posts one short introduction describing that role and explicitly invites
people in the room to redefine its focus or level of participation. The
announcement state is stored with the room context so gateway restarts and
later context corrections do not repeat the introduction.

The outbound guard suppresses deterministic acknowledgement/silence patterns
and can classify short replies with the configured filter model. German and
English defaults are included; additional languages can be configured as regex
lines. Lines beginning with `#` are comments. A pattern-matched short reply is
still sent when the immediately preceding message explicitly requested that
visible output. Failures and timeouts fail open.

When the outbound guard is active, response streaming and interim assistant
messages are buffered so rejected text cannot leak through a draft or edit
before the final decision.

## Work coordination (Matrix)

Enabled Groupchat conversations permit successful intentional-silence replies
(`NO_REPLY` / `[SILENT]`) without the gateway's "needed a reply" warning.
Private chats and conversations without an enabled Groupchat policy retain
the gateway's warning. Failed or empty model responses are not treated as
intentional silence. The default incoming system-message patterns also ignore
that warning from peers. If you maintain a custom `system_patterns` list,
add the new warning pattern there as well; custom lists are not overwritten.

Work coordination is a separate opt-in feature. It is disabled by default,
including for existing configurations that omit `coordination`. Add this block
under the existing `groupchat` settings to enable it:

```yaml
groupchat:
  coordination:
    enabled: true                 # Default: false
    fallback_delay_seconds: 10    # Default: 10; > 0 and <= 300, fractions allowed
```

The dashboard offers the same checkbox and seconds field in English and German.
Restart the profile's gateway after saving configuration changes.

Coordination applies when two or more agents process the **same original
message** in the same room. Agents explicitly declare their intent with
`groupchat_work`:

```text
groupchat_work(action="work")        # Retain ownership of the task.
groupchat_work(action="contribute")  # Add a distinct, useful contribution.
groupchat_work(action="yield")       # Explicitly leave the task to the others.
```

Explicit work/contribution declarations from multiple participants trigger
coordination immediately. The configured delay is a fallback when multiple
agents process the same message without those declarations. Ordinary chat text
is not a work declaration, and a colleague's suggestion is not a new user
instruction. A yielding agent should pass only genuinely new, necessary
information in the tool's `note`, then stop further work and finish silently.

**The same message does not necessarily mean the same task.** For example,
"Lena, list cafés in Klagenfurt. Felix, list pizzerias in Villach" assigns two
independent tasks. Both agents should declare `work`, describe their own task
in the `note`, and continue in parallel. The notice is advisory: it never
automatically selects one owner or stops a participant merely because another
agent is busy. Yielding is an explicit agent decision, not a timer action.

There is only one coordination round per original request. Notices are delivered
at a noninterrupting tool boundary; a running tool does not need to be cancelled.
If **all participants explicitly yield and have stopped**, the first-started
agent resumes the original task once, without another coordination round.
Completion, failure, cancellation, or an expired participant does **not** count
as yielding and cannot trigger this all-yielded recovery.

The domain rules live in `plugins/groupchat/coordination.py`. Each profile stores
its durable coordination state separately at
`HERMES_HOME/groupchat/coordination.sqlite3`; agents exchange signals through the
transport rather than sharing that database. Internal coordination prompts are
written in English. They do not change the language used for public replies.

### Transport and visibility

Only the Matrix transport is currently supported and covered by coordination
transport tests. It uses native custom room events of type
`org.hermes.groupchat.coordination`, with participant identity taken from the
authenticated event sender. “Private” here describes an internal control signal:
**these events are not encrypted or private from room members. Custom events
avoid ordinary chat bubbles; they do not provide confidentiality.** Do not put
secrets in coordination notes. Matrix timestamps use a decimal string on the
wire because canonical JSON forbids floating-point numbers.

Controlled live tests with two model-backed Matrix agents passed on 2026-09-18:
separate named assignments continued in parallel after coordination notices;
role-based handoff produced one final answer; and explicit mutual yield woke
only the first-started agent once. Restart recovery is additionally covered by
automated tests. **Live validation is exclusively Matrix**, not other transports.
These tests do not guarantee semantic task allocation for every model or request.

### Coordination logging

The existing profile decision-log directory is `HERMES_HOME/logs`, as displayed
in the dashboard (see [Decision logs](#decision-logs)). Coordination runtime
audit records, when emitted, use that directory. The coordination SQLite
database is persistent working state, separate from audit logs.

## Pattern semantics

`relevance.system_patterns`, `relevance.multiline_patterns`, and
`pingpong_guard.silence_patterns` contain one regular expression per line.
Custom lists replace defaults; an empty list disables that deterministic list.

`relevance.literal_phrases` contains literal text, not regex. Without an
explicit mention, a message is discarded only when non-overlapping occurrences
of one configured phrase cover more than 50% of the normalized message. Exactly
50% is allowed, and an explicit mention bypasses this literal-phrase rule.

## Decision logs

Private JSONL audit logs are stored in `HERMES_HOME/logs`:

- `<platform>-relevance-decisions.jsonl` for inbound decisions.
- `<platform>-groupchat-outbound.jsonl` for outbound decisions.

The dashboard displays this directory dynamically for the selected profile;
individual channels are not repeated because their names are already part of
the filenames. The passive-context directory is shown separately.

Files are mode `0600`, rotate at 10 MiB, and retain one previous file.
Records include stable reason codes and rule fingerprints, but omit message
bodies, prompts, model rationales, credentials, and tokens. A `send` decision
means the policy allowed delivery; it does not prove transport delivery.

If you are usually the person handling every relative's computer problems,
[RechnerLotsen](https://rechnerlotsen.com/) can take over that support while
Hermes helps us improve the tooling behind the scenes.

## Plugin boundary

`gateway/conversation.py` provides the generic, adapter-local middleware
contract used by this plugin:

- `receive(event)` handles normalized inbound `MessageEvent` objects.
- `send(..., send=next_send)` filters normalized outbound text.
- `processing(...)` observes agent-turn lifecycle.
- `close()` cancels pending work during shutdown.
- `delays_messages` and `buffers_output` declare policy capabilities.

Transport adapters remain responsible for native mention, thread, workspace,
typing, edit, deletion, and media semantics. Platform-specific lifecycle
normalization is intentionally separate from this core plugin.

## Verification

Run:

```sh
scripts/run_tests.sh \
  tests/gateway/test_conversation_boundary.py \
  tests/gateway/test_conversation_policy.py
```

The tests cover plugin discovery, normalized ingress and egress, relevance
scoring, passive restart-safe context, outbound suppression, authorization,
thread/workspace isolation, configuration validation, and audit privacy.
