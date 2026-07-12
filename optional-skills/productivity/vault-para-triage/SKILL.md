---
name: vault-para-triage
description: Capture an Obsidian inbox into a canonical capture event log, stage PARA projections for downstream stores, and learn from Slack feedback.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [obsidian, para, vault, notes, cron, slack, routing, frontmatter]
    category: productivity
    related_skills: [obsidian, slack-surfaces]
---

# Vault PARA Triage

Use this optional skill when the user wants Hermes to clean up an Obsidian inbox
into a PARA-organized note system on a schedule, while keeping a canonical
capture event log, downstream staged projections, an audit trail, and a review
loop.

This skill ships a helper script, `scripts/vault_para_triage.py`, backed by the
shared `hermes_cli.vault_para_triage` module.

## What it does

For each markdown note in the configured inbox folder:

1. ensures YAML frontmatter exists
2. classifies the note into a canonical PARA target folder
3. writes canonical markdown content into a capture event log
4. stages one projected file per enabled downstream store
5. archives the original inbox note into the capture root
6. writes immutable audit and status records
7. marks uncertain routes for review and learns from later feedback

The review loop is designed to pair with the bundled `vault-triage-feedback`
plugin, which exposes `/para-feedback` inside Slack and other chat surfaces.

## Default paths

- Vault path: `OBSIDIAN_VAULT_PATH`, otherwise `~/Documents/Obsidian Vault`
- Config file: `<vault>/.hermes/para-triage.yaml`
- Audit/state root: `<vault>/.hermes/para-triage/`
- Capture root: `<vault>/.hermes/note-capture/`

The capture root is the canonical handoff point for external sync jobs. Hermes
does not write directly into the projected vault or second-brain destinations.

## Locate the helper script

```bash
SCRIPT="$(find ~/.hermes/skills -path '*/vault-para-triage/scripts/vault_para_triage.py' -print -quit)"
```

If `SCRIPT` is empty, install the skill first.

## Install

```bash
hermes skills search vault-para-triage
hermes skills install official/productivity/vault-para-triage
```

Enable the feedback plugin so `/para-feedback` is available:

```bash
hermes plugins enable vault-triage-feedback
```

## Minimal config

Create `<vault>/.hermes/para-triage.yaml`:

```yaml
inbox_dir: Inbox
para_roots:
  projects: Projects
  areas: Areas
  resources: Resources
  archives: Archives
review_dir: Resources/_Inbox Review
capture:
  root: .hermes/note-capture
  source_archive_dir: source-archive
projection:
  stores:
    vault:
      enabled: true
      path_prefix: ""
    second_brain:
      enabled: true
      path_prefix: ""
routing:
  min_confidence: 0.72
  low_confidence_action: review   # one of: review, inbox, auto-file
  rules:
    - name: health-notes
      match_any: ["health", "doctor", "medication"]
      target: Areas/Health
      confidence: 0.95
      reason: Health-related notes belong in Areas/Health.
    - name: taxes
      match_any: ["tax", "hmrc", "invoice"]
      target: Areas/Finance
      confidence: 0.95
```

Add more explicit rules as the vault evolves. The helper also stores learned
examples from reviewer corrections under `.hermes/para-triage/routing_examples.json`.

Projection rules:

- Hermes first decides one canonical logical `target`, such as `Projects/Acme`
  or `Resources/Content Library`.
- For each enabled store, Hermes derives the projected relative path as
  `<path_prefix>/<target>/<filename>`.
- Hermes stages files under
  `.hermes/note-capture/staging/<store>/<entry_id>/<projected-relative-path>`.
- Hermes archives the original note under
  `.hermes/note-capture/source-archive/<entry_id>/<original-relative-path>`.
- Hermes records the canonical event under
  `.hermes/note-capture/events/<entry_id>.json`.
- External sync outside the Hermes trust boundary is responsible for projecting
  staged files into the live vault, iCloud, second brain, or other stores.

Low-confidence behavior:

- `review` — stage uncertain notes to `review_dir`
- `inbox` — keep uncertain notes mapped to their inbox-relative location
- `auto-file` — still stage uncertain notes to the predicted target, then rely on Slack corrections

## Dry-run and status

Preview the next triage pass:

```bash
python3 "$SCRIPT" --vault "$OBSIDIAN_VAULT_PATH" run --dry-run
```

Inspect audit and review state:

```bash
python3 "$SCRIPT" --vault "$OBSIDIAN_VAULT_PATH" status
python3 "$SCRIPT" --vault "$OBSIDIAN_VAULT_PATH" feedback list --limit 20
```

Inspect canonical capture status:

```bash
cat "$OBSIDIAN_VAULT_PATH/.hermes/note-capture/status/latest.json"
```

## Overnight cron job

Use a script-only cron job so the canonical capture logic runs the same way
every night and the output can be delivered directly to Slack.

First install a cron-safe launcher into `~/.hermes/scripts/`:

```bash
python3 "$SCRIPT" --vault "$OBSIDIAN_VAULT_PATH" install-wrapper
```

This prints the generated path, usually:

```text
~/.hermes/scripts/vault-para-triage-nightly.py
```

Then schedule that generated wrapper:

```bash
hermes cron create "0 2 * * *" \
  --name "Vault PARA triage" \
  --deliver slack:C0B6MV5RA06 \
  --script ~/.hermes/scripts/vault-para-triage-nightly.py \
  --no-agent
```

For this rollout, `slack:C0B6MV5RA06` is the explicit `#hermes-ops` channel
target. Using the raw Slack channel ID is safer than relying on name
resolution.

If the scheduler environment on your machine does not preserve
`OBSIDIAN_VAULT_PATH`, the generated wrapper already bakes in the absolute vault
and config path you passed to `install-wrapper`, so cron does not need that env
var later.

If you want a different filename or output mode:

```bash
python3 "$SCRIPT" --vault "$OBSIDIAN_VAULT_PATH" install-wrapper \
  --name para-nightly-review.py \
  --format slack
```

## Slack feedback loop

After the nightly run posts to Slack, review uncertain entries with:

```text
/para-feedback list
/para-feedback approve <entry_id>
/para-feedback correct <entry_id> Areas/Health
/para-feedback ignore <entry_id>
/para-feedback status
```

For an on-demand audit summary, use:

```bash
python3 "$SCRIPT" --vault "$OBSIDIAN_VAULT_PATH" status
```

`approve` and `correct` both teach future runs. `correct` rewrites the staged
projection targets for the current capture event to the supplied canonical
target path.

## Notes

- The helper remembers the vault structure by snapshotting available PARA
  folders on every run into `.hermes/para-triage/structure.json`.
- Canonical captures are logged under `.hermes/note-capture/events/`.
- Projected store health is summarized in `.hermes/note-capture/status/latest.json`.
- All capture audit rows are logged to `.hermes/para-triage/audit.jsonl`.
- Reviewer actions are logged separately to `.hermes/para-triage/feedback.jsonl`.
- This keeps the capability outside Hermes core tools while still making it
  schedulable and auditable.
