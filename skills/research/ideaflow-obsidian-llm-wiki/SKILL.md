---
name: ideaflow-obsidian-llm-wiki
description: "Use when syncing an IdeaFlow/Thoughtstream Knowledge Garden into an Obsidian vault as a fresh LLM Wiki, then maintaining it from periodic heartbeats."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ideaflow, thoughtstream, obsidian, llm-wiki, knowledge-garden, heartbeat, notes]
    category: research
    related_skills: [llm-wiki, obsidian]
---

# IdeaFlow → Obsidian LLM Wiki

## Overview

This skill bridges an IdeaFlow / Thoughtstream Knowledge Garden with a local Obsidian vault that follows the LLM Wiki pattern: IdeaFlow remains the fast capture and upstream idea stream; Obsidian becomes the durable, inspectable, interlinked markdown knowledge garden maintained by the agent.

Use this skill to create a fresh LLM Wiki from an IdeaFlow instance, ingest new IdeaFlow notes on a heartbeat, and keep the Obsidian vault useful as a persistent knowledge base rather than a pile of exports. The workflow is filesystem-first: every durable artifact is markdown, every ingest has raw provenance, and the agent updates `SCHEMA.md`, `index.md`, and `log.md` so future sessions can resume without rediscovering context.

**Security rule:** never commit or paste IdeaFlow tokens into the wiki, scripts, logs, skill files, GitHub issues, or PRs. Use environment variables such as `IDEAFLOW_ACCESS_TOKEN` and local `.env` files excluded from git.

## When to Use

Use this skill when the user:

- Mentions IdeaFlow, Thoughtstream, IdeaFlow Knowledge Garden, IdeaPocket, or `prod-api.ideaflow.app`.
- Wants IdeaFlow notes mirrored into Obsidian.
- Wants a fresh LLM Wiki based on an IdeaFlow account or export.
- Wants a heartbeat that periodically incorporates new captures into a knowledge garden.
- Wants to query or lint an Obsidian wiki whose upstream source is IdeaFlow.

Don't use this for:

- Generic Obsidian note editing with no IdeaFlow source; use `obsidian`.
- Generic LLM Wiki work with local raw sources only; use `llm-wiki`.
- Posting new notes into IdeaFlow only; this skill can do it, but its main purpose is compiling IdeaFlow into a local wiki.

## Architecture

```
Obsidian vault / LLM Wiki
├── SCHEMA.md
├── index.md
├── log.md
├── raw/
│   └── ideaflow/
│       ├── exports/              # full snapshots or user-provided exports
│       ├── notes/YYYY-MM-DD/     # one immutable markdown file per IdeaFlow note
│       └── heartbeat/            # heartbeat reports and run metadata
├── entities/
├── concepts/
├── comparisons/
├── queries/
└── _meta/
    ├── ideaflow-sync-state.json  # last cursor / ids / hashes
    └── ideaflow-field-map.md     # observed API/export shape and mapping notes
```

**Division of labor:**

- **IdeaFlow:** capture stream, upstream note IDs, source timestamps, optional links/tags.
- **Raw layer:** immutable normalized snapshots of IdeaFlow notes.
- **Wiki layer:** agent-written entity, concept, comparison, and query pages.
- **Schema/log/index:** navigation and operational memory.
- **Heartbeat:** scheduled incremental sync + LLM Wiki ingest.

## Required Configuration

Resolve concrete paths before file operations; do not pass shell variables to file tools.

```bash
# Required for direct API sync. Keep this out of git.
export IDEAFLOW_ACCESS_TOKEN="<set-this-in-your-shell-or-env-file>"

# Defaults shown; override as needed.
export IDEAFLOW_API_BASE="https://prod-api.ideaflow.app/v1"
export IDEAFLOW_EXPORT_PATH="$HOME/Downloads/ideaflow-export.json"   # if using export mode
export OBSIDIAN_VAULT_PATH="$HOME/Documents/Obsidian Vault"
export WIKI_PATH="$OBSIDIAN_VAULT_PATH/Knowledge Garden"
```

Recommended local `.gitignore` entries inside the vault if it is a git repo:

```gitignore
.env
_meta/*token*
raw/ideaflow/exports/*.json
raw/ideaflow/exports/*.ndjson
```

## First-Run: Create a Fresh Wiki from IdeaFlow

1. **Load orientation skills:** use this skill together with `llm-wiki` and `obsidian`.
2. **Resolve the vault/wiki path:** use `OBSIDIAN_VAULT_PATH` or `WIKI_PATH`; default to `~/wiki` only if the user has no Obsidian vault preference.
3. **Create the wiki skeleton:** `SCHEMA.md`, `index.md`, `log.md`, `raw/ideaflow/`, `entities/`, `concepts/`, `comparisons/`, `queries/`, `_meta/`.
4. **Discover the IdeaFlow source shape:** prefer an export or read-only API call before writing anything.
5. **Normalize IdeaFlow notes into raw markdown:** one immutable file per upstream note, with source frontmatter.
6. **Compile the first wiki pass:** identify recurring entities/concepts, create durable pages, link them with `[[wikilinks]]`, and update navigation.
7. **Write sync state:** record processed upstream IDs and content hashes in `_meta/ideaflow-sync-state.json`.
8. **Verify:** run the lint checks from `llm-wiki`; ensure no raw token values appear in files.

### Starter SCHEMA.md Additions

Use the normal `llm-wiki` schema template, then add this IdeaFlow-specific section:

```markdown
## IdeaFlow Source Policy

This wiki is compiled from IdeaFlow / Thoughtstream captures. IdeaFlow notes are treated as raw source material unless explicitly promoted into wiki synthesis.

- Raw IdeaFlow files live under `raw/ideaflow/notes/YYYY-MM-DD/`.
- Never edit raw IdeaFlow files after ingest; add corrections to wiki pages instead.
- Preserve upstream IDs, created/updated timestamps, URL, tags, and hash in frontmatter.
- Create wiki pages only for recurring or central ideas; do not create pages for every capture.
- When a capture conflicts with an existing page, preserve both claims with dates and source links.
- Heartbeat runs may update wiki pages automatically, but must append to `log.md` and update `index.md`.
- Private captures stay local. Do not publish the raw layer without explicit user approval.
```

### Raw IdeaFlow Note Format

```markdown
---
source: ideaflow
ideaflow_id: <upstream-note-id>
source_url: <ideaflow-url-if-available>
title: <title-or-first-line>
created: YYYY-MM-DDTHH:MM:SSZ
updated: YYYY-MM-DDTHH:MM:SSZ
ingested: YYYY-MM-DDTHH:MM:SSZ
tags: [ideaflow, capture]
sha256: <hash-of-body>
---

<normalized markdown body>
```

Compute `sha256` over the body below the frontmatter. If the same upstream ID appears again with a different hash, keep the original raw file and add a new version file such as `<slug>--v2.md`, then let the wiki layer reconcile the change.

## IdeaFlow Data Access Modes

IdeaFlow deployments and clients have changed over time. Pick the safest available mode.

### Mode A — User Export (preferred when available)

Ask the user to export IdeaFlow/Thoughtstream notes as JSON, NDJSON, CSV, or markdown. Then:

1. Save the export under `raw/ideaflow/exports/` if the user approves keeping it locally.
2. Inspect the fields without printing sensitive content.
3. Normalize each record into raw note markdown.
4. Process only new or changed records based on upstream ID + content hash.

This mode is robust and avoids depending on private API routes.

### Mode B — Direct API Read (when the user provides a token)

Use only read/list endpoints unless the user explicitly asks to create or update IdeaFlow notes. The IdeaPocket extension has used:

- `IDEAFLOW_API_BASE=https://prod-api.ideaflow.app/v1`
- `Authorization: Bearer $IDEAFLOW_ACCESS_TOKEN`
- `POST /notes/top` for creating a top-level note

Read/list endpoints may be deployment-specific. Discover them defensively:

```bash
BASE="${IDEAFLOW_API_BASE:-https://prod-api.ideaflow.app/v1}"
for path in /users/me /me /notes /notes/top /search; do
  code=$(curl -sS -o /tmp/ideaflow-probe.json -w '%{http_code}' \
    -H "Authorization: Bearer $IDEAFLOW_ACCESS_TOKEN" \
    -H 'Accept: application/json' \
    "$BASE$path")
  printf '%s %s\n' "$code" "$path"
done
```

Do not paste response bodies into chat unless needed; summarize shapes and redact personal content. Record stable field mapping notes in `_meta/ideaflow-field-map.md`.

### Mode C — Local Capture Folder

If IdeaFlow syncs to a local folder, treat that folder as the upstream export. Use filesystem tools to list and read files, then normalize them into `raw/ideaflow/notes/` before touching wiki pages.

## Normalization Recipe

Use `execute_code` when normalizing many records so large exports do not flood context.

```python
from pathlib import Path
import json, hashlib, re, datetime

wiki = Path("/absolute/path/to/Knowledge Garden")
export = Path("/absolute/path/to/ideaflow-export.json")
out = wiki / "raw" / "ideaflow" / "notes"
state_path = wiki / "_meta" / "ideaflow-sync-state.json"
out.mkdir(parents=True, exist_ok=True)
state_path.parent.mkdir(parents=True, exist_ok=True)
state = json.loads(state_path.read_text()) if state_path.exists() else {"processed": {}}

def slugify(s):
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (s or "note").lower()).strip("-")
    return s[:80] or "note"

def body_hash(body):
    return hashlib.sha256(body.encode("utf-8")).hexdigest()

def frontmatter(record, body, ingested):
    title = (record.get("title") or record.get("text", "")[:80]).replace("\n", " ")
    return "\n".join([
        "---",
        "source: ideaflow",
        f"ideaflow_id: {record.get('id', '')}",
        f"source_url: {record.get('url', '')}",
        f"title: {json.dumps(title)[1:-1]}",
        f"created: {record.get('createdAt') or record.get('created_at') or ''}",
        f"updated: {record.get('updatedAt') or record.get('updated_at') or ''}",
        f"ingested: {ingested}",
        "tags: [ideaflow, capture]",
        f"sha256: {body_hash(body)}",
        "---", "",
    ])

records = json.loads(export.read_text())
if isinstance(records, dict):
    records = records.get("notes") or records.get("data") or records.get("items") or []

ingested = datetime.datetime.now(datetime.UTC).isoformat()
new_files = []
for rec in records:
    body = rec.get("markdown") or rec.get("content") or rec.get("text") or rec.get("body") or ""
    if not body.strip():
        continue
    hid = body_hash(body)
    rid = str(rec.get("id") or hid[:12])
    if state["processed"].get(rid) == hid:
        continue
    title = rec.get("title") or body.splitlines()[0][:80]
    day = (rec.get("createdAt") or rec.get("created_at") or ingested)[:10]
    dest_dir = out / day
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(title)}--{rid[:8]}.md"
    dest.write_text(frontmatter(rec, body, ingested) + body + "\n", encoding="utf-8")
    state["processed"][rid] = hid
    new_files.append(str(dest))

state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"new_files": new_files, "count": len(new_files)}, indent=2))
```

After normalization, run the `llm-wiki` ingest operation over the new raw files as a batch.

## Heartbeat Workflow

The heartbeat should be idempotent, quiet when nothing changed, and explicit when it changes the wiki.

### Heartbeat Prompt

Use this as the scheduled job prompt or Obsidian Agent Fleet `HEARTBEAT.md` body:

```markdown
Sync the IdeaFlow Knowledge Garden into the Obsidian LLM Wiki.

1. Resolve WIKI_PATH and OBSIDIAN_VAULT_PATH.
2. Read SCHEMA.md, index.md, and the last 30 entries of log.md.
3. Pull or load new IdeaFlow notes since `_meta/ideaflow-sync-state.json`.
4. Normalize new notes into `raw/ideaflow/notes/YYYY-MM-DD/` with hashes.
5. If no new or changed notes exist, append nothing and report "IdeaFlow heartbeat: no changes".
6. If new notes exist, batch-ingest them using the LLM Wiki rules:
   - update existing entity/concept pages before creating new pages;
   - create pages only for central or recurring ideas;
   - add `[[wikilinks]]`, provenance markers, and confidence where useful;
   - update index.md and log.md;
   - run quick lint for broken links and missing index entries.
7. Report the count of raw notes imported and every wiki file created/updated.
8. Do not reveal tokens or raw private note bodies in the final report.
```

### Hermes Cron Job

Create the job only after the wiki has been initialized and the environment variables are present for the scheduler profile.

```python
cronjob(
  action="create",
  name="ideaflow-obsidian-llm-wiki-heartbeat",
  schedule="0 */6 * * *",
  skills=["ideaflow-obsidian-llm-wiki", "llm-wiki", "obsidian"],
  enabled_toolsets=["file", "terminal", "web"],
  prompt="""Sync IdeaFlow into the Obsidian LLM Wiki using the ideaflow-obsidian-llm-wiki heartbeat workflow. Keep secrets redacted. Only report changed files and issues."""
)
```

If the sync logic becomes stable, move data collection into a script and let the agent handle only the wiki synthesis. Use `no_agent=True` only for pure notifications; wiki maintenance needs reasoning.

## Compiling IdeaFlow Captures into Wiki Pages

For each batch of new raw IdeaFlow notes:

1. Read the batch and identify high-signal themes, entities, decisions, open questions, and repeated concepts.
2. Search existing wiki pages and `index.md` for each candidate before creating anything.
3. Prefer updating existing pages; create new pages only when the idea is central or appears repeatedly.
4. Preserve the human's phrasing as a quote only when the wording matters; otherwise synthesize.
5. Add provenance markers that point back to raw IdeaFlow note files.
6. Update backlinks: if a page links to a newly created page, add a reciprocal contextual link when useful.
7. Update `index.md` one time at the end of the batch.
8. Append one `log.md` entry with counts and file lists.

Suggested wiki page types for IdeaFlow gardens:

- `concept`: recurring ideas, frameworks, principles, hypotheses.
- `entity`: people, products, projects, organizations, codebases.
- `comparison`: tradeoffs, tool choices, competing theories.
- `query`: synthesized answers to prompts asked against the garden.
- `summary`: periodic heartbeat digests or weekly synthesis.

## Optional: Write a Digest Back to IdeaFlow

Only do this when the user explicitly asks for IdeaFlow writeback. The IdeaPocket client has used `POST /notes/top` with a note object containing `id`, `authorId`, `tokens`, `updatedAt`, `insertedAt`, and related fields. Because write APIs may change, prefer a dry run first and require a known user ID.

A safe digest note should contain only:

- heartbeat timestamp,
- count of notes ingested,
- links or titles of public/non-sensitive wiki pages,
- no raw private note bodies,
- no access tokens or local file paths unless the user wants them.

## Lint and Health Checks

Run these after the initial import and periodically:

- No token-like strings in wiki files: search for `IDEAFLOW_ACCESS_TOKEN`, `Authorization: Bearer`, and token prefixes.
- Every wiki page appears in `index.md`.
- Every new/updated wiki page has required frontmatter.
- Every page has at least two useful wikilinks unless it is a raw source.
- No broken `[[wikilinks]]`.
- No raw IdeaFlow file hash drift unless a new version was intentionally created.
- No heartbeat touched 10+ existing wiki pages without user approval, unless the job was explicitly authorized for broad maintenance.
- `log.md` has exactly one entry for each non-empty heartbeat run.

## Common Pitfalls

1. **Leaking the token.** Never write the token into markdown, logs, commits, PR descriptions, or terminal output. Use environment variables and redact command output.
2. **Treating every capture as a page.** IdeaFlow is high-volume capture; the wiki should compile durable concepts, not mirror every thought into the graph.
3. **Skipping orientation.** Always read `SCHEMA.md`, `index.md`, and recent `log.md` before a heartbeat ingest.
4. **Depending on unstable API routes.** Prefer exports or discovered read endpoints. The known public client evidence is strongest for `POST /notes/top`; read routes may differ by deployment.
5. **Editing raw files.** Raw IdeaFlow snapshots are immutable. Version changed notes instead of overwriting.
6. **No sync state.** Without `_meta/ideaflow-sync-state.json`, heartbeats reprocess the same notes and duplicate pages.
7. **Publishing private captures.** A public GitHub wiki should normally exclude `raw/ideaflow/` unless the user explicitly approves public release.
8. **Heartbeat noise.** If nothing changed, say so briefly; do not append empty log entries or rewrite files.

## Verification Checklist

- [ ] `WIKI_PATH` resolves to a concrete directory inside or alongside the Obsidian vault.
- [ ] `SCHEMA.md`, `index.md`, `log.md`, `raw/ideaflow/`, and `_meta/` exist.
- [ ] `_meta/ideaflow-sync-state.json` records processed IDs and hashes.
- [ ] New raw notes have IdeaFlow frontmatter and body hashes.
- [ ] Wiki pages synthesize, cross-link, and cite raw files; they are not raw dumps.
- [ ] `index.md` and `log.md` were updated for non-empty runs.
- [ ] Lint reports no broken links, missing index entries, or leaked secrets.
- [ ] Public GitHub artifacts exclude credentials and private raw captures unless explicitly approved.
