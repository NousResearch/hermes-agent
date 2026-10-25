---
name: lark-cli-patterns
description: "Condensed lark-cli usage patterns, pitfalls, and workarounds for docs/wiki/drive operations. Complements the official lark-* skills by capturing session-tested CLI behavior that isn't in the upstream docs."
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: ["lark-cli"]
metadata:
  hermes:
    tags: [lark-cli, feishu, lark, docs, wiki, drive, pitfalls, workarounds]
    related_skills: [lark-doc, lark-wiki, lark-drive, lark-sheets]
---

# lark-cli Patterns & Pitfalls

> This skill captures session-tested CLI behavior for `lark-cli` operations (docs, wiki, drive, sheets). It complements the official `lark-*` skills by documenting pitfalls discovered through real usage that the upstream docs don't call out.
>
> **Rule**: When operating on Feishu/Lark resources, try `lark-cli` first before reaching for a browser or `web_extract`. Load this skill alongside the relevant `lark-*` skill (`lark-doc`, `lark-wiki`, `lark-drive`, `lark-sheets`).

## Subcommand Prefix: `+` is Required

`lark-cli` **shortcut subcommands** need a `+` prefix:

```bash
# ✅ Correct
lark-cli docs +fetch --doc <token>
lark-cli wiki +node-get --node-token <token>
lark-cli drive +download --file-token <token>

# ❌ Wrong: missing +
lark-cli docs fetch      # unknown subcommand
lark-cli wiki node-get   # unknown subcommand
```

## Path Constraint: Relative Paths Only

**`--output`, `--markdown @file`, `--content @file`, and `drive +upload --file` all enforce relative paths**. Absolute paths are rejected with a hint pointing back at this rule:

```bash
# ❌ Absolute path rejected
lark-cli drive +download --output /tmp/file.pdf
lark-cli docs +update --content @/tmp/payload.xml
# → "unsafe output path: must be a relative path within the current directory"
# → "invalid file path ... --file must be a relative path within the current directory"

# ✅ Correct: cd first, then relative path
cd /tmp && lark-cli drive +download --output ./file.pdf
cd /tmp && lark-cli docs +update --content @./payload.xml
```

**For long payloads (multi-KB XML/Markdown)**, write to a local file then `@-reference` it. This avoids shell-escaping hell and lets `lark-cli` stream the file directly:

```bash
# Write payload to a file in cwd, then cd there
cd /root
lark-cli docs +update --api-version v2 \
  --doc <token> --command append --content @./section.xml

# Don't forget to mv/clean up the file afterwards (rm in cwd is fine)
mv ./section.xml /tmp/section.xml.bak
```

## Identity: Explicit `--as user`

Default `--as auto` often resolves to `bot`. For user resources (wiki, docs, drive), always pass `--as user`:

```bash
# ❌ May run as bot, can't see user resources
lark-cli wiki +node-get --node-token <url>

# ✅ Explicit user identity
lark-cli wiki +node-get --node-token <url> --as user
```

## API Version: v1 is Deprecated

`docs +fetch` and `docs +update` default to v1 API and emit deprecation warnings. **Always add `--api-version v2`**:

```bash
# ❌ Deprecated v1 warning
lark-cli docs +fetch --doc <token>

# ✅ Explicit v2
lark-cli docs +fetch --doc <token> --api-version v2
lark-cli docs +update --doc <token> --mode overwrite --markdown @file.md --api-version v2
```

## Common Error Chain

| Error | Cause | Fix |
|-------|-------|-----|
| `unknown subcommand "get-content"` | Old command name | Use `+fetch` instead |
| `unknown flag "--url"` | Wrong flag name | Use `--node-token` for wiki |
| `unknown flag "--document-token"` | Wrong flag name | Use `--doc` for docs |
| `unknown subcommand "node" for "lark-cli wiki"` | Tried the v2 form `wiki nodes` | Use the `+` shortcut: `wiki +node-list` |
| `unknown flag "--url" for "lark-cli wiki +node-list"` | `+node-list` doesn't take URL | Use `--space-id <NUMERIC_ID>` instead (resolve via `+space-list`) |
| `param err: space_id is not int` | Passed the wiki **token** (e.g. `Vjwmwgyr...`) as `--space-id` | Run `wiki +space-list` to discover the numeric space_id; the wiki URL token is `node_token`, not `space_id` |
| `required flag(s) "type" not set` on `drive +delete` | `+delete` is multi-typed | Add `--type file` (or `docx`, `folder`, `bitable`, `sheet`, `mindnote`, `shortcut`, `slides`) plus `--yes` for high-risk |
| `unsafe file path: --file must be a relative path` | Used absolute path with `drive +upload --file` | `cd` into the file's directory first, use `./filename` (see Path Constraint section) |
| `unsafe output path` | Absolute path in `--output` | `cd` first, use relative path |
| `invalid file path` | Absolute path in `--markdown @...` | `cd` first, use relative path |
| `permission denied` | Running as bot on user resource | Add `--as user` |
| `unknown flag "--overwrite" for "lark-cli api"` | `lark-cli api` lacks `--overwrite` | `rm -f` target file before calling `api --output` |
| HTTP 403 on `drive +download` | Permission gap for that specific token | Fall back to raw API (see below) |

## Subcommand Naming: Prefer `+` Shortcuts

`lark-cli --help` lists both the v2 form (`wiki nodes list`) and the shortcut (`wiki +node-list`). **Always use the `+` shortcut** — it has proper `--as user` handling, sensible JSON output, and is the documented path. The v2 form works but trips on subtle flag differences (e.g. `nodes list` requires `--space-id` as a typed flag, while `+node-list` requires it under the subcommand — order matters and v2 is stricter).

| Concept | v2 form (in `--help`) | Shortcut (use this) |
|---|---|---|
| List wiki nodes | `wiki nodes list` | `wiki +node-list` |
| List wiki spaces | `wiki spaces list` | `wiki +space-list` |
| Create wiki node | `wiki nodes create` | `wiki +node-create` |
| Upload file | `drive files upload` | `drive +upload` |
| Delete file | `drive files delete` | `drive +delete` |

## Wiki URL → space_id Resolution

The user gives you a homepage URL like `https://xxx.feishu.cn/wiki/VjwmwgyrCi9Ai8kX0dkc10bln05`. The `VjwmwgyrCi9Ai8kX0dkc10bln05` is a **node_token** (also the root page's `obj_token`), NOT the `space_id`. `+node-create` and `+node-list` want a **numeric** space_id. Two-step resolution:

```bash
# 1. Find the numeric space_id by enumerating
lark-cli wiki +space-list --as user | jq '.data.spaces[] | {name, space_id}'

# 2. Find the root node_token within that space
lark-cli wiki +node-list --space-id <NUMERIC_ID> --as user | jq '.data.nodes[] | select(.node_type == "origin")'
```

If the user has many spaces, filter by name. The root node usually has `title: "首页"` or `"Home"` and `parent_node_token: ""`.

## Drive Download: 403 Fallback via Raw API

When `lark-cli drive +download --file-token <token>` returns HTTP 403, the raw OpenAPI endpoint often still works:

```bash
# ❌ May 403 on some files
cd /tmp && lark-cli drive +download --file-token <token> --output ./file.pdf --as user

# ✅ Fallback: raw API (also needs relative path + cd first)
cd /tmp && rm -f file.pdf
lark-cli api GET "/open-apis/drive/v1/medias/<file-token>/download" --as user --output ./file.pdf
```

Note: `lark-cli api` does **not** support `--overwrite` — manually `rm -f` the file first. The API response includes `content_type` and `size_bytes` on success.

## Drive Download: 403 Fallback via Authcode URL (Wiki-attached PDFs)

When even `lark-cli api GET` 403s (e.g., for PDFs attached inside Wiki docs that `docs +fetch` returns as `<source href="...authcode...">`), extract the **authcode URL** and download it with plain `curl`. This is the only path that works for some Wiki-embedded course-materials PDFs.

```bash
# 1. Fetch the doc, get the <source href="...authcode..."> for the file
lark-cli docs +fetch --api-version v2 --doc <token> --scope section --start-block-id <id> --format pretty > /tmp/section.txt
grep -oE 'href="https://internal-api-drive-stream[^"]+"' /tmp/section.txt

# 2. Download the authcode URL directly with curl
cd /tmp && curl -sL -o file.pdf '<authcode-url>'
```

The authcode URL embeds an `authcode=...` parameter that authorizes the download without further lark-cli auth headers. Plain `curl` works because the URL itself is signed.

## Update Notifications

`lark-cli` JSON output often contains `_notice.update`:

```json
{
  "_notice": {
    "update": {
      "current": "1.0.48",
      "latest": "1.0.53",
      "message": "lark-cli 1.0.53 available..."
    }
  }
}
```

**Don't ignore silently**. After completing the user's current request, proactively offer to update:

```bash
lark-cli update
```

Then remind the user to **restart the AI Agent** to load the latest Skills.

## Parameter Mapping (Common Confusions)

| What you want | Wrong flag | Correct flag |
|---------------|-----------|-------------|
| Pass a wiki URL | `--url` | `--node-token` (accepts URL) |
| Pass a doc token | `--document-token` | `--doc` |
| Pass a file token | `--file` | `--file-token` |
| Download output path | `--output /abs/path` | `--output ./relative` (after cd) |
| Markdown content file | `--markdown /abs/path` | `--markdown @relative` (after cd) |

## Workflow: Read Wiki → Get Doc Content

When given a wiki URL and need to read the document content:

```bash
# Step 1: Get node info (obj_token, obj_type) from wiki URL
lark-cli wiki +node-get --node-token <wiki_url> --as user
# → returns obj_token, obj_type (docx, file, sheet, etc.)

# Step 2a: If obj_type == docx → read with docs +fetch
lark-cli docs +fetch --doc <obj_token> --as user --api-version v2

# Step 2b: If obj_type == file → download with drive +download
cd /tmp && lark-cli drive +download --file-token <obj_token> --output ./file.pdf --as user
```

## Workflow: Find Review / Tutorial exercises inside course materials

When the user asks for *练习题 / Tutorial / Review* from a course wiki or notes page, do **not** assume the visible summary table is the exercises. First locate the actual Review/Tutorial section and then extract the attached PDF or embedded file token.

Recommended sequence:
1. Use `wiki +node-list` or `docs +fetch --scope outline` to locate the Review/Tutorial node or heading.
2. Read that section with `docs +fetch --scope section --detail with-ids`.
3. If the section contains `<figure><source ... mime="application/pdf">`, treat that as the real exercise handout.
4. Download the PDF by the `source` URL when available; if only a token is present, use the token with `drive +download` / `drive +inspect` as appropriate.

Pitfall: a course note may contain a *summary table of exercise IDs* (e.g. `Review1_2 #27-#31 + Tutorial 4`) that points to exercises elsewhere. That table is a navigation index, not the exercise body itself.

## Workflow: Append a Structured Section (XML `append` to existing doc)

When the user asks to **add a new chapter/section to an existing docx**, the right pattern is XML `append` with a local file:

```bash
# 1. Confirm the doc outline first to find existing chapter boundaries
lark-cli docs +fetch --api-version v2 --doc <token> --scope outline --max-depth 2

# 2. Author the new section as an XML file in cwd
#    Use the same XML grammar as the lark-doc skill: <h1>/<h2>, <p>, <table>+<colgroup>+<thead>/<tbody>+<tr>/<th>/<td>,
#    <ul>/<li>, <callout emoji="..." background-color="..." border-color="..."> for highlights.
#    Tags do NOT get escaped; only text content does (& < > " ').

# 3. dry-run first to validate the payload before writing
cd /root
lark-cli docs +update --api-version v2 \
  --doc <token> --command append \
  --content @./section.xml --dry-run

# 4. Real write
lark-cli docs +update --api-version v2 \
  --doc <token> --command append \
  --content @./section.xml

# 5. Verify by re-fetching the outline (revision_id should increment)
lark-cli docs +fetch --api-version v2 --doc <token> --scope outline --max-depth 2
```

**`append` vs `block_insert_after`**:
- `append` = `block_insert_after --block-id -1` (end of doc, no need to know the last block id)
- Use `block_insert_after` when inserting in the middle (e.g. between existing chapters); you need the target block's id from a previous `+fetch --detail with-ids`

**Why XML over Markdown for `append`**:
- XML supports `<callout>` (colored highlight boxes), `<table>` with `<colgroup>` column widths, controlled `<b>/<span text-color>` styling
- Markdown `append` flattens to plain paragraphs and loses structure
- The default `--doc-format xml` is correct for `+update`; only switch to `markdown` when the user explicitly wants plain MD

**Cleanup pattern** (when `rm` is blocked in protected paths): use `mv` to `/tmp` instead of `rm` for the staged payload file after a successful write.

## Workflow: Small Inline Edits with `str_replace`

For correcting a single field inside a table cell, fixing a typo, or rewording one line — use `str_replace` on the **exact** surrounding XML/text. Don't rebuild the whole block.

```bash
# Match the entire <p id="...">...</p> wrapper so the replacement is unique
lark-cli docs +update --api-version v2 \
  --doc <token> --command str_replace \
  --pattern '<p id="doxcnABC...">some old text</p>' \
  --content '<p id="doxcnABC...">some new text</p>'
```

**Pitfalls**:
- XML mode `--pattern` is **inline only** — cannot match across blocks / across `<p>` boundaries. For a multi-line rewrite, use `block_replace` instead.
- The full `<p id="...">...</p>` wrapper is the unique signature. Matching just the inner text may collide with other blocks.
- After `str_replace`, the `<p>` block id **stays the same** (server preserves the block; only the inner text is swapped). Safe to keep going on the same doc.
- `str_replace` with `--content ""` deletes the matched text.

## Workflow: Insert a Callout Right After an Existing Callout

When you want to add a new tier (e.g. "🆗 备选") to a list of ranked callouts (🥇🥈🥉), anchor on the last existing callout's `id`:

```bash
# 1. Fetch the target section to find the anchor callout's id
lark-cli docs +fetch --api-version v2 \
  --doc <token> --scope section \
  --start-block-id <chapter-h2-id> --detail with-ids

# 2. Stage the new callout as XML (emoji + colors from base palette)
cat > ./eq12_note.xml <<'EOF'
<callout emoji="🆗" background-color="light-yellow" border-color="orange">
  <b>备选：some option</b><br/>
  - 关键参数...<br/>
  - 关键风险...
</callout>
EOF

# 3. Insert AFTER the last callout in the group
cd /root
lark-cli docs +update --api-version v2 \
  --doc <token> --command block_insert_after \
  --block-id "doxcnifOW2MsUQSujAyJ1H4Hbwc" \
  --content @./eq12_note.xml
```

**Pitfall**: `--block-id` for `block_insert_after` is the **anchor** (existing block where you want to insert AFTER), not the new content. If you anchor on a callout that already has something after it, the new content goes between them, which is usually what you want for tiered rankings.

## XML Attribute Quoting: Double-Quote Trap Inside `name="..."`

When a tag attribute value (most commonly `name="..."` on `<source>`, `<img>`, `<bookmark>`) needs to contain a literal double-quote, **`"` inside a double-quoted attribute breaks the parser**. Three valid escape paths; pick the one your content needs:

```xml
<!-- ❌ Broken: unescaped " inside "..." -->
<img name="The "ATX" PSU Guide.pdf" src="..."/>

<!-- ✅ Path 1: XML numeric character reference for " -->
<img name="The &#x22;ATX&#x22; PSU Guide.pdf" src="..."/>

<!-- ✅ Path 2: Unicode escape \u0022 (only when you author XML via Python strings) -->
<img name="The \u0022ATX\u0022 PSU Guide.pdf" src="..."/>

<!-- ✅ Path 3: Switch quoting style if the attribute allows single quotes -->
<!-- (XML spec allows either; lark-cli accepts both) -->
<img name='The "ATX" PSU Guide.pdf' src="..."/>
```

**When does this fire?**  Anytime the file or label you want to reference contains a quote, e.g. `Module "Final" Review.pdf`, `Dr. Smith's notes`, etc. The catalog of escape-only characters in XML text content is `& < >` (and `"` and `'` if the content is inside an attribute). See the `lark-doc` skill's XML reference for the full rule.

**When authoring in Python** (e.g. an f-string building the XML), use `\u0022` (not `\"`) for `"` inside an attribute — Python won't interpret `\"` inside a string that uses double quotes as the string delimiter, but `\u0022` is unambiguous and survives any later refactor.

## Watchdog Cron: `no_agent=true` for Pure-Script Jobs

For scheduled jobs that are pure shell/Python (no reasoning required — backup, sync, log rotation, threshold alerts), use the cron tool with `no_agent=true` and `script=<relative path under ~/.hermes/scripts/>`. The script's stdout **is** the message; empty stdout = silent (no notification). This avoids burning tokens on an LLM loop every tick.

**Anti-pattern**: for a daily mirror script, do NOT use the LLM-driven `prompt=<long instructions>` form. Wasted tokens on a job that doesn't need reasoning, the LLM would re-derive classification logic that the script already encodes, and a misformatted prompt silently no-ops while a script that crashes loudly tells you.

**Pitfall — "no changes" means "no unpushed commits", not "clean working tree".** A naive `git status --porcelain` check returns clean after a successful commit, even if that commit has not been pushed yet. Use `git rev-list --count origin/main..main` to detect locally-committed-but-not-pushed state.

**Pitfall — quiet-mode should still alert on push failure.** Don't silence errors. If `git push` exits non-zero, the cron job should report the truncated stderr so the user sees something went wrong.

## XML Table Column-Width Pattern

When tables need explicit column widths (e.g. dense comparison tables where some columns are narrow labels and others are wide descriptions), use `<colgroup>` + `<col width>`:

```xml
<table>
  <colgroup>
    <col span="1" width="180"/>   <!-- col 1: 180px -->
    <col span="1" width="120"/>   <!-- col 2: 120px -->
    <col span="1" width="600"/>   <!-- col 3: 600px (the long one) -->
  </colgroup>
  <thead>
    <tr><th>Brand model</th><th>Ports</th><th>Storage</th></tr>
  </thead>
  <tbody>
    <tr><td><b>Model A</b></td><td>2× 2.5GbE</td><td>2× 2.5"</td></tr>
  </tbody>
</table>
```

**Notes**:
- `span="1"` is the default; you can write `<col width="180"/>` and skip `span`.
- Widths are in pixels; they are not strictly enforced on narrow screens but the proportional layout holds.
- First row uses `<th>` inside `<thead>`; remaining rows use `<td>` inside `<tbody>`.
- Emoji cells (✅ ❌ 🆗 ⭐) render correctly inside `<td>`.

## Supported `callout` Attributes

```xml
<callout emoji="💡" background-color="light-blue" border-color="blue" text-color="red">
  ...body...
</callout>
```

- `emoji`: any Unicode emoji; common ones render (🏆🥈🥉✅❌🆗⚠️💡📝)
- `background-color`: `gray` + `light-{色}` + `medium-{色}` (7 base colors: red/orange/yellow/green/blue/purple/gray)
- `border-color`: 7 base colors (no `light-` prefix variant)
- `text-color`: 7 base colors (no `light-` prefix)
- Body supports `<b>`, `<br/>`, inline `<span text-color>`, and `<ul>/<li>` blocks.

## Dry-Run Echo Verbosity

`--dry-run` prints the **full request body** to stdout, including the entire XML payload. For a 9KB section this is ~7KB of output. Acceptable for one-shot writes, but if you're iterating on a big payload, capture the output to a file:

```bash
cd /root
lark-cli docs +update --api-version v2 \
  --doc <token> --command append \
  --content @./section.xml --dry-run > /tmp/dryrun.log 2>&1
grep -E '"(ok|error|result)"' /tmp/dryrun.log
```

The dry-run always says `"ok": true` if the payload parses — it does NOT validate against the doc's existing schema. So always re-`+fetch` after the real write to confirm `revision_id` incremented and the new section is in the outline.

## Workflow: Create a Markdown Doc in a Wiki Space (v2)

`docs +create --api-version v2` does **not** have `--wiki-space` (that's v1-only). The v2 path is a two-step workflow:

```bash
# Step 1: Create a wiki node (empty docx)
lark-cli wiki +node-create --space-id <SPACE_ID> --title "My Notes" --as user
# → returns node_token, obj_token, url

# Step 2: Write content into the doc
cd /tmp && lark-cli docs +update --api-version v2 \
  --doc <obj_token> --doc-format markdown \
  --command overwrite --content @notes.md --as user
```

**Tips**:
- Use `--command overwrite` for the first write (replaces all content); use `--command append` to add more later
- `--command overwrite` is destructive — for incremental edits, prefer `block_insert_after` or `str_replace`
- First write `--content` to a temp `.md` file, then pass with `@file` to avoid shell escaping hell
- For v1 (deprecated): `docs +create --wiki-space <id> --markdown @file.md` still works but emits deprecation warning
