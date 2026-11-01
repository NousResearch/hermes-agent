---
name: siyuan-markdown-import
description: "Import markdown documents into the SiYuan second brain with FULL content — createDocWithMd API (not the CLI import which creates empty placeholders)."
version: 1.0.0
author: Fuad Al Fajri
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [SiYuan, Markdown, Import, Notes, Second-Brain]
    category: productivity
    related_skills: [obsidian]
---

# SiYuan Markdown Import (Full Content)

The correct way to import markdown files (reports, research, artifacts) into a SiYuan second brain **with full content** — not an empty placeholder.

## When to Use

- User wants to import a markdown file/report into SiYuan
- User asks to archive notes, research, or artifacts into their second brain
- A SiYuan import produced an empty doc (diagnose & fix)

## ⚠️ Why not the CLI import

`SiYuan-Kernel import md --file <folder>` ONLY creates an empty placeholder doc (title "import", `.sy` file 314 bytes, no markdown content). Proven repeatedly — don't use CLI for content.

## Correct Method: `createDocWithMd` API

```bash
python3 - <<'EOF'
import json, urllib.request
conf = json.load(open('<siyuan-workspace>/conf/conf.json'))
token = conf['api']['token']   # API token ≠ accessAuthCode (UI login)!
md = open('/path/report.md').read()
payload = json.dumps({
    "notebook": "<NOTEBOOK_ID>",
    "path": "/report-title",              # relative path, no .md
    "markdown": md
}).encode()
req = urllib.request.Request("http://127.0.0.1:6806/api/filetree/createDocWithMd",
    data=payload, method="POST",
    headers={"Authorization": f"Token {token}", "Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req, timeout=30).read()))  # code: 0 = success
EOF
```

Note: `data` in the response can be a string (not a dict) — don't call `.get()` on it; check `code` only.

## Verification (REQUIRED)

1. `code: 0` in the response.
2. Check the filesystem (more reliable than `listDocsByPath` API which can return `data: null`):
   ```bash
   ls -lt <siyuan-workspace>/data/<NOTEBOOK_ID>/ | head -5
   ```
   - Correct doc: `.sy` file tens of KB (e.g. 46 KB).
   - Empty doc: 314 bytes.
3. Check title + keyword inside the file:
   ```bash
   python3 -c "
   import json
   d = json.load(open('<siyuan-workspace>/data/<NOTEBOOK_ID>/<DOC_ID>.sy'))
   print(d['Properties']['title'])
   print('keyword' in json.dumps(d))"
   ```

## Reorganize Structure: Move Docs from Subfolder → Notebook Root

If the user wants docs directly visible without clicking folders (flat structure), move documents out of the import subfolder via the `moveDocs` API — NOT drag in the UI:

```bash
TOKEN=$(python3 -c "import json; print(json.load(open('<siyuan-workspace>/conf/conf.json'))['api']['token'])")

# 1. Backup first (REQUIRED before move):
cp -a <siyuan-workspace>/data <siyuan-workspace>/data.bak.$(date +%s)

# 2. Get paths of all nested docs (2+ path segments, not notebook root):
curl -s -X POST http://127.0.0.1:6806/api/query/sql -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stmt":"SELECT box, path FROM blocks WHERE type='"'"'d'"'"' ORDER BY box, path"}'

# 3. Move batch per notebook to root (fromPaths = array of paths, toPath = "/"):
curl -s -X POST http://127.0.0.1:6806/api/filetree/moveDocs -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"fromNotebook":"<BOX>","fromPaths":["/subfolder/doc1.sy","/subfolder/doc2.sy"],"toNotebook":"<BOX>","toPath":"/"}'
```

After move:
- **Parent folder becomes empty** (placeholder with 1 empty paragraph, `type='p'`, content `''`) → remove via `removeDocByID` `{"notebook":"<BOX>","id":"<FOLDER_ID>"}`. Verify first it's empty: SQL `SELECT COUNT(*) FROM blocks WHERE parent_id='<FOLDER_ID>' AND type!='d'` = 0.
- **Don't touch folders that still contain documents** — remove only truly empty ones.
- Final verification: SQL `type='d'` per box → no 3+ segment paths; `listDocsByPath` root shows docs directly.

## Graph View Connectivity (Obsidian-like)

Global Graph (`Alt+9`) / Graph View (`Alt+8`) only shows lines when connections exist in the `refs` table. **Key: refs are only created from real block references `((<doc-id> "title"))`** — `[[name]]` via API is NOT counted (stored as plain text, graph stays dots without lines). Tag graphs also need real tag blocks `#topic#` via `appendBlock`, not `setBlockAttrs custom-tags`.

⚠️ **refs are built at kernel boot** — after adding refs/tags, RESTART the kernel (or `systemctl restart siyuan` + restart the socat tunnel if used). Verify: `SELECT COUNT(*) FROM refs` > 0.

Hub pattern: 1 Index document containing `((id "title"))` links to all documents → star-shaped graph (hub-and-spoke).

## Common Pitfalls

- **API token** is in `conf.json` → `api.token`; `accessAuthCode` (UI password) is NOT the API token.
- `createDocWithMd` with an existing `path` → error; use a unique path (include the date).
- API listens on `127.0.0.1:6806` (default) — access from outside via a proxy/tunnel if needed.
- Header auth: `Authorization: Token <token>` — not Bearer.
- **Imported docs land in SUBFOLDER, not notebook root** (proven): Inbox → `vault-md-root/`, Archive → `arsip/` & `import/`. The notebook filetree may look EMPTY until the subfolder is expanded. Don't conclude data loss — verify first: SQL `SELECT box, COUNT(*) FROM blocks WHERE type='d' GROUP BY box`, kernel boot log (`tree/block count [55/3925]`), or `find <workspace>/data -name '*.sy' | wc -l`.
- Endpoint `loadFiles` does NOT exist → 404 "page not found"; list docs with `listDocsByPath` (which may return `data: null` when empty).

## Verification Checklist

- [ ] `code: 0` returned by createDocWithMd
- [ ] `.sy` file is tens of KB (not 314 bytes)
- [ ] Title + expected content present in the `.sy` JSON
- [ ] After moveDocs: no 3+ segment paths; empty placeholder folders removed
- [ ] After adding refs/tags: kernel restarted, `refs` count > 0
