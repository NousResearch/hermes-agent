# Hermes for Excel — Bridge Protocol

Contributor-facing description of the HTTP surface and the model contract, for
upstream integration with hermes-agent. Everything is local-loopback; the
bridge binds 127.0.0.1 only.

## Components

```
Excel task pane (Office.js)  →  bridge :8787 (node, zero deps)  →  Hermes API server :8642/v1
                                        ↘  Docling :8200 (document parsing, optional)
```

The bridge is stateless: every request carries the full conversation context.

## HTTP surface

### GET /api/health

Returns `{ok, service, port, llmBaseUrl, llmModel, doclingBaseUrl, time, hermes:{ok,status}, docling:{ok,status}}`.
`ok` is true only when both upstreams answer.

### POST /api/chat

Request body:

```jsonc
{
  "prompt": "user text",
  "history": [{"role": "user|assistant", "content": "..."}],   // last 12 turns
  "workbook": {"activeSheet": "...", "sheets": [{"name": "...", "usedRange": "A1:F120", "rowCount": 120, "columnCount": 6}]},
  "selection": {"address": "Sheet1!H23", "values": [[...]], "formulas": [[...]], "rowCount": 1, "columnCount": 1},
  "files": [{"name": "x.pdf", "type": "application/pdf", "size": 1234, "base64": "..."}],
  "tool_results": [{"range": "Bank Rec!A1:B6", "values": [[...]], "formulas": [[...]], "truncated": true, "error": "..."}],  // loop rounds only
  "parsed_files": [ /* echo of a previous response's parsed_files */ ],   // loop rounds only
  "loop_count": 0
}
```

Response body:

```jsonc
{
  "message": "short user-facing reply",
  "actions": [ /* normalized actions, see below */ ],
  "files": [ /* per-file extraction summaries */ ],
  "parsed_files": [ /* present only when actions contain read_range; echo back next round */ ],
  "source": "llm" | "file-parser" | "fallback"
}
```

The bridge always answers HTTP 200 with a structured body; failures surface in
`source` and `message`, never as fabricated data.

### POST /api/export

`{"name": "report", "values": [["matrix"]]}` → writes RFC-4180 CSV under the
add-in's `exports/` folder, returns `{ok, path}`. This is the bridge's only
file-writing path and runs only on an explicit export action.

## Actions

| type | fields | executed by |
|---|---|---|
| `write_cells` | `start_cell`, `values`, `allow_overwrite`, `auto_format` | pane |
| `create_sheet` | `name`, `values` | pane |
| `format_cells` | `range`, `style[]`, fonts/borders/number formats | pane |
| `conditional_format` | `range`, `operator`, `value`, `value2?`, `fill_color`, `font_color` | pane |
| `execute_office_js` | `explanation`, `code` (no nested Excel.run) | pane |
| `read_range` | `range`, `reason` | pane (loop) |
| `export` | `name`, `values` | pane → POST /api/export |

`conditional_format` applies a native Office.js cell-value rule so the model
never has to hand-author code (the largest single source of unparseable JSON
from a local model). `operator` is one of `lessThan`, `lessThanOrEqual`,
`greaterThan`, `greaterThanOrEqual`, `equalTo`, `notEqualTo`, `between`,
`notBetween` (common synonyms like `below`/`>=` are normalized bridge-side);
`value2` is used only for `between`/`notBetween`. Colors default to a light-red
fill (`#FFC7CE`) with dark-red font (`#9C0006`). For percentage columns the
underlying cell value is a decimal, so "below 25%" is `value: 0.25`.

The legacy `write: {mode, name, values}` response shape is still accepted and
converted to actions.

### The read loop

If a response contains `read_range` actions, the pane executes the reads
(capped 300×30 per range, `truncated: true` when clipped), re-POSTs the same
prompt/history plus accumulated `tool_results`, `parsed_files`, and
`loop_count + 1`. Maximum 5 rounds; at the budget the bridge instructs the
model to answer with available data and strips further `read_range` actions.

### Formula anchoring contract

The model authors all in-table formulas **table-local**: as if the table's
top-left cell were A1 (header row 1, first data row 2 → `=B2*C2`,
`=SUM(D2:D4)`). The bridge rebases relative references to the resolved
`start_cell` in `normalizeActions` (see `broker/formula-rebase.mjs`):

- `$`-anchored halves never shift (`$B$2` fixed, `B$2` shifts column only).
- Sheet-qualified refs and ranges (`Sheet1!B5`, `'Bank Rec'!A1:B5`) never shift.
- String literals (`"See B2"`) pass through untouched.
- A ref that would leave the grid is left unchanged rather than clamped.
- `create_sheet` tables anchor at A1 (no shift). When the model omits
  `start_cell` on `write_cells`, the bridge anchors at the user's selection.

This is the **single** rebasing layer. The pane writes `values` verbatim;
adding a second translation layer double-shifts formulas.

## Model contract (system prompt, enforced bridge-side)

- JSON only; bracket-repair plus one corrective retry handle near-valid output.
- **Complete the whole task in one reply.** There is no follow-up turn except
  the read_range loop. The model must not defer work ("this will take a couple
  of actions") — that leaves multi-step builds half-finished, since the pane
  applies what it gets and stops.
- **No prose in cells.** Cell values are short labels, numbers, or formulas
  (~40 chars). Paragraph-length "QA Notes" rows both produce terrible sheets and
  destabilize the local model into stopping mid-JSON; long text goes in `message`.
- **Prefer native actions over `execute_office_js`.** `write_cells`,
  `create_sheet`, `format_cells`, and `conditional_format` are reliable;
  embedding Office.js code in a JSON string is the top cause of unparseable
  replies. Reserve `execute_office_js` for structural changes with no native
  action, and prefer single-quoted JS string literals there to cut escaping.
- Workbook changes happen ONLY through actions. The agent must not use its own
  tools or filesystem. **Deployment requirement:** run the API server platform
  with file/terminal/code-execution toolsets disabled (hermes-agent config:
  `platform_toolsets.api_server` — keep cognitive toolsets like memory/vision).
- Never invent, estimate, or placeholder financial numbers; missing data is
  reported, not filled in.
- Attached file text arrives pre-extracted; the model must not claim it cannot
  read the file.

## Verification

- `node --test broker/server.test.mjs` — 30 unit tests over parsing, normalization, prompt assembly, and
  the formula-rebase module (the H23 regression and conditional_format normalization are pinned).
- `node broker/smoke.mjs` — live cases against a running bridge: health, A1 table,
  H23 anchor regression, multi-turn, the medium multi-action build, and export.
- `node broker/debug-llm.mjs <body.json>` — dump the raw model reply for a
  saved request body.
