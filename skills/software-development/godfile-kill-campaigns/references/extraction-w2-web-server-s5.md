# Extraction w2 — web_server s5 mixed-case adjudication (w2a, 2026-08-05)

Second instance of the MIXED coverage case (after cli s2): c2 covered by BOTH
witnesses (head-to-head), c12 only w1a, c17 only w1b. Verdict:
`C:/tmp/tg-campaign/godfile/web_server/extraction/w2/s5-w2a.json`
(schema `run-godfile-extraction-w2-witness.v1`, canonical per cluster).

## Coverage + canonical
- c2 `ws_auth_gates` (10 methods + `_LOOPBACK_HOSTS`, move=40) → `hermes_cli/ws_auth_gate_mixin.py` — both witnesses; canonical **w1b**.
- c12 `console_ws` (7 top-level + nested `run_command`/`start_command` + 6 consts, move=36) → `hermes_cli/console_ws_mixin.py` — **w1a only**.
- c17 `spa_mount` (3 top-level + nested `_esc`/`no_frontend`/`_serve_index`/`serve_css`/`_ImmutableAssetFiles.get_response`/`serve_spa` + `_IMMUTABLE_ASSET_CACHE_CONTROL`, move=36) → `hermes_cli/spa_mount_mixin.py` — **w1b only**.

## Head-to-head tie-breaks that decided c2 → w1b (bodies byte-identical between witnesses)
1. Blank-line separator fidelity: live has 2 blanks after `_LOOPBACK_HOSTS`; w1b preserves 2, w1a has 4 (+2 extra). (w1a's file is 273 vs w1b 264 lines; the 9-line gap = 7-line docstring + 2 blanks.)
2. Re-export comment convention: the godfile's existing blocks carry `# noqa: E402,F401 — legacy re-exports; tests call these via web_server.<name>` (whatsapp_onboarding block at line 8412); w1b copies it verbatim, w1a invents `# noqa: E402`.
3. w1a's `from fastapi import WebSocket` top-level import is harmless-but-unneeded (annotations are strings); w1b has none.
w1a's c2 mixin remains a valid drop-in (identical bodies) if the merger prefers w1a wholesale.

## Byte-fidelity checker — the parenthesized multi-line shim block (regex that works)
Naive `^\s*from hermes_cli\.web_server import .*$` single-line strip misses
parenthesized import blocks and false-flags verbatim bodies (w1b's
`_render_active_theme_bootstrap_css` +7 lines, `mount_spa` +6 lines). Concrete fix:

```python
SHIM = re.compile(r"^\s*from hermes_cli\.web_server import .*$")
SHIM_BLOCK = re.compile(r"^\s*from hermes_cli\.web_server import \($")
def clean_shims(body):
    out, i = [], 0
    while i < len(body):
        ln = body[i]
        if SHIM.match(ln) or SHIM_BLOCK.match(ln):
            if SHIM_BLOCK.match(ln):
                i += 1
                while i < len(body) and not body[i].rstrip().endswith(")"):
                    i += 1
                i += 1  # closing paren line
            else:
                i += 1
            if i < len(body) and body[i].strip() == "":  # shim + trailing blank
                i += 1
            continue
        out.append(ln); i += 1
    return out
```
Results: w1a 24/24, w1b 15/15 verbatim; 0 leftovers in new web_server.py; 0 staying-def/const diffs; undefined-name AST scan (module scope ∪ all defs/args/targets ∪ builtins) clean on all 4 mixins.

## Parameter-collision false-positive (do not cry "rename")
w1b's `mount_spa` shim imports the module-global `app` while the function
parameter is `application` (routes register via `@application.get(...)`; the
global `app` serves `app.state.auth_required` reads). Live's def at 15529 is
ALREADY `def mount_spa(application: FastAPI):` — no rename. Rule: verify the
LIVE def line + body usage of both names before flagging; a shim importing a
module-global whose name differs from the parameter is correct.

## console_ws route re-registration seam (REQUIRED whenever c12 ships)
The `@app.websocket("/api/console")` decorator moved out with `console_ws`;
web_server.py must register `app.websocket("/api/console")(console_ws)`
BETWEEN `app.include_router(_dashboard_auth_router)` and `mount_spa(app)`
(ahead of the SPA catch-all `/{full_path:path}`). w1a placed it at line
16039/16041. Proof: tests/hermes_cli/test_web_server_console_ws.py passes
through the real app in the applied tree.

## Union assembly (canonical = w1b patch + w1a's c12 pieces)
1. Apply `s5-impl-w1b.patch` wholesale (git-format, plain `git apply`).
2. Delete the console region: banner `# ---...` line .. end of `console_ws`
   def (557 lines in the w1b-applied tree; find the banner by scanning up from
   the first console def for `^# -{5,}`).
3. Insert w1a's `from hermes_cli.console_ws_mixin import (...)` re-export
   block after the spa_mount_mixin re-export block.
4. Insert the route registration (comment + `app.websocket("/api/console")(console_ws)`)
   immediately before `mount_spa(app)`.
5. Copy `hermes_cli/console_ws_mixin.py` + `tests/hermes_cli/test_web_server_ws_auth_extraction.py`
   from w1a's `new/` tree.
Ready-made assembler: `C:/tmp/tg-campaign/godfile/web_server/extraction/w2/build_union_s5_w2a.py`.
importlib dry-run gotcha: module-level constants (`WT`, `WS`, `W1B_WT`) are bound at import —
re-bind `bu.WS` after overriding `bu.WT` or the run reads the original path.
Alternative composition (equally valid): w1a wholesale (c2+c12) + w1b's c17 additions.
Apply modes: w1a raw `diff -ruN` orig/new-prefixed → plain `git apply` (never `-p2`);
w1b git-format → plain `git apply`. Both zero-offset at HEAD 0577116f83.

## Evidence numbers (fresh worktrees at HEAD, --basetemp, venv 3.11.15/pytest 9.1.1, HERMES_SERVE_HEADLESS unset)
- w1a tree: 85 passed (43 shipped + console_ws/ws_auth/ws_tickets/pty/ws_client regressions).
- w1b tree: 92 passed (77 + 15 theme subset).
- Canonical UNION tree: 111 passed.
- Baseline theme suite on unmodified live repo: 15/15 (the one first-run failure was the
  HERMES_SERVE_HEADLESS env leak — 404 on /chat via mount_spa's no-frontend path).

## Env + hygiene notes
- `HERMES_SERVE_HEADLESS=1` leaks from the parent shell on this host; `unset` before
  dashboard/theme/console runs (see SKILL.md baseline-verify doctrine).
- Both sandbox origs were byte-identical to the live working tree (md5 9fd0e9f7...); the
  working tree vs HEAD differs by exactly 1 line (`_write_platform_enabled` removal at 8412)
  — patches apply at BOTH.
- w1b patch carries 5 inert `Binary files` (__pycache__) lines — git apply tolerates; strip
  for patch(1)-based tooling.
- Shared extraction/w2 dir carries sibling witnesses' artifacts (s2-w2a.json, wt-s1-w2b, ...) —
  shard-suffix your files, never delete a sibling's.
- No open PRs referenced the three mixin modules (gh pr list --search, read-only) — interlock clear.
