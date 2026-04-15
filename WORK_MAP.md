# Work Map

Central status/map file for recent repo work.
Update this file whenever a session makes meaningful changes so the next session can resume without digging through chat logs.

## Current Status
- recorded_at: 2026-04-16 08:46:41 KST
- branch: `feat/firecrawl-document-parsing-web-extract`
- working_tree: clean
- focus: Firecrawl document parsing support + Camofox VNC URL scheme fix

## Latest Completed Changes

### 2026-04-16 — Firecrawl document parsing + Camofox VNC fix
1. `ebb73837` — `feat: add Firecrawl document parsing options to web_extract`
   - file: `tools/web_tools.py`
   - tests: `tests/tools/test_web_extract_document_parsing.py`
   - status: done
   - note: added Firecrawl document parsing options to `web_extract`.

2. `10981a7a` — `feat: support Firecrawl PDF parsers in web_crawl`
   - file: `tools/web_tools.py`
   - tests: `tests/tools/test_web_extract_document_parsing.py`
   - status: done
   - note: `web_crawl` now forwards Firecrawl PDF parsing options including `pdf_mode` and `pdf_max_pages`.

3. `3dc2fab0` — `fix: preserve camofox VNC URL scheme`
   - file: `tools/browser_camofox.py`
   - tests: `tests/tools/test_browser_camofox_persistence.py`
   - status: done
   - note: VNC URL generation now preserves the configured `CAMOFOX_URL` scheme instead of forcing `http`.

## Verification
- `source venv/bin/activate && python -m pytest tests/tools/test_web_extract_document_parsing.py -q`
  - result: `3 passed`
- `source venv/bin/activate && python -m pytest tests/tools/test_browser_camofox_persistence.py tests/tools/test_browser_camofox_state.py -q`
  - result: `29 passed`

## Open Threads
- This file did not exist before this session; it is now the default central map/status file for repo changes.
- If future sessions change the repo, append or refresh this file in the same session.
- Full-suite verification was not run in this recording step.

## Next Practical Step
1. Keep `WORK_MAP.md` updated whenever a branch lands meaningful changes.
2. If the branch changes again, add the new commit, touched files, and verification command/results here.
3. Run broader test coverage before merge if the change surface expands beyond targeted tool behavior.
