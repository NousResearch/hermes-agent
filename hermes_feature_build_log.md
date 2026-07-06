# Hermes Feature Build Log

- **Date/Time:** 2026-07-06T14:49:00-07:00
- **Machine Name:** Laptop 1
- **Repo Path:** c:\Users\ralle\projects\Hermes
- **Current Branch:** main
- **Objective:** Build two Hermes features in order:
  1. A new core API tool named `http_request`
  2. A native `github` toolset built on top of that HTTP foundation

## Master Checklist

### Track A — http_request core tool
- [x] A1 Check repo state
- [x] A2 Create feature branch
- [x] A3 Inspect reference files and patterns
- [x] A4 Implement tools/http_request.py
- [x] A5 Wire into toolsets.py as opt-in api toolset
- [x] A6 Add tests in tests/tools/test_http_request_tool.py
- [x] A7 Run focused tests
- [x] A8 Run nearby regression tests
- [x] A9 Review diff for cleanliness
- [x] A10 Commit Track A

### Track B — github native toolset
- [ ] B1 Confirm branch strategy for Track B
- [ ] B2 Implement tools/github_tools.py
- [ ] B3 Add GitHub auth/helper layer
- [ ] B4 Add first GitHub wrapper tools
- [ ] B5 Wire into toolsets.py as opt-in github toolset
- [ ] B6 Add tests in tests/tools/test_github_tools.py
- [ ] B7 Run focused tests
- [ ] B8 Run combined tests
- [ ] B9 Review diff for cleanliness
- [ ] B10 Commit Track B

### Final
- [ ] F1 Final implementation summary
- [ ] F2 Suggested PR titles
- [ ] F3 Suggested PR descriptions
- [ ] F4 Manual review checklist for me

---

## Step A1: Check Repo State
- **Status:** Completed
- **Details:** Cloned `C:\Users\ralle\AppData\Local\hermes\hermes-agent` to `c:\Users\ralle\projects\Hermes`. Repo is clean. Current branch is `main`.

## Step A2: Create Feature Branch
- **Status:** Completed
- **Details:** Created and checked out a new feature branch `feat/http-request-tool` from `main`. No pre-existing branch with this name was found.

## Step A3: Inspect Reference Files and Patterns
- **Status:** Completed
- **Details:**
  - **Tool Registration**: Registered at module level via `registry.register()` from [registry.py](file:///c:/Users/ralle/projects/Hermes/tools/registry.py) with name, toolset, schema, handler, check_fn, requires_env, is_async, description, emoji, and max_result_size_chars.
  - **check_fn usage**: Passed during registration to check tool dependencies or environmental requirements dynamically. Cache-protected for 30s in registry. Returns `False` or raises exceptions (caught as `False`) if unavailable.
  - **JSON Responses**: Tool handlers must return a serialized JSON string. Helper functions `tool_result` and `tool_error` in [registry.py](file:///c:/Users/ralle/projects/Hermes/tools/registry.py) minimize boilerplate.
  - **Toolset Definitions**: Structured in [toolsets.py](file:///c:/Users/ralle/projects/Hermes/toolsets.py) using the static `TOOLSETS` dict and composed via `resolve_toolset(name)`.
  - **SSRF / URL Safety**: Handled by `is_safe_url` / `async_is_safe_url` in [url_safety.py](file:///c:/Users/ralle/projects/Hermes/tools/url_safety.py). Restricts requests to public HTTP/HTTPS destinations, blocking private, CGNAT, link-local, loopback IP ranges, and cloud metadata sentinels (e.g. `169.254.169.254`, `metadata.google.internal`). Can be bypassed globally via `security.allow_private_urls: true` in config (except for cloud metadata sentinels which are always blocked). Redirect-based bypass is mitigated by re-validating the target of each redirect using `httpx` event hooks.
  - **Nearby Test Patterns**: Standard pytest files inside [tests/tools/](file:///c:/Users/ralle/projects/Hermes/tests/tools/). Example tests like `test_x_search_tool.py` patch outgoing requests using `monkeypatch` to mock backend endpoints, verifying payloads, status codes, timeouts, and error handling.
  - **Requests vs Httpx**: Standard sync tools (e.g., `x_search_tool.py`) use `requests` while async tools (e.g. `vision_tools.py`) and tools requiring redirect hooks use `httpx`. Given the redirect security requirement to protect against redirect SSRF bypass, using `httpx` with `_ssrf_redirect_guard` is preferred.

## Step A4: Implement tools/http_request.py
- **Status:** Completed
- **Details:** Created [http_request.py](file:///c:/Users/ralle/projects/Hermes/tools/http_request.py) containing the `http_request` core tool. Implemented method validation, safe URL resolution using `is_safe_url` and `normalize_url_for_request`, event-hook based redirect validation, headers / query / json_body / form_body support, timeout handling, content-type verification, bearer auth with secret redaction, and response body size truncation. Registered under the `api` toolset via module-level self-registration.

## Step A5: Wire http_request into toolsets.py as opt-in api toolset
- **Status:** Completed
- **Details:** Modified [toolsets.py](file:///c:/Users/ralle/projects/Hermes/toolsets.py) to define the new `api` toolset. Grouped the `http_request` tool within it, with the description "Direct structured HTTP API access". The tool is kept opt-in (not added to `_HERMES_CORE_TOOLS`) as requested, ensuring maximum safety and clear architectural isolation.

## Step A6: Add tests in tests/tools/test_http_request_tool.py
- **Status:** Completed
- **Details:** Created [test_http_request_tool.py](file:///c:/Users/ralle/projects/Hermes/tests/tools/test_http_request_tool.py) containing comprehensive deterministic pytest coverage. The suite validates invalid methods, localhost/private blocking, cloud metadata blocking, public URLs (mocked), JSON parsing, POST body payloads, custom header inclusion, query string parsing, environment token auth + credential redaction in responses/URLs, timeout handling, connection errors, truncation behavior, and non-200 responses.

## Step A7: Run focused tests for http_request
- **Status:** Completed
- **Details:** Executed the focused pytest suite using `uv run --extra dev pytest tests/tools/test_http_request_tool.py -v --tb=short`. All 13 test cases passed cleanly.

## Step A8: Run nearby regression tests
- **Status:** Completed
- **Details:** Executed nearby regression tests in `tests/tools/test_url_safety.py`, `tests/tools/test_web_tools_config.py`, and `tests/tools/test_tool_search.py` using `uv run --extra dev pytest tests/tools/test_url_safety.py tests/tools/test_web_tools_config.py tests/tools/test_tool_search.py -v --tb=short`. All 213 test cases passed successfully.

## Step A9: Review diff for cleanliness
- **Status:** Completed
- **Details:** Ran `git status` and `git diff` to inspect modifications. Verified that `toolsets.py` contains only the minimal 6-line insertion for the `api` toolset, and the untracked files `tools/http_request.py` and `tests/tools/test_http_request_tool.py` conform strictly to Python styling guidelines and contain no stray debug statements, unused imports, or credential leakage vectors.

## Step A10: Commit Track A
- **Status:** Completed
- **Details:** Committed the Track A implementation, tests, toolset configuration, and build log updates to the branch `feat/http-request-tool`.
- **Commit Message:** `feat: add http_request API tool`
- **Commit Hash:** `467f71023bde8d7ee2d01b1fe98b630304795ff9`

