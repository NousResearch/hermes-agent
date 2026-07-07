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
- [x] B1 Confirm branch strategy for Track B
- [x] B2 Implement tools/github_tools.py
- [x] B3 Add GitHub auth/helper layer
- [x] B4 Add first GitHub wrapper tools
- [x] B5 Wire into toolsets.py as opt-in github toolset
- [x] B6 Add tests in tests/tools/test_github_tools.py
- [x] B7 Run focused tests
- [ ] B8 Run combined tests
- [ ] B9 Review diff for cleanliness
- [ ] B10 Commit Track B

### Track C — github_add_issue_comment
- [x] C1 Check branch, status, and files
- [x] C2 Implement github_add_issue_comment_tool
- [x] C3 Register tool schema/handler/toolset
- [x] C4 Add mocked tests in tests/tools/test_github_tools.py
- [x] C5 Verify compilation and run regression suite
- [x] C6 Review diff for cleanliness
- [x] C7 Commit and log Track C

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
- **Commit Hash:** `59102b9e2a2ceb7074c8289f25e11b9be5b54779`

## Step B1: Confirm branch strategy for Track B
- **Status:** Completed
- **Details:** Switched to branch `feat/github-toolset` based on the clean amended Track A commit `59102b9e2a2ceb7074c8289f25e11b9be5b54779`. Verified that the working tree is clean.

## Step B2: Implement tools/github_tools.py
- **Status:** Completed
- **Details:** Created [github_tools.py](file:///c:/Users/ralle/projects/Hermes/tools/github_tools.py) with the base module skeleton and API request helper on top of the secure `http_request` foundation. Defined default base URL, token env var configs, `_get_github_token` resolver, `check_github_requirements` checklist check, and the `github_api_request` helper function which passes target REST paths and standard headers to `http_request_tool`.

## Step B3: Add GitHub auth/helper layer
- **Status:** Completed
- **Details:** Expanded [github_tools.py](file:///c:/Users/ralle/projects/Hermes/tools/github_tools.py) to implement robust auth, request-shaping, repository parsing, and error normalization. Added `parse_owner_repo` to resolve repository URLs/paths into `(owner, repo)` tuples, and `get_github_error_message` to extract detailed user-facing descriptions from status codes or standard GitHub API error payloads.

## Step B4: Add first GitHub wrapper tools
- **Status:** Completed
- **Details:** Added and self-registered four read-only core GitHub wrapper tools in [github_tools.py](file:///c:/Users/ralle/projects/Hermes/tools/github_tools.py): `github_get_issue`, `github_list_issues` (which excludes PRs), `github_get_pull_request`, and `github_list_pull_requests`. Each tool is configured to parse inputs via `parse_owner_repo`, query the API via `github_api_request`, and filter/format payloads to return only relevant metadata (redacting token/header noise). Registered under the `github` toolset with requirement checks mapping to `check_github_requirements`.

## Step B5: Wire into toolsets.py as opt-in github toolset
- **Status:** Completed
- **Details:** Modified [toolsets.py](file:///c:/Users/ralle/projects/Hermes/toolsets.py) to define the opt-in `"github"` toolset in `TOOLSETS`. Listed the four wrapper tools under `"tools"`. This exposes the GitHub capabilities dynamically in sessions where the `github` toolset is requested, while keeping them isolated from core/default toolsets.

## Step B6: Add tests in tests/tools/test_github_tools.py
- **Status:** Completed
- **Details:** Created [test_github_tools.py](file:///c:/Users/ralle/projects/Hermes/tests/tools/test_github_tools.py) implementing a mocked test suite covering: check requirements gating (token lookup), owner/repo string and URL parsing including trailing `.git` sanitation, error parsing from response JSON/text, all four core tool success paths, invalid inputs, non-200 and failed requests, and specific issue-vs-PR filtering (PRs correctly omitted in `github_list_issues`).

## Step B7: Run focused tests
- **Status:** Completed
- **Details:** Executed the focused pytest suite using `uv run --extra dev pytest tests/tools/test_github_tools.py -v --tb=short`. All 10 test cases passed cleanly.

## Step C1: Check Branch, Status, and Files
- **Status:** Completed
- **Details:** Checked git status, active branch (`feat/github-toolset`), and read the relevant implementation files `tools/github_tools.py`, `tests/tools/test_github_tools.py`, and `toolsets.py`.

## Step C2: Implement github_add_issue_comment_tool
- **Status:** Completed
- **Details:** Added `github_add_issue_comment_tool` to `tools/github_tools.py`, implementing a safe POST request to `/repos/{owner}/{repo}/issues/{issue_number}/comments` via `github_api_request` and returning id, html_url, body, created_at, updated_at, and author (from user.login).

## Step C3: Register Tool Schema/Handler/Toolset
- **Status:** Completed
- **Details:** Registered the schema `GITHUB_ADD_ISSUE_COMMENT_SCHEMA`, handler `_handle_github_add_issue_comment`, and connected it in the registry under the `"github"` toolset in `tools/github_tools.py`. Added it to the static `"github"` toolset definition in `toolsets.py`.

## Step C4: Add Mocked Tests in tests/tools/test_github_tools.py
- **Status:** Completed
- **Details:** Added four deterministic mocked unit tests: `test_github_add_issue_comment_success`, `test_github_add_issue_comment_invalid_repo`, `test_github_add_issue_comment_non_200_response`, and `test_github_add_issue_comment_failed_http_request`.

## Step C5: Verify Compilation and Run Regression Suite
- **Status:** Completed
- **Details:** Ran `python -m py_compile` to verify syntax check. Executed focused tests (14 passed) and combined regression suite (240 passed in 15.37s).

## Step C6: Review Diff for Cleanliness
- **Status:** Completed
- **Details:** Verified `git diff` for minimal changes, lack of debug artifacts, and adherence to styling.

## Step C7: Commit and Log Track C
- **Status:** Completed
- **Details:** Committed the changes with message `feat: add github_add_issue_comment tool`. Commit hash: `c7621d48385afb1a2feb690248ce646024a5f08e`.

## Step D1: Inspect Branch, GitHub Module, Tests, and Toolset Wiring
- **Status:** Completed
- **Details:** Re-checked repo state on `main`, confirmed the native GitHub implementation lived in `tools/github_tools.py`, reviewed `tests/tools/test_github_tools.py`, and verified the opt-in `github` toolset wiring in `toolsets.py` before expanding scope.

## Step D2: Save Repo-Local Expansion Plan
- **Status:** Completed
- **Details:** Wrote the workflow expansion plan to `C:\Users\ralle\projects\Hermes\.hermes\plans\2026-07-07_103747-github-workflow-expansion.md`, documenting architecture, implementation order, verification commands, and exit criteria.

## Step D3: Add github_create_issue
- **Status:** Completed
- **Details:** Added `github_create_issue_tool` with compact response shaping, schema/handler/registry wiring, and inclusion in the `github` toolset. The tool POSTs to `/repos/{owner}/{repo}/issues` and supports `title`, optional `body`, `labels`, and `assignees`.

## Step D4: Add github_add_pull_request_review_comment
- **Status:** Completed
- **Details:** Added `github_add_pull_request_review_comment_tool` for inline PR feedback via `/repos/{owner}/{repo}/pulls/{pull_number}/comments`, supporting `body`, `commit_id`, `path`, `line`, and optional `side`, with compact returned metadata for agents.

## Step D5: Add github_list_workflow_runs
- **Status:** Completed
- **Details:** Added `github_list_workflow_runs_tool` for GitHub Actions inspection via `/repos/{owner}/{repo}/actions/runs`, including optional `status` filtering and compact workflow-run summaries.

## Step D6: Add github_get_workflow_run
- **Status:** Completed
- **Details:** Added `github_get_workflow_run_tool` for detailed GitHub Actions run inspection via `/repos/{owner}/{repo}/actions/runs/{run_id}` returning branch, SHA, status, conclusion, timestamps, and run attempt.

## Step D7: Add github_rerun_workflow
- **Status:** Completed
- **Details:** Added `github_rerun_workflow_tool` for Actions reruns via `/repos/{owner}/{repo}/actions/runs/{run_id}/rerun`, returning an acknowledgement payload with `accepted`, `repository`, and `run_id` when the API accepts the rerun.

## Step D8: Add Mocked Tests for New Workflow Tools
- **Status:** Completed
- **Details:** Extended `tests/tools/test_github_tools.py` with deterministic mocked coverage for create-issue success/failure cases, PR review comment success/failure cases, workflow-run listing and detail shaping, and workflow rerun success/error handling.

## Step D9: Run Focused GitHub Verification
- **Status:** Completed
- **Details:** Ran `python -m py_compile tools/github_tools.py tests/tools/test_github_tools.py toolsets.py` successfully, then ran `uv run --extra dev pytest tests/tools/test_github_tools.py -v --tb=short`. Result: **26 passed**.

## Step D10: Run Combined Regression Suite
- **Status:** Completed
- **Details:** Ran `uv run --extra dev pytest tests/tools/test_http_request_tool.py tests/tools/test_github_tools.py tests/tools/test_url_safety.py tests/tools/test_web_tools_config.py tests/tools/test_tool_search.py -v --tb=short`. Result: **252 passed**, **8 warnings**, **0 failures**.

## Step D11: Review Diff for Cleanliness
- **Status:** Completed
- **Details:** Reviewed `git diff` and confirmed the intentional change set covered `tools/github_tools.py`, `tests/tools/test_github_tools.py`, `toolsets.py`, `hermes_feature_build_log.md`, and the saved plan file. No debug code or unrelated file edits were introduced.

