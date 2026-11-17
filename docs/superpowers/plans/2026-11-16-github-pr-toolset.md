# `github_pr` toolset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A narrow, shell-free GitHub tool for the Product Education teammate: open PRs on `everafterai/product-skills`, read PRs and commit logs on `everafterai/ea-monorepo`, never merge, never push to a default branch, never touch settings.

**Architecture:** One module, `tools/github_pr_tool.py`, registering two tools in two toolsets so RBAC gates reads and writes independently (the `jira_api` / `jira_api_write` pattern): `github_pr_read` (toolset `github_pr`) and `github_pr_write` (toolset `github_pr_write`). Both take an `op` from a closed set. Every call resolves the repo and op against a per-repo action allowlist in the profile's `config.yaml` (`github_pr.repos`) and refuses **before any request** when either is missing. HTTP goes straight to the GitHub REST API over aiohttp, proxy-aware like `tools/jira_api_tool.py`. The token comes only from `agent.secret_scope.get_secret("GITHUB_PR_TOKEN")`.

**Tech Stack:** Python 3.11, aiohttp, GitHub REST API v3 (`X-GitHub-Api-Version: 2022-11-28`), pytest via `scripts/run_tests.sh`.

**Spec:** `docs/superpowers/specs/2026-11-16-product-education-teammate-design.md` — "Dependencies → 4. GitHub tool" and "Autonomy levels" (`merge_skill_pr: off`, not built in v1).

## Global Constraints

- Credentials only via `get_secret("GITHUB_PR_TOKEN")`; never `os.getenv` / `os.environ`, no dotenv self-heal (it writes `os.environ`, which leaks across profiles under multiplex). `UnscopedSecretError` = not configured.
- Must not depend on the GitHub MCP.
- No merge op exists. An allowlist naming any action outside the known set, or `autonomy.actions.merge_skill_pr` set to anything but `off`, is a config error and both tools refuse every call with a message naming it.
- Never push to the repo's default branch: looked up live (`GET repos/{repo}` → `default_branch`) before every branch-writing op; if the lookup fails, refuse.
- Not in any built-in role (`gateway/tool_access.py` unchanged). Deployment needs both toolsets in `platform_toolsets.slack`.
- Tests: `scripts/run_tests.sh <path>` only, never without a path.

## Config shape (profile `config.yaml`)

```yaml
github_pr:
  repos:
    everafterai/product-skills:
      actions: [read_pr, read_commits, create_branch, commit_files, open_pr]
      branch_prefix: educator/      # optional: write ops only touch branches under it
    everafterai/ea-monorepo:
      actions: [read_pr, read_commits]
```

Actions and the ops they unlock:

| Action | Tool | Ops |
|---|---|---|
| `read_pr` | `github_pr_read` | `list_prs`, `get_pr`, `pr_files`, `pr_diff`, `pr_comments`, `pr_status` |
| `read_commits` | `github_pr_read` | `commits` |
| `create_branch` | `github_pr_write` | `create_branch` |
| `commit_files` | `github_pr_write` | `commit_files` |
| `open_pr` | `github_pr_write` | `open_pr` |

Repo keys match case-insensitively.

## Review Focus

1. **Writing to the default branch by another name** — `commit_files` with `branch: "Main"` or `"refs/heads/main"`: refused (compare case-insensitively after stripping `refs/heads/`; refuse names starting `refs/`).
2. **Workflow injection** — committing `.github/workflows/x.yml` to a feature branch runs a push-triggered workflow with repo secrets: refuse every path under `.github/`.
3. **Path traversal in file paths and branch names** — `../x`, `/abs`, `a//b`, `x.lock`, `-flag`: refused before any request.
4. **Unscoped secret under multiplex** — `get_secret` raises `UnscopedSecretError`: both the handler and `check_fn` treat it as "not configured", never crash.
5. **Huge diffs** — a monorepo PR diff of megabytes: truncated with an explicit `truncated: true`, not dumped.

---

### Task 1: Config + guards (pure, no network)

**Files:**
- Create: `tools/github_pr_tool.py`
- Test: `tests/tools/test_github_pr_tool.py`

**Interfaces — Produces:**
- `READ_OPS: dict[str, str]` op → action; `WRITE_OPS: dict[str, str]` op → action; `KNOWN_ACTIONS: frozenset[str]`.
- `_load_repo_policy() -> tuple[dict[str, dict], str | None]` — `({repo_lower: {"actions": frozenset, "branch_prefix": str}}, config_error_or_None)`; reads `hermes_cli.config.read_raw_config_readonly()`.
- `_authorize(repo: str, op: str, ops: dict[str, str]) -> tuple[dict | None, str | None]` — returns `(repo_policy, None)` or `(None, error)`.
- `_valid_branch(name: str) -> bool`, `_valid_file_path(path: str) -> bool`.

- [ ] **Step 1: Write failing tests** (monkeypatch `read_raw_config_readonly`):
  - `test_unconfigured_repo_refused` — `_authorize("acme/x", "get_pr", READ_OPS)` → error containing `"not configured"`.
  - `test_action_not_in_allowlist_refused` — monorepo with `[read_pr]`, op `commits` → error naming `read_commits`.
  - `test_repo_key_case_insensitive` — config `EverAfterAI/Product-Skills`, call `everafterai/product-skills` → allowed.
  - `test_unknown_action_is_config_error` / `test_merge_action_is_config_error` — `actions: [read_pr, merge_pr]` → every `_authorize` fails with a message containing `"merge"` and `"not built"`.
  - `test_merge_skill_pr_autonomy_above_off_is_config_error` — `autonomy.actions.merge_skill_pr: auto` → refused; `off` → fine.
  - `test_branch_validation` — accepts `educator/fix-voice`; rejects `""`, `../x`, `a//b`, `-x`, `x.lock`, `refs/heads/x`, `a b`, `x/`.
  - `test_file_path_validation` — accepts `plugins/release-comms/skills/x/SKILL.md`; rejects `/abs`, `a/../b`, `.github/workflows/x.yml`, `.GitHub/x`, `""`, `a//b`.
- [ ] **Step 2:** `scripts/run_tests.sh tests/tools/test_github_pr_tool.py` → FAIL (module missing).
- [ ] **Step 3:** Implement the above in `tools/github_pr_tool.py`.
- [ ] **Step 4:** Re-run → PASS.

### Task 2: Read tool

**Interfaces — Consumes:** Task 1. **Produces:**
- `_github_token() -> str` — `get_secret("GITHUB_PR_TOKEN")`, `""` on `UnscopedSecretError`/unset.
- `async _github_request(method, path, token, *, params=None, json_body=None, accept=None) -> {"status": int, "text": str}` — the only network seam (tests monkeypatch it).
- `async _github_pr_read_handler(args, **kw) -> str`; `_check_github_pr_read() -> bool`.

- [ ] **Step 1: Failing tests** (fake `_github_request` records calls):
  - `test_read_refused_before_any_request` — unconfigured repo → error and the fake asserts it was never called.
  - `test_missing_token_not_configured` and `test_unscoped_secret_not_configured` (fake `get_secret` raising `UnscopedSecretError`) → error containing `"GITHUB_PR_TOKEN"`.
  - `test_get_pr_compacts` — returns `number, title, body, state, merged, head.ref, base.ref, html_url`; path `repos/everafterai/ea-monorepo/pulls/7`.
  - `test_pr_diff_uses_diff_accept_and_truncates` — `accept == "application/vnd.github.diff"`; a 200k body comes back with `truncated: True` and ≤ `_MAX_DIFF_CHARS` (60000).
  - `test_pr_comments_merges_three_sources` — issue comments, review comments (`path`, `line`), reviews (`state`).
  - `test_pr_status_reads_head_sha_checks` — calls `commits/<sha>/status` and `commits/<sha>/check-runs`.
  - `test_commits_passes_filters` — `sha`, `path`, `since`, `per_page` capped at 100.
  - `test_http_error_shaped` — 404 → error containing `404`.
  - `test_bad_number_refused` — `number: 0` / `"x"` → error, no request.
  - `test_check_fn` — false with no token, false with no repo granting a read action, true with both; false (no raise) on `UnscopedSecretError`.
  - `test_registered_toolsets` — `github_pr_read` in `github_pr`; `toolsets.TOOLSETS["github_pr"]["tools"] == ["github_pr_read"]`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement + register + add `github_pr` to `toolsets.py` and `hermes_cli/tools_config.py`. **Step 4:** Run → PASS.

### Task 3: Write tool

**Interfaces — Consumes:** Tasks 1–2. **Produces:** `async _github_pr_write_handler(args, **kw) -> str`, `_check_github_pr_write() -> bool`, `async _default_branch(repo, token) -> str | None`.

Flows:
- `create_branch(repo, branch, from_branch?)`: default-branch lookup → refuse `branch == default` → `GET git/ref/heads/{from or default}` → `POST git/refs {ref: refs/heads/<branch>, sha}`.
- `commit_files(repo, branch, message, files[{path, content} | {path, delete: true}])` (1–50 files, ≤ 1 MB total content): refuse default → `GET git/ref/heads/{branch}` → `GET git/commits/{sha}` → `POST git/trees {base_tree, tree}` (delete = `sha: null`) → `POST git/commits {message, tree, parents: [sha]}` → `PATCH git/refs/heads/{branch} {sha, force: false}`.
- `open_pr(repo, head, title, body?, base?, draft?)`: refuse `head == default`; base defaults to the default branch → `POST pulls`.
- `branch_prefix` (when set) applies to `branch` / `head`.

- [ ] **Step 1: Failing tests:**
  - `test_write_refused_on_read_only_repo` — monorepo `[read_pr, read_commits]`, `open_pr` → refused, no request.
  - `test_commit_to_default_branch_refused` — default `main`; `branch` in `main`, `Main` → refused after only the `GET repos/...` lookup; never a ref/tree/commit call.
  - `test_default_branch_lookup_failure_refuses` — lookup 500 → refused.
  - `test_branch_prefix_enforced` — prefix `educator/`, branch `feature/x` → refused, no request.
  - `test_commit_files_flow` — exact call sequence and the tree entries (content + `sha: None` delete), `force: False`.
  - `test_commit_rejects_github_dir_and_bad_files` — `.github/workflows/x.yml`, empty list, 51 files, oversize → refused, no request.
  - `test_create_branch_flow` and `test_open_pr_flow` — paths and bodies; `open_pr` with `head == default` refused.
  - `test_unknown_op_refused` — `op: "merge"` → error mentioning merge is not available.
  - `test_write_check_fn` and `test_registered_write_toolset` — as for read.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement + register + `github_pr_write` in `toolsets.py` / `tools_config.py`. **Step 4:** Run → PASS.

### Task 4: Docs + inventory guard

- [ ] Add to `tests/test_fork_feature_inventory.py`: `WIRING` rows for `github_pr_read` (`toolset="github_pr"`), `github_pr_write` (`toolset="github_pr_write"`), and `get_secret("GITHUB_PR_TOKEN")`; add both names to `test_fork_toolsets_are_registered`.
- [ ] CLAUDE.md "Fork-added toolsets" entry incl. config shape, merge refusal, default-branch + `.github/` guards, token scopes, `platform_toolsets.slack` note.
- [ ] Run `scripts/run_tests.sh tests/tools/test_github_pr_tool.py tests/test_fork_feature_inventory.py`, then `scripts/run_tests.sh tests/tools/` and diff failures against the clean tree.
- [ ] Commit, push, open PR against `main`.
