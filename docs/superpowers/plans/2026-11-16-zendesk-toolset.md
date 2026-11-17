# Zendesk toolset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Product Education teammate a draft-only Zendesk Help Center tool: free reads of the KB, and an unpublished-draft create gated by a per-profile autonomy level (`off` / `signoff` / `veto` / `auto`).

**Architecture:** Two registry toolsets in one module, modelled on `tools/jira_api_tool.py`: `zendesk` (`zendesk_api`, GET-only behind a Help Center path allowlist) and `zendesk_write` (`zendesk_create_draft`, a *structured* tool — the model never supplies a method or path — whose single HTTP chokepoint re-checks a curated `(method, path-regex)` write allowlist and forces `draft: true`). A new generic `agent/autonomy.py` resolves the action's level from the profile's `config.yaml` (mtime-cached `read_raw_config`), computes veto windows in working hours, and stores pending veto actions. `tools/release_asset_gate.py` holds the two external signals the levels need: the Notion Release Assets row (sign-off / Hold) and the Slack announcement (❌ veto).

**Tech Stack:** Python, aiohttp (Zendesk, Notion, Slack HTTP), `agent.secret_scope.get_secret`, `hermes_cli.config.read_raw_config`, `agent.data_access_audit.record_access`.

**Spec:** `docs/superpowers/specs/2026-11-16-product-education-teammate-design.md` — "Dependencies → 2. Zendesk tool" and "The boundary → Autonomy levels". API reference: the `release-comms:zendesk-draft` skill (`SKILL.md`, `references/zendesk-api.md`).

## Global Constraints

- Hub: `https://{ZENDESK_SUBDOMAIN}.zendesk.com`; basic auth `"{ZENDESK_EMAIL}/token:{ZENDESK_API_TOKEN}"`.
- Every credential (`ZENDESK_*`, `NOTION_API_KEY`/`NOTION_API_TOKEN`, `SLACK_BOT_TOKEN`) is read ONLY via `agent.secret_scope.get_secret`; `UnscopedSecretError` = not configured. No `os.getenv`, no dotenv loading.
- Never publish, archive, delete, change permissions, or touch tickets. Write = `POST …/sections/{id}/articles.json` with `article.draft = true` and `notify_subscribers = false`, nothing else.
- Edit to a published article: Guide Team Publishing has **no public API** for work-in-progress (Zendesk staff, community thread 6482703860506), so the tool always creates a separate draft and returns `reconcile_required: true` with the live article.
- Autonomy config (profile `config.yaml`): `autonomy.paused`, `autonomy.actions.<action>`, `autonomy.announce_channel`, `autonomy.veto_wait_working_hours` (default 4), `autonomy.working_hours` (default `{timezone: UTC, start: 9, end: 18, days: [mon,tue,wed,thu,fri]}`). Absent action → `signoff`. Paused → `off` stays `off`, everything else becomes `signoff`.
- Built levels: `zendesk_create_draft` = all four; `zendesk_publish` = `{off}`; any other action = `{off, signoff}`. A configured level outside the built set (or not one of the four) is refused with a clear error naming the action, the level and the built levels.
- Sign-off signal: Release Assets row with `Status` in `zendesk.approved_statuses` (default `["Approved"]`), `Asset type = Help article`, `Destination = {subdomain}.zendesk.com`, parent data source/database = `zendesk.release_assets_data_source_id` (required; unset → refuse). `Published` rows are always refused. Hold: `Status` in `zendesk.hold_statuses` (default `["Hold"]`).
- Neither toolset is in any built-in role. Both must be listed in `platform_toolsets.slack` to reach the model.
- Tests: `scripts/run_tests.sh <path>` only.

## Review Focus

1. A pagination URL handed back from `links.next` (full `https://…zendesk.com/…?page[after]=…`) — expected to work on `zendesk_api`; a full URL on any other host must be refused. (Task 2 test `test_read_accepts_own_host_full_url_and_refuses_other_hosts`.)
2. A traversal or ticket path dressed as Help Center (`api/v2/help_center/../tickets.json`, `api/v2/tickets.json`, `api/v2/help_center/articles/1/../../../users.json`) — refused before any request. (Task 2 `test_read_refuses_tickets_and_traversal`.)
3. Sign-off withdrawn between announcement and execution of a veto action (row moved to `Hold`), or a ❌ added — must not execute, pending record marked `vetoed`. (Task 4 `test_veto_execute_refuses_on_hold_or_x_reaction`.)
4. Config edited to `zendesk_publish: auto` or `zendesk_create_draft: autopilot` — refuse with an error naming the built levels, never fall through to acting. (Task 1 `test_unbuilt_level_is_refused`.)
5. Veto announced Friday afternoon — the window must not elapse over the weekend. (Task 1 `test_working_hours_skip_weekend`.)

---

### Task 1: `agent/autonomy.py` — levels, working hours, pending store

**Files:**
- Create: `agent/autonomy.py`
- Test: `tests/agent/test_autonomy.py`

**Interfaces — Produces:**
- `LEVELS = ("off", "signoff", "veto", "auto")`, `DEFAULT_LEVEL = "signoff"`, `BUILT_LEVELS: dict[str, frozenset[str]]`
- `class AutonomyConfigError(ValueError)`
- `resolve_level(action: str, config: dict | None = None) -> str` (config defaults to `read_raw_config()`; raises `AutonomyConfigError`)
- `@dataclass WorkingHours(timezone: str, start: int, end: int, days: frozenset[int])`, `working_hours_from_config(config) -> WorkingHours`
- `add_working_hours(start: datetime, hours: float, wh: WorkingHours) -> datetime` (aware datetimes)
- `veto_settings(config) -> tuple[str, float, WorkingHours]` → `(announce_channel, wait_hours, wh)`
- `save_pending(action: str, record: dict) -> str` (returns `pending_id`), `load_pending(action, pending_id) -> dict | None`, `update_pending(action, pending_id, **fields) -> None`, `list_pending(action) -> list[dict]` — files under `get_hermes_home()/autonomy/pending/<action>/<pending_id>.json`, atomic write; `pending_id` validated `^[0-9a-f]{12}$`.

- [ ] Tests: `test_default_level_is_signoff`, `test_reads_configured_level`, `test_paused_caps_at_signoff_but_keeps_off`, `test_unbuilt_level_is_refused` (`zendesk_publish: auto`, `zendesk_create_draft: autopilot`, `post_release_update: veto` all raise with the built levels in the message), `test_working_hours_within_day`, `test_working_hours_skip_weekend` (Fri 16:00 + 4h, Mon–Fri 9–18 UTC → Mon 11:00), `test_working_hours_start_outside_hours`, `test_pending_roundtrip_and_list`, `test_pending_id_rejects_traversal`.
- [ ] Run `scripts/run_tests.sh tests/agent/test_autonomy.py` → FAIL (module missing).
- [ ] Implement.
- [ ] Run again → PASS. Commit `feat(autonomy): per-action autonomy levels, working-hours veto window, pending store`.

### Task 2: `zendesk_api` read tool + write chokepoint

**Files:**
- Create: `tools/zendesk_tool.py`; Modify: `toolsets.py` (add `zendesk`, `zendesk_write` after `jira_write`)
- Test: `tests/tools/test_zendesk_tool.py`

**Interfaces — Produces:**
- `_zendesk_creds() -> tuple[str, str, str]` → `(base_url, email, token)`; `("", "", "")` when unset, invalid subdomain, or `UnscopedSecretError`.
- `async _zendesk_request(base, email, token, path, method="GET", json_body=None) -> dict` → `{status, text}`
- `_normalize_read_path(raw: str, base: str) -> str | None`, `_read_allowed(path) -> bool`, `_write_allowed(method, path) -> bool`
- `async _zendesk_write(method, path, body) -> dict` — the only write path: refuses unless `_write_allowed` and `body["article"]["draft"] is True` and `body["notify_subscribers"] is False`.
- Tool `zendesk_api(path)` in toolset `zendesk`; `check_fn=_check_zendesk` (all three creds resolvable).

Read allowlist (path part, before `?`; `{HC}` = `api/v2/help_center/(?:<locale>/)?`, locale `[a-z]{2}(?:-[a-z0-9]{2,4})?`, `.json` optional): categories[/id], categories/id/(sections|articles), sections[/id], sections/id/articles, articles[/id], articles/id/translations[/locale], articles/id/attachments, articles/search, user_segments[/id], locales, `api/v2/guide/permission_groups[/id]`. Reject whitespace, `..`, `//`, `#`, `\`.

- [ ] Tests: `test_registered_toolsets_are_separate`, `test_read_passes_path_and_parses_json`, `test_read_accepts_own_host_full_url_and_refuses_other_hosts`, `test_read_refuses_tickets_and_traversal`, `test_read_unconfigured`, `test_creds_unscoped_secret_is_not_configured`, `test_creds_use_get_secret_not_environ` (monkeypatch `get_secret`), `test_write_chokepoint_refuses_non_draft_and_other_paths`.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat(zendesk): read-only Help Center tool and draft-only write chokepoint`.

### Task 3: `tools/release_asset_gate.py` — Notion row + Slack veto signals

**Files:**
- Create: `tools/release_asset_gate.py`
- Test: `tests/tools/test_release_asset_gate.py`

**Interfaces — Produces:**
- `async fetch_notion_page(page_id: str) -> dict` (GET `https://api.notion.com/v1/pages/{id}`, `Notion-Version: 2025-09-03`, token `get_secret("NOTION_API_KEY") or get_secret("NOTION_API_TOKEN")`; raises `GateError` on any failure)
- `class GateError(RuntimeError)`
- `@dataclass RowState(status: str, asset_type: str, destination: str, url: str)`
- `row_state(page: dict, expected_parent: str) -> RowState` — raises `GateError` if the parent's `data_source_id`/`database_id` (dashes ignored) ≠ expected.
- `async post_announcement(channel: str, text: str, thread_ts: str = "") -> str` (returns ts; `GateError` on failure)
- `async veto_reactions(channel: str, ts: str) -> list[str]` — user ids that reacted `x` (also `no_entry`, `no_entry_sign`), excluding the bot's own id (`auth.test`); `GateError` on failure (callers fail closed).

- [ ] Tests (aiohttp mocked at `_http_json`): `test_row_state_reads_select_props`, `test_row_state_refuses_foreign_parent`, `test_fetch_requires_notion_token`, `test_veto_reactions_ignores_bot_and_other_emoji`, `test_veto_reactions_error_raises`.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat(autonomy): Release Assets sign-off and Slack veto signals`.

### Task 4: `zendesk_create_draft` with autonomy levels

**Files:**
- Modify: `tools/zendesk_tool.py`
- Test: `tests/tools/test_zendesk_create_draft.py`

**Interfaces — Consumes:** Tasks 1–3. **Produces:** tool `zendesk_create_draft` in toolset `zendesk_write`.

Args: `section_id` (digits), `title`, `body` (HTML), `permission_group_id` (int), `user_segment_id` (int|null, optional), `locale` (default `en-us`), `release_asset_row_id`, `revision_of_article_id` (optional), `pending_id` (execute a veto-pending action), `list_pending` (bool).

Flow: resolve level (error → refuse) → `off` refuses → validate args → if `revision_of_article_id`: GET the article (must exist; default `section_id` to its section) → level:
- `signoff`: row required, gate passes (Approved, Help article, destination, parent) → create.
- `veto`: row required and not Published/Hold; if already Approved → create now; else `save_pending` with the full request + `not_before = add_working_hours(now, wait)`, post announcement, return `status: "pending"`. Re-call with `pending_id`: re-resolve level (`off` refuses; `signoff` needs Approved), not before `not_before`, no ❌, row not Hold/Published → create with the **stored** request, reply in the announcement thread, mark done.
- `auto`: row optional (if given: not Published/Hold, type/destination checked) → create, then best-effort summary to `announce_channel`.
- veto/auto executions → `record_access(tool="zendesk_create_draft", action="autonomy_<level>", target=…)`.
Result: `{ok, article_id, html_url, draft: true, level, reconcile_required, reconcile_with}`.

- [ ] Tests: `test_off_refuses`, `test_signoff_requires_row`, `test_signoff_refuses_unapproved_published_wrong_type_or_parent`, `test_signoff_creates_forced_draft`, `test_revision_creates_separate_draft_and_flags_reconcile`, `test_veto_announces_and_queues`, `test_veto_execute_too_early`, `test_veto_execute_refuses_on_hold_or_x_reaction`, `test_veto_execute_uses_stored_request_and_audits`, `test_auto_creates_and_announces`, `test_paused_drops_auto_to_signoff`, `test_publish_level_config_error_refuses`.
- [ ] Run → FAIL. Implement. Run → PASS. Commit `feat(zendesk): zendesk_create_draft honours autonomy levels`.

### Task 5: Wiring guards + docs

**Files:** Modify `tests/test_fork_feature_inventory.py` (WIRING entries for `zendesk_api` `_READ_ALLOWLIST`, `zendesk_create_draft` `_WRITE_ALLOWLIST`, `agent/autonomy.py` `BUILT_LEVELS`; add `zendesk`, `zendesk_write` to `test_fork_toolsets_are_registered`), `CLAUDE.md` "Fork-added toolsets" entry.

- [ ] Run `scripts/run_tests.sh tests/test_fork_feature_inventory.py tests/tools/test_zendesk_tool.py tests/tools/test_zendesk_create_draft.py tests/tools/test_release_asset_gate.py tests/agent/test_autonomy.py` → PASS; then `tests/tools/` and diff failures against a clean tree.
- [ ] Commit, push, PR.

---

## Review round 1 (2026-11-17) — decisions and changes

Reviewer: the education-teammate session. All items addressed on the same branch.

**Decisions recorded**
- **Always a separate draft for edits to published articles.** Guide Team Publishing's work-in-progress state has no public API, and the classic API either goes live (PUT on a published translation) or unpublishes (`draft: true`). `revision_of_article_id` creates a new draft in that article's section with `reconcile_required: true`.
- **Any ❌ (`x` / `no_entry` / `no_entry_sign`) reaction by anyone other than the bot vetoes** a pending action. We don't check the reactor's role; over-vetoing is the safe direction.
- **Content comes from the row.** `zendesk_create_draft` takes no title or body. The title is the row's title property. The body is the row's Notion blocks, rendered by `tools/notion_blocks_html.py`. We chose the blocks API over `GET v1/pages/<id>/markdown` because rendering structured blocks is safe by construction (escaped text, fixed tag set, scheme-checked `href`). Notion's enhanced markdown carries XML-like tags that would need a post-hoc sanitizer.
- **Images:** refused by default. Notion file URLs expire, and attachment upload isn't built. `omit_images=true` is an explicit opt-in: images are left out, and their captions are returned (`images_omitted`) and shown in the Slack announcement/report for a human to add in Zendesk. Attachment upload is a follow-up.
- **Notion write-guard precondition:** every level except `off` is refused unless `notion.write_guard.enabled: true`. The Notion write guard (spec 5a) owns enforcement behind that key and must deploy first.
- **Visibility:** the permission group and user segment come from the revised article, otherwise from `zendesk.default_permission_group_id` (required) and `default_user_segment_id`. Never from the model.

**Changes**
- `agent/autonomy.py`:
  - pending records split into a `signed` part (HMAC-SHA256, host-local 0600 key `autonomy/.hmac_key`) and mutable state; tampered records are refused and audited;
  - `exclusive()` O_EXCL lock per key (left behind on a crash so a possibly-completed write is never repeated);
  - per-key `ledger_*` (one draft per row);
  - `veto_expiry_hours` (default 72 after `not_before`).
- Veto execution re-fetches the row and refuses as `stale` if the rendered content's sha256 differs from the announcement. `run_due_pending()` (tool flag `run_due=true`, or `python -m tools.zendesk_tool run-due`) executes due actions and expires old ones.
- The gate refuses archived and trashed rows (`RowStopped`, which counts as a veto for pending actions).
- Slack text: every interpolated value is escaped (`& < >`), and links are emitted only for https URLs on the Zendesk or Notion hosts. Zendesk links are constructed from integer ids.
- Read query keys are allowlisted (`_READ_QUERY_KEYS`). `include`, unknown keys and `;` are refused.
- `agent/file_safety.is_protected_data_path` covers `<home>/autonomy/**` for read, write, patch and search.

## Review round 2 (2026-11-17): OAuth instead of API tokens

Zendesk is removing API tokens ([support article 10851263566234](https://support.zendesk.com/hc/en-us/articles/10851263566234)). Since Phase 2 (2026-10-27) no new token can be created, and Phase 3 (2027-04-30) deactivates every remaining token. So the Basic `email/token:` auth is gone with **no fallback**.

- **Grant:** `client_credentials` against `POST https://<sub>.zendesk.com/oauth/tokens`, which requires a confidential client. The body is `grant_type`, `client_id`, `client_secret`, `scope`, and `expires_in: 1800`. We request the lifetime explicitly because the response may omit `expires_in`; Zendesk allows 300–172800. The grant returns no refresh token, so we simply re-request.
- **Scopes:** reads request `hc:read`; the draft create requests `hc:read hc:write`. This deliberately splits the reviewer's single `hc:read hc:write` so a read path never holds a write-capable token. We never request global `read`/`write` or ticket scopes.
- **Credentials:** `ZENDESK_SUBDOMAIN`, `ZENDESK_OAUTH_CLIENT_ID`, `ZENDESK_OAUTH_CLIENT_SECRET`, read via `get_secret` only. `ZENDESK_EMAIL` and `ZENDESK_API_TOKEN` are no longer read.
- **Cache:** in memory, keyed by (profile home, base URL, client id, scope). A token is refreshed 120 s before it expires. On a 401 the cached token is dropped and the call is retried exactly once.
- **No leaks:** the token and secret are scrubbed from every error, and transport exceptions are reduced to their type name. The token request uses the fixed host with `allow_redirects=False`.
- **Correction to the request** ("the client must be created by a service user with Help Center manager only — not an account admin"):
  - Per the [OAuth Clients API](https://developer.zendesk.com/api-reference/ticketing/oauth/oauth_clients/), clients can be created by "Admins, and agents with the Manage APIs permission".
  - The token "inherits the permissions of the user associated with the OAuth client".
  - So the service user is an **agent with a custom role that grants Manage APIs**, plus the Guide role Help Center manager. It is not an admin, but it is not Guide-only either.
  - The client's own `scope` field caps it at `hc:read hc:write`.
  - Not verified: whether removing Manage APIs after the client is created keeps its tokens working, and whether `hc:read` covers `guide/permission_groups` and `user_segments`.
- **Tests:** `tests/tools/test_zendesk_oauth.py`.
