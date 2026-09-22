feat(skills): add directus optional skill — Directus 11 CRUD, schema, and the policy permission chain

## What does this PR do?

Adds **`directus`**, an optional skill that drives a Directus 11+ instance over its REST API: collections and fields, item CRUD, and — the part #4176 is really about — the Directus 11 permission chain, `role → directus_access → policy → directus_permissions`, including static tokens for agent service accounts. One stdlib-only Python script plus a SKILL.md and a permissions reference; no Directus SDK, no pip installs, no new core surface.

Every feature the issue lists maps to a subcommand:

```bash
directus_admin.py collections create agent_heartbeats --field 'agent_id:string!' --field 'payload:json'
directus_admin.py items list agent_tasks --filter '{"status":{"_eq":"queued"}}' --sort=-date_created
directus_admin.py permissions create --policy <id> --collection agent_tasks --action read --fields '*'
directus_admin.py access grant --policy <id> --role <id>
directus_admin.py token set <user-id>
directus_admin.py bootstrap-agent dao-07 --collections agent_tasks,agent_heartbeats --actions read,create --email dao-07@example.com
```

**The v11 permission model is enforced, not just documented.** The issue's central technical point is that Directus 11 moved permissions off roles and onto policies. That is exactly the kind of claim a script gets wrong silently, so `/server/info` is read before any policy command and a Directus 10 server is refused *by version, with the reason* ("permissions attach directly to a role there") instead of returning a bewildering 404 on `/policies`. `check` prints the detected model (`policy-based (v11+)` / `role-based (v10)`), and `roles create` warns that the role it just made grants nothing until a policy is linked — the single most common cause of "my token still gets a 403".

**`bootstrap-agent` is the issue's use case in one command.** 44 DAO agents means the six-object dance (policy → one permission per collection × action → role → `directus_access` link → service user → static token) is run 44 times, and any step skipped produces a 403 that looks like all the others. The command walks the chain in order and prints every id it created. The policy it creates never gets `admin_access` (which bypasses every permission row) or `app_access` (Studio sign-in, which an API-only account does not need); a test pins both, because a convenience helper that quietly hands out admin is worse than no helper.

**The `fields: []` trap the issue names is handled at the boundary.** `permissions create` defaults `fields` to `["*"]`, and an *explicitly* empty `--fields` gets a `warning:` line saying Directus reads it as the primary key only. That failure is otherwise invisible: the request succeeds, the read succeeds, and every row comes back holding nothing but an `id`.

**Static tokens go through the API, as the issue says.** `token set` issues `PATCH /users/{id}`, generates a `secrets.token_urlsafe(32)` value when none is given, prints it once and says so. The reference records why the SQL route does not work: Directus hashes and caches on write, so a direct `UPDATE` on `directus_users.token` leaves a value that never authenticates.

**Errors carry the cause, not just the status.** Directus' `errors[].message` and `extensions.code` are unwrapped, then one line of diagnosis is attached — 403 → the access/policy chain plus the two commands that inspect it; 404 → collection names are case-sensitive and a table created with raw DDL is invisible to `/items` until Directus has a `directus_collections` row; 422 → an unknown or missing field. A reverse-proxy HTML error page is named as such instead of surfacing as a `JSONDecodeError`. 429 and 5xx retry three times with a 1/3/6 s back-off.

**Why this approach.** The footprint ladder in `AGENTS.md` puts a CLI-shaped script + skill (rung 2) above a new core tool: this is config/infra work expressible as commands, so it costs zero tokens on every API call and nothing in the tool schema. The instance URL is a `metadata.hermes.config` key (`directus.url`) because it is behavioural config, not a secret; only the credentials go in `.env`. `optional-skills/` rather than `skills/` because the skill needs a running Directus instance and admin credentials — official, but not loaded by default.

## Related Issue

Fixes #4176

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 🔒 Security fix
- [ ] 📝 Documentation update
- [ ] ✅ Tests (adding or improving test coverage)
- [ ] ♻️ Refactor (no behavior change)
- [x] 🎯 New skill (bundled or hub)

## Changes Made

- **`optional-skills/web-development/directus/SKILL.md`** (new, 148 lines) — hardline format: 50-char description, `## When to Use` / `Prerequisites` / `How to Run` / `Quick Reference` / `Procedure` / `Pitfalls` / `Verification`; `platforms: [linux, macos, windows]`; author credits the requester first; `related_skills` (`airtable`, `notion`, `har-derived-api-client`, `docker-management`) all resolve; one `metadata.hermes.config` key, `directus.url`. Prose routes every command through `terminal` and names `read_file` for quoting results back.
- **`optional-skills/web-development/directus/scripts/directus_admin.py`** (new, 1,120 lines, stdlib only, Python 3.9+) — subcommands `check`, `collections`, `fields`, `items`, `policies`, `permissions`, `roles`, `access`, `users`, `token`, `bootstrap-agent`. Bearer-token client with an injectable opener; an email/password login is exchanged for an access token exactly once per process and never written to disk; `limit`/`offset` pagination behind `--all` with a `MAX_PAGES` stop that tells the caller to narrow the filter; `--data`/`--filter`/`--rule`/`--validation` accept inline JSON, `@file`, or `-`; `--dry-run` on every mutating command prints the endpoint and payload without sending it; `--yes` required for all three deletes; `directus_*` system collections refused before the request. JSON on stdout, `[directus] …` progress and `warning: …` on stderr, and every failure a single `error:` line with exit 2 — never a traceback.
- **`optional-skills/web-development/directus/references/directus-11-permissions.md`** (new, 93 lines) — the junction diagram, what each system table actually holds, the four meanings of `fields`, row-level rule syntax with the `$CURRENT_USER` variables, how public access is expressed (an access row with both `role` and `user` null), static-token mechanics, DDL-created tables, and a five-step 403 triage.
- **`tests/skills/test_directus_skill.py`** (new, 475 lines, 33 tests, stdlib + pytest + monkeypatch, no live network) — frontmatter contract, stdlib-only import guard, config-vs-secret placement, URL normalisation (localhost stays `http`), setting precedence, bearer header, login-once, 403/non-JSON/429-with-back-off error paths, short-page pagination, the v10 version guard, permission-payload defaults and the empty-`fields` warning, field-spec parsing, primary-key declaration, the system-collection guard, and nine CLI end-to-end runs over a fake HTTP layer — including `bootstrap-agent` asserting the exact nine-call order and that the created policy has `admin_access: false`.
- **`.env.example`** — one delimited `directus skill (optional)` block with the three credential vars, commented out, plus a note that the URL is a setting rather than a secret. No edits outside that block.
- **`website/docs/user-guide/skills/optional/web-development/web-development-directus.md`** (new, generated by `website/scripts/generate-skill-docs.py`), plus the one new row in `website/docs/reference/optional-skills-catalog.md` and the one new entry in `website/sidebars.ts`.

Nothing else: no Python core, tool, toolset, gateway, prompt, or config-schema changes.

**Deliberately left out:** running the docs generator also rewrites ~190 unrelated pages (pre-existing drift — Windows backslashes in committed `Path` cells, a stale `merge-reconciler` sidebar entry, alphabetisation). Only the three lines belonging to this skill were kept, so the diff stays reviewable; the drift is someone's separate cleanup.

## How to Test

**1. Automated (no network, no live instance):**

```bash
scripts/run_tests.sh tests/skills/test_directus_skill.py -q                       # 33 passed
scripts/run_tests.sh tests/skills/test_authoring_standards.py -q                  # 1202 passed (hardline, incl. the new skill)
scripts/run_tests.sh tests/skills tests/website tests/test_plugin_skills.py -q    # 1735 passed
ruff check optional-skills/web-development/directus tests/skills/test_directus_skill.py   # All checks passed!
```

**2. The CLI without touching a server** — every mutating command answers `--dry-run` with the exact request it would send:

```bash
S=optional-skills/web-development/directus/scripts/directus_admin.py
python3 $S bootstrap-agent dao-42 --collections agent_heartbeats,agent_tasks --actions read,create --email dao-42@example.com --dry-run
python3 $S collections create agent_heartbeats --field 'agent_id:string!' --field 'payload:json' --dry-run
python3 $S permissions create --policy p1 --collection agent_tasks --action read --fields "" --dry-run   # emits the fields:[] warning
python3 $S collections delete directus_users --yes                                                       # refused: system collection
python3 $S items delete agent_tasks 7                                                                    # refused: needs --yes
```

**3. Against a real instance** — a throwaway Directus 11 container is enough:

```bash
docker run --rm -p 8055:8055 \
  -e ADMIN_EMAIL=admin@example.com -e ADMIN_PASSWORD=d1r3ctu5 \
  -e SECRET=replace-me directus/directus:11

export DIRECTUS_URL=http://localhost:8055 DIRECTUS_EMAIL=admin@example.com DIRECTUS_PASSWORD=d1r3ctu5

python3 $S check                      # version 11.x, "policy-based (v11+)", admin_access true
python3 $S collections create agent_heartbeats --field 'agent_id:string!' --field 'beat_at:timestamp' --field 'payload:json'
python3 $S fields list agent_heartbeats                       # id (primary key) + the three fields
python3 $S items create agent_heartbeats --data '{"agent_id":"dao-07","beat_at":"2026-09-21T10:00:00Z"}'
python3 $S items list agent_heartbeats --filter '{"agent_id":{"_eq":"dao-07"}}'
python3 $S bootstrap-agent dao-07 --collections agent_heartbeats --actions read,create --email dao-07@example.com
```

Then prove the chain actually granted what it claims — the printed token is a working, non-admin identity:

```bash
python3 $S check --token <static_token_from_the_previous_output>      # admin_access: false
python3 $S items list agent_heartbeats --token <static_token>         # allowed
python3 $S collections list --token <static_token>                    # 403, with the access/policy chain explained
python3 $S access list --role <role_id>                               # one link
python3 $S permissions list --policy <policy_id>                      # one row per collection × action
```

**4. Version guard** — point it at a Directus 10 instance: `check` exits 1 and says `policy-based` is unavailable, and `policies list` refuses with the detected version instead of a 404.

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat(skills): …`)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate — no open PR or existing skill mentions Directus
- [x] My PR contains **only** changes related to this feature (one commit, no unrelated doc-generator drift)
- [x] I've run the tests — via `scripts/run_tests.sh` rather than bare `pytest`, as `AGENTS.md` requires for CI parity (credentials unset, `TZ=UTC`, temp `HERMES_HOME`, per-file subprocess isolation): new file 33 passed; `tests/skills tests/website tests/test_plugin_skills.py` 1735 passed; authoring standards 1202 passed
- [x] I've added tests for my changes — `tests/skills/test_directus_skill.py`, invariant contracts only (no change-detectors, no source-text assertions)
- [x] I've tested on my platform: Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic, Python 3.11.16

### Documentation & Housekeeping

- [x] I've updated relevant documentation — generated skill page, optional-skills catalog row, sidebar entry, and `.env.example`
- [x] `cli-config.yaml.example` — N/A: per-skill settings are declared in SKILL.md frontmatter and stored under `skills.config.<key>`; no per-skill keys live in that file
- [x] `CONTRIBUTING.md` / `AGENTS.md` — N/A: no architecture or workflow change
- [x] I've considered cross-platform impact — the script is pure stdlib with no POSIX-only primitives (no `fcntl`, `os.setsid`, hardcoded `/tmp`, or shell-outs), so `platforms: [linux, macos, windows]` is honest; paths come from `pathlib`/`open` only
- [x] Tool descriptions/schemas — N/A: no tool behaviour changed; the skill adds no model-visible tool

## For New Skills

- [x] Placement is deliberate — **not** bundled: it needs a running Directus instance and admin credentials, so it ships in `optional-skills/web-development/` and installs with `hermes skills install official/web-development/directus`
- [x] SKILL.md follows the standard format (frontmatter, `## When to Use` triggers, procedure steps, pitfalls, verification) and passes `tests/skills/test_authoring_standards.py`
- [x] No external dependencies — Python standard library only, no Directus SDK, no pip install, no `curl`
- [ ] I've tested the skill end-to-end with `hermes --toolsets skills -q "…"` — **not run here:** this environment has no Directus instance to point it at. The script itself is exercised end-to-end through its real `main()` over a faked HTTP layer (nine CLI tests) and with `--dry-run` against the real argument parser; step 3 above is the live check for a reviewer who has an instance.

## Screenshots / Logs

`bootstrap-agent --dry-run`, the whole chain before a single request is sent:

```json
{
  "dry_run": true,
  "agent": "dao-42",
  "steps": [
    {"step": "create_policy",     "endpoint": "POST /policies",       "name": "dao-42 policy", "app_access": false},
    {"step": "create_permission", "endpoint": "POST /permissions",    "collection": "agent_heartbeats", "action": "read",   "fields": ["*"]},
    {"step": "create_permission", "endpoint": "POST /permissions",    "collection": "agent_heartbeats", "action": "create", "fields": ["*"]},
    {"step": "create_permission", "endpoint": "POST /permissions",    "collection": "agent_tasks",      "action": "read",   "fields": ["*"]},
    {"step": "create_permission", "endpoint": "POST /permissions",    "collection": "agent_tasks",      "action": "create", "fields": ["*"]},
    {"step": "create_role",       "endpoint": "POST /roles",          "name": "dao-42 role"},
    {"step": "link_access",       "endpoint": "POST /access",         "note": "role -> policy via directus_access"},
    {"step": "create_user",       "endpoint": "POST /users",          "email": "dao-42@example.com"},
    {"step": "set_static_token",  "endpoint": "PATCH /users/{id}",    "note": "static token cannot be set via SQL"}
  ]
}
```

The guards, each one line on stderr with exit 2:

```
warning: fields is empty for the read permission: Directus reads that as the primary key ONLY, not as every field. Pass --fields '*' for all fields.
error: Refusing to delete the system collection 'directus_users'. Directus system tables are managed through their own endpoints (/users, /roles, /policies, /permissions).
error: Refusing to delete agent_tasks/7 without --yes. Deleting in Directus is not undoable.
error: --filter is not valid JSON (Expecting property name enclosed in double quotes at char 1): {bad
```

Test run:

```
=== Summary: 1 files, 33 tests passed, 0 failed (100% complete) in 0.7s (64 workers) ===
=== Summary: 51 files, 1735 tests passed, 0 failed (100% complete) in 13.2s (64 workers) ===
```
