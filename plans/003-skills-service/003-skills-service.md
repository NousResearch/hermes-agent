# Plan 003 — Hermes Skills Service

**Status:** DRAFT (2026-05-18)
**Run after:** Plan 001-0 (HermesIdentity dataclass — scope resolution requires knowing who is asking)
**Blocks:** Plan 001-A (scoped skills in SaaS mode — this replaces its storage backend)
**Date:** 2026-05-18
**Branch:** feat/003-skills-service (to be created)

---

## Context

### The current state of skills

Hermes skills today are Markdown files (`SKILL.md`) on a local filesystem, discovered via `get_all_skills_dirs()` in `agent/skill_utils.py`. The machinery already implements CSS-like resolution: `~/.hermes/skills/` is always first (most specific), followed by `skills.external_dirs` in config.yaml order. The `blake-cowork-plugins` repo has a separate `build/commands/` tree with `execute-plan.md`, `architect.md`, etc. that are conceptually skills but not in SKILL.md format and not connected to Hermes's discovery.

**The problem is not one problem — it is four:**

1. **No canonical scope model.** There is no machine-readable label that says "this skill is global / team / personal." The order of `external_dirs` is implicit scope — change the order and everything breaks silently.

2. **No single registry.** Skills live in at least five places today: `hermes-agent/skills/` (bundled), `optional-skills/` (opt-in), `~/.hermes/skills/` (personal), `~/Documents/agentic-hub/skills/` (external_dir), and `blake-cowork-plugins/build/commands/` (format mismatch). An agent on a new machine or a new team member starts with zero skills.

3. **No collaboration primitive.** A skill born in personal scope has no upgrade path to team scope. Blake must manually copy files and manage git commits across multiple repos.

4. **No service boundary.** Skills and Hermes are tightly coupled — the agent reads SKILL.md directly from disk. A future team or SaaS deployment can't share skills across machines without either a shared filesystem (fragile) or S3 (inflexible and not git-native).

### Why Atlas is the reference architecture

Atlas (Plan 001) solved an analogous problem for memory: instead of scattering facts across tool outputs and session logs, it made memory a first-class service with a standard interface (MCP), a clear data model (RDF triples with provenance), and a separation between the service (Atlas) and the consumers (Hermes, Claude Code, Cursor).

**The same separation applies to skills:**

- **Skills Service** = stores, versions, and serves SKILL.md bundles with scope metadata
- **Hermes** = orchestrates agents, loads skills as context, delegates execution
- **Interface** = MCP (same pattern as Atlas)

Hermes should own agentic orchestration. It should not own skill storage.

### Why Git is the right backend for skills

Skills are Markdown. Git is the canonical store for Markdown. Git already provides:
- Versioning (SHA per commit)
- Diffing and review (PRs for team promotion)
- Access control (repo permissions per scope)
- Replication (push/pull to remote)
- Conflict detection (merge conflicts vs. silent overwrite)

S3 is appropriate for binary blobs or hot-path reads in stateless deployments. For skills — human-readable, rarely changed, best reviewed via diff — Git is strictly better.

---

## Key Design Insights

1. **Scope is a first-class property, not an implicit directory order.** Every SKILL.md carries `metadata.hermes.scope: personal | team | global` in its frontmatter. The resolution algorithm reads scope from the file, not from which directory it was found in. This makes scope portable: the same file in a different directory still has the right scope.

2. **The Skills Service is a thin read-through Git cache, not a database.** Its job is to (a) aggregate skills from multiple Git remotes, (b) enforce the CSS resolution algorithm across scopes, (c) serve skill content via MCP, and (d) gate writes back to the appropriate Git remote with optional PR workflow. It holds no state that isn't derivable from the underlying Git repos.

3. **Skills promotion is the collaboration primitive.** A personal skill can be promoted to team scope with one command: `skill_manage(action='promote', name='foo', target_scope='team')`. This creates a branch, opens a PR, and — after merge — the skill becomes available to all team members on next sync. This is the atomic unit of team skill co-development.

4. **The `SkillSource` ABC already exists.** `tools/skills_hub.py:294` defines an interface with `search()`, `fetch()`, `inspect()`, and `source_id()`. The Skills Service is a new `SkillSource` implementation — `RegistrySkillSource` — that understands scope, Git remotes, and the three-tier resolution model. This is the minimal surface area for integration.

5. **`config.yaml` `skills.registries` replaces `skills.external_dirs` without breaking it.** `external_dirs` stays as a legacy fallback for users who haven't migrated. New users get a structured `registries` block. `get_all_skills_dirs()` is updated to read `registries` first in scope order (personal → team → global), then append legacy `external_dirs`.

6. **No S3 until `HERMES_MODE=saas`.** Local teams use Git repos. S3 is gated behind the SaaS mode flag (Plan 001-D) and becomes an alternate `SkillSource` implementation (`S3SkillSource`) for stateless ECS deployments where local disk is ephemeral.

---

## Phase Index

| Phase | Title | Effort (1-5) | Risk (1-5) | Priority | Dependencies | Status |
|-------|-------|-------------|-----------|----------|--------------|--------|
| 003-A | `skills.registries` config + updated resolution | 2 | 1 | P0 | Plan 001-0 | Not started |
| 003-B | Scope annotations + promotion CLI | 2 | 2 | P0 | 003-A | Not started |
| 003-C | `RegistrySkillSource` + MCP surface | 3 | 2 | P0 | 003-B | Not started |
| 003-D | Git sync + advisory write locks | 2 | 3 | P1 | 003-C | Not started |
| 003-E | `blake-cowork-plugins` migration + SKILL.md conversion | 2 | 1 | P1 | 003-A | Not started |
| 003-F | `S3SkillSource` (HERMES_MODE=saas only) | 3 | 2 | P2 | 003-C, Plan 001-D | Not started |

## Execution Sequence

```
003-A (config + resolution)
  └─> 003-B (scope + promotion)
        └─> 003-C (RegistrySkillSource + MCP)
              └─> 003-D (git sync + locks)
              └─> 003-F (S3 — saas only, parallel with 003-D)
003-A → 003-E (cowork-plugins migration — parallelizable with 003-B)
```

003-E can run any time after 003-A; it does not block 003-B/C/D.
003-F does not start until Plan 001-D (cloud storage) is complete.

---

## Phase Details

### Phase 003-A — `skills.registries` config + updated resolution

**What:** Introduces the `skills.registries` config block in `~/.hermes/config.yaml` and updates `agent/skill_utils.py:get_all_skills_dirs()` to read from it. No behavior change for users with `external_dirs` only — full backward compatibility.

**Config shape:**
```yaml
skills:
  registries:
    - scope: personal
      name: blake-personal
      path: ~/.hermes/skills
      writable: true
      auto_sync: false

    - scope: team
      name: predicate-team
      path: ~/Documents/predicate-ventures/skills
      remote: git@github.com:predicate-ventures/hermes-skills.git
      writable: true
      auto_sync: true
      promote_requires_pr: true

    - scope: global
      name: hermes-official
      path: ~/Documents/hermes-agent/skills
      writable: false

  # Legacy — still honored, appended after registries
  external_dirs:
    - ~/Documents/agentic-hub/skills
```

**Resolution algorithm change:** `get_all_skills_dirs()` returns `[personal_paths..., team_paths..., global_paths..., legacy_external_dirs...]`. First-found-wins is unchanged. Scope metadata is now available on each path via a `RegistryEntry` dataclass.

**Files to modify:**
- `agent/skill_utils.py` — `get_all_skills_dirs()`, add `get_skill_registries()`, add `RegistryEntry` dataclass
- `~/.hermes/config.yaml` (user-space, not repo) — add `registries` block

**Acceptance criteria:**
- [ ] `get_all_skills_dirs()` returns dirs in personal → team → global → external order
- [ ] `external_dirs`-only config produces identical behavior to today
- [ ] `registries`-only config produces correct scope-ordered dirs
- [ ] Both present: registries first, external_dirs appended
- [ ] `RegistryEntry` carries `scope`, `name`, `path`, `writable`, `remote`, `auto_sync`, `promote_requires_pr`
- [ ] `hermes skills list` shows `[personal]` / `[team]` / `[global]` annotation on each skill

### Phase 003-B — Scope annotations + promotion CLI

**What:** Adds `scope` annotation to `skills_list` output and `skill_view` results. Adds `skill_manage(action='promote')` to copy a personal skill into the team registry, optionally opening a Git PR.

**New `skill_manage` action:**
```python
skill_manage(action='promote', name='writing-plans', target_scope='team')
# → copies ~/.hermes/skills/software-development/writing-plans/ to team registry dir
# → if promote_requires_pr: git checkout -b promote/writing-plans, git add, git commit, git push, gh pr create
# → if not promote_requires_pr: git add, git commit, git push origin main
```

**New frontmatter field (advisory, not enforced):**
```yaml
metadata:
  hermes:
    scope: personal   # personal | team | global
    # Note: scope is advisory only — the registry the file lives in determines
    # effective scope. This field aids migration tooling and human readers.
```

**Files to modify:**
- `tools/skill_manager_tool.py` — add `promote` action, `scope` output on list
- `agent/skill_commands.py` — surface scope in `_list_skills()` response
- `agent/skill_utils.py` — add `get_skill_scope_for_path(path) -> str` helper

**Acceptance criteria:**
- [ ] `skills_list()` result for each skill includes `scope: personal | team | global`
- [ ] Skills that shadow same-named skills in lower scopes show `shadowing: [team, global]`
- [ ] `skill_manage(action='promote', name='foo')` copies file to team registry dir
- [ ] If `promote_requires_pr: true`, a git branch is created and PR opened via `gh pr create`
- [ ] If target registry `writable: false`, promotion raises `PermissionError`
- [ ] Global scope write from an agent turn always raises `PermissionError` regardless of config

### Phase 003-C — `RegistrySkillSource` + MCP surface

**What:** Implements `RegistrySkillSource(SkillSource)` — the new `SkillSource` adapter that serves skills from the structured registry model. Exposes an MCP server (`skills-service`) with three tools: `list_skills`, `view_skill`, `promote_skill`. This is the service boundary — Hermes (and any other MCP client) talks to the Skills Service via MCP, not by reading SKILL.md files directly.

**Why MCP:** Atlas's success as an external memory service is entirely because it surfaces over MCP. Any agent (Claude Code, Cursor, rooben-pro) can call `mcp_atlas_ask` without knowing Atlas's internals. Skills Service should be the same: `mcp_skills_view_skill(name='writing-plans')` works from any connected agent.

**Architecture:**
```
Hermes (agent turn)
  └─> skill_view(name='foo')
        └─> if HERMES_SKILLS_SERVICE_URL set:
              mcp_skills_view_skill(name='foo')   ← Skills Service MCP
            else:
              read from local filesystem (today's behavior)
```

**Service spec (FastAPI + MCP, mirroring Atlas pattern):**
```python
# MCP tools exposed:
list_skills(scope_filter=None, tag_filter=None) -> list[SkillMeta]
view_skill(name: str, scope: str = None) -> SkillContent  # returns SKILL.md text
promote_skill(name: str, from_scope: str, to_scope: str) -> PromotionResult
search_skills(query: str) -> list[SkillMeta]
```

**Service lives at:** `~/Documents/hermes-skills-service/` (new repo, same pattern as `army-of-one/`)

**Port:** `localhost:8001` (Atlas is 8000)

**Auth:** LAN bearer token (same pattern as Atlas v1)

**Files to create (new repo):**
- `hermes-skills-service/main.py` — FastAPI app + MCP server
- `hermes-skills-service/registry.py` — reads `registries` config, aggregates dirs
- `hermes-skills-service/resolver.py` — CSS resolution algorithm (scope priority)
- `hermes-skills-service/git_ops.py` — pull, push, branch, PR creation via `gh`
- `hermes-skills-service/mcp_tools.py` — MCP tool definitions
- `hermes-skills-service/config.py` — reads `~/.hermes/config.yaml` for registries block

**Files to modify (hermes-agent repo):**
- `agent/skill_utils.py` — add `HERMES_SKILLS_SERVICE_URL` env check; fall through to local if unset

**Acceptance criteria:**
- [ ] `list_skills()` MCP tool returns all skills across registries with scope annotation
- [ ] `view_skill(name='writing-plans')` returns correct SKILL.md content, resolving personal override of team/global
- [ ] `promote_skill(name='foo', from_scope='personal', to_scope='team')` copies + PR opens
- [ ] Service starts via `make up` in `hermes-skills-service/`
- [ ] Hermes `skill_view` falls through to local filesystem when service URL not set
- [ ] Hermes `skill_view` uses MCP when `HERMES_SKILLS_SERVICE_URL` is set
- [ ] MCP tools registered and callable via `mcp_skills_*` prefix

### Phase 003-D — Git sync + advisory write locks

**What:** Adds startup auto-sync (`git pull --ff-only`) for registries with `auto_sync: true`. Adds a POSIX `flock`-based advisory write lock to prevent concurrent agent turns from colliding on the same skill file. Adds a `git push` step after any write to a writable registry with a remote configured.

**Startup sync (hermes-agent):**
```python
# hermes_cli/main.py — on startup
for reg in get_skill_registries():
    if reg.auto_sync and reg.remote:
        subprocess.run(
            ["git", "-C", str(reg.path), "pull", "--ff-only", "--quiet"],
            timeout=5, capture_output=True
        )
        # Non-blocking: errors are warnings, not fatal
```

**Write lock (skills service + local):**
```python
# tools/skill_lock.py
@contextmanager
def skill_write_lock(skill_name: str, registry_path: Path, timeout: int = 5):
    lock_file = registry_path / ".hub" / f"{skill_name}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except IOError:
            raise RuntimeError(f"Skill '{skill_name}' is being modified. Retry.")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
```

**Auto-push after write:**
```python
# After skill_manage create/edit/patch in a registry with remote:
# git add <skill_path>
# git commit -m "skill(personal): update {name}" 
# git push origin HEAD (non-blocking, warn on failure)
```

**Files to modify:**
- `hermes_cli/main.py` — add startup auto-sync loop
- `tools/skill_manager_tool.py` — wrap writes in `skill_write_lock`
- `agent/skill_utils.py` — add `get_skill_registry_for_path(path)` to know which remote to push to

**Files to create:**
- `tools/skill_lock.py` — `skill_write_lock` context manager

**Acceptance criteria:**
- [ ] Registries with `auto_sync: true` pull on Hermes startup; failure is a warning not a crash
- [ ] Two concurrent `skill_manage(action='edit')` calls on the same skill name: second call raises `RuntimeError` within `timeout` seconds
- [ ] After a `skill_manage` write to a registry with `remote` set, git commits and attempts push
- [ ] Push failure is logged as warning; skill write still succeeds locally

### Phase 003-E — `blake-cowork-plugins` migration + SKILL.md conversion

**What:** Converts the high-value commands in `blake-cowork-plugins/build/commands/` into SKILL.md format and registers the `blake-cowork-plugins` repo as a `scope: team` registry entry. This makes `execute-plan`, `architect`, and related commands discoverable via `skills_list` and loadable via `skill_view`.

**Files to convert (priority order):**
1. `build/commands/execute-plan.md` → `build/commands/execute-plan/SKILL.md` (role: execution)
2. `build/commands/architect.md` → `build/commands/architect/SKILL.md` (role: orchestration)
3. `daily-digest.md` → `daily-digest/SKILL.md` (role: reference)
4. `morning-brief.md` → `morning-brief/SKILL.md` (role: reference)

**Conversion rules:**
- Add YAML frontmatter: `name`, `description`, `metadata.hermes.role`, `metadata.hermes.scope: team`
- Do NOT move existing body content — wrap it
- Add `metadata.hermes.calls` / `called_by` composition graph entries per existing `writing-plans` governance

**Registry entry to add to `~/.hermes/config.yaml`:**
```yaml
skills:
  registries:
    - scope: team
      name: blake-cowork
      path: ~/Documents/blake-cowork-plugins
      remote: git@github.com:blakeaber/blake-cowork-plugins.git
      writable: true
      auto_sync: true
      promote_requires_pr: false  # Blake owns this repo
```

**Files to create:**
- `blake-cowork-plugins/build/commands/execute-plan/SKILL.md`
- `blake-cowork-plugins/build/commands/architect/SKILL.md`
- `blake-cowork-plugins/daily-digest/SKILL.md`
- `blake-cowork-plugins/morning-brief/SKILL.md`

**Acceptance criteria:**
- [ ] `skills_list()` shows `execute-plan`, `architect`, `daily-digest`, `morning-brief` with `scope: team`
- [ ] `skill_view(name='execute-plan')` loads the converted SKILL.md
- [ ] Existing `writing-plans` skill frontmatter updated: `metadata.hermes.calls: [execute-plan]` (replaces the inline Stage 2/3 description that duplicates the original command)
- [ ] All 4 converted SKILL.md files pass frontmatter validation (name, description, metadata.hermes.role all present)

### Phase 003-F — `S3SkillSource` (HERMES_MODE=saas only)

**What:** Implements `S3SkillSource(SkillSource)` for stateless ECS deployments where there is no local Git clone. Skills are stored in S3 under `s3://hermes-skills/{scope}/{tenant_id}/{skill_name}/SKILL.md`. The `RegistrySkillSource` delegates to `S3SkillSource` when `HERMES_MODE=saas` is set and no local path is available for a registry entry.

**Scope resolution in SaaS mode:**
```
personal: s3://hermes-skills/personal/{user_id}/{name}/SKILL.md
team:     s3://hermes-skills/team/{team_id}/{name}/SKILL.md
global:   s3://hermes-skills/global/{name}/SKILL.md
```

**CSS resolution:** same algorithm, but `boto3.get_object` calls instead of `Path.read_text`. First successful read wins.

**Write path:** personal and team writes go to S3 directly (no Git remote on ECS). Git sync is handled by a separate CI job that pushes S3 changes back to the team Git repo weekly (or on promotion events).

**Dependencies:** Plan 001-D (cloud storage backend — S3 bucket, IAM roles, tenant_id in context)

**Files to create:**
- `hermes-skills-service/sources/s3_source.py` — `S3SkillSource(SkillSource)`

**Files to modify:**
- `hermes-skills-service/registry.py` — detect `HERMES_MODE=saas`, switch to `S3SkillSource`

**Acceptance criteria:**
- [ ] `HERMES_MODE=saas` set + S3 bucket configured → `skill_view('foo')` reads from S3
- [ ] `HERMES_MODE=saas` not set → `skill_view('foo')` reads from local Git registry (no S3 calls)
- [ ] Scope resolution order is identical in both modes
- [ ] Personal skill write in SaaS mode goes to `s3://hermes-skills/personal/{user_id}/` not team or global

---

## System Architecture (post-003)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HERMES AGENT (orchestration)                                               │
│                                                                             │
│  skill_view(name) / skill_manage(action)                                   │
│       │                                                                     │
│       ├── HERMES_SKILLS_SERVICE_URL set?                                   │
│       │     YES → mcp_skills_* MCP calls ─────────────────────────────┐   │
│       │     NO  → local filesystem (today's behavior, unchanged)       │   │
│       │                                                                 │   │
└───────────────────────────────────────────────────────────────────────┼───┘
                                                                         │
┌────────────────────────────────────────────────────────────────────────▼───┐
│  SKILLS SERVICE  (localhost:8001 — new repo: hermes-skills-service/)       │
│                                                                             │
│  MCP tools: list_skills / view_skill / promote_skill / search_skills       │
│                                                                             │
│  RegistrySkillSource                                                        │
│    reads: skills.registries from ~/.hermes/config.yaml                     │
│    resolves: personal → team → global (CSS specificity)                    │
│    delegates to: GitRegistrySource (local) or S3SkillSource (saas mode)   │
│                                                                             │
│  flock advisory write lock per skill name                                  │
│  git pull on auto_sync registries                                          │
│  git push + optional PR on writes to registries with remotes               │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                         │
         ▼                    ▼                         ▼
┌────────────────┐  ┌──────────────────────┐  ┌────────────────────────────┐
│ PERSONAL       │  │ TEAM                 │  │ GLOBAL                     │
│ ~/.hermes/     │  │ ~/Documents/         │  │ ~/Documents/               │
│   skills/      │  │   predicate-ventures/│  │   hermes-agent/skills/     │
│ writable       │  │   skills/            │  │ writable: false            │
│ no remote      │  │ remote: github.com/  │  │ remote: hermes-agent repo  │
│                │  │   predicate-ventures │  │                            │
└────────────────┘  └──────────────────────┘  └────────────────────────────┘
         │                    │
         └────────────────────┴──────────> S3 (HERMES_MODE=saas only)
                                            s3://hermes-skills/{scope}/{id}/
```

---

## Relationship to Existing Plans

### Atlas (Plan 012) — Hermes→Atlas Memory Connector
Atlas solved memory as an external service. Plan 003 applies the same pattern to skills.
- **Atlas** = external memory service → MCP → Hermes
- **Skills Service** = external skill registry → MCP → Hermes
- **Same pattern**: FastAPI + MCP + local-first + Git-backed + `HERMES_MODE=saas` gated cloud path

These are **peer services** to Hermes, not Hermes internals.

### Hermes Plan 001 (Multi-User SaaS)
- Plan 001-0 (HermesIdentity) is a prerequisite: scope resolution needs `user_id` and `team_id` to determine which personal/team registries to read
- Plan 001-A (scoped skills) is **superseded by Plan 003** — 003 is the full implementation of what 001-A scoped as a feature flag. Update 001-A in PROGRESS.md to reference 003.
- Plan 001-D (cloud storage) must precede Plan 003-F (S3SkillSource)

### Hermes Plan 002 (Self-Organization)
- Plan 002 defines `~/.hermes/STRUCTURE.md` and runtime workspace isolation
- Skills Service respects the STRUCTURE.md layout — `~/.hermes/skills/` remains the personal registry root
- No conflict; 003-A reads the same directory 002 defines

---

## Open Questions (Blake must resolve before Phase 003-C begins)

**Q-3.1:** Should the Skills Service be a new standalone repo (`hermes-skills-service/`) mirroring Atlas's pattern, or a package within `hermes-agent/` that can be started as a subprocess?
- **Recommended:** Standalone repo (`~/Documents/hermes-skills-service/`). Mirrors Atlas exactly. Enables independent deployment, versioning, and consumption by non-Hermes agents (Claude Code, Cursor).
- **Alternative:** `hermes-agent/services/skills/` — simpler initial setup, but tight coupling.

**Q-3.2:** Is `promote_requires_pr: true` the right default for team registries, or should it be `false` until Blake has a team?
- **Recommended:** `false` for now (Blake is solo). Add to `skills.registries` config when team grows. PR workflow is ready but not required.

**Q-3.3:** Should `blake-cowork-plugins` be a `scope: team` or `scope: personal` registry?
- **Recommended:** `scope: team` — it's the canonical shared-command repo. Even solo, framing it as team ensures the right behavior when teammates are added.

**Q-3.4:** What is the right port for the Skills Service? Atlas uses 8000.
- **Recommended:** `8001`. Leaves room for a third service at `8002` (likely a future self-improvement / eval service).

---

## Budget Estimate

| Component | Local mode | SaaS mode (~10 teams) |
|-----------|------------|-----------------------|
| Skills Service (EC2 t3.micro) | $0 (localhost) | ~$8/mo |
| S3 (skill bundles) | $0 | ~$0.50/mo |
| Git remotes | $0 (GitHub free) | $0 |
| **Total** | **$0** | **~$8.50/mo** |

Skills storage is fundamentally cheap. The cost model is entirely in compute (worker fleet), not storage.
