# ADR-0001 — Platform-level skill resolution in profile mode (skill stub cross-profile contract)

**Status:** accepted
**Date:** 2026-06-23
**Driver:** kanban `t_39362dc9` (SPEC-6 from `t_cbf829f4/REPORT.md` §2.7);
the Jun 20 dispatch burst included `totum-orchestrator` workers that died
because their `HERMES_HOME` was pinned to `~/.hermes/profiles/<name>/` and
the dispatcher could not resolve the `totum-platform` skill name (it lived
only on the spec-writer profile's local skills/ and at the platform level
in `~/.hermes/skills/`). 41 worker logs in the 7d window ended with
`Error: Unknown skill(s): X, Y, Z` — same root cause.

## Context

Hermes skill resolution historically lived in three places, each with its
own ad-hoc list of search roots:

| Caller | Search roots |
|---|---|
| `agent.skill_utils.get_all_skills_dirs()` | `[<HERMES_HOME>/skills/] + get_external_skills_dirs()` |
| `tools.skills_tool.skill_view()` | `[<HERMES_HOME>/skills/ (module-global `SKILLS_DIR`)] + get_external_skills_dirs()` |
| `tools.skills_tool._find_all_skills()` | `[SKILLS_DIR] + get_external_skills_dirs()` |
| `agent.skill_commands.scan_skill_commands()` | `[SKILLS_DIR] + get_external_skills_dirs()` |

In **default mode** (`HERMES_HOME = ~/.hermes`), all three resolve to the
same `~/.hermes/skills/` and behaviour is correct. In **profile mode**
(`HERMES_HOME = ~/.hermes/profiles/<name>/`), only the profile's local
skills/ is searched. The platform-level `~/.hermes/skills/` is invisible
to profile-mode workers — including all kanban-spawned workers, because
`hermes_cli/kanban_db.py:_default_spawn` pins the worker's `HERMES_HOME`
to the assignee profile.

This is the dispatcher code path for `--skills <name>`:

```
hermes chat -q "work kanban task <id>"
  → cli.py builds HermesCLI
  → build_preloaded_skills_prompt(['<name>'])
  → _load_skill_payload('<name>')
  → skill_view('<name>')
  → _find_skill('<name>') across the three search roots
  → if not found: raise ValueError("Unknown skill(s): <name>")
```

When the search roots only see the profile's skills/, the worker crashes
with the `Unknown skill(s):` ValueError and the dispatcher marks the task
`crashed` with `pid N exited with code 1`. This accounts for a
material fraction of the 7d dispatcher burst:

| Skill name | Distinct failing logs (7d) |
|---|---:|
| `code-craftsman`, `code-craftsman-toolkit` | 4 |
| `totum-platform-audit` | 3 |
| `fleet-coach`, `fleet-coach-toolkit` | 2 |
| `spec-writer`, `Wags`, `sdlc-review`, `test-driven-development`, `totum-coordinator` | 1 each |

All of these names live (or should live) at the platform level or in
profiles whose workers can pin them in `--skills`. With no platform-level
visibility, every profile that does not install the skill locally dies
on first use.

## Decision

**Make the platform-level `~/.hermes/skills/` automatically visible to
profile-mode workers.** Resolution precedence becomes:

1. **Local** — `<HERMES_HOME>/skills/` (active profile's `skills/` in
   profile mode, or platform `~/.hermes/skills/` in default mode).
2. **Platform** — `<default_hermes_root>/skills/`, included in profile
   mode only. In default mode the platform dir IS the local dir, so it
   is not added twice.
3. **External** — paths from `skills.external_dirs` in config.yaml.

First match wins on skill lookup. The walk-up uses
`hermes_constants.get_default_hermes_root()`, which already handles
standard `~/.hermes`, Docker `/opt/data`, and the `~/.hermes/profiles/<name>`
profile layout uniformly.

### Single point of truth

`agent.skill_utils.get_all_skills_dirs()` is the canonical resolution
function. The three duplicates in `tools.skills_tool` and
`agent.skill_commands` are now built explicitly:

```python
# tools/skills_tool.py: skill_view + _find_all_skills
# agent/skill_commands.py: scan_skill_commands
dirs = [SKILLS_DIR]                              # local (monkeypatch-friendly)
dirs.append(platform_skills) if should_include else None
dirs.extend(get_external_skills_dirs())           # external
```

The explicit platform-dir walk-up is duplicated in three call sites
**on purpose**: the alternative — switching them all to
`get_all_skills_dirs()` — breaks the existing test pattern of
monkeypatching `tools.skills_tool.SKILLS_DIR`. The test that catches this
is `tests/agent/test_skill_commands.py::TestScanSkillCommands::test_loads_skill_invocation_from_symlinked_skill_dir`.
The three-call-site duplication costs ~30 lines of code in exchange for
preserving test ergonomics; the cleaner refactor (tests use
`monkeypatch.setenv("HERMES_HOME", ...)` instead of patching `SKILLS_DIR`)
is follow-up work.

### Stub convention

A skill at the platform level is a **stub** if its frontmatter declares
`metadata.hermes.stub: true`. Stubs are 1-2 paragraph pointers at the
canonical skill content. The dispatcher emits a single WARN log per loaded
stub via `agent.skill_commands._log_stub_fallback`:

```
[Skill stub] Loaded platform-level stub for 'totum-platform-audit'
from ~/.hermes/skills/totum-platform-audit.
Canonical content at ~/.hermes/profiles/spec-writer/skills/totum-platform-audit/SKILL.md.
```

The marker is the preferred signal; a path+size heuristic
(`<4 KB SKILL.md under platform/skills/`) is a defensive backstop for
unannotated stubs that predate the convention.

The two platform-level stubs introduced by this ADR are
`~/.hermes/skills/totum-platform/SKILL.md` and
`~/.hermes/skills/totum-platform-audit/SKILL.md`. Both declare
`metadata.hermes.stub: true` and `metadata.hermes.canonical_source`.

### Properties this ADR does NOT change

- Skill *content* — the stubs contain no methodology, only pointer text.
  The canonical content lives in the spec-writer profile and is owned
  there. Editing methodology in the platform stub would diverge profiles.
- Skill *installation* — profiles that need the canonical content must
  still install it via `hermes skills install totum-platform-audit` (or
  equivalent). The stub does not auto-install; it only unblocks the
  dispatcher so a worker that lacks the skill can still spawn.
- Skill *disable* — `skills.disabled` and `skills.platform_disabled` in
  config.yaml still win over platform stubs. A profile that opts out of a
  skill stays opted out.
- Skill *external_dirs precedence* — `skills.external_dirs` is still
  read AFTER the platform dir. Operators who want to shadow a platform
  stub with a profile-specific version can use external_dirs (or just
  install the skill locally and let local win via precedence).

## Consequences

### Positive

- **Closes the Jun 20 crash class.** 41 worker logs with
  `Error: Unknown skill(s): ...` no longer reproduce for any of the 12
  listed skill names; the dispatcher now resolves them via the platform
  stub or via external_dirs.
- **Single contract.** All callers resolve skills the same way
  (local → platform → external). Future skill discovery changes have one
  place to land.
- **Operator visibility.** The `[Skill stub]` WARN log tells operators
  which tasks loaded a stub, so they can decide whether to install the
  canonical skill or accept the stub.
- **Backward-compatible test pattern.** Existing tests that
  monkeypatch `tools.skills_tool.SKILLS_DIR` keep working — the local
  override wins by precedence.

### Negative / Trade-offs

- **Three near-duplicate platform-dir insertions.** Acceptable because
  switching to `get_all_skills_dirs()` breaks test compatibility and the
  duplicated block is small (~15 lines per call site, well-commented).
- **Profile-mode workers now see more skills.** If an operator's
  intention was for a profile to be strictly isolated from the platform
  skills/, they can opt out per-profile via `skills.platform_disabled`
  in config.yaml. This is the same opt-out mechanism used today for
  platform-specific skill disabling.
- **Stub heuristics may false-positive.** A real skill under 4 KB
  loaded from `~/.hermes/skills/` (without a stub marker) will trigger
  the `[Skill stub heuristic]` warning. The marker is the
  authoritative signal; the heuristic exists only to surface unannotated
  stubs that predate this convention. Operators who hit a false positive
  should add `metadata.hermes.stub: false` to suppress the heuristic, or
  move the skill out of `~/.hermes/skills/` if it really is canonical.

## Alternatives considered

1. **No fix; let workers crash on Unknown skill.** Rejected. The crash
   class is the bug this ADR exists to close. Doing nothing keeps the
   41-log/week baseline error rate.

2. **Fix only the two named skills (totum-platform, totum-platform-audit)
   by adding stubs at the profile level on each profile.** Rejected.
   That solves 3 of the 41 logs and creates a maintenance burden — every
   new profile would need stub copies of every cross-profile skill.

3. **Switch all four resolution sites to `get_all_skills_dirs()` and
   update the existing `SKILLS_DIR` monkeypatch tests to use
   `monkeypatch.setenv("HERMES_HOME", ...)`.** Rejected for this ADR —
   higher risk, larger diff, and the existing test pattern is widely
   used in the codebase. Logged as follow-up cleanup.

4. **Make the dispatcher pass through `--external-skills-dir` for every
   worker instead of relying on HERMES_HOME.** Rejected. The dispatcher
   doesn't own the skill-resolution path; the resolution layer should
   be self-sufficient. ADR-0001 is the right place to fix the lookup
   chain.

## Follow-up

- [ ] Refactor `_find_all_skills`, `skill_view`, and `scan_skill_commands`
      to share a single `get_all_skills_dirs(local_override=...)` helper
      so the platform-dir walk-up lives in one place. Update the
      `SKILLS_DIR`-monkeypatch tests to use `monkeypatch.setenv`. See
      SPEC-6 follow-up ticket (to be created).
- [ ] Audit the 12 skill names in the 7d crash log and add stubs at
      the platform level for any that do not yet have them. New SPEC
      beyond SPEC-6 scope; parked here for the operator to triage.
- [ ] Add a `gap-detection` signal that fires when a worker task loads
      a stub more than N times in a 7d window — operator signal that
      the canonical skill needs installing on more profiles.

## References

- `hermes-agent/agent/skill_utils.py:get_all_skills_dirs()` (resolution
  contract)
- `hermes-agent/tools/skills_tool.py:skill_view()` and
  `_find_all_skills()` (dispatcher skill lookup path)
- `hermes-agent/agent/skill_commands.py:scan_skill_commands()` and
  `build_preloaded_skills_prompt()` (slash-command discovery + preload
  prompt assembly + stub-fallback logging)
- `hermes-agent/hermes_cli/kanban_db.py:_default_spawn()` (worker
  spawn-time `HERMES_HOME` pin — unchanged by this ADR; this ADR makes
  profile-mode HERMES_HOME compatible with platform-level skills instead
  of touching the spawner)
- `~/.hermes/skills/totum-platform/SKILL.md` and
  `~/.hermes/skills/totum-platform-audit/SKILL.md` (the two stubs)
- `tests/agent/test_platform_level_skill_resolution.py` (9 new tests
  pinning the contract)
