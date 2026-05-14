# Hermes BMAD Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an optional, project-aware BMAD-METHOD adapter for Hermes that can load and invoke BMAD workflows from projects that already contain `_bmad/`, without turning every Hermes session or agent into BMAD by default.

**Architecture:** Implement a thin adapter layer that discovers `_bmad/` from the current working directory, resolves BMAD module/config/customization metadata, exposes BMAD skills through Hermes skill listing and slash invocation only when relevant, and leaves durable multi-agent orchestration to existing Hermes systems such as profiles, delegation, cron, and Kanban. The adapter is gated by config and project detection; it does not globally inject BMAD prompts into every session.

**Tech Stack:** Python 3.11+, Hermes Agent, YAML/TOML parsing, existing Hermes skills infrastructure, existing Hermes CLI/gateway slash-command infrastructure, pytest.

---

## Product stance

This is **Variant B: Hermes BMAD Adapter**, not “Hermes becomes BMAD”. BMAD is treated as an optional project methodology pack that Hermes can understand when a project explicitly has `_bmad/` or the user explicitly invokes a BMAD skill.

### Non-negotiable boundaries

- BMAD must be **opt-in by project or explicit command**, not global always-on guidance.
- BMAD content must not be stuffed wholesale into the system prompt.
- Existing Hermes skills, Superpowers skills, Kanban, delegation, cron, and profiles remain first-class.
- The adapter must work with BMAD installed locally in a project; no mandatory npm install at Hermes runtime.
- The adapter must preserve Hermes profile isolation via `get_hermes_home()` and must not write user/global state unless explicitly requested.
- The first implementation must be read-mostly: discover, index, view, and explicit invocation only. Kanban mapping and system prompt index exposure come after the adapter is trustworthy.

### First supported use cases

1. User is in a project with `_bmad/` and asks “run BMAD help” or invokes `/bmad-help`.
2. User invokes a workflow such as `/bmad-prd`, `/bmad-create-architecture`, `/bmad-quick-dev`.
3. `skills_list` can show BMAD skills only when a BMAD project is active. System prompt index exposure is a later, feature-flagged phase.
4. `skill_view` can read BMAD skill content and bundled resources safely.
5. Hermes can use BMAD agent/persona skills as invocation prompts, but does not automatically route all work through them.

### Later use cases deliberately kept out of the first implementation

- Automatic BMAD flow selection for every coding request.
- Full BMAD party mode with persistent subagent conversation state.
- Automatic conversion of BMAD epics/stories into Hermes Kanban tasks.
- BMAD npm installer integration.
- Writing BMAD customization files from Hermes UI.

These are planned extension phases after the adapter has solid discovery/invocation semantics.

---

## Implementation location recommendation

Keep this plan and design material in:

```text
/home/ere/projects/hermes-bmad-adapter/
```

Reason: this is a cross-project research/design effort and should not dirty `/home/ere/.hermes/hermes-agent` before approval. When implementation starts, create an isolated worktree from the Hermes repo and copy this plan into that worktree under either:

```text
docs/superpowers/plans/2026-05-14-hermes-bmad-adapter.md
```

or keep it externally and reference it from the PR. I recommend copying it into the implementation worktree once Ere approves execution, because future agents working in the repo will find it through normal project docs.

---

## Executor review amendments accepted

Executor reviewed this plan as the implementation/coder agent and raised cache/cwd and phase-order risks. These amendments supersede any later task ordering that appears more aggressive.

### Revised execution phases

**Phase 0 — Reality check against real BMAD**
- Inspect at least one real `_bmad/` project layout before codifying assumptions.
- Confirm whether `manifest.yaml` lists module paths and skill paths; use manifest-driven discovery when available.
- Validate config layer names and merge behavior against real BMAD output.

**Phase 1 — Core read-only adapter**
- Data models, discovery, config resolver, skill loader, resource safety, project index.
- Include config gates (`enabled`, `allowed_roots`, `disabled_skills`) before any user-visible exposure.
- Discovery must stop at the nearest git root by default so a parent workspace `_bmad/` does not leak into nested repositories.

**Phase 2 — Explicit tool surface only**
- `skills_list` and `skill_view("bmad:...")` only.
- Do not touch `prompt_builder.py` in this phase.
- Ensure BMAD still appears when local Hermes skills dir is empty.

**Phase 3 — Explicit slash invocation**
- `/bmad-*` commands only after BMAD slash-command cache is keyed by active project identity and fingerprint.
- Keep Hermes skill command cache separate from BMAD command cache or make the composite cache explicitly project-aware.

**Phase 4 — Prompt index, feature flagged**
- `bmad.expose_in_skill_index` defaults to `false`.
- Enable only after cache keys include active project root, BMAD root, adapter config subset, and a file-level BMAD fingerprint.
- The fingerprint must cover `_bmad/**/SKILL.md`, `_bmad/config*.toml`, `_bmad/custom/config*.toml`, and `_bmad/_config/manifest.yaml`; `_bmad/` directory mtime alone is insufficient.

**Phase 5 — Docs and broad validation**
- Document the adapter and its opt-in/project-scoped semantics.
- Run targeted BMAD tests, touched integration tests, and a real fixture validation pass.

### Additional hard requirements from review

- `find_bmad_root()` gets `stop_at_git_root=True` by default.
- Gating/filtering happens in `agent.bmad.index` before prompt/slash/tool integrations consume BMAD data.
- `skill_view("bmad:...")` must intercept the `bmad:` namespace before plugin-qualified skill dispatch.
- Linked resource reads are restricted to `references/`, `templates/`, `scripts/`, and `assets/`, and must reject symlink escapes.
- Add tests for invalid/missing frontmatter, duplicate skill names, cwd/project changes in one process, no local Hermes skills + active BMAD, symlink/path traversal, and git-root discovery boundaries.
- Treat project-local BMAD content as semi-trusted project instructions, closer to `AGENTS.md` than bundled Hermes skills. Invocation payloads must label BMAD instructions as project-provided and scoped to the current task.

---

## File structure to create or modify

### New core adapter package

- Create: `agent/bmad/__init__.py`
  - Public adapter package marker.

- Create: `agent/bmad/models.py`
  - Dataclasses for discovered BMAD project, module, skill, agent roster entry, and resource metadata.
  - No filesystem scanning here.

- Create: `agent/bmad/discovery.py`
  - Finds project root and `_bmad/` root from a starting cwd.
  - Reads `_bmad/_config/manifest.yaml` when present.
  - Scans `_bmad/core/` and `_bmad/bmm/` for skill directories containing `SKILL.md`.

- Create: `agent/bmad/config.py`
  - Resolves BMAD TOML config layers:
    1. `_bmad/config.toml`
    2. `_bmad/config.user.toml`
    3. `_bmad/custom/config.toml`
    4. `_bmad/custom/config.user.toml`
  - Implements BMAD merge semantics: scalar override, table deep merge, arrays of tables keyed by `code` or `id`, other arrays append.

- Create: `agent/bmad/skill_loader.py`
  - Parses BMAD `SKILL.md` frontmatter/body.
  - Builds Hermes-compatible `BmadSkill` objects.
  - Resolves safe resource paths under a BMAD skill directory.

- Create: `agent/bmad/invocation.py`
  - Builds the user-message payload used when invoking BMAD skills.
  - Substitutes BMAD path variables such as `{project-root}`, `{skill-root}`, `{skill-name}`.
  - Adds runtime note that BMAD is project-scoped and not global policy.

- Create: `agent/bmad/index.py`
  - Provides high-level functions used by Hermes integrations:
    - `get_active_bmad_project(start_path=None) -> BmadProject | None`
    - `list_bmad_skills(start_path=None) -> list[BmadSkill]`
    - `get_bmad_skill(identifier, start_path=None) -> BmadSkill | None`
    - `build_bmad_skill_message(identifier, user_instruction='', start_path=None, task_id=None) -> str | None`

### Hermes integration points

- Modify: `hermes_cli/config.py`
  - Add `bmad` config defaults:
    - `enabled: true`
    - `auto_detect: true`
    - `expose_in_skill_index: false`
    - `expose_slash_commands: true`
    - `max_indexed_skills: 80`
    - `allowed_roots: []`
    - `disabled_skills: []`
  - Keep defaults safe: BMAD appears only when `_bmad/` is detected or explicit invocation names a BMAD project skill.

- Future modify: `agent/prompt_builder.py` (not PR1)
  - Future Phase 4 only: include a compact “BMAD project skills” category when `bmad.expose_in_skill_index` is explicitly enabled.
  - Do not include BMAD skill bodies in the system prompt.
  - Cache key must include active BMAD project root, BMAD root, adapter config subset, and file-level BMAD fingerprint. Directory mtimes are insufficient.

- Modify: `agent/skill_commands.py`
  - Extend slash command scanning to include project BMAD slash commands such as `/bmad-help`, `/bmad-prd`, `/bmad-quick-dev`.
  - Extend `resolve_skill_command_key()` to resolve BMAD commands without colliding with normal Hermes skills.
  - Extend `build_skill_invocation_message()` to route BMAD commands through `agent.bmad.invocation`.

- Modify: `tools/skills_tool.py`
  - Extend `skills_list` to show BMAD project skills in a separate category named `bmad-project` when active.
  - Extend `skill_view` to support identifiers like `bmad:bmad-prd` and linked resource reads.
  - Preserve path traversal protections for linked files.

- Modify: `hermes_cli/skills_config.py` only if existing enable/disable config cannot express `bmad.disabled_skills` cleanly.
  - Prefer not modifying this file in the first pass unless tests prove necessary.

### CLI/docs

- Modify: `website/docs/user-guide/skills/index.md` if present; otherwise add a new doc:
  - Create: `website/docs/user-guide/features/bmad-adapter.md`
  - Explain what the adapter does, how to install BMAD in a project, how Hermes detects it, and what it does not do.

- Modify: `website/docs/reference/cli-commands.md`
  - Mention BMAD slash commands are project-derived and available only when `_bmad/` is detected.

- Modify: `website/docs/user-guide/configuration.md`
  - Document the `bmad` config keys.

### Tests

- Create: `tests/agent/bmad/test_discovery.py`
- Create: `tests/agent/bmad/test_config.py`
- Create: `tests/agent/bmad/test_skill_loader.py`
- Create: `tests/agent/bmad/test_invocation.py`
- Create: `tests/agent/bmad/test_prompt_builder_integration.py`
- Create: `tests/agent/bmad/test_skill_commands_integration.py`
- Modify: `tests/tools/test_skills_tool.py`
  - Add BMAD project skill list/view cases if `skills_tool` is extended there.

---

## Adapter data model

### Task 1: Add BMAD dataclasses

**Files:**
- Create: `agent/bmad/__init__.py`
- Create: `agent/bmad/models.py`
- Test: `tests/agent/bmad/test_discovery.py`

- [ ] **Step 1: Write the failing dataclass import test**

Add `tests/agent/bmad/test_discovery.py`:

```python
from pathlib import Path

from agent.bmad.models import BmadProject, BmadSkill


def test_bmad_models_are_importable():
    project = BmadProject(
        project_root=Path('/workspace/app'),
        bmad_root=Path('/workspace/app/_bmad'),
        manifest_path=Path('/workspace/app/_bmad/_config/manifest.yaml'),
        config={},
        agents=[],
        skills=[],
    )
    skill = BmadSkill(
        name='bmad-help',
        description='Help choose the right BMAD workflow.',
        skill_dir=Path('/workspace/app/_bmad/core/bmad-help'),
        skill_file=Path('/workspace/app/_bmad/core/bmad-help/SKILL.md'),
        module='core',
        category='core',
        frontmatter={'name': 'bmad-help'},
    )

    assert project.bmad_root.name == '_bmad'
    assert skill.identifier == 'bmad:bmad-help'
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/ere/.hermes/hermes-agent
python -m pytest tests/agent/bmad/test_discovery.py::test_bmad_models_are_importable -q -o 'addopts='
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent.bmad'`.

- [ ] **Step 3: Create the package and dataclasses**

Create `agent/bmad/__init__.py`:

```python
"""Project-aware BMAD-METHOD adapter for Hermes.

The adapter is intentionally optional. It only exposes BMAD skills when a
project-local `_bmad/` installation is detected or a caller explicitly requests
BMAD discovery from a project path.
"""
```

Create `agent/bmad/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BmadAgent:
    code: str
    name: str
    title: str = ''
    icon: str = ''
    team: str = ''
    description: str = ''
    module: str = ''


@dataclass(frozen=True)
class BmadSkill:
    name: str
    description: str
    skill_dir: Path
    skill_file: Path
    module: str
    category: str
    frontmatter: dict[str, Any] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        return f'bmad:{self.name}'

    @property
    def slash_command(self) -> str:
        return f'/{self.name}'


@dataclass(frozen=True)
class BmadProject:
    project_root: Path
    bmad_root: Path
    manifest_path: Path | None
    config: dict[str, Any]
    agents: list[BmadAgent]
    skills: list[BmadSkill]
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/agent/bmad/test_discovery.py::test_bmad_models_are_importable -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/bmad/__init__.py agent/bmad/models.py tests/agent/bmad/test_discovery.py
git commit -m "feat: add BMAD adapter data models"
```

---

## Project discovery

### Task 2: Detect project-local `_bmad/`

**Files:**
- Create: `agent/bmad/discovery.py`
- Modify: `tests/agent/bmad/test_discovery.py`

- [ ] **Step 1: Add failing discovery tests**

Append to `tests/agent/bmad/test_discovery.py`:

```python
from agent.bmad.discovery import find_bmad_root, find_project_root


def test_find_bmad_root_from_nested_directory(tmp_path):
    project = tmp_path / 'project'
    nested = project / 'src' / 'feature'
    bmad = project / '_bmad'
    nested.mkdir(parents=True)
    (bmad / '_config').mkdir(parents=True)

    assert find_bmad_root(nested) == bmad
    assert find_project_root(nested) == project


def test_find_bmad_root_returns_none_without_bmad(tmp_path):
    nested = tmp_path / 'project' / 'src'
    nested.mkdir(parents=True)

    assert find_bmad_root(nested) is None
    assert find_project_root(nested) is None


def test_find_bmad_root_stops_at_git_root_by_default(tmp_path):
    workspace_bmad = tmp_path / '_bmad'
    project = tmp_path / 'project'
    nested = project / 'src' / 'feature'
    workspace_bmad.mkdir()
    (project / '.git').mkdir(parents=True)
    nested.mkdir(parents=True)

    assert find_bmad_root(nested) is None
    assert find_bmad_root(nested, stop_at_git_root=False) == workspace_bmad
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/agent/bmad/test_discovery.py -q -o 'addopts='
```

Expected: FAIL with import error for `agent.bmad.discovery`.

- [ ] **Step 3: Implement discovery**

Create `agent/bmad/discovery.py`:

```python
from __future__ import annotations

from pathlib import Path


def _normalize_start(start_path: str | Path | None) -> Path:
    if start_path is None:
        return Path.cwd().resolve()
    path = Path(start_path).expanduser().resolve()
    return path if path.is_dir() else path.parent


def _is_git_root(path: Path) -> bool:
    return (path / '.git').exists()


def find_bmad_root(
    start_path: str | Path | None = None,
    *,
    stop_at_git_root: bool = True,
) -> Path | None:
    current = _normalize_start(start_path)
    for candidate in (current, *current.parents):
        bmad_root = candidate / '_bmad'
        if bmad_root.is_dir():
            return bmad_root
        if stop_at_git_root and _is_git_root(candidate):
            return None
    return None


def find_project_root(start_path: str | Path | None = None) -> Path | None:
    bmad_root = find_bmad_root(start_path)
    return bmad_root.parent if bmad_root else None
```

- [ ] **Step 4: Run discovery tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_discovery.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/bmad/discovery.py tests/agent/bmad/test_discovery.py
git commit -m "feat: detect project-local BMAD installations"
```

---

## BMAD config resolution

### Task 3: Resolve BMAD TOML config layers

**Files:**
- Create: `agent/bmad/config.py`
- Create: `tests/agent/bmad/test_config.py`

- [ ] **Step 1: Write failing merge tests**

Create `tests/agent/bmad/test_config.py`:

```python
from agent.bmad.config import merge_bmad_values, resolve_bmad_config


def test_merge_scalars_tables_and_agent_arrays_by_code():
    base = {
        'communication_language': 'English',
        'nested': {'a': 1, 'b': 2},
        'agents': [
            {'code': 'bmad-agent-pm', 'name': 'John', 'title': 'PM'},
            {'code': 'bmad-agent-dev', 'name': 'Amelia'},
        ],
        'tags': ['base'],
    }
    override = {
        'communication_language': 'Romanian',
        'nested': {'b': 3, 'c': 4},
        'agents': [
            {'code': 'bmad-agent-pm', 'name': 'Ion'},
            {'code': 'bmad-agent-architect', 'name': 'Winston'},
        ],
        'tags': ['override'],
    }

    merged = merge_bmad_values(base, override)

    assert merged['communication_language'] == 'Romanian'
    assert merged['nested'] == {'a': 1, 'b': 3, 'c': 4}
    assert merged['agents'] == [
        {'code': 'bmad-agent-pm', 'name': 'Ion', 'title': 'PM'},
        {'code': 'bmad-agent-dev', 'name': 'Amelia'},
        {'code': 'bmad-agent-architect', 'name': 'Winston'},
    ]
    assert merged['tags'] == ['base', 'override']


def test_resolve_bmad_config_reads_layers_in_order(tmp_path):
    bmad = tmp_path / '_bmad'
    custom = bmad / 'custom'
    custom.mkdir(parents=True)
    (bmad / 'config.toml').write_text('communication_language = "English"\n[paths]\noutput = "docs"\n')
    (bmad / 'config.user.toml').write_text('communication_language = "Romanian"\n')
    (custom / 'config.toml').write_text('[paths]\nplanning = "planning-artifacts"\n')
    (custom / 'config.user.toml').write_text('user_name = "Ere"\n')

    config = resolve_bmad_config(bmad)

    assert config['communication_language'] == 'Romanian'
    assert config['paths'] == {'output': 'docs', 'planning': 'planning-artifacts'}
    assert config['user_name'] == 'Ere'
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/agent/bmad/test_config.py -q -o 'addopts='
```

Expected: FAIL with missing module or missing functions.

- [ ] **Step 3: Implement config resolver**

Create `agent/bmad/config.py`:

```python
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any


def _array_key(items: list[Any]) -> str | None:
    if not items or not all(isinstance(item, dict) for item in items):
        return None
    for key in ('code', 'id'):
        if all(key in item for item in items):
            return key
    return None


def merge_bmad_values(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result:
                result[key] = merge_bmad_values(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    if isinstance(base, list) and isinstance(override, list):
        key = _array_key(base) if _array_key(base) == _array_key(override) else None
        if not key:
            return copy.deepcopy(base) + copy.deepcopy(override)

        result = [copy.deepcopy(item) for item in base]
        index = {item[key]: idx for idx, item in enumerate(result)}
        for item in override:
            item_key = item[key]
            if item_key in index:
                result[index[item_key]] = merge_bmad_values(result[index[item_key]], item)
            else:
                index[item_key] = len(result)
                result.append(copy.deepcopy(item))
        return result

    return copy.deepcopy(override)


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open('rb') as f:
        data = tomllib.load(f)
    return data if isinstance(data, dict) else {}


def resolve_bmad_config(bmad_root: str | Path) -> dict[str, Any]:
    root = Path(bmad_root)
    layers = [
        root / 'config.toml',
        root / 'config.user.toml',
        root / 'custom' / 'config.toml',
        root / 'custom' / 'config.user.toml',
    ]
    resolved: dict[str, Any] = {}
    for layer in layers:
        resolved = merge_bmad_values(resolved, _read_toml(layer))
    return resolved
```

- [ ] **Step 4: Run config tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_config.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/bmad/config.py tests/agent/bmad/test_config.py
git commit -m "feat: resolve BMAD project config layers"
```

---

## Skill discovery and resource safety

### Task 4: Discover BMAD skills from `_bmad/`

**Files:**
- Create: `agent/bmad/skill_loader.py`
- Modify: `agent/bmad/discovery.py`
- Create: `tests/agent/bmad/test_skill_loader.py`

- [ ] **Step 1: Write failing skill discovery tests**

Create `tests/agent/bmad/test_skill_loader.py`:

```python
from agent.bmad.discovery import build_bmad_fingerprint, discover_bmad_skills
from agent.bmad.skill_loader import load_bmad_skill, resolve_bmad_resource


def _write_bmad_skill(root, module, name, description='BMAD skill'):
    skill_dir = root / module / name
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text(f'''---
name: {name}
description: {description}
---

# {name}

Use {{project-root}} and {{skill-root}} safely.
''')
    (skill_dir / 'references').mkdir()
    (skill_dir / 'references' / 'guide.md').write_text('# Guide\n')
    return skill_dir


def test_discover_bmad_skills_under_core_and_bmm(tmp_path):
    bmad = tmp_path / '_bmad'
    _write_bmad_skill(bmad, 'core', 'bmad-help', 'Help choose workflows')
    _write_bmad_skill(bmad, 'bmm', 'bmad-prd', 'Create or validate a PRD')

    skills = discover_bmad_skills(bmad)

    assert [skill.name for skill in skills] == ['bmad-help', 'bmad-prd']
    assert skills[0].module == 'core'
    assert skills[1].module == 'bmm'
    assert skills[1].identifier == 'bmad:bmad-prd'


def test_load_bmad_skill_rejects_path_traversal_resource(tmp_path):
    bmad = tmp_path / '_bmad'
    skill_dir = _write_bmad_skill(bmad, 'core', 'bmad-help')
    skill = load_bmad_skill(skill_dir, module='core')

    assert skill is not None
    assert skill.name == 'bmad-help'
    assert resolve_bmad_resource(skill, '../secret.txt') is None


def test_resolve_bmad_resource_allows_only_linked_resource_dirs(tmp_path):
    bmad = tmp_path / '_bmad'
    skill_dir = _write_bmad_skill(bmad, 'core', 'bmad-help')
    (skill_dir / 'notes.md').write_text('private note')
    skill = load_bmad_skill(skill_dir, module='core')

    assert resolve_bmad_resource(skill, 'references/guide.md') == skill_dir / 'references' / 'guide.md'
    assert resolve_bmad_resource(skill, 'notes.md') is None


def test_resolve_bmad_resource_rejects_symlink_escape(tmp_path):
    bmad = tmp_path / '_bmad'
    outside = tmp_path / 'outside.md'
    outside.write_text('outside')
    skill_dir = _write_bmad_skill(bmad, 'core', 'bmad-help')
    link = skill_dir / 'references' / 'outside.md'
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        import pytest
        pytest.skip(f'symlinks unavailable: {exc}')
    skill = load_bmad_skill(skill_dir, module='core')

    assert resolve_bmad_resource(skill, 'references/outside.md') is None


def test_bmad_fingerprint_changes_when_nested_skill_changes(tmp_path):
    bmad = tmp_path / '_bmad'
    skill_dir = _write_bmad_skill(bmad, 'core', 'bmad-help')
    before = build_bmad_fingerprint(bmad)
    (skill_dir / 'SKILL.md').write_text('''---
name: bmad-help
description: Changed.
---

# Changed
''')
    after = build_bmad_fingerprint(bmad)

    assert before != after


def test_bmad_fingerprint_includes_config_and_manifest(tmp_path):
    bmad = tmp_path / '_bmad'
    _write_bmad_skill(bmad, 'core', 'bmad-help')
    before = build_bmad_fingerprint(bmad)
    (bmad / 'config.toml').write_text('communication_language = "Romanian"\n')
    (bmad / '_config').mkdir()
    (bmad / '_config' / 'manifest.yaml').write_text('modules: []\n')
    after = build_bmad_fingerprint(bmad)

    assert before != after
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/agent/bmad/test_skill_loader.py -q -o 'addopts='
```

Expected: FAIL because `discover_bmad_skills` and `load_bmad_skill` do not exist.

- [ ] **Step 3: Implement skill loader and discovery**

Create `agent/bmad/skill_loader.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.skill_utils import extract_skill_description, parse_frontmatter
from agent.bmad.models import BmadSkill


def load_bmad_skill(skill_dir: str | Path, module: str, category: str | None = None) -> BmadSkill | None:
    root = Path(skill_dir)
    skill_file = root / 'SKILL.md'
    if not skill_file.is_file():
        return None
    raw = skill_file.read_text(encoding='utf-8')
    frontmatter, _body = parse_frontmatter(raw)
    name = str(frontmatter.get('name') or root.name).strip()
    description = extract_skill_description(frontmatter) or ''
    if not name:
        return None
    return BmadSkill(
        name=name,
        description=description,
        skill_dir=root,
        skill_file=skill_file,
        module=module,
        category=category or module,
        frontmatter=frontmatter,
    )


_ALLOWED_RESOURCE_DIRS = {'references', 'templates', 'scripts', 'assets'}


def _has_symlink_component(path: Path, stop_at: Path) -> bool:
    current = path
    parts = []
    while current != stop_at and current != current.parent:
        parts.append(current)
        current = current.parent
    return any(part.is_symlink() for part in parts)


def resolve_bmad_resource(skill: BmadSkill, relative_path: str) -> Path | None:
    rel = Path(relative_path)
    if rel.is_absolute() or '..' in rel.parts:
        return None
    if not rel.parts or rel.parts[0] not in _ALLOWED_RESOURCE_DIRS:
        return None

    skill_root = skill.skill_dir.resolve()
    requested_unresolved = skill.skill_dir / rel
    requested = requested_unresolved.resolve()
    try:
        requested.relative_to(skill_root)
    except ValueError:
        return None
    if _has_symlink_component(requested_unresolved, skill.skill_dir):
        return None
    return requested if requested.is_file() else None
```

Append to `agent/bmad/discovery.py`:

```python
from agent.bmad.models import BmadSkill
from agent.bmad.skill_loader import load_bmad_skill


def discover_bmad_skills(bmad_root: str | Path) -> list[BmadSkill]:
    root = Path(bmad_root)
    discovered: list[BmadSkill] = []
    for module in ('core', 'bmm'):
        module_root = root / module
        if not module_root.is_dir():
            continue
        for skill_file in sorted(module_root.rglob('SKILL.md')):
            skill = load_bmad_skill(skill_file.parent, module=module)
            if skill:
                discovered.append(skill)
    return discovered


def build_bmad_fingerprint(bmad_root: str | Path) -> tuple[tuple[str, int, int], ...]:
    root = Path(bmad_root)
    patterns = [
        '**/SKILL.md',
        'config*.toml',
        'custom/config*.toml',
        '_config/manifest.yaml',
    ]
    entries: dict[str, tuple[str, int, int]] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
                rel = path.relative_to(root).as_posix()
            except OSError:
                continue
            entries[rel] = (rel, stat.st_mtime_ns, stat.st_size)
    return tuple(entries[key] for key in sorted(entries))
```

- [ ] **Step 4: Run skill loader tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_skill_loader.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/bmad/discovery.py agent/bmad/skill_loader.py tests/agent/bmad/test_skill_loader.py
git commit -m "feat: discover BMAD project skills"
```

---

## High-level BMAD index

### Task 5: Build the active BMAD project index

**Files:**
- Create: `agent/bmad/index.py`
- Modify: `tests/agent/bmad/test_discovery.py`

- [ ] **Step 1: Add failing active project test**

Append to `tests/agent/bmad/test_discovery.py`:

```python
from agent.bmad.index import get_active_bmad_project, list_bmad_skills


def test_get_active_bmad_project_returns_config_agents_and_skills(tmp_path):
    project = tmp_path / 'app'
    bmad = project / '_bmad'
    (bmad / '_config').mkdir(parents=True)
    (bmad / 'config.toml').write_text('[[agents]]\ncode = "bmad-agent-pm"\nname = "John"\ntitle = "PM"\n')
    skill_dir = bmad / 'core' / 'bmad-help'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text('''---
name: bmad-help
description: Help choose workflows.
---

# Help
''')

    active = get_active_bmad_project(project)

    assert active is not None
    assert active.project_root == project
    assert active.config['agents'][0]['code'] == 'bmad-agent-pm'
    assert active.agents[0].name == 'John'
    assert active.skills[0].name == 'bmad-help'
    assert list_bmad_skills(project)[0].identifier == 'bmad:bmad-help'
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/agent/bmad/test_discovery.py::test_get_active_bmad_project_returns_config_agents_and_skills -q -o 'addopts='
```

Expected: FAIL because `agent.bmad.index` is missing.

- [ ] **Step 3: Implement index functions**

Create `agent/bmad/index.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.bmad.config import resolve_bmad_config
from agent.bmad.discovery import discover_bmad_skills, find_bmad_root, find_project_root
from agent.bmad.models import BmadAgent, BmadProject, BmadSkill


def _agents_from_config(config: dict[str, Any]) -> list[BmadAgent]:
    agents = config.get('agents') or []
    result: list[BmadAgent] = []
    for item in agents:
        if not isinstance(item, dict):
            continue
        code = str(item.get('code') or '').strip()
        name = str(item.get('name') or '').strip()
        if not code or not name:
            continue
        result.append(BmadAgent(
            code=code,
            name=name,
            title=str(item.get('title') or ''),
            icon=str(item.get('icon') or ''),
            team=str(item.get('team') or ''),
            description=str(item.get('description') or ''),
            module=str(item.get('module') or ''),
        ))
    return result


def get_active_bmad_project(start_path: str | Path | None = None) -> BmadProject | None:
    bmad_root = find_bmad_root(start_path)
    project_root = find_project_root(start_path)
    if not bmad_root or not project_root:
        return None
    config = resolve_bmad_config(bmad_root)
    manifest = bmad_root / '_config' / 'manifest.yaml'
    return BmadProject(
        project_root=project_root,
        bmad_root=bmad_root,
        manifest_path=manifest if manifest.exists() else None,
        config=config,
        agents=_agents_from_config(config),
        skills=discover_bmad_skills(bmad_root),
    )


def list_bmad_skills(start_path: str | Path | None = None) -> list[BmadSkill]:
    project = get_active_bmad_project(start_path)
    return project.skills if project else []


def get_bmad_skill(identifier: str, start_path: str | Path | None = None) -> BmadSkill | None:
    normalized = identifier.removeprefix('bmad:').lstrip('/')
    for skill in list_bmad_skills(start_path):
        if skill.name == normalized:
            return skill
    return None
```

- [ ] **Step 4: Run active project tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_discovery.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/bmad/index.py tests/agent/bmad/test_discovery.py
git commit -m "feat: index active BMAD project metadata"
```

---

## Invocation payloads

### Task 6: Build safe BMAD skill invocation messages

**Files:**
- Create: `agent/bmad/invocation.py`
- Modify: `agent/bmad/index.py`
- Create: `tests/agent/bmad/test_invocation.py`

- [ ] **Step 1: Write failing invocation tests**

Create `tests/agent/bmad/test_invocation.py`:

```python
from agent.bmad.invocation import build_bmad_skill_message


def test_build_bmad_skill_message_substitutes_paths_and_instruction(tmp_path):
    project = tmp_path / 'app'
    bmad = project / '_bmad'
    skill_dir = bmad / 'core' / 'bmad-help'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text('''---
name: bmad-help
description: Help choose workflows.
---

# Help

Read {project-root}/_bmad/config.toml and use {skill-root}.
''')

    message = build_bmad_skill_message('bmad-help', 'what next?', start_path=project)

    assert message is not None
    assert '[IMPORTANT: The user invoked BMAD project skill "bmad-help"' in message
    assert str(project) in message
    assert str(skill_dir) in message
    assert 'what next?' in message
    assert '{project-root}' not in message
    assert '{skill-root}' not in message


def test_build_bmad_skill_message_returns_none_when_no_project(tmp_path):
    assert build_bmad_skill_message('bmad-help', start_path=tmp_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/agent/bmad/test_invocation.py -q -o 'addopts='
```

Expected: FAIL because invocation module is missing.

- [ ] **Step 3: Implement invocation builder**

Create `agent/bmad/invocation.py`:

```python
from __future__ import annotations

from pathlib import Path

from agent.bmad.index import get_active_bmad_project, get_bmad_skill


def _replace_bmad_vars(text: str, *, project_root: Path, skill_root: Path, skill_name: str) -> str:
    return (
        text
        .replace('{project-root}', str(project_root))
        .replace('{skill-root}', str(skill_root))
        .replace('{skill-name}', skill_name)
    )


def build_bmad_skill_message(
    identifier: str,
    user_instruction: str = '',
    start_path: str | Path | None = None,
    task_id: str | None = None,
) -> str | None:
    project = get_active_bmad_project(start_path)
    if not project:
        return None
    skill = get_bmad_skill(identifier, start_path=project.project_root)
    if not skill:
        return None

    raw = skill.skill_file.read_text(encoding='utf-8')
    body = _replace_bmad_vars(
        raw,
        project_root=project.project_root,
        skill_root=skill.skill_dir,
        skill_name=skill.name,
    )
    instruction = user_instruction.strip()
    runtime = (
        f'[IMPORTANT: The user invoked BMAD project skill "{skill.name}" from '
        f'{project.project_root}. Follow this BMAD skill for this task only. '
        'Do not treat BMAD as global policy for unrelated future tasks.]'
    )
    parts = [runtime, body]
    if instruction:
        parts.append(f'\nUser instruction:\n{instruction}')
    if task_id:
        parts.append(f'\nSession/task id: {task_id}')
    return '\n\n'.join(parts)
```

Modify `agent/bmad/index.py` to re-export for convenience:

```python
# Do not import invocation at module top; invocation imports index.
```

No code change is required in `index.py` if direct imports use `agent.bmad.invocation`.

- [ ] **Step 4: Run invocation tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_invocation.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/bmad/invocation.py tests/agent/bmad/test_invocation.py
git commit -m "feat: build BMAD skill invocation payloads"
```

---

## Hermes config gates

### Task 7: Add `bmad` config defaults

**Files:**
- Modify: `hermes_cli/config.py`
- Create: `tests/agent/bmad/test_config.py` additions

- [ ] **Step 1: Add failing config default test**

Append to `tests/agent/bmad/test_config.py`:

```python
from hermes_cli.config import DEFAULT_CONFIG


def test_default_config_has_safe_bmad_gates():
    bmad = DEFAULT_CONFIG['bmad']

    assert bmad['enabled'] is True
    assert bmad['auto_detect'] is True
    assert bmad['expose_in_skill_index'] is False
    assert bmad['expose_slash_commands'] is True
    assert bmad['max_indexed_skills'] == 80
    assert bmad['allowed_roots'] == []
    assert bmad['disabled_skills'] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/agent/bmad/test_config.py::test_default_config_has_safe_bmad_gates -q -o 'addopts='
```

Expected: FAIL with `KeyError: 'bmad'`.

- [ ] **Step 3: Add config defaults**

Modify `hermes_cli/config.py` inside `DEFAULT_CONFIG`:

```python
    'bmad': {
        'enabled': True,
        'auto_detect': True,
        'expose_in_skill_index': False,
        'expose_slash_commands': True,
        'max_indexed_skills': 80,
        'allowed_roots': [],
        'disabled_skills': [],
    },
```

- [ ] **Step 4: Run config tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_config.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hermes_cli/config.py tests/agent/bmad/test_config.py
git commit -m "feat: add BMAD adapter config gates"
```

---

## Config hardening and scope controls

### Task 8: Enforce allowed roots and disabled BMAD skills

**Execution-order correction:** This task is physically placed before all user-visible integration tasks. Implement it before `prompt_builder`, slash commands, or `skills_tool` consume BMAD data.

**Files:**
- Modify: `agent/bmad/index.py`
- Modify: `agent/bmad/discovery.py`
- Create: `tests/agent/bmad/test_config_gates.py`

- [ ] **Step 1: Write failing gate tests**

Create `tests/agent/bmad/test_config_gates.py`:

```python
from unittest.mock import patch

from agent.bmad.index import list_bmad_skills


def _project_with_skill(tmp_path):
    project = tmp_path / 'app'
    skill_dir = project / '_bmad' / 'core' / 'bmad-help'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text('''---
name: bmad-help
description: Help.
---

# Help
''')
    return project


def test_bmad_disabled_hides_project_skills(tmp_path):
    project = _project_with_skill(tmp_path)
    with patch('hermes_cli.config.load_config', return_value={'bmad': {'enabled': False}}):
        assert list_bmad_skills(project) == []


def test_disabled_bmad_skill_is_filtered(tmp_path):
    project = _project_with_skill(tmp_path)
    with patch('hermes_cli.config.load_config', return_value={'bmad': {'enabled': True, 'disabled_skills': ['bmad-help']}}):
        assert list_bmad_skills(project) == []


def test_allowed_roots_blocks_outside_project(tmp_path):
    project = _project_with_skill(tmp_path)
    allowed = str(tmp_path / 'other')
    with patch('hermes_cli.config.load_config', return_value={'bmad': {'enabled': True, 'allowed_roots': [allowed]}}):
        assert list_bmad_skills(project) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/agent/bmad/test_config_gates.py -q -o 'addopts='
```

Expected: FAIL because `list_bmad_skills` ignores config gates.

- [ ] **Step 3: Implement gates in index layer**

Modify `agent/bmad/index.py`:

```python
def _bmad_adapter_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return (load_config() or {}).get('bmad') or {}
    except Exception:
        return {}


def _path_allowed(project_root: Path, allowed_roots: list[str]) -> bool:
    if not allowed_roots:
        return True
    resolved = project_root.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(Path(root).expanduser().resolve())
            return True
        except ValueError:
            continue
    return False
```

Apply in `get_active_bmad_project` before returning a project:

```python
    adapter_config = _bmad_adapter_config()
    if adapter_config.get('enabled', True) is not True:
        return None
    if not _path_allowed(project_root, list(adapter_config.get('allowed_roots') or [])):
        return None
```

Apply disabled skill filtering in `list_bmad_skills`:

```python
    adapter_config = _bmad_adapter_config()
    disabled = set(adapter_config.get('disabled_skills') or [])
    return [skill for skill in project.skills if skill.name not in disabled]
```

- [ ] **Step 4: Run gate tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_config_gates.py tests/agent/bmad -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/bmad/index.py tests/agent/bmad/test_config_gates.py
git commit -m "feat: gate BMAD adapter by config and project roots"
```

---

## Prompt index integration

### Future Task P1: Show BMAD skills in the system skill index only after project-aware cache semantics are proven

**Execution gate:** Do not execute this task in PR1. This is Phase 4 work. PR1 must keep `bmad.expose_in_skill_index: false` and rely on explicit surfaces only: `skills_list`, `skill_view`, and optionally `/bmad-*` after slash cache tests pass.

**Files:**
- Future modify: `agent/prompt_builder.py`
- Future create: `tests/agent/bmad/test_prompt_builder_integration.py`

**Future implementation requirements:**

- Add BMAD to `build_skills_system_prompt()` only when `bmad.expose_in_skill_index` is explicitly true.
- Do not include BMAD skill bodies in the system prompt; only compact name + description entries.
- Cache keys must include:
  - active project root;
  - active BMAD root;
  - adapter config subset affecting exposure;
  - file-level BMAD fingerprint.
- The BMAD fingerprint must be derived from relevant files, not `_bmad/` directory mtime. Include at minimum:
  - `_bmad/**/SKILL.md`
  - `_bmad/config*.toml`
  - `_bmad/custom/config*.toml`
  - `_bmad/_config/manifest.yaml`
- Tests must prove no leak when changing cwd/project in the same Python process.

**Future test cases:**

- `test_bmad_prompt_index_disabled_by_default`
- `test_bmad_prompt_index_enabled_only_with_feature_flag`
- `test_bmad_prompt_cache_changes_when_cwd_changes`
- `test_bmad_prompt_cache_changes_when_nested_skill_changes`
- `test_bmad_prompt_index_does_not_include_skill_body`

**Do not copy any older directory-mtime snippet into implementation.** `_bmad_root.stat().st_mtime_ns` is explicitly insufficient.

---

## Slash command integration

### Task 9: Add BMAD project slash commands

**Execution gate:** This is Phase 3. Do not implement until Phase 1 gates and Phase 2 explicit `skills_tool` surfaces pass. If implemented in PR1, it must include project-aware slash-command cache tests.

**Files:**
- Modify: `agent/skill_commands.py`
- Create: `tests/agent/bmad/test_skill_commands_integration.py`

- [ ] **Step 1: Write failing slash command tests**

Create `tests/agent/bmad/test_skill_commands_integration.py`:

```python
from unittest.mock import patch

import agent.skill_commands as sc_mod
from agent.skill_commands import build_skill_invocation_message, get_skill_commands, resolve_skill_command_key


def _make_bmad_project(tmp_path):
    project = tmp_path / 'app'
    skill_dir = project / '_bmad' / 'core' / 'bmad-help'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text('''---
name: bmad-help
description: Help choose workflows.
---

# Help
''')
    return project


def test_bmad_project_skill_registers_slash_command(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    monkeypatch.chdir(project)

    with patch.object(sc_mod, '_skill_commands', {}), patch.object(sc_mod, '_skill_commands_platform', None):
        commands = get_skill_commands()

    assert '/bmad-help' in commands
    assert commands['/bmad-help']['source'] == 'bmad-project'
    assert resolve_skill_command_key('bmad-help') == '/bmad-help'


def test_bmad_slash_invocation_builds_bmad_message(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    monkeypatch.chdir(project)

    with patch.object(sc_mod, '_skill_commands', {}), patch.object(sc_mod, '_skill_commands_platform', None):
        message = build_skill_invocation_message('/bmad-help', 'what next?')

    assert message is not None
    assert 'BMAD project skill "bmad-help"' in message
    assert 'what next?' in message


def test_bmad_slash_commands_rescan_when_project_changes(tmp_path, monkeypatch):
    project = _make_bmad_project(tmp_path)
    other = tmp_path / 'other'
    other.mkdir()

    with patch.object(sc_mod, '_skill_commands', {}), patch.object(sc_mod, '_skill_commands_platform', None):
        monkeypatch.chdir(project)
        assert '/bmad-help' in get_skill_commands()

        monkeypatch.chdir(other)
        assert '/bmad-help' not in get_skill_commands()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/agent/bmad/test_skill_commands_integration.py -q -o 'addopts='
```

Expected: FAIL because BMAD commands are not registered.

- [ ] **Step 3: Extend skill command scanning**

Modify `agent/skill_commands.py` without folding BMAD commands into the existing process-global `_skill_commands` cache. Keep the normal Hermes skill cache as-is, and add a separate BMAD command cache keyed by active project identity + fingerprint:

```python
_bmad_skill_commands_cache: dict[tuple, dict[str, dict]] = {}


def _active_bmad_command_cache_key() -> tuple | None:
    try:
        from agent.bmad.index import get_active_bmad_project
        from agent.bmad.discovery import build_bmad_fingerprint
    except Exception:
        return None
    project = get_active_bmad_project()
    if not project:
        return None
    return (
        str(project.project_root.resolve()),
        str(project.bmad_root.resolve()),
        build_bmad_fingerprint(project.bmad_root),
    )


def _get_bmad_skill_commands() -> dict[str, dict]:
    key = _active_bmad_command_cache_key()
    if key is None:
        return {}
    cached = _bmad_skill_commands_cache.get(key)
    if cached is not None:
        return dict(cached)

    from agent.bmad.index import list_bmad_skills
    commands: dict[str, dict] = {}
    for skill in list_bmad_skills():
        commands[f'/{skill.name}'] = {
            'name': skill.name,
            'description': skill.description,
            'skill_dir': str(skill.skill_dir),
            'source': 'bmad-project',
        }
    _bmad_skill_commands_cache.clear()
    _bmad_skill_commands_cache[key] = commands
    return dict(commands)
```

In `get_skill_commands()`, build the normal cached Hermes result first, then merge BMAD commands dynamically into a shallow composite result so cwd/project changes are reflected:

```python
    result = dict(_skill_commands)
    for key, value in _get_bmad_skill_commands().items():
        result.setdefault(key, value)
    return result
```

Do not store BMAD commands inside `_skill_commands` unless `_skill_commands` itself is rekeyed by active BMAD project.

Modify `build_skill_invocation_message()` near the existing skill lookup:

```python
    if skill_info.get('source') == 'bmad-project':
        from agent.bmad.invocation import build_bmad_skill_message
        return build_bmad_skill_message(skill_info['name'], user_instruction, task_id=task_id)
```

- [ ] **Step 4: Run slash command tests**

Run:

```bash
python -m pytest tests/agent/bmad/test_skill_commands_integration.py tests/agent/test_skill_commands.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/skill_commands.py tests/agent/bmad/test_skill_commands_integration.py
git commit -m "feat: expose BMAD project skills as slash commands"
```

---

## skills_list and skill_view integration

### Task 10: Show and inspect BMAD skills through tools

**Files:**
- Modify: `tools/skills_tool.py`
- Modify: `tests/tools/test_skills_tool.py`

- [ ] **Step 1: Add failing tool tests**

Append to `tests/tools/test_skills_tool.py`:

```python
def test_skills_list_includes_active_bmad_project_skill_even_without_local_skills(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch
    import tools.skills_tool as skills_tool

    project = tmp_path / 'app'
    empty_hermes_skills = tmp_path / 'empty-hermes-skills'
    skill_dir = project / '_bmad' / 'core' / 'bmad-help'
    skill_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text('''---
name: bmad-help
description: Help choose workflows.
---

# Help
''')
    monkeypatch.chdir(project)

    with patch.object(skills_tool, 'SKILLS_DIR', empty_hermes_skills):
        result = json.loads(skills_tool.skills_list())

    names = [skill['name'] for skill in result['skills']]
    assert 'bmad:bmad-help' in names


def test_skill_view_reads_bmad_project_skill_resource(tmp_path, monkeypatch):
    import json
    import tools.skills_tool as skills_tool

    project = tmp_path / 'app'
    skill_dir = project / '_bmad' / 'core' / 'bmad-help'
    ref_dir = skill_dir / 'references'
    ref_dir.mkdir(parents=True)
    (skill_dir / 'SKILL.md').write_text('''---
name: bmad-help
description: Help choose workflows.
---

# Help
''')
    (ref_dir / 'guide.md').write_text('# Guide\n')
    monkeypatch.chdir(project)

    skill_result = json.loads(skills_tool.skill_view('bmad:bmad-help'))
    ref_result = json.loads(skills_tool.skill_view('bmad:bmad-help', 'references/guide.md'))

    assert skill_result['success'] is True
    assert skill_result['name'] == 'bmad:bmad-help'
    assert 'Help choose workflows' in skill_result['content']
    assert ref_result['success'] is True
    assert '# Guide' in ref_result['content']
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/tools/test_skills_tool.py::test_skills_list_includes_active_bmad_project_skill_even_without_local_skills tests/tools/test_skills_tool.py::test_skill_view_reads_bmad_project_skill_resource -q -o 'addopts='
```

Expected: FAIL because `skills_tool` does not know `bmad:` identifiers.

- [ ] **Step 3: Extend skills tool**

Modify `tools/skills_tool.py`:

- In `skills_list`, refactor the flow so BMAD skills are collected even when `SKILLS_DIR` does not exist or normal Hermes skill discovery returns empty. Do not append after an early return. Use this structure:

```python
    skills = []

    # Existing Hermes skill discovery populates skills when local/external skills exist.
    # It must not return before BMAD discovery gets a chance to run.
    if SKILLS_DIR.exists() or external_skill_dirs:
        skills.extend(_collect_regular_hermes_skills(...))

    try:
        from agent.bmad.index import list_bmad_skills
        for skill in list_bmad_skills():
            skills.append({
                'name': skill.identifier,
                'description': skill.description,
                'category': 'bmad-project',
                'path': str(skill.skill_dir),
            })
    except Exception:
        pass

    return json.dumps({'success': True, 'skills': skills, 'count': len(skills)})
```

Preserve the existing response shape and existing regular-skill behavior; the important change is eliminating early returns before BMAD collection.

- In `skill_view`, intercept `bmad:` before plugin-qualified skill dispatch or normal skill lookup. Resource reads must use `resolve_bmad_resource()` with allowlisted top-level directories and symlink-escape rejection:

```python
    if name.startswith('bmad:'):
        from agent.bmad.index import get_bmad_skill
        from agent.bmad.skill_loader import resolve_bmad_resource
        skill = get_bmad_skill(name)
        if not skill:
            return json.dumps({'success': False, 'error': f'BMAD skill not found: {name}'})
        if file_path:
            resource = resolve_bmad_resource(skill, file_path)
            if not resource:
                return json.dumps({'success': False, 'error': 'Resource not found or outside BMAD skill directory'})
            return json.dumps({'success': True, 'name': name, 'file_path': file_path, 'content': resource.read_text(encoding='utf-8')})
        return json.dumps({
            'success': True,
            'name': name,
            'description': skill.description,
            'content': skill.skill_file.read_text(encoding='utf-8'),
            'path': str(skill.skill_file),
            'skill_dir': str(skill.skill_dir),
        })
```

Adapt field names to the existing `skills_tool.py` response structure so existing tests continue passing.

- [ ] **Step 4: Run skills tool tests**

Run:

```bash
python -m pytest tests/tools/test_skills_tool.py tests/tools/test_skill_view_path_check.py -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/skills_tool.py tests/tools/test_skills_tool.py
git commit -m "feat: expose BMAD project skills through skills tools"
```

---

## Documentation

### Task 12: Add user and config docs

**Files:**
- Create: `website/docs/user-guide/features/bmad-adapter.md`
- Modify: `website/docs/user-guide/configuration.md`
- Modify: `website/docs/reference/cli-commands.md`

- [ ] **Step 1: Write docs page**

Create `website/docs/user-guide/features/bmad-adapter.md`:

```markdown
---
title: BMAD Adapter
description: Use project-local BMAD-METHOD skills from Hermes without making BMAD global behavior.
---

# BMAD Adapter

Hermes can detect a project-local BMAD-METHOD installation and expose its workflows as project-scoped skills. The adapter is optional: BMAD appears only when your current working directory is inside a project that contains `_bmad/`, or when you explicitly invoke a BMAD project skill from that project.

## What it does

- Detects `_bmad/` by walking upward from the current working directory.
- Reads BMAD skills from `_bmad/core/` and `_bmad/bmm/`.
- Shows BMAD skills in `skills_list` under `bmad-project`. System prompt skill-index exposure is disabled by default and deferred to a future phase.
- Adds slash commands such as `/bmad-help`, `/bmad-prd`, and `/bmad-quick-dev` when those skills exist in the active project.
- Supports `skills_list` and `skill_view` identifiers such as `bmad:bmad-help`.

## What it does not do

- It does not turn every Hermes session into a BMAD session.
- It does not inject all BMAD instructions into the system prompt.
- It does not install BMAD for you.
- It does not automatically convert BMAD stories into Kanban tasks in the first version.

## Configuration

```yaml
bmad:
  enabled: true
  auto_detect: true
  expose_in_skill_index: false  # future prompt-index exposure; keep false for PR1
  expose_slash_commands: true
  max_indexed_skills: 80
  allowed_roots: []
  disabled_skills: []
```

Set `bmad.enabled: false` to disable the adapter. Use `allowed_roots` to restrict BMAD discovery to approved workspaces.
```

- [ ] **Step 2: Document config keys in configuration reference**

Add a `BMAD adapter` subsection to `website/docs/user-guide/configuration.md`:

```markdown
### BMAD adapter

The `bmad` section controls project-local BMAD-METHOD discovery.

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `true` | Enables the adapter. BMAD is still project-scoped and requires `_bmad/`. |
| `auto_detect` | `true` | Detect `_bmad/` from the current working directory. |
| `expose_in_skill_index` | `false` | Show active BMAD project skills in the system skill index. Keep this off until project-aware prompt-cache semantics are proven. |
| `expose_slash_commands` | `true` | Register project BMAD slash commands such as `/bmad-help`. |
| `max_indexed_skills` | `80` | Caps BMAD skills listed in the compact system prompt index. |
| `allowed_roots` | `[]` | Optional list of workspace roots where BMAD discovery is allowed. Empty means no root restriction. |
| `disabled_skills` | `[]` | BMAD skill names to hide from list, view, prompt index, and slash commands. |
```

- [ ] **Step 3: Document dynamic slash commands**

Add to `website/docs/reference/cli-commands.md` near skills/slash commands:

```markdown
### Project BMAD commands

When the BMAD adapter is enabled and the current directory is inside a project containing `_bmad/`, Hermes registers BMAD project skills as slash commands. Examples include `/bmad-help`, `/bmad-prd`, and `/bmad-quick-dev`, depending on the project's installed BMAD modules. These commands are dynamic and disappear outside BMAD projects.
```

- [ ] **Step 4: Run docs sanity checks**

Run:

```bash
python -m pytest tests/website/test_generate_skill_docs.py -q -o 'addopts='
```

Expected: PASS or SKIP depending on local website test dependencies. If it fails due to unrelated pre-existing website dependencies, record the failure in the implementation notes and run targeted Python tests from previous tasks.

- [ ] **Step 5: Commit**

```bash
git add website/docs/user-guide/features/bmad-adapter.md website/docs/user-guide/configuration.md website/docs/reference/cli-commands.md
git commit -m "docs: document BMAD adapter"
```

---

## Manual integration test with real BMAD checkout

### Task 13: Validate against `/home/ere/BMAD-METHOD` or a real BMAD project

**Files:**
- No code files unless a defect is found.
- Add implementation notes to PR body or `docs/superpowers/plans/2026-05-14-hermes-bmad-adapter.md` if this plan is copied into the repo.

Split validation by phase so PR1 does not accidentally require slash commands.

### PR1 required validation: discovery + skills tools

- [ ] **Step 1: Create a disposable BMAD-like fixture project**

Run:

```bash
cd /tmp
rm -rf hermes-bmad-fixture
mkdir -p hermes-bmad-fixture/_bmad/core/bmad-help/references
cat > hermes-bmad-fixture/_bmad/core/bmad-help/SKILL.md <<'EOF'
---
name: bmad-help
description: Help choose the right BMAD workflow.
---

# BMAD Help

Use {project-root} and {skill-root}.
EOF
cat > hermes-bmad-fixture/_bmad/core/bmad-help/references/guide.md <<'EOF'
# Guide
EOF
```

Expected: fixture created under `/tmp/hermes-bmad-fixture`.

- [ ] **Step 2: Verify Python adapter discovery manually**

Run:

```bash
cd /tmp/hermes-bmad-fixture
PYTHONPATH=/home/ere/.hermes/hermes-agent python - <<'PY'
from agent.bmad.index import list_bmad_skills
print([(s.identifier, s.description) for s in list_bmad_skills()])
PY
```

Expected output contains:

```text
[('bmad:bmad-help', 'Help choose the right BMAD workflow.')]
```

- [ ] **Step 3: Verify `skills_list` and `skill_view` manually**

Run:

```bash
cd /tmp/hermes-bmad-fixture
PYTHONPATH=/home/ere/.hermes/hermes-agent python - <<'PY'
import json
from tools.skills_tool import skills_list, skill_view
listed = json.loads(skills_list())
print(any(s.get('name') == 'bmad:bmad-help' for s in listed.get('skills', [])))
view = json.loads(skill_view('bmad:bmad-help'))
print(view.get('success') is True)
ref = json.loads(skill_view('bmad:bmad-help', 'references/guide.md'))
print(ref.get('success') is True)
PY
```

Expected output:

```text
True
True
True
```

- [ ] **Step 4: Verify normal project remains unaffected for required PR1 surfaces**

Run:

```bash
cd /tmp
PYTHONPATH=/home/ere/.hermes/hermes-agent python - <<'PY'
from agent.bmad.index import list_bmad_skills
print(list_bmad_skills())
PY
```

Expected output:

```text
[]
```

### Optional validation only if Task 9 slash commands are included

- [ ] **Optional Step 5: Verify slash command payload manually**

Run this only if Phase 3 / Task 9 is implemented:

```bash
cd /tmp/hermes-bmad-fixture
PYTHONPATH=/home/ere/.hermes/hermes-agent python - <<'PY'
from agent.skill_commands import build_skill_invocation_message, get_skill_commands
print('/bmad-help' in get_skill_commands())
msg = build_skill_invocation_message('/bmad-help', 'what next?')
print('BMAD project skill "bmad-help"' in msg)
print('/tmp/hermes-bmad-fixture' in msg)
PY
```

Expected output:

```text
True
True
True
```

- [ ] **Optional Step 6: Verify normal project remains unaffected for slash commands**

Run this only if Phase 3 / Task 9 is implemented:

```bash
cd /tmp
PYTHONPATH=/home/ere/.hermes/hermes-agent python - <<'PY'
from agent.skill_commands import get_skill_commands
print('/bmad-help' in get_skill_commands())
PY
```

Expected output:

```text
False
```

- [ ] **Step 7: Commit fixes if manual validation finds defects**

If any manual step fails due to adapter logic, write a regression test first, fix the code, run the targeted tests, then commit:

```bash
git add agent/bmad tests/agent/bmad tools/skills_tool.py agent/skill_commands.py
```

Only include `agent/skill_commands.py` if optional Phase 3 is implemented. Do not include `agent/prompt_builder.py` in PR1.

```bash
git commit -m "fix: handle BMAD adapter validation edge case"
```

---

## Full test pass

### Task 14: Run targeted and broad tests

**Files:**
- No new files unless failures require fixes.

- [ ] **Step 1: Run all BMAD adapter tests**

Run:

```bash
cd /home/ere/.hermes/hermes-agent
python -m pytest tests/agent/bmad -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 2: Run touched integration test areas**

Run:

```bash
python -m pytest \
  tests/agent/test_skill_commands.py \
  tests/agent/test_prompt_builder.py \
  tests/tools/test_skills_tool.py \
  tests/tools/test_skill_view_path_check.py \
  -q -o 'addopts='
```

Expected: PASS.

- [ ] **Step 3: Run full test suite if runtime is acceptable**

Run:

```bash
python -m pytest tests/ -q -o 'addopts='
```

Expected: PASS. If full suite is too slow or has unrelated pre-existing failures, capture the failing test names and logs in the PR notes and provide the targeted test pass evidence.

- [ ] **Step 4: Commit any test-stability fixes**

If fixes are needed, commit them separately:

```bash
git add <changed-files>
git commit -m "fix: stabilize BMAD adapter tests"
```

---

## Future phase plan: BMAD + Hermes Kanban

Do not implement this until the adapter has shipped and Ere approves the next phase.

### Phase 2 proposed scope

- Add a command/tool to convert BMAD story files into Hermes Kanban tasks.
- Map BMAD agent roles to Hermes profiles through config:

```yaml
bmad:
  profiles:
    analyst: default
    pm: default
    architect: default
    dev: coder
    reviewer: coder
```

- Add explicit command, not automatic behavior:

```bash
hermes bmad import-stories --project . --board ./kanban.db
```

- Keep user approval before creating or assigning tasks.

### Phase 2 likely files

- Create: `hermes_cli/bmad.py`
- Modify: `hermes_cli/main.py`
- Modify: `hermes_cli/kanban.py` only if existing APIs cannot create linked tasks programmatically.
- Create: `tests/hermes_cli/test_bmad_cli.py`

---

## Self-review

### Spec coverage

- Optional BMAD adapter, not full BMAD takeover: covered by config gates, project detection, and runtime note.
- Project-local `_bmad/`: covered by discovery and index tasks.
- BMAD config/customization foundation: covered by TOML config resolver; per-skill `customize.toml` can be added after base skill invocation is stable.
- Skills exposure: PR1 covers `skills_list` and `skill_view`; slash commands are Phase 3 after project-aware cache tests; prompt builder is future Phase 4 behind `bmad.expose_in_skill_index: false`.
- Multi-agent integration: deliberately deferred to Phase 2 so the first implementation is stable and non-invasive.
- Documentation: covered by BMAD adapter docs and config reference.
- Testing: covered with focused unit tests, integration tests, manual fixture validation, and broader touched-area tests.

### Placeholder scan

The plan contains no open implementation placeholders. Deferred work is explicitly scoped as a later phase and not required for the first adapter delivery.

### Type consistency

The plan consistently uses:

- `BmadProject`
- `BmadSkill`
- `BmadAgent`
- `bmad:<skill-name>` identifiers
- `/bmad-*` slash commands
- `bmad-project` skill category

### Main implementation risk

The highest-risk integration points are project identity and process-global caches:

- `agent/prompt_builder.py` has in-process and disk snapshot caching that is not inherently cwd/project-aware.
- `agent/skill_commands.py` has process-global command caching keyed around platform, not active project.
- Gateway sessions may not have the same obvious cwd semantics as CLI sessions.

Do not enable BMAD prompt-index exposure until cache keys include active project root, BMAD root, adapter config subset, and a file-level BMAD fingerprint. Directory mtime on `_bmad/` is not enough because nested `SKILL.md` edits may not update the parent directory mtime. If this is hard to guarantee in a first pass, keep BMAD available only through explicit tool/list/view and slash invocation surfaces.

---

## Execution handoff

### PR1 implementation checkpoint — 2026-05-14

Status: **PR1 implemented and merged into `main`** in `/home/ere/.hermes/hermes-agent`.

Merged commit:

```text
4a75ad4b4 feat: add project-scoped BMAD adapter
```

Implemented PR1 scope:

- Phase 0 reality check against real BMAD layout in `/home/ere/projects/analytics`.
- Phase 1 read-only adapter package under `agent/bmad/`:
  - discovery and git-root boundary behavior;
  - manifest-aware project/module/skill discovery;
  - TOML config resolver and merge semantics;
  - skill loader and strict frontmatter handling;
  - safe linked-resource handling with traversal/symlink escape rejection;
  - high-level project index and config gates.
- Phase 2 explicit tools surface:
  - `tools/skills_tool.py` lists active project BMAD skills as `bmad:<skill-name>` only when cwd is inside a BMAD project;
  - `skill_view("bmad:...")` resolves project-local BMAD skill content/resources and prepends a semi-trusted project-scope banner.
- Config defaults in `hermes_cli/config.py` with safe gating, including `bmad.expose_in_skill_index: false`.
- User docs for BMAD adapter, config reference, and CLI reference.

Implemented as Phase 3 follow-up after PR1 checkpoint:

- `agent/skill_commands.py` now exposes active project `/bmad-*` commands from a separate BMAD command cache keyed by active project root, BMAD root, adapter slash config subset, and file-level BMAD fingerprint.
- `build_skill_invocation_message()` routes BMAD slash commands through `agent.bmad.invocation` so the payload remains project-scoped and semi-trusted.
- CLI help/dispatch/completion and all identified TUI slash-command catalog/dispatch/completion routes now consume `get_skill_commands()` so project-derived BMAD commands are visible at execution time without being stored in the normal Hermes skill-command cache.
- BMAD slash command names are normalized with the same slug rules as regular skills, restricted to `/bmad-*`, and active BMAD project commands override same-key regular skills only while the BMAD project is active.
- `resolve_skill_command_key()` accepts both bare and slash-prefixed command names while preserving hyphen/underscore compatibility.

Explicitly **not** implemented:

- Phase 4 prompt-builder/system skill-index exposure via `agent/prompt_builder.py`.
- BMAD Kanban mapping, npm installer integration, persistent BMAD party mode, or global BMAD prompt injection.

Validation already run after merge:

```text
python -m pytest tests/agent/bmad tests/tools/test_bmad_skills_tool.py tests/tools/test_skills_tool.py tests/tools/test_skill_view_path_check.py -q -o 'addopts='
# 114 passed, 1 skipped

python -m pytest tests/website/test_generate_skill_docs.py -q -o 'addopts='
# 7 passed

python -m pytest tests/agent/test_prompt_builder.py tests/tools/test_skills_tool.py tests/tools/test_skill_view_path_check.py tests/tools/test_bmad_skills_tool.py tests/agent/bmad -q -o 'addopts='
# 236 passed, 1 skipped

python -m pytest tests/agent/bmad tests/agent/test_skill_commands.py tests/agent/test_skill_commands_reload.py -q -o 'addopts='
# 79 passed

python -m pytest tests/tools/test_bmad_skills_tool.py tests/tools/test_skill_view_path_check.py -q -o 'addopts='
# 8 passed, 1 skipped

python -m py_compile agent/skill_commands.py cli.py tui_gateway/server.py
# exit 0

python -m pytest tests/test_tui_gateway_server.py tests/cli/test_cli_background_tui_refresh.py tests/hermes_cli/test_tui_resume_flow.py -q -o 'addopts='
# 197 passed

Manual disposable fixture validation covered BMAD discovery, `skills_list`, `skill_view`, `/bmad-help` registration/invocation, and non-BMAD cwd isolation:
# [('bmad:bmad-help', 'Help choose the right BMAD workflow.')]
# True
# True
# True
# False  (normal scan cache intentionally does not contain project BMAD commands)
# True   (`get_skill_commands()` dynamic composite contains `/bmad-help`)
# True
# True
# []
# False
```

Post-restart smoke check:

```text
/home/ere/projects/analytics:
  skills_list success: true
  BMAD skills count: 68
  has bmad:bmad-quick-dev: true
  skill_view(bmad:bmad-quick-dev): true

/tmp:
  BMAD skills count: 0
```

Gateway restart checkpoint:

- `hermes-gateway.service` active after restart.
- `hermes-gateway-ana.service` active after restart.
- `hermes-gateway-coder.service` active after restart.

A new agent can resume from this plan by first re-syncing the live repository state:

```bash
cd /home/ere/.hermes/hermes-agent
git status --short
git log --oneline -5
python -m pytest tests/agent/bmad tests/tools/test_bmad_skills_tool.py tests/tools/test_skills_tool.py tests/tools/test_skill_view_path_check.py -q -o 'addopts='
```

Recommended next work, in order:

1. Ask Executor to review the Phase 3 slash-command diff before fan-in commit/push if more assurance is needed.
2. If working on PR1/Phase 3 follow-up bugs, stay within `agent/bmad/**`, `tools/skills_tool.py`, `agent/skill_commands.py`, docs, and the BMAD tests unless a failing test proves another integration point is necessary.
3. For Phase 4 prompt index exposure, do not touch `agent/prompt_builder.py` until file-level BMAD fingerprinting and cache-key invalidation are designed and tested.
4. Keep `bmad.expose_in_skill_index: false` unless Phase 4 is fully implemented and validated.
5. Treat project-local `_bmad/` content as semi-trusted project instruction, never global policy.

Preferred mode remains **Subagent-Driven**, one fresh subagent per phase/task with review between tasks, because this touches prompt caching, skills, tools, and docs.
