---
name: skillhub-install
description: "Use when installing skills from SkillHub (skillhub.cn) marketplace. Implements SkillSource adapter with router injection for native Hermes integration, downloads ZIP packages, validates security via core scanner, converts frontmatter, and installs through quarantine pipeline."
version: 2.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [skills, marketplace, skillhub, install, security, skillsource]
    related_skills: [hermes-agent]
---

# SkillHub Install

## Overview

Install skills from the SkillHub (skillhub.cn) marketplace into Hermes using a **SkillSource adapter** that integrates natively with Hermes' built-in skill installation pipeline.

This skill implements the `SkillSource` ABC from `tools.skills_hub` and uses **dynamic router injection** to register SkillHub as a first-class source alongside GitHub and ClawHub. The installation script (`scripts/install_skill.py`) provides both:

1. **Standalone CLI** — run directly without Hermes runtime
2. **Router integration** — injects `SkillHubSource` into the source router for future `hermes skills install skillhub:<slug>` support

The full security pipeline is preserved: path-traversal protection → quarantine staging → core security scanner → lockfile recording → audit logging.

SkillHub is a Chinese-optimized skills marketplace with 35,000+ skills. This skill handles the complete pipeline: discover → download → validate → scan → install.

## When to Use

- User asks to install a skill from SkillHub or skillhub.cn
- User provides a SkillHub URL (e.g., `https://skillhub.cn/skills/baidu-search`)
- User mentions a skill slug and SkillHub as the source
- Browsing or searching SkillHub for available skills

Don't use for:
- Skills from ClawHub or other marketplaces (use `hermes skills install` directly)
- Creating new skills (use `skill_manage` action='create')
- Skills already in the local `~/.hermes/skills/` directory

## Quick Start

### Standalone Script (Primary Method)

```bash
# Install by slug
python3 scripts/install_skill.py baidu-search

# Install from SkillHub URL
python3 scripts/install_skill.py https://skillhub.cn/skills/baidu-search

# Specify category
python3 scripts/install_skill.py my-skill --category productivity

# Force reinstall
python3 scripts/install_skill.py my-skill --force

# Non-interactive (skip confirmation prompts, for CI/cron)
python3 scripts/install_skill.py my-skill --yes
```

### CLI Arguments

```
python3 scripts/install_skill.py <slug-or-url> [OPTIONS]

Positional:
  slug-or-url         Skill slug or SkillHub URL

Options:
  --category CAT      Target category (default: auto-detect from skill metadata)
  --force             Reinstall even if already present
  --yes, -y           Skip confirmation prompts (non-interactive mode)
  --help              Show help message
```

## API Information

| Endpoint | URL |
|----------|-----|
| API Base | `https://api.skillhub.cn` |
| Skill Details | `GET /api/v1/skills/{slug}` |
| Download ZIP | `GET /api/v1/download?slug={slug}` |
| Skill List | `GET /api/skills?page=1&pageSize=20&keyword=xxx&category=xxx` |
| Website | https://skillhub.cn |

## Architecture: SkillSource Adapter

### Core Design

The installer implements the `SkillSource` interface from `tools.skills_hub`:

```python
from tools.skills_hub import SkillSource, SkillBundle, SkillMeta

class SkillHubSource(SkillSource):
    def source_id(self) -> str:
        return "skillhub"
    
    def trust_level_for(self, identifier: str) -> str:
        return "community"  # All SkillHub skills are community-trust
    
    def inspect(self, identifier: str) -> Optional[SkillMeta]:
        # Fetch metadata from SkillHub API
        ...
    
    def search(self, query: str, limit: int = 10) -> List[SkillMeta]:
        # Search SkillHub API
        ...
    
    def fetch(self, identifier: str) -> Optional[SkillBundle]:
        # Download ZIP, validate, extract to in-memory bundle
        # Convert frontmatter (OpenClaw → Hermes)
        # Return SkillBundle ready for quarantine
        ...
```

### Router Injection

When installing, the script dynamically injects `SkillHubSource` into the source router:

```python
def _install_via_do_install(slug, category, force, skip_confirm):
    """Register SkillHub in the router and delegate to do_install."""
    import tools.skills_hub as hub
    from hermes_cli.skills_hub import do_install
    
    original = hub.create_source_router
    
    def _router_with_skillhub(auth=None):
        sources = original(auth)
        # Inject SkillHubSource if not already present
        if not any(getattr(s, "source_id", lambda: "")() == "skillhub" for s in sources):
            sources.append(SkillHubSource())
        return sources
    
    # Temporarily replace router
    hub.create_source_router = _router_with_skillhub
    try:
        do_install(slug, category=category, force=force, skip_confirm=skip_confirm)
        return True
    finally:
        # Restore original router
        hub.create_source_router = original
```

### Graceful Degradation

If Hermes core modules are unavailable (standalone mode), the script falls back to a direct installation path:

```python
def install(slug, category, force, skip_confirm):
    if _install_via_do_install(slug, category, force, skip_confirm):
        return 0  # Success via router injection
    return _install_direct(slug, category, force, skip_confirm)  # Fallback
```

`_install_direct()` manually calls the same shared installer building blocks (`quarantine_bundle`, `_scan_quarantine`, `install_from_quarantine`, etc.) to preserve the full security pipeline.

## Installation Pipeline

### Step 1: Fetch & Validate

```python
bundle = src.fetch(slug)
# - Download ZIP from SkillHub API
# - Validate every member with _validate_bundle_rel_path() (core helper)
# - Reject path traversal, absolute paths, Windows drive letters
# - Skip files > 500KB, non-UTF-8 binaries
# - Convert frontmatter (OpenClaw → Hermes format)
# - Return SkillBundle with all validated assets
```

**Security**: Uses the same `_validate_bundle_rel_path()` helper as ClawHub, ensuring consistent path validation across all sources.

### Step 2: Quarantine

```python
from tools.skills_hub import quarantine_bundle
q_path = quarantine_bundle(bundle)
# Writes bundle to: ~/.hermes/skills/.hub/quarantine/<skill-name>/
```

### Step 3: Security Scan

```python
def _scan_quarantine(q_path, bundle):
    """Prefers scan_skill_cached (newer cores); degrades to scan_skill."""
    try:
        from tools.skills_guard import scan_skill_cached
        result, _provenance = scan_skill_cached(
            q_path, source=bundle.identifier, cache_dir=cache_dir
        )
        return result
    except ImportError:
        from tools.skills_guard import scan_skill
        return scan_skill(q_path, source=bundle.identifier)
```

The core scanner checks for:
- Suspicious system calls (`os.system`, `subprocess`, `eval`, `exec`)
- Dangerous file operations (`shutil.rmtree`, `rm -rf`)
- Sensitive path access (`/etc/passwd`, `.ssh/`)
- Hard-coded credentials (`API_KEY`, `SECRET`)

**Verdict levels**: `safe` | `caution` | `dangerous` | `blocked`

### Step 4: Confirmation (Interactive)

Unless `--yes` or `--force` is specified:

```python
if not force and not skip_confirm:
    print("You are installing a third-party skill at your own risk.")
    answer = input(f"Install '{bundle.name}'? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        shutil.rmtree(q_path, ignore_errors=True)
        return 0  # Cancelled
```

**Non-TTY handling**: In cron jobs or CI (piped stdin), `input()` raises `EOFError`/`KeyboardInterrupt`, which the script catches and treats as a refusal. Use `--yes` flag for automation.

### Step 5: Install

```python
from tools.skills_hub import install_from_quarantine, HubLockFile, append_audit_log

install_dir = install_from_quarantine(q_path, bundle.name, category, bundle, scan_result)
# - Moves from quarantine to: ~/.hermes/skills/<category>/<skill-name>/
# - Records in lockfile: ~/.hermes/skills/.hub/lock.json
# - Appends to audit log: ~/.hermes/skills/.hub/audit.log
```

## Frontmatter Conversion

SkillHub packages use OpenClaw-style frontmatter. The installer converts to Hermes format:

```python
def _convert_frontmatter(md_content: str) -> str:
    """Rewrite frontmatter with:
    - OpenClaw → Hermes references
    - Tags moved under metadata.hermes.tags (Title-Cased)
    - YAML-safe scalars (quotes values containing :, #, quotes, etc.)
    - Preserves version/license/author/platforms/prerequisites
    """
    ...
```

### YAML Scalar Escaping

Untrusted frontmatter may contain special characters that break YAML parsing:

```python
def _yaml_scalar(value: str) -> str:
    """Return YAML-safe representation.
    
    Quotes values containing: : # ' " or leading/trailing whitespace.
    Escapes backslashes and quotes inside quoted strings.
    Clean values stay unquoted for readability.
    """
    if (value == "" or value != value.strip()
            or value[:1] in "!&*[]{}#|>@`\"'%,-?:"
            or ": " in value or value.endswith(":") or " #" in value):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value
```

**Example**: `description: Search: the web #1` → `description: "Search: the web #1"`

## Common Pitfalls

1. **Route B duplication trap** — When integrating external registries, the instinct is to build a standalone script (Route B) that mirrors the entire security pipeline. This creates maintenance burden and security divergence risk. Use Route A: implement `SkillSource` ABC + dynamic router injection instead. See `references/route-a-migration.md` for the decision framework.

2. **Doc-code drift after major rewrites** — After rewriting code (e.g., migrating Route B → Route A), SKILL.md often has 20+ inconsistencies with the new implementation. Before committing:
   - Line-by-line cross-check every function name, class name, pipeline step
   - Verify test count claims match actual `grep -c "def test_"`
   - Confirm pipeline order in docs matches actual code flow
   - Check that code examples use actual variable names (not idealized ones)
   - Use a checklist: function signatures, test counts, file paths, CLI args, architectural claims

3. **Hard-coding `~/.hermes/skills`**. Always use `hermes_constants.get_hermes_home() / "skills"` or the `HERMES_HOME` env var. Hard-coded paths break profile isolation and native Windows layouts.

2. **Using `zipfile.extractall()` without validation**. This accepts any ZIP member path including `../../etc/passwd`. Always validate each member with `_validate_bundle_rel_path()` (core helper) before extraction.

3. **Only copying `scripts/` directory**. SkillHub packages may include `references/`, `templates/`, `assets/`, and other directories. The installer preserves ALL validated files, not just scripts.

4. **Skipping quarantine and writing directly to skills dir**. This bypasses security scanning, lockfile recording, and audit logging. Always stage in quarantine first, scan, then move.

5. **Ignoring security scan findings**. A `caution` verdict doesn't mean "safe enough to ignore" — review the findings before confirming. A `dangerous` verdict should only be force-installed if you've personally reviewed the code.

6. **Generating invalid YAML from untrusted input**. SkillHub frontmatter may contain special characters (`:`, `#`, quotes) that break YAML parsing. Always escape values with `_yaml_scalar()` before emitting frontmatter.

7. **Hard-depending on Hermes core scanner**. When running standalone (outside hermes-agent repo), `tools.skills_guard` is unavailable. Use `_scan_quarantine()` which gracefully degrades: tries `scan_skill_cached` → `scan_skill`. This ensures the installer works in all environments.

8. **`input()` in non-TTY environments**. Interactive prompts like `input("Continue? [y/N]: ")` raise `EOFError`/`KeyboardInterrupt` in cron jobs, CI, or piped stdin. The installer catches these exceptions and treats them as a refusal. Use `--yes` flag for automation.

9. **`__pycache__` committed to Git**. The `scripts/` and `tests/` directories generate `.pyc` files when running tests. A `.gitignore` is included in both directories to exclude `__pycache__/` and `*.pyc`.

10. **Not implementing SkillSource interface**. Route B (standalone script) duplicates security logic and maintenance burden. Route A (SkillSource adapter + router injection) integrates natively with Hermes core and automatically benefits from scanner updates.

## Verification Checklist

Before committing the skill:

- [ ] **Architecture**: Implements `SkillSource` ABC (Route A), not standalone script (Route B)
- [ ] **Router injection**: Uses dynamic `_router_with_skillhub()` wrapper, not core code modification
- [ ] **Security**: Uses core `_validate_bundle_rel_path()`, `_scan_quarantine()`, `install_from_quarantine()`
- [ ] **Frontmatter**: Implements `_convert_frontmatter()` with YAML-safe `_yaml_scalar()` escaping
- [ ] **Graceful degradation**: `_scan_quarantine()` falls back through `scan_skill_cached` → `scan_skill` → None
- [ ] **TTY handling**: `input()` wrapped in `try/except (EOFError, KeyboardInterrupt)`
- [ ] **CLI flags**: Supports `--yes`, `--category`, `--force`
- [ ] **No hard-coded paths**: Uses `hermes_constants.get_hermes_home()` or `HERMES_HOME` env var
- [ ] **Doc-code consistency**: Cross-check SKILL.md against actual implementation:
  - Every function name, class name, pipeline step matches
  - Test count matches `grep -c "def test_" tests/test_*.py`
  - Pipeline order in docs matches actual code flow
  - Code examples use actual variable names (not idealized)
- [ ] **Build artifacts cleaned**:
  ```bash
  find . -name "__pycache__" -o -name ".pytest_cache" -exec rm -rf {} +
  ```
- [ ] **No stale files**: Old test files, outdated reference docs deleted
- [ ] **`.gitignore` present**: At root and in subdirectories to prevent future accumulation
- [ ] Script runs without errors: `python3 scripts/install_skill.py --help`
- [ ] All 39 unit tests pass: `python3 -m pytest tests/test_skillhub_install.py -v`
- [ ] SKILL.md starts with `---` at byte 0 (no leading blank line)
- [ ] Description ≤ 1024 chars and starts with "Use when ..."
- [ ] Name ≤ 64 chars, lowercase + hyphens
- [ ] File size ≤ 100,000 chars
- [ ] Structure: Overview → When to Use → body → Common Pitfalls → Verification Checklist
- [ ] `metadata.hermes.{tags, related_skills}` present in frontmatter

## Running Tests

```bash
# Requires pytest (install: pip install pytest)
cd ~/.hermes/skills/productivity/skillhub-install

# With Hermes venv python (recommended — has pytest pre-installed):
~/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_skillhub_install.py -v

# Or with system python:
python3 -m pytest tests/test_skillhub_install.py -v
```

If pytest is missing, the test file will fail with `ModuleNotFoundError`. The test file includes a graceful import fallback at the top.

### Test Coverage (39 tests)

| Test Class | Count | Coverage |
|------------|-------|----------|
| `TestSkillSourceInterface` | 2 | `source_id()`, `trust_level_for()` |
| `TestSlugParsing` | 5 | URL extraction, bare slugs, edge cases |
| `TestInspect` | 4 | Metadata fetch, fallback, API failures |
| `TestSearch` | 3 | Keyword search, deduplication, limits |
| `TestFetch` | 4 | Bundle download, ZIP validation, frontmatter conversion |
| `TestFrontmatterConversion` | 7 | OpenClaw → Hermes, YAML escaping, metadata preservation |
| `TestYAMLScalar` | 6 | Clean values, special chars, escaping |
| `TestTagExtraction` | 3 | Top-level tags, bins → requires, deduplication |
| `TestRouterInjection` | 2 | Dynamic router, graceful degradation |
| `TestCoreScanner` | 2 | `scan_skill_cached` → `scan_skill` fallback |
| `TestNoHardCodedHome` | 1 | No `expanduser` or `.hermes/skills` in source |

## Reference

- Architecture patterns from Hermes core: `references/hermes-internal-architecture.md`
- Graceful scanner degradation pattern: `references/graceful-degradation.md`
- YAML escaping for untrusted frontmatter: `references/yaml-escaping.md`

## File Structure

```
skillhub-install/
├── SKILL.md                              # This file
├── scripts/
│   ├── install_skill.py                  # Main installer (579 lines)
│   └── .gitignore                        # Excludes __pycache__, *.pyc, .pytest_cache
├── tests/
│   ├── test_skillhub_install.py          # 39 unit tests
│   └── .gitignore                        # Excludes __pycache__, *.pyc, .pytest_cache
└── references/
    ├── route-a-migration.md              # Route A vs Route B decision framework
    ├── graceful-degradation.md           # Scanner fallback pattern
    ├── hermes-internal-architecture.md   # SkillSource, router, quarantine
    └── yaml-escaping.md                  # _yaml_scalar() pattern
```
