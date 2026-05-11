# Plan: Skill Description i18n (中文技能描述)

**Created:** 2026-05-11 12:50  
**Status:** planned  
**Goal:** Add locale-aware skill descriptions so `hermes skills list` and the system prompt `<available_skills>` can display Chinese descriptions when `display.language` is set to `zh`.

---

## Current Context

Skill descriptions flow through 3 surfaces, all English-only:

| Surface | File | Line | Format |
|---------|------|------|--------|
| System prompt `<available_skills>` | `agent/prompt_builder.py` | 845 | `- name: desc` (60-char trunc) |
| Tool `skills_list()` | `tools/skills_tool.py` | 594 | `{"description": "..."}` |
| CLI `hermes skills list` | `hermes_cli/skills_hub.py` | 807 | **no description shown** (name/category/source/trust/status only) |

Description extraction is centralized in `agent/skill_utils.py:extract_skill_description()` — a single 9-line function that reads `frontmatter["description"]` and truncates to 60 chars.

Snapshot caching (`agent/prompt_builder.py` → `.skills_prompt_snapshot.json`) bypasses the live filesystem after first scan. Cache key is `(skills_dir_mtime, tools, toolsets)`. Does **not** include language.

---

## Proposed Approach

Add `description_<lang>` fields to SKILL.md frontmatter. Config-driven locale resolution with fallback to English.

### Why this approach (vs alternatives)

| Alternative | Verdict |
|-------------|---------|
| External i18n files per skill dir | Over-engineered for a single field |
| Full skill body translation | Too heavy; not what user asked for |
| `description_zh` in frontmatter | ✅ Minimal, backward-compatible, no new deps |

### Config

```yaml
# ~/.hermes/config.yaml
display:
  language: zh   # default: en. Controls skill description locale.
```

---

## Files to Change

### 1. `agent/skill_utils.py` — Core extraction logic

**`extract_skill_description()` (line 426):** Add `language="en"` parameter.

```python
def extract_skill_description(frontmatter: Dict[str, Any], language: str = "en") -> str:
    # 1. Try locale-specific field first: description_zh
    if language and language != "en":
        locale_key = f"description_{language}"
        raw_desc = frontmatter.get(locale_key, "")
        if raw_desc:
            desc = str(raw_desc).strip().strip("'\"")
            if desc:
                if len(desc) > 60:
                    return desc[:57] + "..."
                return desc
    # 2. Fall back to description
    raw_desc = frontmatter.get("description", "")
    ...
```

Also add a helper to read language from config:

```python
def get_display_language() -> str:
    """Read display.language from config. Returns 'en' on any error."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("display", {}).get("language", "en") or "en"
    except Exception:
        return "en"
```

### 2. `agent/prompt_builder.py` — System prompt + snapshot

**`_parse_skill_file()` (line 604):** Accept `language` param, pass to `extract_skill_description()`.

**`_build_snapshot_entry()`:** Add `language` to stored description (the snapshot stores one description string — it should store the resolved one for the current language).

**`build_skills_system_prompt()` (line 654):**  
- Read `display.language` early.
- Pass it through to all description extraction.
- **Critical:** Include language in the snapshot cache key so switching language triggers a rescan. Change the cache key from `(skills_dir, tools, toolsets)` to `(skills_dir, tools, toolsets, language)`.

**Snapshot file (`_load_skills_snapshot` / `_write_skills_snapshot`):**  
- Add `"language": "zh"` to the snapshot metadata.
- On load, if `snapshot["language"] != current_language`, treat as cache miss (return None).

**Snapshot invalidation (line 546):** The manifest check at line 546 uses `_build_skills_manifest(skills_dir)` which returns mtime/size dict. Add language to the composite key so `snapshot["language"] != current_language` → miss.

### 3. `tools/skills_tool.py` — Tool interface

**`_find_all_skills()` (line 549):**  
- Read `display.language` once.
- Pass it to `extract_skill_description()` via the frontmatter parsing path.

Actually, `_find_all_skills` calls `_parse_frontmatter()` then reads `frontmatter.get("description")` directly at line 594. Need to replace that direct read with a call to `extract_skill_description(frontmatter, language)`.

**`skill_view()`:** The full skill content is returned as-is (no locale filtering on the body). But the `description` field in the JSON response should respect locale. Currently `skill_view()` returns frontmatter fields directly — needs locale-aware description resolution in the response dict.

### 4. `hermes_cli/skills_hub.py` — CLI table

**`do_list()` (line 761):**  
- Add a `Description` column to the Rich table (between Category and Source).
- Populate it with `skill["description"]` (already locale-resolved from `_find_all_skills`).

This is a UX improvement that makes the i18n visible — currently `hermes skills list` shows no description at all.

### 5. `hermes_cli/config.py` — Config defaults (optional)

Add `display.language` to the default config block if one exists, defaulting to `en`. Not strictly required since `load_config().get("display", {}).get("language", "en")` handles missing keys.

### 6. `hermes_cli/setup.py` or `hermes setup` wizard (optional stretch)

Add `display.language` to the interactive setup wizard so users can set it without editing YAML.

### 7. SKILL.md authoring documentation

Update `tools/skills_tool.py` docstring (line ~10, "Metadata" section) and the `hermes-agent-skill-authoring` skill to document `description_zh` as a supported frontmatter field.

---

## Step-by-Step Plan

1. **Add `get_display_language()` helper** to `agent/skill_utils.py`
2. **Modify `extract_skill_description()`** to accept language param and check `description_<lang>`
3. **Update `_find_all_skills()`** in `tools/skills_tool.py` to use locale-aware description
4. **Update `skill_view()`** in `tools/skills_tool.py` for locale-aware description in response
5. **Update `_parse_skill_file()` and `build_skills_system_prompt()`** in `agent/prompt_builder.py` to pass language through
6. **Add language to snapshot cache key** — invalidate on language change
7. **Add Description column** to `do_list()` in `hermes_cli/skills_hub.py`
8. **Add `display.language` to config defaults** if a defaults block exists
9. **Add Chinese descriptions** to a few bundled skills as proof-of-concept (e.g., `hermes-agent`, `plan`, `writing-plans`)
10. **Run existing tests**, add new tests for locale fallback
11. **Write CHANGELOG / commit**

---

## Validation / Testing

### Manual
```bash
# Default (en)
hermes skills list

# After setting display.language: zh in config.yaml
hermes skills list    # should show Chinese descriptions

# System prompt injection — start a chat and check /config shows the right language
hermes -s hermes-agent  # should load skill with Chinese description
```

### Automated tests to add
- `test_extract_skill_description_falls_back_to_en` — no `description_zh`, returns `description`
- `test_extract_skill_description_uses_zh` — `description_zh` present, language=zh
- `test_extract_skill_description_ignores_zh_when_language_en` — `description_zh` present but language=en
- `test_snapshot_cache_invalidates_on_language_change` — change language → rescan
- `test_skills_list_returns_localized_descriptions` — integration test

### Existing tests that should still pass
- `tests/tools/test_skills_tool.py`
- `tests/agent/test_curator_reports.py`
- `tests/agent/test_curator.py`
- `tests/agent/test_skill_commands_reload.py`

---

## Risks & Tradeoffs

| Risk | Mitigation |
|------|------------|
| Snapshot cache doesn't invalidate on language switch → stale descriptions | Add `language` to cache key + snapshot metadata field |
| Chinese text in system prompt increases token usage (~10-30% per description) | Truncation at 60 chars limits impact; user opts in by setting `display.language` |
| Some SKILL.md files don't have `description_zh` → mixed language output | Graceful fallback to `description` (English); no errors |
| `display.language` config key might conflict with future use | Namespaced under `display.` which already has `skin`; `language` is natural |
| External skill dirs (read-only) can't be translated | They fall back to English; prioritized local skills override |

---

## Open Questions

1. **Language code format:** `zh` or `zh-CN`? Leaning toward `zh` for simplicity — can add `zh-CN` → `zh` normalization later.
2. **Should `hermes setup` wizard include language selection?** Stretch goal. Not required for MVP.
3. **Should tool descriptions also be localizable?** Out of scope for this PR — those are code-level strings, much larger effort.

---

## Files Summary

| File | Change | Risk |
|------|--------|------|
| `agent/skill_utils.py` | Add `get_display_language()`, modify `extract_skill_description()` | Low |
| `agent/prompt_builder.py` | Pass language through + snapshot cache key | Medium (cache logic) |
| `tools/skills_tool.py` | `_find_all_skills()`, `skill_view()` use locale-aware description | Low |
| `hermes_cli/skills_hub.py` | Add Description column to `do_list()` | Low |
| `hermes_cli/config.py` (optional) | Add `display.language` default | Low |
| `tools/skills_tool.py` docstring | Document `description_zh` | Trivial |
| A few bundled SKILL.md files | Add `description_zh` as demo | Trivial |
| `tests/tools/test_skills_tool.py` | New tests | Low |
