# skill_intake V2.1 — Modular Refactor Proposal

## Conclusion

Do a light modular refactor, not a rewrite.

Current `tools/skill_intake_tool.py` is functional and test-covered, but it is beginning to concentrate too many responsibilities in one file. The right next step is to extract helpers into a small module cluster while preserving the current external tool interface.

## Refactor goals

1. keep public behavior stable
2. preserve current test suite
3. reduce cognitive load per file
4. make further V2.1 work easier (provenance, eval planning, richer fixtures)

## Recommended target layout

```text
tools/
├── skill_intake_tool.py          # thin tool entrypoint + registry registration
├── skill_intake/
│   ├── __init__.py
│   ├── schemas.py                # tool schema + enums/constants if desired
│   ├── source_resolution.py      # source classification + provenance + remote fetch
│   ├── discovery.py              # local/remote discovery + candidate typing
│   ├── semantics.py              # semantic scoring + section extraction + discard logic
│   ├── conversion.py             # draft generation + out_dir persistence + OpenClaw-like asset handling
│   ├── governance.py             # re-audit + admit + policy helpers
│   └── reporting.py              # optional: common result-shaping helpers
```

## Suggested extraction sequence

### Step 1
Extract source/provenance helpers.

Move:
- `_classify_source`
- `_github_blob_to_raw`
- `_build_remote_candidates`
- `_fetch_remote_text_entries`
- `_build_local_provenance`

Target:
- `tools/skill_intake/source_resolution.py`

### Step 2
Extract discovery helpers.

Move:
- `_collect_local_files`
- `_safe_read_text`
- `_discover_text_entries`
- `_discover_local`
- OpenClaw-like candidate typing hook

Target:
- `tools/skill_intake/discovery.py`

### Step 3
Extract semantic/conversion helpers.

Move:
- `_extract_title_and_body`
- `_extract_description`
- `_split_discarded_lines`
- `_extract_semantic_sections`
- `_build_skill_md`
- `_collect_openclaw_assets`
- `_write_converted_skill_artifacts`

Target:
- `tools/skill_intake/semantics.py`
- `tools/skill_intake/conversion.py`

### Step 4
Extract governance helpers.

Move:
- `_compatibility_for_remote`
- `_compatibility_from_discovery`
- `_security_report`
- `_quality_report`
- `_recommendation`
- `re_audit_converted_candidate`
- `admit_skill_candidate`

Target:
- `tools/skill_intake/governance.py`

### Step 5
Leave `skill_intake_tool.py` as a thin wrapper.

Keep in file:
- tool schema import/reference
- mode dispatch
- `registry.register(...)`

## Non-goals

Do NOT in the same refactor:
- redesign output contracts
- change policy semantics
- change test expectations unless fixing a real bug
- add major new features

## Why this refactor is enough

Because the system is already functionally complete for V2.
The risk now is maintainability, not missing core workflow.

This modular split gives you:
- easier review
- easier targeted tests
- easier future extensions
- lower chance of accidental regressions

## Recommended order after refactor

1. modular refactor
2. richer provenance
3. larger fixtures
4. eval planning
5. sandbox / CI hardening
