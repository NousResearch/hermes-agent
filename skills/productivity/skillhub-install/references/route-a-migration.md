# Route A Migration Decision Record

## Context

When building a skill that integrates an external registry into Hermes, there are two architectural routes:

- **Route A**: Implement `SkillSource` ABC + dynamic router injection into Hermes core
- **Route B**: Standalone script that mirrors the entire security pipeline independently

## Decision: Route A

### Why Route A is correct (even when Route B seems safer)

1. **No duplicated security logic** — Route B must re-implement quarantine, scanning, lockfile, audit logging. Route A reuses `tools.skills_hub.quarantine_bundle()`, `tools.skills_guard.scan_skill_cached()`, `install_from_quarantine()`, `HubLockFile`, `append_audit_log()`.

2. **Automatic core updates** — When Hermes improves the scanner or changes the pipeline, Route A benefits automatically. Route B requires manual sync.

3. **Correct architecture** — Dynamic router injection (`hub.create_source_router = wrapper`) is the official extension point. It's not a hack; it's the intended plugin pattern.

4. **Less code, fewer bugs** — Route A: ~579 lines. Route B: ~651 lines. The 70+ extra lines were all duplicated pipeline logic.

### When Route B might seem appealing (and why to resist)

- "I don't want to depend on Hermes core" — But the skill ships WITH Hermes. It already has core access.
- "It's safer as standalone" — Duplicating security logic is LESS safe (divergence risk).
- "The reviewer might reject core changes" — Dynamic injection doesn't modify core files. It's a runtime monkey-patch, not a permanent fork.

### Router injection pattern

```python
import tools.skills_hub as hub
from hermes_cli.skills_hub import do_install

original = hub.create_source_router

def _router_with_skillhub(auth=None):
    sources = original(auth)
    if not any(getattr(s, "source_id", lambda: "")() == "skillhub" for s in sources):
        sources.append(SkillHubSource())
    return sources

hub.create_source_router = _router_with_skillhub
try:
    do_install(slug, ...)
finally:
    hub.create_source_router = original
```

Key details:
- Use `getattr(s, "source_id", lambda: "")()` for defensive check (some sources may not have the method)
- Always restore original router in `finally` block
- The `install()` function should try `_install_via_do_install()` first, fall back to `_install_direct()` for standalone mode

## Comparison methodology

When evaluating two implementations side-by-side:
1. Read both codebases completely
2. Create a comparison table: architecture, HTTP client, line count, test count, integration method
3. Identify what each does better (not just "ours vs theirs")
4. Quantify: code reduction, test coverage delta, maintenance cost
5. Make decision based on long-term maintenance, not short-term convenience
