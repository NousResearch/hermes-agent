# RAM-only skill source proposal

Status: upstream proposal, NOT an official released capability.
Base: ecf9655b5aacbc826ebbe9bf11a6db5cbbcb214d.

`PluginContext.register_skill_source(name, *, list_skills, load_skill)` registers a plugin-owned source. Both callbacks are synchronous and called in the current profile scope. `list_skills()` returns bounded metadata rows (`name`, `description`, opaque `uri`). `load_skill(uri)` returns a dictionary with text `content` and `name`. Source handles use the existing ownership ledger and dispose/unload lifecycle.

The adapter must authorize before network IO, request only explicitly granted subtrees, validate every returned URI, recheck current grants on load, impose transport timeouts, and never persist remote text or credentials. Neither discovery nor content is cached by this source implementation. Native path-backed skills retain precedence; ambiguous remote canonical names fail closed. Reserved built-in/plugin command names are not overridden.

Interactive commands and invocation use the source without a filesystem path; messaging menus and existing manual remote invocation are unchanged. Remote content is inert: no inline shell preprocessing, script execution, or support-file materialization. This proposal does not expose remote support files through native `skill_view`, nor imply compatibility with scripts that require physical files. Existing scoped remote read tools remain available.

Validation:
- Selected official-runner suite: 77 tests passed, plus the dedicated plugin-host skill source test (1 passed). All 11 official lint/health checks passed. Catalog/load tests cover revocation before and during load, profile isolation, disabled skills and canonical plugin keys, fail-closed remote collisions, lifecycle, no local config injection and no usage writes.
- RPC tests exercise `commands.catalog`, `complete.slash`, and `command.dispatch` with `type=skill`.
- These are not a rendered Desktop/TUI end-to-end acceptance test.

Release gate: production must keep the official deployed Core unchanged until an official compatible release exists. Only then may a pinned official release plus matching extension artifact be staged, rendered TUI/Desktop behavior verified, and promoted through GitOps. Preserve native operational sessions and all remote knowledge on rollback; restore prior image and extension pins rather than deleting data.
