# Extraction w2 — monkeypatch-transparency adjudication (web_server s2, head-to-head)

Case: BOTH witnesses extracted the SAME two clusters (c16 memory_provider_setup,
c17 memory_provider_native) of `hermes_cli/web_server.py` with identical cluster
coverage. Both were 100% byte-faithful (35/35 vs 36/36 moved bodies +
2 constants), both `git apply --check` clean vs HEAD in a fresh worktree, both
test suites green (24+5+3+4 vs 42+5+3+4, fresh worktree + `--basetemp`). The
decisive discriminator was NOT fidelity or tests — it was **whether the moved
modules still observe `monkeypatch.setattr(web_server, ...)`**. The losing
witness had a LATENT regression: no current test trips it, so green suites did
not discriminate. This is the "patching contract is the live behavior" doctrine
made concrete.

## Step 1 — enumerate in-module helper calls in the moved bodies
For every moved function, grep its body for calls to module-level helpers
(config functions, cross-cluster helpers). web_server s2 findings:
`_read_memory_provider_existing_values`→load_config/get_hermes_home;
`_env_lookup`→load_env; `_save_memory_provider_native_config`→
load_config/get_hermes_home/save_config; `_discover_memory_provider_statuses`→
load_config; `_write_memory_provider_config_values`→save_env_value;
`_require_memory_provider_ready`/`_install_memory_provider_setup`→
_discover_memory_provider_statuses; `_memory_provider_dependencies_installed`/
`_install_memory_provider_pip_dependencies`→_dependency_importable.

## Step 2 — classify each moved module's binding for those names
- **Direct import** (`from hermes_cli.config import load_config`): BYPASS —
  in-module calls hit the imported object; `monkeypatch.setattr(web_server,
  "load_config", fake)` is invisible to them → behavior change vs the
  pre-extraction single-module namespace (where the module global WAS
  web_server.load_config).
- **`late()` proxy** (`load_config = late("load_config")` via web_deps):
  OBSERVED — resolves the web_server attribute at call time, so the patch is
  seen. Matches the repo's established seam (cron.py precedent:
  `load_config = late("load_config")`).
Adjudicate on the established seam: the late() witness preserves behavior; the
direct-import witness has a latent regression. Verify the patching tests only
exercise NON-moved paths (e.g. test_web_server.py:1721 patches
web_server.load_config but calls a staying function) — that's why both suites
pass and the discriminator must be structural, not empirical.

## Step 3 — rebinding of test-patched moved names
Repo tests patch `web_server._dependency_importable` (test_web_server.py:644)
and `web_server._discover_memory_provider_statuses`
(test_plugins_hub_perf_guard.py:17-19). Two valid ways to move such a name:
- keep the def in web_server + `late()` proxy from the moved module (w1a for
  _dependency_importable), or
- move the def as `_impl`, rebind the module name to `late("_name")`, re-export
  `_name_impl as _name` from web_server (w1b for both names).
INVALID: moving the def and leaving the module name un-rebound — in-module
callers (e.g. `_require_memory_provider_ready`→_discover) bypass web_server
patches (w1a's native module did exactly this for _discover).

## Step 4 — plan fidelity on shared helpers
`_dependency_importable` was a plan-listed c16 member used ONLY by moved
methods (live lines 5350/5377, both inside the moved span) → per brief rules it
must MOVE with the cluster. The witness that kept it in web_server (17/18)
deviated from the plan; functionally safe via the proxy but plan-incomplete.
The full-move witness (18/18, `_impl`+late rebinding) won the cluster.

## Fidelity-check pitfalls (cost a false DIFFERS this session)
- **AST name→segment map OVERWRITE**: building `{name: (lineno, end_lineno)}`
  from module bodies where later `Assign` nodes (`_x = late("_x")` rebinding at
  module bottom) OVERWRITE the FunctionDef entry → the check reports DIFFERS on
  a verbatim body. Track FunctionDefs and Assigns in SEPARATE dicts, and match
  `{name}_impl` aliases (`key = name if name in f2 else f"{name}_impl"`).
- **Logger-name convention is DESTINATION-DIR-specific**: all 7 existing
  web_routers modules (cron, mcp, profiles, sessions, skills, tools,
  whatsapp_onboarding) use `_log = logging.getLogger("hermes_cli.web_server")`
  — the module being extracted FROM — to keep pre-extraction log routing.
  `getLogger(__name__)` in a moved module CHANGES log routing (records under
  the new module name) and contradicts the precedent the witness itself cited.
  Check sibling modules in the destination dir before blessing either choice;
  the cli-mixin `getLogger(__name__)` is NOT universal.
- **Re-export placement**: web_server convention = extraction-point re-export
  block with `# noqa: E402,F401 — legacy re-exports; tests call these via
  web_server.<name>` (git.py:2753, cron.py:11232). Top-of-file re-export blocks
  (with `_impl as` aliases) work and pass tests but deviate stylistically →
  recommended fix, not a rejection.

## Full-suite baseline protocol (canonical candidate shows failures not in its manifest)
1. `cp <patched-file> /tmp/applied.py` BEFORE reverting.
2. `git checkout -- <file>` → pristine HEAD; run the failing tests. Identical
   failure set = pre-existing (env/SPA-assets dependent), NOT a regression.
3. Restore trap: re-applying the patch FAILS all-or-nothing with "already
   exists in working directory" if the patch's NEW untracked files still exist
   — `rm` those files first, then `git apply`, then `diff -q <file>
   /tmp/applied.py` to confirm byte-identical restoration.
4. Record BOTH failure sets (patched + baseline) in the verdict JSON.

## Verdict shape used (web_server s2-w2a)
Per-cluster `canonical_witness` (both → w1b), byte_fidelity table per witness,
plan_fidelity, module_conventions (logger/late-seam/reexport-placement with
CONFORMS/DEVIATES), tests (fresh worktree + --basetemp, incl. full-suite +
baseline), git_apply_check_vs_head, freshness (shard region byte-identical
orig↔live↔HEAD even though orig predates an unrelated later commit),
`contradicted[]` (one entry per divergence axis with w1a/w1b/canonical),
`required_fix[]` (severity required/recommended, file+line+fix). Freshness
note: the sandbox orig (17700 lines) predates the whatsapp_onboarding
extraction commit (live 17223) — only the SHARD REGION (5177-6001) must be
byte-identical, which it was, so both patches applied cleanly to HEAD.
