"""Targets, file scope and the rule table: the one place a rule's id, scope and fix text live.

A rule's ``fix`` is the text the agent reads at the moment it trips the rule, so it names the
repo-specific remedy rather than restating the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

# Master switch, honoured everywhere the ratchet runs (CI, `scripts/check`, the git hooks). It is
# read from main, never from the branch under test, so a PR cannot relax its own check: CI reads
# the merge commit's first parent; local runs read `origin/main` (falling back to the merge-base).
# A flip reaches an open PR on its next CI run (a re-run reuses the old merge commit). "blocking": findings fail;
# "advisory": findings print, the run passes; "off": the ratchet is skipped. Flip it with a
# one-line commit to main; a single rule can instead be demoted with ``blocking=False`` below.
ENFORCEMENT = "blocking"

# Per-unit targets. A unit already over its target keeps its own current value as its cap
# (measured on the base revision) and may only go down; new units must meet the target.
TARGETS = {
    "CC": 20,  # cyclomatic complexity per function (ruff C901 numbers for Python)
    "FUNC_LINES": 300,
    "NESTING": 6,  # nested control-flow blocks inside one function
    "FILE_LINES": 2000,
}

METRIC_FIX = {
    "CC": "split it: extract phases into a topical sibling (`<stem>_<topic>.py`), or replace a"
    " branch ladder with a dict/table -> handler",
    "FUNC_LINES": "extract phases into named helpers or a topical sibling module",
    "NESTING": "return early, or extract the inner block into a helper",
    "FILE_LINES": "offset the growth in the same PR: move an existing function or class into a"
    " topical sibling `<stem>_<topic>.py` (moved code keeps its cap; see root AGENTS.md), or put"
    " new tests in a new `test_<stem>_<topic>.py`; new behaviour never goes into a facade",
    "MEASURE": "the file must parse and measure (a file that cannot be measured cannot be"
    " judged); fix the syntax, or report the measurer bug",
}

PY_EXCLUDE = ("website/*", "*/node_modules/*", "node_modules/*")
TS_EXCLUDE = PY_EXCLUDE + ("*.d.ts", "*/dist/*", "*/build/*", "*/generated/*", "*.generated.ts")
TESTS = ("tests/*", "*/tests/*", "tests-js/*", "*/conftest.py")
# One-shot processes that serve a single profile (CI/release scripts, skill scripts run as a
# child with a scoped env, installers): the profile-scope rules don't apply to them.
SINGLE_PROFILE = TESTS + ("scripts/*", "skills/*", "optional-skills/*", "evals/*", "setup.py")


@dataclass(frozen=True)
class Rule:
    id: str
    title: str
    fix: str
    source: str  # "ruff" | "ast" | "regex"
    exclude: tuple[str, ...] = TESTS
    blocking: bool = True
    include: tuple[str, ...] = ()  # when set, the rule applies only to these paths
    pattern_id: str = ""  # regex rules: id in scripts/ci/profile_scope_patterns.json
    hosted: bool = False  # applies only to modules the hosted entry point imports (hosted_graph)


# The hosted entry point: the modules a shared (multi-tenant) process starts from. HX015 judges
# every module these import at import time (``hosted_graph``). There is no hosted boot module
# yet, so the roots are what it will load whatever its shape: the inbound turn path, the turn
# loop, the scope modules and the four registries ``registry_freeze`` freezes. The hosted boot,
# when it lands, is added here (and any root it makes redundant dropped) in the same PR. A test
# fails when a root no longer names a module, so a rename cannot silently empty the graph.
# Modules a root loads lazily on the hosted path are roots too: the built-in platform adapters
# ``gateway.run`` loads through ``importlib`` and the agent constructor ``AIAgent.__init__`` imports.
HOSTED_ENTRY: tuple[str, ...] = (
    "gateway.run",
    "gateway.platforms.api_server",
    "gateway.platforms.webhook",
    "gateway.platforms.msgraph_webhook",
    "run_agent",
    "agent.agent_init",
    "model_tools",
    "hermes_scope",
    "hermes_scope_cache",
    "agent.runtime_scope",
    "registry_freeze",
    "tools.registry",
    "tools.mcp_tool",
    "providers",
    "hermes_cli.plugins",
    # Imported inside functions of the audited roots below and run while the process serves
    # (the turn path, GatewayRunner methods, housekeeping ticks, agent construction): the graph
    # does not follow function-body imports, so each is a root. tests/scripts/
    # test_code_health_lazy_roots.py fails on a new lazy import until it is placed here or in
    # OFF_HOSTED_PATH.
    "agent.anthropic_adapter",
    "agent.bedrock_adapter",
    "agent.context_pin",
    "agent.credential_source",
    "agent.credits_tracker",
    "agent.curator",
    "agent.delegation_backends",
    "agent.memory_manager",
    "agent.memory_manager_tools",
    "agent.moa_loop",
    "agent.native_compaction",
    "agent.oneshot_footprint",
    "agent.provider_route_hooks",
    "agent.reasoning_timeouts",
    "agent.review_idle_queue",
    "agent.safe_worker_policy",
    "cron.scheduler_provider",
    "gateway.channel_directory",
    "gateway.host_attach",
    "gateway.host_rendezvous",
    "gateway.media_policy",
    "gateway.message_timestamps",
    "gateway.profile_routing",
    "gateway.relay.egress",
    "gateway.restart_notice",
    "gateway.run_delivery_queue_watch",
    "gateway.run_housekeeping_disk",
    "gateway.session_api_turn",
    "gateway.session_db_recovery",
    "gateway.status",
    "hermes_cli.commands_platforms",
    "hermes_cli.config_effective",
    "hermes_cli.debug",
    "hermes_cli.gateway_multiplex_mode",
    "hermes_cli.mem_trim",
    "hermes_cli.model_normalize",
    "hermes_cli.model_switch",
    "hermes_cli.nous_auth_keepalive",
    "hermes_cli.observability.shared_metrics_process",
    "hermes_cli.plugins_cadence",
    "hermes_cli.profiles",
    "hermes_cli.tools_config",
    "hermes_state",
    "hermes_state_registry",
    "plugins.context_engine",
    "plugins.teams_pipeline.runtime",
    "tools.approval_yolo",
    "tools.checkpoint_maintenance",
    "tools.checkpoint_manager",
    "tools.env_probe",
    "tools.mcp_oauth",
    "tools.mcp_tool_discovery",
    "tools.terminal_scope",
    "tui_gateway.launch_profile_policy",
)

# Roots whose function-body imports are audited (tests/scripts/test_code_health_lazy_roots.py): each
# in-repo module they import inside a function is a root above or listed in OFF_HOSTED_PATH.
LAZY_IMPORTS_AUDITED: tuple[str, ...] = ("gateway.run", "run_agent", "agent.agent_init")

# Lazily imported by an audited root but never while the process serves tenants: process boot
# before the first turn, process exit, or a CLI-only entry. Each needs its reason; a module that
# starts running during serving moves to HOSTED_ENTRY.
OFF_HOSTED_PATH: dict[str, str] = {
    "agent/ssl_verify.py": "run_agent.main only: the one-shot CLI entry, not the hosted path",
    "gateway/config_loader.py": "gateway.run.main: reads the host's gateway config once at boot",
    "gateway/lifecycle_ledger.py": "process exit: records how the process ended",
    "hermes_cli/boot_bootstrap.py": "gateway.run.main: install bootstrap before serving starts",
    "hermes_cli/cpu_profile.py": "process exit: writes the --cpu-profile output",
    "hermes_cli/gateway_migrate.py": "a string constant for a log line; no state",
    "hermes_cli/resource_limits.py": "gateway.run.main: malloc arena limit before any thread",
    "hermes_cli/stdio.py": "gateway.run.main: Windows stdio encoding at boot",
}

# Heuristic rules print as warnings until a frozen replay (`python -m scripts.code_health.replay`)
# shows most of their hits on merged PRs are real problems; flip a rule to blocking with that
# evidence in the PR. Reviewers' samples put these between ~12% and ~60% precision.
WARN_UNTIL_REPLAYED = False

RULES: tuple[Rule, ...] = (
    # Swallowed exceptions: the top statically-catchable bug cause in the fix sample.
    Rule("S110", "try/except/pass", "log it or narrow the except; a silent pass hides the"
         " failure the next bug report is about", "ruff"),
    # ruff (pinned, --ignore-noqa) passes a handler that re-raises or logs the traceback; a
    # message-only log such as `logger.warning("...: %s", e)` still counts as swallowed.
    Rule("BLE001", "blind `except Exception`", "catch the exceptions this code can actually"
         " raise; a deliberate catch-all boundary must re-raise or log the traceback"
         " (`logger.exception(...)` or `exc_info=True`; `logger.warning(..., e)` is not enough)."
         " `# noqa` is ignored here: waive with `# health: allow BLE001 -- <why>`", "ruff"),
    Rule("E722", "bare `except:`", "catch a named exception (bare except also swallows"
         " KeyboardInterrupt/SystemExit)", "ruff"),
    # Correctness basics.
    # tui_gateway method modules get the server's names injected (`bind_module(globals(), ...)`).
    Rule("F821", "undefined name", "import or define it", "ruff", exclude=("tui_gateway/*",)),
    Rule("F823", "local used before assignment", "assign before use or declare global",
         "ruff", exclude=()),
    Rule("F811", "redefinition of unused name", "remove the shadowed definition", "ruff",
         exclude=()),
    Rule("F841", "unused local variable", "delete it or use it", "ruff", exclude=()),
    Rule("RUF006", "dangling asyncio task", "keep a reference to the task (it can be garbage"
         " collected mid-run) and handle its exception", "ruff"),
    # Hangs.
    Rule("ASYNC230", "blocking open() in async def", "`await asyncio.to_thread(...)`", "ruff"),
    Rule("ASYNC240", "blocking path call in async def", "`await asyncio.to_thread(...)`",
         "ruff"),
    Rule("S113", "HTTP request without timeout", "pass `timeout=`", "ruff"),
    Rule("HX006", "subprocess/urlopen without timeout", "pass `timeout=`; an asyncio"
         " `proc.communicate()` goes inside `asyncio.wait_for(..., timeout=...)` (a child or"
         " socket that never answers hangs forever); after `proc.kill()` reap with `proc.wait()`,"
         " never a bare `communicate()` (it blocks while a grandchild holds the pipe)", "ast"),
    Rule("HX007", "sync config I/O inside async def", "`await asyncio.to_thread(load_config)`"
         " or read it before entering the event loop", "ast"),
    Rule("HX008", "asyncio.get_event_loop()", "`asyncio.get_running_loop()` inside a"
         " coroutine; `asyncio.run()` at an entry point", "ast"),
    Rule("HX009", "isinstance(r, Exception) on gather results", "check `BaseException`:"
         " `gather(return_exceptions=True)` also returns CancelledError", "ast"),
    # Profile scope (one process serves many profiles).
    Rule("HX001", "hardcoded Hermes home", "`get_hermes_home()` for paths,"
         " `display_hermes_home()` for user-facing text (`hermes_constants`); a path that"
         " deliberately lives under the user's home, not the profile (e.g. the profiles root),"
         " takes `# health: allow HX001 -- <why>`", "ast",
         blocking=WARN_UNTIL_REPLAYED,
         exclude=SINGLE_PROFILE + ("hermes_constants.py",)),
    Rule("HX002", "new HERMES_* environment variable", "behavioural settings go in"
         " config.yaml, secrets through the secret scope; `.env` is for credentials only",
         "ast", exclude=SINGLE_PROFILE),
    Rule("HX004", "UnscopedSecretError falls back to os.getenv", "bind the owning profile's"
         " secret scope at the spawn site; never read the launch profile's environment",
         "ast", exclude=SINGLE_PROFILE),
    Rule("HX005", "home/config/env captured at import time", "resolve it at call time"
         " (`get_hermes_home()`, `load_config()`) or key the slot by `hermes_home_key()`;"
         " module constants hold the launch profile's value", "ast",
         exclude=SINGLE_PROFILE),
    Rule("HX012", "raw threading.Thread", "`spawn_context_thread(...)` so the thread keeps"
         " the caller's profile scope (ContextVars do not cross a bare Thread)", "ast",
         blocking=WARN_UNTIL_REPLAYED,
         exclude=SINGLE_PROFILE),
    Rule("PS-P05", "child env built from os.environ", "`served_profile_child_env()`; the"
         " child otherwise inherits the launch profile's home and secrets", "regex",
         exclude=SINGLE_PROFILE, blocking=WARN_UNTIL_REPLAYED, pattern_id="P05"),
    Rule("PS-P06", "raw platform credential getenv", "read it through the profile's secret"
         " scope (`get_secret`)", "regex", exclude=SINGLE_PROFILE,
         blocking=WARN_UNTIL_REPLAYED, pattern_id="P06"),
    Rule("HX014", "legacy profile setter", "bind a scope:"
         " `agent.runtime_scope.bind_profile_scope(home)` moves home, secrets and terminal policy"
         " together (`bind_turn(session_id, turn_id, fence)` inside it for a turn); `bind_state(scope, ...)` for a host"
         " that binds less on purpose. A raw `set_hermes_home_override` / `set_secret_scope` /"
         " `set_terminal_scope` / `install_profile_terminal_scope` call is migrated when its function"
         " is next edited", "ast",
         exclude=SINGLE_PROFILE + ("agent/runtime_scope.py", "hermes_constants.py",
                                   "agent/secret_scope.py", "tools/terminal_scope.py")),
    Rule("HX015", "process-wide mutable state on the hosted path", "one hosted process serves"
         " every tenant, so this state is shared by all of them: key it per agent with"
         " `hermes_scope_cache.ScopedCache(name)` (raises with no scope bound), or, when it is"
         " identical for every tenant (a catalog, tool schemas), declare it with"
         " `tenant_neutral(value, reason=\"...\")` / `@tenant_neutral(reason=\"...\")`."
         " Registries are frozen after hosted boot (`registry_freeze`); a per-agent toggle is a"
         " filter applied from the bound scope, never a registration", "ast", hosted=True,
         # The freeze flag is process-wide by definition: one boot freezes every tenant's view.
         exclude=TESTS + ("registry_freeze.py",)),
    # Process identity.
    Rule("HX003", "process identity from argv substrings", "use"
         " `gateway.status.looks_like_gateway_command_line` /"
         " `hermes_cli.update_cmd._hermes_holder_subcommand` and match full cmdlines", "ast",
         blocking=WARN_UNTIL_REPLAYED),
    # Config truthiness.
    Rule("HX010", "bool() of a config/env string", "`bool(\"false\")` is True. Is it set?"
         " `os.getenv(\"X\") is not None` (or `!= \"\"` to treat empty as unset). A boolean"
         " flag? `is_truthy_value(...)`, which is False for any non-flag string such as a key",
         "ast", blocking=WARN_UNTIL_REPLAYED),
    # Structure.
    Rule("HX011", "if/elif ladder on one name", "use a dict/table -> handler (`_SLASH_DISPATCH`"
         " is the shape)", "ast", exclude=()),
    # Tests: a fixed delay is a race against the runner's load (the usual flake).
    Rule("HX013", "wall-clock sleep in a test", "wait for the condition, not the clock: an"
         " Event the code under test sets (`threading.Event` / `asyncio.Event`, `.wait(timeout=)`),"
         " or poll with `tests.waiting.wait_until(predicate, timeout=...)` (`async_wait_until` in a"
         " coroutine; a poll loop's interval under a deadline or a `range(n)` bound passes); a stand-in that must hang"
         " blocks on an Event the test sets at teardown."
         " `sleep(0)` to yield is fine. A delay that IS the behaviour under test takes"
         " `# health: allow HX013 -- <why>`", "ast", include=TESTS, exclude=()),
)

RULES_BY_ID = {rule.id: rule for rule in RULES}
RUFF_CODES = tuple(rule.id for rule in RULES if rule.source == "ruff")
ALLOW_SYNTAX = "# health: allow <RULE> -- <why>"


def _matches(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch(path, glob) for glob in globs)


def rule_applies(rule: Rule, path: str) -> bool:
    if rule.include and not _matches(path, rule.include):
        return False
    return not _matches(path, rule.exclude)


def in_scope(path: str) -> str | None:
    """Language of a tracked path the engine measures, or None."""
    if path.endswith(".py"):
        return None if _matches(path, PY_EXCLUDE) else "py"
    if path.endswith((".ts", ".tsx", ".mts", ".cts")):
        return None if _matches(path, TS_EXCLUDE) else "ts"
    return None
