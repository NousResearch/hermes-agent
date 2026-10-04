"""Targets, file scope and the rule table: the one place a rule's id, scope and fix text live.

A rule's ``fix`` is the text the agent reads at the moment it trips the rule, so it names the
repo-specific remedy rather than restating the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

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
    "FILE_LINES": "split along `<stem>_<topic>` (facade + siblings, see root AGENTS.md); new"
    " behaviour goes in a new or topical sibling, never appended to a facade",
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
    pattern_id: str = ""  # regex rules: id in scripts/ci/profile_scope_patterns.json


RULES: tuple[Rule, ...] = (
    # Swallowed exceptions: the top statically-catchable bug cause in the fix sample.
    Rule("S110", "try/except/pass", "log it or narrow the except; a silent pass hides the"
         " failure the next bug report is about", "ruff"),
    Rule("BLE001", "blind `except Exception`", "catch the exceptions this code can actually"
         " raise; if a boundary truly needs a catch-all, log it with context", "ruff"),
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
         " `display_hermes_home()` for user-facing text (`hermes_constants`)", "ast",
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
         exclude=SINGLE_PROFILE),
    Rule("PS-P05", "child env built from os.environ", "`served_profile_child_env()`; the"
         " child otherwise inherits the launch profile's home and secrets", "regex",
         exclude=SINGLE_PROFILE, pattern_id="P05"),
    Rule("PS-P06", "raw platform credential getenv", "read it through the profile's secret"
         " scope (`get_secret`)", "regex", exclude=SINGLE_PROFILE, pattern_id="P06"),
    # Process identity.
    Rule("HX003", "process identity from argv substrings", "use"
         " `gateway.status.looks_like_gateway_command_line` /"
         " `hermes_cli.update_cmd._hermes_holder_subcommand` and match full cmdlines", "ast"),
    # Config truthiness.
    Rule("HX010", "bool() of a config/env string", "`bool(\"false\")` is True: parse it"
         " (`is_truthy_value`) or compare explicitly", "ast"),
    # Structure.
    Rule("HX011", "if/elif ladder on one name", "use a dict/table -> handler (`_SLASH_DISPATCH`"
         " is the shape)", "ast", exclude=()),
)

RULES_BY_ID = {rule.id: rule for rule in RULES}
RUFF_CODES = tuple(rule.id for rule in RULES if rule.source == "ruff")
ALLOW_SYNTAX = "# health: allow <RULE> -- <why>"


def _matches(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch(path, glob) for glob in globs)


def rule_applies(rule: Rule, path: str) -> bool:
    return not _matches(path, rule.exclude)


def in_scope(path: str) -> str | None:
    """Language of a tracked path the engine measures, or None."""
    if path.endswith(".py"):
        return None if _matches(path, PY_EXCLUDE) else "py"
    if path.endswith((".ts", ".tsx", ".mts", ".cts")):
        return None if _matches(path, TS_EXCLUDE) else "ts"
    return None
