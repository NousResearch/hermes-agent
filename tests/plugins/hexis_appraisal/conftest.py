"""Test scaffolding for the hexis_appraisal suite.

Inserts the repo root at the front of sys.path so `from hexis_appraisal
import store` resolves regardless of pytest's cwd, and makes the host
hermes-agent package importable (for agent.plugin_llm test fakes) when it
isn't already. sys.path mutation is acceptable in test scaffolding — the
prohibition is on plugin code. No literal user paths: the host location is
resolved via $HERMES_HOME with the conventional fallback.
"""

import os
import re
import sys
from pathlib import Path

# Repo root FIRST on sys.path so `agent`/`hermes_cli` resolve to this
# checkout's code, shadowing any editable install of another branch.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

# Load plugins/hexis_appraisal as the importable package `hexis_appraisal`,
# mirroring the host loader idiom (hermes_cli/plugins.py: spec_from_file_location
# with submodule_search_locations) so `from hexis_appraisal import store` and
# the package's own relative imports behave identically to the standalone
# layout. One module, one identity: never import it as `plugins.hexis_appraisal`
# — that would create a second module object with separate state. The module
# is registered in sys.modules BEFORE exec_module so relative imports inside
# the package resolve during execution.
if "hexis_appraisal" not in sys.modules:
    import importlib.util

    _PLUGIN_DIR = _REPO_ROOT / "plugins" / "hexis_appraisal"
    _spec = importlib.util.spec_from_file_location(
        "hexis_appraisal",
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["hexis_appraisal"] = _module
    _spec.loader.exec_module(_module)


# ---------------------------------------------------------------------------
# SAFE-03: structural directive-language detection (shared surface — plan
# 03-02 extends the corpus with reflection trust-hint lines)
# ---------------------------------------------------------------------------

# Rendered content lines must read as observations. These patterns catch
# STRUCTURAL directive language — lines that read as instructions to the
# agent. Quoted payload text (double-quoted spans; single-quoted search
# phrases on the memory-searches line) is reported model output and exempt.
DIRECTIVE_PATTERNS = [
    # imperative/directive line openers
    re.compile(
        r"(?im)^\s*-?\s*(you (should|must|need to|have to|shall)"
        r"|do not|don't|never|always)\b"
    ),
    re.compile(
        r"(?im)^\s*-?\s*(do this|prioritize|reach out|next step|make sure"
        r"|remember to|consider doing|act on|execute|run)\b"
    ),
    # second-person directive phrasing anywhere in a rendered advisory line
    re.compile(r"(?im)\byou (should|must)( now)?\b"),
]

# Every content line of a rendered block must carry one of these
# observational labels ("- trust note:" is forward-compat for 03-02 REFL-05).
ALLOWED_LABEL_PREFIXES = (
    "- instinct:",
    "- observation:",
    "- contradiction (",
    "- possible memory searches:",
    "- gut reaction:",
    "- trust note:",
)

_DQUOTED_SPAN_RE = re.compile(r'"[^"]*"')
_SQUOTED_SPAN_RE = re.compile(r"'[^']*'")


def _strip_quoted_spans(line):
    """Remove reported-material spans before directive matching.

    Double-quoted spans are stripped everywhere; single-quoted spans only on
    the memory-searches line (search phrases are single-quoted by render.py —
    stripping single quotes globally would mangle contractions like "it's").
    """
    line = _DQUOTED_SPAN_RE.sub(" ", line)
    if line.startswith("- possible memory searches:"):
        line = _SQUOTED_SPAN_RE.sub(" ", line)
    return line


def assert_no_directive_language(block):
    """Assert a rendered block contains no structural directive language.

    Skips the sentinel + framing lines; every remaining content line must
    (a) start with an allowlisted observational label and (b) match no
    DIRECTIVE_PATTERNS entry after quoted spans are removed. The failure
    message names the offending line.
    """
    assert isinstance(block, str) and block.strip(), "expected a rendered block"
    lines = block.split("\n")
    assert lines[0].startswith("[hexis appraisal]"), (
        "missing sentinel header: %r" % lines[0]
    )
    for line in lines[2:]:  # 0 = sentinel, 1 = framing
        if not line.strip():
            continue
        assert line.startswith(ALLOWED_LABEL_PREFIXES), (
            "non-observational line (fails label allowlist): %r" % line
        )
        stripped = _strip_quoted_spans(line)
        for pattern in DIRECTIVE_PATTERNS:
            found = pattern.search(stripped)
            assert found is None, (
                "directive language %r in line: %r" % (found.group(0), line)
            )


def _ensure_host_on_path():
    try:
        import agent.plugin_llm  # noqa: F401

        return
    except ImportError:
        pass
    home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
    agent_dir = home / "hermes-agent"
    if (agent_dir / "agent" / "plugin_llm.py").exists():
        agent_path = str(agent_dir)
        if agent_path not in sys.path:
            sys.path.insert(0, agent_path)


_ensure_host_on_path()
