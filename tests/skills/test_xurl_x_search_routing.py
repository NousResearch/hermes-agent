"""Behavioral contract for xurl / x_search routing guidance.

These tests assert structural invariants (required topics + mutual exclusivity
of responsibility), not frozen prose snapshots.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
XURL_SKILL = REPO_ROOT / "skills" / "social-media" / "xurl" / "SKILL.md"
X_SEARCH_DOC = REPO_ROOT / "website" / "docs" / "user-guide" / "features" / "x-search.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_any(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


def test_xurl_skill_routes_by_intent_not_interchangeably():
    text = _read(XURL_SKILL)
    lowered = text.lower()

    # Both surfaces named so agents can choose by capability.
    assert "x_search" in lowered
    assert "xurl" in lowered

    # x_search is discovery / read-only public research.
    assert _contains_any(text, "read-only public", "public x discovery", "broad public")
    # xurl owns authenticated / write / exact API work.
    assert _contains_any(
        text,
        "authenticated",
        "exact or authenticated",
        "exact api",
        "account actions",
        "write action",
    )
    # Writes must not be evidenced by x_search answers.
    assert _contains_any(
        text,
        "never treat an `x_search` answer",
        "never evidence",
        "proves the action",
        "x api response",
    )
    # Prefer x_search over xurl search for broad public discovery when available.
    assert "x_search" in lowered and "xurl search" in lowered
    assert _contains_any(text, "prefer `x_search`", "use `x_search` instead", "route by intent")


def test_xurl_agent_workflow_prefers_x_search_for_broad_discovery():
    text = _read(XURL_SKILL)
    # Workflow must preflight intent before xurl search.
    assert "xurl search" in text.lower()
    assert _contains_any(text, "check intent", "before using `xurl search`")
    assert _contains_any(text, "broad public", "public x discovery")
    assert _contains_any(text, "write action", "authenticated account", "exact api")


def test_x_search_doc_separates_discovery_from_account_actions():
    text = _read(X_SEARCH_DOC)
    lowered = text.lower()

    assert "x_search" in lowered
    assert "xurl" in lowered
    # Explicit comparison section or equivalent boundary language.
    assert _contains_any(text, "vs `xurl`", "vs xurl", "two different x surfaces")
    assert _contains_any(text, "read-only public", "public x discovery")
    assert _contains_any(
        text,
        "posting",
        "replying",
        "liking",
        "dm",
        "media upload",
        "deleting",
    )
    assert _contains_any(
        text,
        "authenticated",
        "exact or authenticated",
        "account actions",
        "state-changing",
    )
    # Write confirmation must come from xurl / X API, not x_search.
    assert _contains_any(
        text,
        "confirmed by `xurl`",
        "xurl` output",
        "x api response",
        "never evidence",
    )
    assert _contains_any(text, "switch to the `xurl`", "switch to `xurl`", "xurl skill")
