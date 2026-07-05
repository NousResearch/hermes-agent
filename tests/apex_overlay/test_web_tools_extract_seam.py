"""Seam-test + behavior test for apex_overlay.web_tools_extract (hc-401 SEAM A).

Pins the upstream symbols the seam binds to so an upstream rename turns a
*silently disarmed guard* (web_extract hanging on ddgs) into a *loud CI failure*:

* ``tools.web_tools._get_extract_backend`` — the seam calls it to learn the
  configured extract backend.
* ``tools.web_tools._is_backend_available`` — the seam calls it to learn whether
  that backend is usable.
* ``tools.registry.registry._tools["web_extract"]`` — the entry whose
  ``.check_fn`` the seam swaps.

Behavior proven:
* after apply(), web_extract's check_fn is the overlay gate;
* with a search-only backend (ddgs) the gate returns False (web_extract hidden);
* with an extract-capable + available backend (exa) it returns True;
* web_search is untouched (still gated by the broad check_web_api_key).

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter).
"""

from __future__ import annotations

import inspect

from apex_overlay import web_tools_extract


# ---------------------------------------------------------------------------
# Seam assertions — pin the patched/consumed symbols
# ---------------------------------------------------------------------------

def test_seam_targets_exist_in_web_tools():
    import tools.web_tools as web_tools

    get_backend = getattr(web_tools, web_tools_extract._TARGET_EXTRACT_BACKEND_FN, None)
    assert get_backend is not None, (
        "tools.web_tools._get_extract_backend is gone — the web_extract "
        "capability gate can no longer learn the configured extract backend. "
        "Update apex_overlay.web_tools_extract._TARGET_EXTRACT_BACKEND_FN."
    )
    # zero-arg callable
    assert len(inspect.signature(get_backend).parameters) == 0

    is_avail = getattr(web_tools, web_tools_extract._TARGET_BACKEND_AVAILABLE_FN, None)
    assert is_avail is not None, (
        "tools.web_tools._is_backend_available is gone — update "
        "apex_overlay.web_tools_extract._TARGET_BACKEND_AVAILABLE_FN."
    )
    params = list(inspect.signature(is_avail).parameters)
    assert params and params[0] == "backend", (
        f"_is_backend_available first param changed to {params!r}; the overlay "
        f"passes backend positionally."
    )


def test_seam_target_web_extract_registered():
    from tools.registry import registry

    # web_tools import registers the tool; force it.
    import tools.web_tools  # noqa: F401

    entry = registry._tools.get(web_tools_extract._TOOL_NAME)
    assert entry is not None, (
        "web_extract is not in the registry — the seam has nothing to gate. "
        "Update apex_overlay.web_tools_extract._TOOL_NAME or check the "
        "registration in tools/web_tools.py."
    )
    assert hasattr(entry, "check_fn"), "ToolEntry lost its .check_fn attribute"


def test_apply_swaps_check_fn_and_is_idempotent():
    from tools.registry import registry

    web_tools_extract._APPLIED = False
    assert web_tools_extract.apply() is True

    entry = registry._tools["web_extract"]
    assert entry.check_fn is web_tools_extract.check_web_extract_available
    assert getattr(entry.check_fn, web_tools_extract._MARK, False)

    # Idempotent: second apply is a no-op and does not double-wrap.
    assert web_tools_extract.apply() is True
    assert registry._tools["web_extract"].check_fn is web_tools_extract.check_web_extract_available


# ---------------------------------------------------------------------------
# Behavior — the gate hides web_extract on search-only backends
# ---------------------------------------------------------------------------

def test_gate_false_for_search_only_backend(monkeypatch):
    """ddgs (search-only) → gate returns False, so web_extract is hidden."""
    import tools.web_tools as web_tools

    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "ddgs")
    # even if ddgs is 'available', it's not extract-capable → False
    monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: True)
    assert web_tools_extract.check_web_extract_available() is False


def test_gate_true_for_extract_capable_available_backend(monkeypatch):
    """exa (extract-capable) + available → gate returns True."""
    import tools.web_tools as web_tools

    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "exa")
    monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: b == "exa")
    assert web_tools_extract.check_web_extract_available() is True


def test_gate_false_for_extract_capable_but_unavailable_backend(monkeypatch):
    """firecrawl configured but unavailable → False (don't advertise a dead tool)."""
    import tools.web_tools as web_tools

    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "firecrawl")
    monkeypatch.setattr(web_tools, "_is_backend_available", lambda b: False)
    assert web_tools_extract.check_web_extract_available() is False


def test_gate_fails_closed_on_error(monkeypatch):
    """Any probe error resolves to False (hide the tool) rather than raising."""
    import tools.web_tools as web_tools

    def _boom():
        raise RuntimeError("config blew up")

    monkeypatch.setattr(web_tools, "_get_extract_backend", _boom)
    assert web_tools_extract.check_web_extract_available() is False


def test_web_search_check_fn_untouched():
    """web_search must keep its own (broad) check_fn — only web_extract is gated."""
    from tools.registry import registry
    import tools.web_tools as web_tools

    web_tools_extract._APPLIED = False
    web_tools_extract.apply()

    search_entry = registry._tools.get("web_search")
    assert search_entry is not None
    assert search_entry.check_fn is web_tools.check_web_api_key, (
        "web_search's check_fn changed — the seam must only touch web_extract."
    )


def test_plugin_register_applies_seam():
    from tests.apex_overlay.conftest import run_plugin_register_with_stubbed_seams

    called = run_plugin_register_with_stubbed_seams("_web_extract_plugin_under_test")
    assert "web_tools_extract" in called, (
        "plugin.register() must call web_tools_extract.apply()"
    )
