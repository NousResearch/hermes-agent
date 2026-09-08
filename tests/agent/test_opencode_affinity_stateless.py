"""Tests for the OpenCode session-affinity header, including the stateless
fallback that synthesizes an opaque key when no conversation context is
available (commit-message generation via ``hermes -z``, standalone cron jobs).

Bug #105841: one-shot/stateless calls run outside any conversation turn, so
``get_affinity_scope()`` / ``get_conversation_context()`` are both ``None`` and
``opencode_session_headers`` returned ``{}`` — OpenCode then 400s with
``MissingSessionID``. The fix synthesizes a per-call opaque key so the header is
always present for OpenCode targets.
"""
from __future__ import annotations

import re

# Standalone-runner shim (pytest unavailable in some sandboxes). Assertions are
# pytest-compatible so this file also runs under ``pytest tests/agent/``.
_PASSED = 0
_FAILED = 0


class _Monkey:
    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, target, attr, value):
        prev = getattr(target, attr)
        self._undo.append((target, attr, prev))
        setattr(target, attr, value)

    def undo(self) -> None:
        for target, attr, prev in reversed(self._undo):
            setattr(target, attr, prev)
        self._undo.clear()


def _import_affinity():
    import agent.opencode_affinity as aff
    return aff


def _import_portal_tags():
    import agent.portal_tags as pt
    return pt


_UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


# ---- Tests -----------------------------------------------------------------


def test_stateless_opencode_target_gets_synthesized_header():
    """No affinity scope, no conversation context, no explicit session_id
    (the one-shot / commit-gen state) -> header MUST still be present with an
    opaque per-call key (previously returned {} -> MissingSessionID 400)."""
    mp = _Monkey()
    try:
        aff = _import_affinity()
        pt = _import_portal_tags()
        mp.setattr(aff, "is_opencode_target", lambda provider, base_url: True)
        mp.setattr(pt, "get_affinity_scope", lambda: None)
        mp.setattr(pt, "get_conversation_context", lambda: None)

        headers = aff.opencode_session_headers("opencode-go", "https://opencode.ai/v1")
        assert aff.OPENCODE_SESSION_HEADER in headers, "header missing for stateless opencode target"
        key = headers[aff.OPENCODE_SESSION_HEADER]
        assert isinstance(key, str) and key, "synthesized key must be a non-empty string"
        assert _UUID_HEX_RE.match(key), f"synthesized key must be a uuid4 hex, got {key!r}"
    finally:
        mp.undo()


def test_stateless_synthesized_keys_differ_per_call():
    """Each stateless call gets its own opaque key (no unintended session
    sharing between independent one-shot requests)."""
    mp = _Monkey()
    try:
        aff = _import_affinity()
        pt = _import_portal_tags()
        mp.setattr(aff, "is_opencode_target", lambda provider, base_url: True)
        mp.setattr(pt, "get_affinity_scope", lambda: None)
        mp.setattr(pt, "get_conversation_context", lambda: None)

        k1 = aff.opencode_session_headers("opencode-go", "https://opencode.ai/v1")[
            aff.OPENCODE_SESSION_HEADER]
        k2 = aff.opencode_session_headers("opencode-go", "https://opencode.ai/v1")[
            aff.OPENCODE_SESSION_HEADER]
        assert k1 != k2, "per-call synthesized keys must differ"
    finally:
        mp.undo()


def test_conversation_context_wins_over_synthesis():
    """When a conversation context IS available, it is used (synthesis is only a
    fallback for the stateless case) -- preserves conversation prompt-cache
    warmth."""
    mp = _Monkey()
    try:
        aff = _import_affinity()
        pt = _import_portal_tags()
        mp.setattr(aff, "is_opencode_target", lambda provider, base_url: True)
        mp.setattr(pt, "get_affinity_scope", lambda: None)
        mp.setattr(pt, "get_conversation_context", lambda: "conv-session-123")

        headers = aff.opencode_session_headers("opencode-go", "https://opencode.ai/v1")
        assert headers[aff.OPENCODE_SESSION_HEADER] == "conv-session-123", \
            "conversation context must win over synthesis"
    finally:
        mp.undo()


def test_explicit_session_id_wins_when_no_context():
    """Explicit session_id argument is used when no ambient context exists
    (preserves the documented fallback order)."""
    mp = _Monkey()
    try:
        aff = _import_affinity()
        pt = _import_portal_tags()
        mp.setattr(aff, "is_opencode_target", lambda provider, base_url: True)
        mp.setattr(pt, "get_affinity_scope", lambda: None)
        mp.setattr(pt, "get_conversation_context", lambda: None)

        headers = aff.opencode_session_headers(
            "opencode-go", "https://opencode.ai/v1", session_id="explicit-sess-456")
        assert headers[aff.OPENCODE_SESSION_HEADER] == "explicit-sess-456", \
            "explicit session_id must win over synthesis"
    finally:
        mp.undo()


def test_non_opencode_target_returns_empty():
    """Non-OpenCode targets never get the header (synthesis only applies to
    OpenCode targets)."""
    mp = _Monkey()
    try:
        aff = _import_affinity()
        mp.setattr(aff, "is_opencode_target", lambda provider, base_url: False)

        headers = aff.opencode_session_headers("openai", "https://api.openai.com/v1")
        assert headers == {}, "non-opencode target must return {}"
    finally:
        mp.undo()


def test_merge_synthesizes_header_into_kwargs():
    """merge_opencode_session_headers injects the synthesized header into the
    call kwargs for stateless opencode targets (the fix path for commit-gen)."""
    mp = _Monkey()
    try:
        aff = _import_affinity()
        pt = _import_portal_tags()
        mp.setattr(aff, "is_opencode_target", lambda provider, base_url: True)
        mp.setattr(pt, "get_affinity_scope", lambda: None)
        mp.setattr(pt, "get_conversation_context", lambda: None)

        kwargs = {"extra_headers": {"x-request-id": "abc"}}
        result = aff.merge_opencode_session_headers(
            kwargs, provider="opencode-go", base_url="https://opencode.ai/v1")
        merged = result.get("extra_headers", {})
        assert aff.OPENCODE_SESSION_HEADER in merged, \
            "merge must inject synthesized header for stateless opencode target"
        assert merged.get("x-request-id") == "abc", "caller headers must be preserved"
        key = merged[aff.OPENCODE_SESSION_HEADER]
        assert _UUID_HEX_RE.match(key), f"synthesized key must be uuid4 hex, got {key!r}"
    finally:
        mp.undo()


def test_merge_preserves_caller_header_over_synthesis():
    """When the caller already set x-opencode-session, that value wins
    (setdefault semantics) -- synthesis never overrides an explicit value."""
    mp = _Monkey()
    try:
        aff = _import_affinity()
        pt = _import_portal_tags()
        mp.setattr(aff, "is_opencode_target", lambda provider, base_url: True)
        mp.setattr(pt, "get_affinity_scope", lambda: None)
        mp.setattr(pt, "get_conversation_context", lambda: None)

        caller_key = "caller-provided-session"
        kwargs = {"extra_headers": {aff.OPENCODE_SESSION_HEADER: caller_key}}
        result = aff.merge_opencode_session_headers(
            kwargs, provider="opencode-go", base_url="https://opencode.ai/v1")
        merged = result.get("extra_headers", {})
        assert merged[aff.OPENCODE_SESSION_HEADER] == caller_key, \
            "caller's explicit header must win over synthesis (setdefault)"
    finally:
        mp.undo()


# ---- Standalone runner -----------------------------------------------------


def _run():
    import inspect
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and inspect.isfunction(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", type(exc).__name__, str(exc)[:200])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run() else 0)
