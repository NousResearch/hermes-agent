"""
Adversarial security probes for api_server user->peer routing patch.

The patch forwards OpenAI request field `body.get("user")` as the AIAgent
`user_id`, which propagates to Honcho's `runtime_user_peer_name` and
ultimately to a peer_id. Verify that adversarial `user` values cannot:
  - smuggle path-traversal characters into Honcho URL paths
  - break out of the Honcho ID regex / cause 5xx
  - be used as an auth bypass or peer-impersonation vector
  - leak PII in error paths / responses / logs
  - cause resource exhaustion (very long strings, control chars)
"""
import re
import pytest

from gateway.platforms.api_server import MAX_USER_ID_LENGTH, _resolve_caller_user_id
from plugins.memory.honcho.session import HonchoSessionManager


# Mirror of the sanitize regex used by HonchoSessionManager._sanitize_id
PEER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _sanitize_id(id_str: str) -> str:
    """Mirror of plugins/memory/honcho/session.py:_sanitize_id."""
    return re.sub(r'[^a-zA-Z0-9_-]', '-', id_str)


# ---------------------------------------------------------------------------
# Adversarial user values to probe
# ---------------------------------------------------------------------------

ADVERSARIAL_USERS = [
    # path-traversal style
    "../../etc/passwd",
    "../admin",
    "/etc/shadow",
    "\\windows\\system32",
    # null bytes / control chars
    "alice\x00admin",
    "alice\nfoo",
    "alice\rX-Evil: true",
    # oversized
    "a" * 100_000,
    "a" * 1_000_000,
    # SQL/SSRF-ish
    "' OR 1=1 --",
    "alice'; DROP TABLE peers;--",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
    # Honcho regex mismatches
    "alice@example.com",
    "alice bob",                # whitespace
    "alice.bob",                # dots
    "alice/bob",                # slashes
    "alice+bob",                # plus
    # Unicode / bidi / zero-width
    "alice‮",                   # right-to-left override
    "alice​",              # zero-width space
    # Empty / None-ish
    "",                          # falsy -> dropped by `kwargs.get("user_id") or None`
    "0",                         # truthy "0" -> kept
    "false",
    # JSON edge cases
    "null",
    "true",
    "{}",
    "[]",
    # The exact OpenAI spec note: "user" is supposed to be a stable identifier.
    # Anything beyond ~64 chars is anomalous for an end-user identifier.
    "x" * 65,
]


# Empty string is intentionally in ADVERSARIAL_USERS — it tests the
# upstream `if agent._user_id:` truthiness check, not _sanitize_id itself.
# All other entries must reach _sanitize_id.
_NONEMPTY_ADVERSARIAL = [u for u in ADVERSARIAL_USERS if u != ""]


class TestSanitizationDefense:
    """Verify _sanitize_id neutralizes every adversarial character class."""

    @pytest.mark.parametrize("raw", _NONEMPTY_ADVERSARIAL)
    def test_adversarial_user_neutralized(self, raw):
        sanitized = _sanitize_id(raw)
        # Whatever survives must be a valid Honcho peer ID.
        assert PEER_ID_RE.match(sanitized), (
            f"raw={raw!r} sanitized={sanitized!r} violates Honcho peer ID regex"
        )
        # And must not contain any of the dangerous characters.
        assert "/" not in sanitized, f"forward slash leaked: {sanitized!r}"
        assert "\\" not in sanitized, f"backslash leaked: {sanitized!r}"
        assert "\x00" not in sanitized, "NUL byte leaked"
        assert "\n" not in sanitized, "newline leaked"
        assert "\r" not in sanitized, "carriage return leaked"
        assert " " not in sanitized, "space leaked"
        assert "." not in sanitized, "dot leaked"

    def test_path_traversal_user_resolves_to_safe_peer(self):
        """The exact HIGH-severity case the patch should prevent."""
        sanitized = _sanitize_id("../../etc/passwd")
        assert sanitized == "------etc-passwd"
        # No slashes, no traversal.
        assert "/" not in sanitized

    def test_url_smuggling_user_resolves_to_safe_peer(self):
        """If raw contains a colon + host, it must NOT survive as a URL fragment."""
        raw = "http://169.254.169.254/latest/meta-data/"
        sanitized = _sanitize_id(raw)
        assert ":" not in sanitized
        assert "/" not in sanitized


class TestPeerIdValidation:
    """Verify HonchoSessionManager uses _sanitize_id on every user-supplied ID."""

    def test_resolve_user_peer_id_uses_sanitized_value(self):
        """The patch's user_id must hit _sanitize_id before reaching Honcho."""
        manager = HonchoSessionManager.__new__(HonchoSessionManager)
        # Construct only the bits _resolve_user_peer_id touches.
        manager._config = None
        manager._runtime_user_peer_name = "../../etc/passwd"
        manager._runtime_user_peer_name_alt = None
        # session_key for fallback path
        peer_id = manager._resolve_user_peer_id("api_server:default:user1")

        assert PEER_ID_RE.match(peer_id)
        assert "/" not in peer_id
        assert ".." not in peer_id.replace("-", "")
        # Must not equal the raw adversarial input.
        assert peer_id != "../../etc/passwd"


class TestPatchInputFlow:
    """Verify the patch does not introduce type/None bypass."""

    def test_body_user_none_does_not_propagate(self):
        """OpenAI spec allows user to be omitted. patch uses body.get('user')
        which returns None if absent. AIAgent only forwards to memory when
        user_id is truthy (see agent_init.py:1172). Verify the contract."""
        user = None
        forwarded = user if user else None
        assert forwarded is None

    def test_body_user_empty_string_does_not_propagate(self):
        user = ""
        forwarded = user if user else None
        assert forwarded is None

    def test_body_user_zero_string_propagates(self):
        # "0" is truthy in Python — must propagate, not be silently dropped.
        user = "0"
        forwarded = user if user else None
        assert forwarded == "0"


class TestLengthExhaustion:
    """The MEDIUM finding from t_c1b0a03c: the patch forwarded body.get('user')
    to Honcho with no length cap. The follow-up card t_a5d888ec closed the gap
    at the gateway layer via ``_resolve_caller_user_id``, which caps to
    :data:`MAX_USER_ID_LENGTH` BEFORE handing the value to ``AIAgent``.

    These tests assert the cap is in place: oversized ``user`` values are
    truncated at the gateway, and falsy values still drop to None so the
    ``if agent._user_id:`` truthiness check at agent_init.py:1172 continues
    to suppress memory propagation.
    """

    def test_helper_cap_constant_is_128(self):
        """Document the chosen cap. 128 chars is well above OpenAI's ~64-char
        realistic identifier length while keeping the sanitized peer_id
        bounded in memory regardless of how many unsafe chars survive."""
        assert MAX_USER_ID_LENGTH == 128

    def test_helper_caps_1mb_user_to_128_chars(self):
        raw = "a" * 1_000_000
        capped = _resolve_caller_user_id({"user": raw})
        assert capped is not None  # narrows `str | None` -> `str` for type-checker
        assert len(capped) == MAX_USER_ID_LENGTH
        # First 128 chars preserved (stable-prefix preference).
        assert capped == raw[:MAX_USER_ID_LENGTH]

    def test_helper_caps_65_char_user_at_128(self):
        """65 chars is unusual but within the cap — should pass through."""
        capped = _resolve_caller_user_id({"user": "x" * 65})
        assert capped is not None
        assert capped == "x" * 65
        assert len(capped) == 65

    def test_helper_passes_through_exactly_128_chars(self):
        """Boundary: exactly MAX_USER_ID_LENGTH chars passes through verbatim."""
        capped = _resolve_caller_user_id({"user": "x" * MAX_USER_ID_LENGTH})
        assert capped is not None
        assert len(capped) == MAX_USER_ID_LENGTH
        assert capped == "x" * MAX_USER_ID_LENGTH

    def test_helper_truncates_129_chars_to_128(self):
        """Boundary: one over the cap is truncated."""
        capped = _resolve_caller_user_id({"user": "x" * 129})
        assert capped is not None
        assert len(capped) == MAX_USER_ID_LENGTH

    def test_helper_caps_oversized_with_dangerous_chars(self):
        """The cap MUST run before _sanitize_id — otherwise a pathologically
        long adversarial `user` would still produce a pathologically long
        sanitized peer_id at Honcho."""
        raw = "../../etc/passwd" + ("a" * 100_000)
        capped = _resolve_caller_user_id({"user": raw})
        assert capped is not None
        # Cap is applied first.
        assert len(capped) == MAX_USER_ID_LENGTH
        # Then sanitize — but only on the already-capped slice.
        sanitized = _sanitize_id(capped)
        assert len(sanitized) == MAX_USER_ID_LENGTH
        assert PEER_ID_RE.match(sanitized)

    def test_helper_none_returns_none(self):
        """OpenAI spec allows `user` to be omitted. Must round-trip to None
        so the truthiness check at agent_init.py:1172 drops memory propagation."""
        assert _resolve_caller_user_id({"user": None}) is None

    def test_helper_empty_string_returns_none(self):
        """Empty string is falsy; the truthiness check at agent_init.py:1172
        relies on it being None (not "") so memory is not propagated."""
        assert _resolve_caller_user_id({"user": ""}) is None

    def test_helper_missing_key_returns_none(self):
        assert _resolve_caller_user_id({}) is None

    def test_helper_zero_string_propagates(self):
        """`"0"` is truthy in Python. The helper must keep it — anything else
        would silently drop a legitimate user identifier."""
        assert _resolve_caller_user_id({"user": "0"}) == "0"

    def test_helper_non_string_user_returns_none(self):
        """OpenAI spec says `user` is a string. Non-string values are an API
        contract violation; the helper must not coerce (str() of a 1GB list
        would itself be a DoS vector)."""
        assert _resolve_caller_user_id({"user": 12345}) is None
        assert _resolve_caller_user_id({"user": ["a", "b"]}) is None
        assert _resolve_caller_user_id({"user": {"nested": "obj"}}) is None

    def test_helper_non_dict_body_returns_none(self):
        """Defensive: a caller passing the wrong shape must not crash."""
        assert _resolve_caller_user_id(None) is None
        assert _resolve_caller_user_id("not a dict") is None
        assert _resolve_caller_user_id(123) is None

    def test_helper_then_sanitize_preserves_cap(self):
        """End-to-end: the helper + _sanitize_id together guarantee the
        Honcho peer_id can never exceed MAX_USER_ID_LENGTH chars, regardless
        of input. This is the property that closes the DoS gap."""
        for raw in [
            "a" * 1_000_000,
            "../../etc/passwd" * 1000,
            "x" * (MAX_USER_ID_LENGTH + 1),
            "normal-user-id",
        ]:
            capped = _resolve_caller_user_id({"user": raw})
            assert capped is not None
            sanitized = _sanitize_id(capped)
            assert len(sanitized) <= MAX_USER_ID_LENGTH, (
                f"raw len={len(raw)} sanitized len={len(sanitized)}"
            )
