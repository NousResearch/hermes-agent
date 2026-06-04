"""Tests for acp_client.transport — the default-off transport selection seam.

Pure decision logic: no ``acp`` import and nothing launched.  ``acp``
availability is injected so these tests run identically with or without the
optional extra installed.
"""

from acp_client.transport import (
    LAUNCH_GUARD_ENV_VAR,
    TRANSPORT_ACP,
    TRANSPORT_ENV_VAR,
    TRANSPORT_PTY,
    resolve_transport,
)

_AVAIL = lambda: True  # noqa: E731 - injected acp-available predicate
_UNAVAIL = lambda: False  # noqa: E731


class TestDefaultOff:
    def test_empty_env_defaults_to_pty(self):
        d = resolve_transport({}, acp_available_fn=_AVAIL)
        assert d.transport == TRANSPORT_PTY
        assert d.requested == TRANSPORT_PTY
        assert not d.is_acp
        assert not d.fell_back
        assert d.refusal is None

    def test_non_acp_value_is_pty(self):
        d = resolve_transport(
            {TRANSPORT_ENV_VAR: "claude", LAUNCH_GUARD_ENV_VAR: "1"},
            acp_available_fn=_AVAIL,
        )
        assert d.transport == TRANSPORT_PTY
        assert not d.fell_back  # never requested acp

    def test_pty_does_not_consult_acp_available(self):
        # Default lane must not even need to know whether acp is importable.
        def _boom():
            raise AssertionError("acp_available must not be called on the PTY path")

        d = resolve_transport({}, acp_available_fn=_boom)
        assert d.transport == TRANSPORT_PTY


class TestAcpRequested:
    def test_acp_unavailable_falls_back_with_refusal(self):
        d = resolve_transport(
            {TRANSPORT_ENV_VAR: "acp", LAUNCH_GUARD_ENV_VAR: "1"},
            acp_available_fn=_UNAVAIL,
        )
        assert d.transport == TRANSPORT_PTY
        assert d.requested == TRANSPORT_ACP
        assert d.fell_back
        assert d.refusal and "not installed" in d.refusal

    def test_acp_without_launch_guard_falls_back_with_refusal(self):
        d = resolve_transport({TRANSPORT_ENV_VAR: "acp"}, acp_available_fn=_AVAIL)
        assert d.transport == TRANSPORT_PTY
        assert d.fell_back
        assert d.refusal and LAUNCH_GUARD_ENV_VAR in d.refusal

    def test_acp_with_both_gates_selects_acp(self):
        d = resolve_transport(
            {TRANSPORT_ENV_VAR: "acp", LAUNCH_GUARD_ENV_VAR: "1"},
            acp_available_fn=_AVAIL,
        )
        assert d.transport == TRANSPORT_ACP
        assert d.is_acp
        assert not d.fell_back
        assert d.refusal is None

    def test_launch_guard_accepts_truthy_aliases(self):
        for val in ("1", "true", "yes", "on", "TRUE", "On"):
            d = resolve_transport(
                {TRANSPORT_ENV_VAR: "acp", LAUNCH_GUARD_ENV_VAR: val},
                acp_available_fn=_AVAIL,
            )
            assert d.is_acp, val

    def test_launch_guard_rejects_other_values(self):
        for val in ("0", "false", "", "no", "maybe"):
            d = resolve_transport(
                {TRANSPORT_ENV_VAR: "acp", LAUNCH_GUARD_ENV_VAR: val},
                acp_available_fn=_AVAIL,
            )
            assert d.transport == TRANSPORT_PTY, val
            assert d.fell_back, val

    def test_acp_aliases_are_normalized(self):
        for alias in ("acp", "ACP", "acp-client", "acp_client", "  acp  "):
            d = resolve_transport(
                {TRANSPORT_ENV_VAR: alias, LAUNCH_GUARD_ENV_VAR: "1"},
                acp_available_fn=_AVAIL,
            )
            assert d.is_acp, alias


class TestNeverRaises:
    def test_returns_data_not_exceptions(self):
        # The seam never raises; refusal is carried as data for the caller.
        d = resolve_transport({TRANSPORT_ENV_VAR: "acp"}, acp_available_fn=_UNAVAIL)
        assert isinstance(d.refusal, str)
