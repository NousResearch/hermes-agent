import pytest

from tests.fakes.mcp_oauth_peer import (
    FakeOAuthMCPPeer,
    KnownCredentialLoss,
    OAuthFailurePoint,
    capture_oauth_state,
    raise_known_mutation,
    seed_old_oauth_state,
)


_GH_76590 = pytest.mark.xfail(
    strict=True,
    raises=KnownCredentialLoss,
    reason="GH #76590: CLI reauthorization deletes active credentials before success",
)


@pytest.mark.parametrize(
    ("failure_point", "broken_labels"),
    [
        pytest.param(
            OAuthFailurePoint.PROTECTED_RESOURCE_DISCOVERY,
            ("MISSING", "MISSING", "MISSING"),
            marks=_GH_76590,
        ),
        pytest.param(
            OAuthFailurePoint.DYNAMIC_CLIENT_REGISTRATION,
            ("MISSING", "PARTIAL", "PARTIAL"),
            marks=_GH_76590,
        ),
        pytest.param(
            OAuthFailurePoint.MCP_INITIALIZATION,
            ("NEW", "PARTIAL", "PARTIAL"),
            marks=_GH_76590,
        ),
    ],
)
def test_cli_failed_reauth_preserves_active_state(
    tmp_path, monkeypatch, capsys, failure_point, broken_labels
):
    from hermes_cli import mcp_config
    from tools.mcp_oauth_manager import reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path.resolve()))
    reset_manager_for_tests()
    before = seed_old_oauth_state(tmp_path, "reports")
    peer = FakeOAuthMCPPeer(failure_point)
    monkeypatch.setattr(mcp_config, "_probe_single_server", peer.probe)

    result = mcp_config._reauth_oauth_server(
        "reports", {"url": "https://mcp.invalid/mcp", "auth": "oauth"}
    )

    output = capsys.readouterr().out
    assert result is False
    assert "Authentication failed" in output
    assert "ACCESS_TOKEN_FOR_TEST_ONLY" not in output
    assert peer.connect_timeouts == [315.0]
    raise_known_mutation(
        before=before,
        after=capture_oauth_state(tmp_path, "reports"),
        expected_labels=broken_labels,
        surface="cli",
        failure_point=failure_point,
    )
