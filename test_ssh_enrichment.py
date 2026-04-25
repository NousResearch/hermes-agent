"""
Tests for ssh_enrichment.py
===========================
Run with:  pytest test_ssh_enrichment.py -v

Test cases:
  TC1 — Happy path: Interface flapping → show log (Cisco IOS)
  TC2 — Device unreachable: connection timeout (Cisco IOS)
  TC3 — Authentication failure (FortiGate-simulated)
  TC4 — Output overflow: huge syslog capped at MAX_OUTPUT_CHARS
  TC5 — Linux/FRR: BGP down → vtysh + ip route (multi-command session)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_env(monkeypatch):
    """Set valid SSH credentials via env vars."""
    monkeypatch.setenv("NETBOX_SSH_USERNAME", "noc_operator")
    monkeypatch.setenv("NETBOX_SSH_PASSWORD", "lab_secret_password")


# ---------------------------------------------------------------------------
# TC1 — Happy path: Cisco IOS interface flapping
# ---------------------------------------------------------------------------

def test_tc1_happy_path_interface_flapping(mock_env, monkeypatch):
    """
    TC1: Interface flapping on core-rtr-01 (Cisco IOS).
    Alert trigger = 'interface_flapping'.
    Expected: status=success, commands_run non-empty, outputs have show log.
    """
    from ssh_enrichment import SSHEnricher, AlertContext, SSHEnrichmentResult

    # Mock Netmiko ConnectHandler
    mock_conn = MagicMock()
    mock_conn.send_command_timing.side_effect = [
        # show interfaces status
        """Port      Name  Status     Vlan   Duplex  Speed Type
Gi0/1     UPLINK-01  connected trunk     a-full a-1000 10/100/1000BaseTX
Gi0/0     UPLINK-02  notconnect 1        auto   auto   10/100/1000BaseTX
""",
        # show log (last 50 lines — simulated)
        """%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up
[last 8 flap events repeated]
%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet0/1, changed state to down
%LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to up
Interface GigabitEthernet0/1 has been flapping for 00:03:12"""
    ]
    mock_conn.disconnect = MagicMock()

    with patch("ssh_enrichment._get_netmiko_connection", return_value=mock_conn):
        ctx = AlertContext(
            alert_id="alert-001",
            device_name="core-rtr-01",
            device_ip="10.1.1.1",
            device_type="cisco_ios",
            severity="P2",
            alert_message="Interface GigabitEthernet0/1 is flapping",
        )
        result = SSHEnricher().enrich(
            device_ip="10.1.1.1",
            device_type="cisco_ios",
            alert_context=ctx,
            trigger="interface_flapping",
        )

    # Assertions
    assert isinstance(result, SSHEnrichmentResult)
    assert result.status == "success", f"Expected success, got {result.status}: {result.error_reason}"
    assert result.device_ip == "10.1.1.1"
    assert result.device_type == "cisco_ios"
    assert "show interfaces status" in result.commands_run
    assert "show log" in result.commands_run
    assert "flapping" in result.outputs.get("show log", "").lower()
    assert result.truncated is False
    assert result.duration_seconds < 30  # sanity check


# ---------------------------------------------------------------------------
# TC2 — Device unreachable: timeout
# ---------------------------------------------------------------------------

def test_tc2_device_unreachable(mock_env, monkeypatch):
    """
    TC2: peer-rtr-02 (10.1.2.1) does not respond — timeout.
    Expected: status=unreachable, no crash, pipeline continues.
    """
    from ssh_enrichment import SSHEnricher, SSHEnrichmentResult

    with patch("ssh_enrichment._get_netmiko_connection") as mock_get:
        from netmiko import NetmikoTimeoutException
        mock_get.side_effect = NetmikoTimeoutException("Connection timed out")

        result = SSHEnricher().enrich(
            device_ip="10.1.2.1",
            device_type="cisco_ios",
            trigger="bgp_down",
        )

    assert isinstance(result, SSHEnrichmentResult)
    assert result.status == "unreachable", f"Expected unreachable, got {result.status}"
    assert result.device_ip == "10.1.2.1"
    assert "timeout" in result.error_reason.lower()
    assert result.duration_seconds > 0
    # Outputs should be empty on unreachable
    assert result.outputs == {}
    assert result.commands_run == ["show bgp summary", "show bgp neighbors"]


# ---------------------------------------------------------------------------
# TC3 — Authentication failure
# ---------------------------------------------------------------------------

def test_tc3_auth_failure(mock_env, monkeypatch):
    """
    TC3: FortiGate fw-edge-01 credentials wrong.
    Expected: status=auth_failed, no credential info in error_reason.
    """
    from ssh_enrichment import SSHEnricher, SSHEnrichmentResult

    with patch("ssh_enrichment._get_netmiko_connection") as mock_get:
        from netmiko import NetmikoAuthenticationException
        mock_get.side_effect = NetmikoAuthenticationException(
            "Bad authentication attempt"
        )

        result = SSHEnricher().enrich(
            device_ip="10.1.3.1",
            device_type="fortinet",
            trigger="default",
        )

    assert isinstance(result, SSHEnrichmentResult)
    assert result.status == "auth_failed", f"Expected auth_failed, got {result.status}"
    assert result.device_ip == "10.1.3.1"
    # Must NOT leak the actual password in the error reason
    assert "lab_secret_password" not in (result.error_reason or "")
    assert "authentication" in result.error_reason.lower()


# ---------------------------------------------------------------------------
# TC4 — Output overflow: massive syslog capped
# ---------------------------------------------------------------------------

def test_tc4_output_overflow_capped(mock_env, monkeypatch):
    """
    TC4: leaf-sw-03 show log returns > MAX_OUTPUT_CHARS.
    Expected: output truncated at MAX_OUTPUT_CHARS, truncated=True flag set.
    """
    from ssh_enrichment import SSHEnricher, SSHEnrichmentResult, MAX_OUTPUT_CHARS

    # Generate a log larger than MAX_OUTPUT_CHARS
    huge_log = ("LINE_OF_OUTPUT\n" * 1000)  # ~13K chars >> 5000
    assert len(huge_log) > MAX_OUTPUT_CHARS

    mock_conn = MagicMock()
    # Force show log via explicit_commands (trigger="ospf_down" maps to different commands)
    mock_conn.send_command_timing.side_effect = [huge_log]
    mock_conn.disconnect = MagicMock()

    with patch("ssh_enrichment._get_netmiko_connection", return_value=mock_conn):
        result = SSHEnricher().enrich(
            device_ip="10.1.4.1",
            device_type="cisco_ios",
            explicit_commands=["show log"],   # force the show log command
        )

    assert isinstance(result, SSHEnrichmentResult)
    assert result.status == "success"
    assert result.truncated is True, "Expected truncated=True when output exceeds limit"

    # Check the output is actually capped
    show_log_output = result.outputs.get("show log", "")
    assert len(show_log_output) <= MAX_OUTPUT_CHARS + 100, \
        f"Output should be capped near {MAX_OUTPUT_CHARS}, got {len(show_log_output)}"
    assert "TRUNCATED" in show_log_output, \
        f"Expected '[TRUNCATED]' marker in output, got: {show_log_output[:100]!r}"


# ---------------------------------------------------------------------------
# TC5 — Linux/FRR: BGP blackhole
# ---------------------------------------------------------------------------

def test_tc5_linux_frr_bgp_blackhole(mock_env, monkeypatch):
    """
    TC5: vrr-rtr-linux-01 BGP peer went down → blackhole risk.
    Commands: vtysh show bgp summary + ip route show
    Expected: both commands succeed, BGP idle and route gap identified.
    """
    from ssh_enrichment import SSHEnricher, AlertContext, SSHEnrichmentResult

    vtysh_bgp_output = """IPV4 BGP Summary
Neighbor        AS    Up/Down    State    Reason
10.2.5.1     65001    00:00:07  Idle     Idle (Admin)
"""

    ip_route_output = """default via 10.1.0.1 dev eth0
10.2.0.0/16 via 10.1.0.254 dev eth0
10.2.5.0/24 dev bgp0  proto bird  scope link
"""

    mock_conn = MagicMock()
    # vtysh is called first, then ip route
    mock_conn.send_command_timing.side_effect = [vtysh_bgp_output, ip_route_output]
    mock_conn.disconnect = MagicMock()

    with patch("ssh_enrichment._get_netmiko_connection", return_value=mock_conn):
        ctx = AlertContext(
            alert_id="alert-005",
            device_name="vrr-rtr-linux-01",
            device_ip="10.1.5.1",
            device_type="linux",
            severity="P1",
            alert_message="BGP session down on vrr-rtr-linux-01",
        )
        result = SSHEnricher().enrich(
            device_ip="10.1.5.1",
            device_type="linux",
            alert_context=ctx,
            trigger="bgp_down",
        )

    assert isinstance(result, SSHEnrichmentResult)
    assert result.status == "success", f"Expected success, got {result.status}: {result.error_reason}"
    assert result.device_ip == "10.1.5.1"
    assert result.device_type == "linux"

    # Both commands must have been called
    assert len(result.commands_run) == 2
    vtysh_cmd = 'vtysh -c "show bgp summary"'
    assert vtysh_cmd in result.commands_run, f"Expected {vtysh_cmd!r} in commands_run"

    # Verify BGP idle peer in vtysh output
    vtysh_out = result.outputs.get(vtysh_cmd, "")
    assert "10.2.5.1" in vtysh_out
    assert "Idle" in vtysh_out
    assert "65001" in vtysh_out

    # Verify route gap in ip route output (10.2.5.0/24 via bgp0 — peer down = blackhole)
    ip_cmd = "ip route show"
    ip_out = result.outputs.get(ip_cmd, "")
    assert "10.2.5.0/24" in ip_out
    # The route is still present in the kernel (blackhole risk — FRR lost the peer)
    assert result.truncated is False


# ---------------------------------------------------------------------------
# Additional unit tests
# ---------------------------------------------------------------------------

def test_command_allowlist_rejects_unknown_command(mock_env, monkeypatch):
    """
    A command NOT in the allowlist must be silently dropped.
    'show running-config' is NOT in CISCO_ALLOWLIST.
    """
    from ssh_enrichment import SSHEnricher

    mock_conn = MagicMock()
    mock_conn.send_command_timing.return_value = "running config output"
    mock_conn.disconnect = MagicMock()

    with patch("ssh_enrichment._get_netmiko_connection", return_value=mock_conn):
        result = SSHEnricher().enrich(
            device_ip="10.1.1.1",
            device_type="cisco_ios",
            explicit_commands=["show version", "show running-config"],
        )

    # show running-config should be rejected — not in allowlist
    assert "show running-config" not in result.commands_run
    assert "show version" in result.commands_run
    # And Netmiko should only have been called once (for show version)
    assert mock_conn.send_command_timing.call_count == 1


def test_infer_trigger_from_bgp_message():
    """Heuristic trigger inference from alert message text."""
    from ssh_enrichment import SSHEnricher

    enricher = SSHEnricher.__new__(SSHEnricher)  # cheap — no init needed

    assert enricher._infer_trigger_from_message(
        "BGP peer 10.2.5.1 is down"
    ) == "bgp_down"

    assert enricher._infer_trigger_from_message(
        "OSPF neighbor FULL to DOWN on Gi0/0/1"
    ) == "ospf_down"

    assert enricher._infer_trigger_from_message(
        "Interface Gi0/1 is flapping"
    ) == "interface_flapping"

    assert enricher._infer_trigger_from_message(
        "High CPU usage detected"
    ) == "high_cpu"

    assert enricher._infer_trigger_from_message(
        "Random unrelated alert"
    ) == "default"


def test_cred_required_without_env(monkeypatch):
    """No credentials in env → must raise ValueError at init."""
    import os
    # Clear env vars
    monkeypatch.delenv("NETBOX_SSH_USERNAME", raising=False)
    monkeypatch.delenv("NETBOX_SSH_PASSWORD", raising=False)

    from ssh_enrichment import SSHEnricher
    with pytest.raises(ValueError, match="SSH credentials not set"):
        SSHEnricher()


def test_cred_via_explicit_args():
    """Credentials passed directly to __init__ — env vars not needed."""
    from ssh_enrichment import SSHEnricher, SSHEnrichmentResult
    import os

    mock_conn = MagicMock()
    mock_conn.send_command_timing.return_value = "show version output"
    mock_conn.disconnect = MagicMock()

    with patch("ssh_enrichment._get_netmiko_connection", return_value=mock_conn):
        # Pass credentials explicitly — no env vars needed
        enricher = SSHEnricher(username="noc_op", password="secret123")
        result = enricher.enrich(
            device_ip="10.1.1.1",
            device_type="cisco_ios",
            explicit_commands=["show version"],
        )

    assert result.status == "success"
    # error_reason should be None on success — never reference leaked creds
    assert result.error_reason is None, \
        f"error_reason should be None on success, got: {result.error_reason}"


def test_sanitise_output_strips_ansi():
    """ANSI escape sequences must be stripped from output."""
    from ssh_enrichment import _sanitise_output

    raw = "\x1b[1mBold\x1b[0m \x1b[32mGreen\x1b[0m\r\nNormal\x1b[K"
    clean, truncated = _sanitise_output(raw)
    assert "\x1b" not in clean
    assert "Bold" in clean
    assert "Green" in clean
    assert "\r" not in clean
    assert truncated is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
