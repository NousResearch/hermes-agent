"""Generated service behavior for opt-in systemd memory limits (#123176)."""

from __future__ import annotations

from gateway.config import GatewayConfig, coerce_systemd_memory_limit
from hermes_cli import gateway as gateway_cli


def test_memory_limits_emit_directives_when_configured(monkeypatch):
    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_config",
        lambda: GatewayConfig.from_dict(
            {"systemd_memory_high": "3G", "systemd_memory_max": "6G"}
        ),
        raising=False,
    )

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "MemoryAccounting=yes" in unit
    assert "MemoryHigh=3G" in unit
    assert "MemoryMax=6G" in unit


def test_no_memory_directives_by_default(monkeypatch):
    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_config",
        lambda: GatewayConfig.from_dict({}),
        raising=False,
    )

    unit = gateway_cli.generate_systemd_unit(system=False)

    assert "MemoryHigh=" not in unit
    assert "MemoryMax=" not in unit
    assert "MemoryAccounting=" not in unit


def test_invalid_memory_limits_are_ignored():
    for raw in [None, "", "  ", "infinity", "0", "-1G", "abc", "10X", "150%", True, 1.5]:
        assert coerce_systemd_memory_limit(raw) is None
        config = GatewayConfig.from_dict({"systemd_memory_high": raw, "systemd_memory_max": raw})
        assert config.systemd_memory_high is None
        assert config.systemd_memory_max is None


def test_valid_memory_limits_roundtrip():
    for raw, expected in [("3G", "3G"), ("512m", "512M"), ("25%", "25%"), (1024, "1024"), ("1.5G", "1.5G")]:
        assert coerce_systemd_memory_limit(raw) == expected
    config = GatewayConfig.from_dict({"systemd_memory_high": "3G", "systemd_memory_max": "50%"})
    assert config.systemd_memory_high == "3G"
    assert config.systemd_memory_max == "50%"
    restored = GatewayConfig.from_dict(config.to_dict())
    assert restored.systemd_memory_high == "3G"
    assert restored.systemd_memory_max == "50%"


def test_system_unit_reads_memory_limits_from_target_home(tmp_path, monkeypatch):
    caller_home = tmp_path / "caller"
    target_home = tmp_path / "target"
    caller_home.mkdir()
    target_home.mkdir()
    (caller_home / "config.yaml").write_text("gateway: {}\n", encoding="utf-8")
    (target_home / "config.yaml").write_text(
        "gateway:\n  systemd_memory_high: 3G\n  systemd_memory_max: 6G\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(caller_home))
    monkeypatch.setattr(
        gateway_cli,
        "_system_service_identity",
        lambda _user: ("service", "service", str(tmp_path / "account"), 1001),
    )
    monkeypatch.setattr(
        gateway_cli,
        "_hermes_home_for_target_user",
        lambda _home: str(target_home),
    )

    unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="service")

    assert "MemoryAccounting=yes" in unit
    assert "MemoryHigh=3G" in unit
    assert "MemoryMax=6G" in unit
