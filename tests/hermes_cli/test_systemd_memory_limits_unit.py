"""Generated service behavior for the opt-in systemd memory ceiling (#81625).

The gateway rewrites its own unit file whenever the install drifts, so an
operator-added ``MemoryMax=``/``MemoryHigh=`` is wiped on the next start and
the leak they were containing takes the whole machine down again. The ceiling
has to come from config so the generator emits it itself.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, coerce_systemd_memory_limit
from hermes_cli import gateway as gateway_cli


def _unit_with(monkeypatch, data: dict) -> str:
    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_config",
        lambda: GatewayConfig.from_dict(data),
        raising=False,
    )
    return gateway_cli.generate_systemd_unit(system=False)


class TestLimitCoercion:
    @pytest.mark.parametrize(
        "value", ["8G", "6144M", "2147483648", "80%", "infinity", " 8G "]
    )
    def test_supported_systemd_sizes_are_kept(self, value):
        assert coerce_systemd_memory_limit(value) == value.strip()

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "",
            "   ",
            0,
            True,
            "8 GB",
            "eight",
            "8G\nExecStartPre=/bin/rm -rf /",
            "-8G",
            "8Z",
        ],
    )
    def test_unusable_values_resolve_to_absent(self, value):
        """A typo must leave the unit unlimited, never half a directive.

        The value is interpolated straight into the unit file, so rejecting
        an unrecognised token is also the injection guard.
        """
        assert coerce_systemd_memory_limit(value) == ""

    def test_integer_bytes_are_accepted(self):
        assert coerce_systemd_memory_limit(2147483648) == "2147483648"


class TestGeneratedUnit:
    def test_unset_config_emits_no_memory_directives(self, monkeypatch):
        """Default stays exactly as it is today: no ceiling, no surprise kills."""
        unit = _unit_with(monkeypatch, {})

        assert "MemoryMax=" not in unit
        assert "MemoryHigh=" not in unit

    def test_configured_limits_land_in_the_service_section(self, monkeypatch):
        unit = _unit_with(
            monkeypatch,
            {"systemd_memory_high": "6G", "systemd_memory_max": "8G"},
        )

        assert "MemoryHigh=6G" in unit
        service = unit.split("[Service]", 1)[1].split("[Install]", 1)[0]
        assert "MemoryMax=8G" in service

    def test_either_limit_can_be_set_alone(self, monkeypatch):
        unit = _unit_with(monkeypatch, {"systemd_memory_max": "8G"})

        assert "MemoryMax=8G" in unit
        assert "MemoryHigh=" not in unit

    def test_nested_gateway_section_is_honoured(self, monkeypatch):
        unit = _unit_with(monkeypatch, {"gateway": {"systemd_memory_max": "4G"}})

        assert "MemoryMax=4G" in unit

    def test_rejected_value_leaves_the_unit_unlimited(self, monkeypatch):
        unit = _unit_with(monkeypatch, {"systemd_memory_max": "8 GB"})

        assert "MemoryMax=" not in unit

    def test_system_unit_carries_the_limits_too(self, tmp_path, monkeypatch):
        """The system unit is a separate template — it must not drift."""
        target_home = tmp_path / "target"
        target_home.mkdir()
        (target_home / "config.yaml").write_text(
            "gateway:\n  systemd_memory_max: 8G\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "caller"))
        monkeypatch.setattr(
            gateway_cli,
            "_system_service_identity",
            lambda _user: ("service", "service", str(tmp_path / "account")),
        )
        monkeypatch.setattr(
            gateway_cli,
            "_hermes_home_for_target_user",
            lambda _home: str(target_home),
        )

        unit = gateway_cli.generate_systemd_unit(system=True, run_as_user="service")

        assert "MemoryMax=8G" in unit
