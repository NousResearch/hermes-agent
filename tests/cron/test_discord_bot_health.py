from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "discord-bot-health.py"


def _module():
    spec = importlib.util.spec_from_file_location("discord_bot_health", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_services_parses_supplied_systemctl_output(monkeypatch) -> None:
    """discover_services must parse the systemctl output we supply via the
    fake subprocess.run, not this host's actual service configuration.

    The supplied stdout contains three hermes-gateway units plus one
    unrelated unit. We assert the three parsed (name, service) pairs are
    present in the result, the unrelated unit is excluded, and the bare
    kensei mapping (hermes-gateway.service -> kensei) is exercised
    alongside named specialist units (kensei-review, quan).

    Runtime discovery also merges a known-inventory fallback; that merge is
    intentional behaviour we do not weaken here, so we assert the parsed
    units are a subset of the full result rather than an exact equality.
    """
    module = _module()

    def fake_run(command, **kwargs):
        assert command == [
            "systemctl",
            "list-unit-files",
            "--type=service",
            "--no-legend",
            "hermes-gateway*.service",
        ]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "hermes-gateway.service disabled enabled\n"
                "hermes-gateway-quan.service disabled enabled\n"
                "hermes-gateway-kensei-review.service disabled enabled\n"
                "unrelated.service enabled enabled\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    parsed = module.discover_services()

    # Bare kensei unit: hermes-gateway.service -> kensei.
    assert ("kensei", "hermes-gateway.service") in parsed
    # Named specialist units parsed from supplied systemctl output.
    assert ("kensei-review", "hermes-gateway-kensei-review.service") in parsed
    assert ("quan", "hermes-gateway-quan.service") in parsed
    # The unrelated unit must be filtered out by the parser.
    assert not any(svc == "unrelated.service" for _, svc in parsed)
    # Result is sorted (runtime contract preserved).
    assert parsed == sorted(parsed)
