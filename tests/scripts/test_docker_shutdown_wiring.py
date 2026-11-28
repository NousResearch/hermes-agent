"""Behavioral tests for dynamic gateway container shutdown wiring."""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "docker" / "cont-finish.d" / "50-stop-gateways"


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_shutdown_hook_exits_without_gateway_slots(tmp_path: Path) -> None:
    service_root = tmp_path / "service"
    service_root.mkdir()

    result = subprocess.run(
        ["sh", str(HOOK)],
        env={**os.environ, "GATEWAY_SERVICE_ROOT": str(service_root)},
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr


def test_stale_slot_does_not_cancel_healthy_slot_wait(tmp_path: Path) -> None:
    service_root = tmp_path / "service"
    service_root.mkdir()
    healthy = service_root / "gateway-healthy"
    stale = service_root / "gateway-stale"
    healthy.mkdir()
    stale.mkdir()

    log = tmp_path / "calls.log"
    marker = tmp_path / "healthy-teardown.marker"
    desired_state = tmp_path / "gateway_state.json"
    desired_state.write_text('{"enabled": true}\n', encoding="utf-8")

    setuidgid = _write_executable(
        tmp_path / "s6-setuidgid",
        "#!/bin/sh\nshift\nexec \"$@\"\n",
    )
    s6_svc = _write_executable(
        tmp_path / "s6-svc",
        "#!/bin/sh\nprintf 'svc:%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "case \"$*\" in *gateway-stale) exit 111;; esac\n",
    )
    s6_svwait = _write_executable(
        tmp_path / "s6-svwait",
        "#!/bin/sh\nprintf 'wait:%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
        "case \"$*\" in *gateway-stale) exit 111;; esac\n"
        "sleep 0.15\nprintf done > \"$HEALTHY_MARKER\"\n",
    )

    result = subprocess.run(
        ["sh", str(HOOK)],
        env={
            **os.environ,
            "CALL_LOG": str(log),
            "GATEWAY_SERVICE_ROOT": str(service_root),
            "HEALTHY_MARKER": str(marker),
            "S6_SETUIDGID": str(setuidgid),
            "S6_SVC": str(s6_svc),
            "S6_SVWAIT": str(s6_svwait),
        },
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("svc:") and "gateway-healthy" in line for line in calls)
    assert any(line.startswith("svc:") and "gateway-stale" in line for line in calls)
    assert any(line.startswith("wait:") and "gateway-healthy" in line for line in calls)
    assert any(line.startswith("wait:") and "gateway-stale" in line for line in calls)
    assert marker.read_text(encoding="utf-8") == "done"
    assert desired_state.read_text(encoding="utf-8") == '{"enabled": true}\n'
