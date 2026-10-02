"""Static regression checks for dynamic gateway container shutdown wiring."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
HOOK = ROOT / "docker" / "cont-finish.d" / "50-stop-gateways"


def test_dynamic_gateway_shutdown_hook_is_installed() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    hook = HOOK.read_text(encoding="utf-8")

    assert "COPY --chmod=0755 docker/cont-finish.d/50-stop-gateways /etc/cont-finish.d/50-stop-gateways" in dockerfile
    assert "ENV S6_KILL_FINISH_MAXTIME=45000" in dockerfile
    assert "/run/service/gateway-*" in hook
    assert "s6-svc -d" in hook
    assert "for slot do" in hook
    assert "s6-svwait -D -t 40000" in hook
    assert "desired_state" in hook


def test_dynamic_gateway_shutdown_hook_exits_without_gateway_slots() -> None:
    hook = HOOK.read_text(encoding="utf-8")

    assert '[ "$#" -gt 0 ] || exit 0' in hook
