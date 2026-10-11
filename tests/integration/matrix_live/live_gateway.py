"""The handle that live gateway fixtures yield.

The conftest and the test modules both import this module by its dotted path. pytest loads
the conftest itself as a plugin module named ``conftest``, so a class defined there would
exist twice, and the failure hook's ``isinstance`` check would not recognise a gateway that a
test module had built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from testcontainers.core.container import DockerContainer

from tests.fakes.fake_llm_provider import FakeLLMServer


@dataclass(frozen=True)
class LiveGateway:
    container: DockerContainer
    model: FakeLLMServer
    home: Path

    def restart(self, *, wait_for_checkpoint: bool = True) -> None:
        from tests.integration.matrix_live.conftest import _wait_for

        gateway_log = self.home / "logs" / "gateway.log"
        log_offset = len(gateway_log.read_text(encoding="utf-8"))
        self.container.get_wrapped_container().restart(timeout=5)
        if not wait_for_checkpoint:
            return

        def connected() -> bool:
            log = gateway_log.read_text(encoding="utf-8")[log_offset:]
            return (
                "Matrix: connected after initial dispatch checkpoint" in log
                and "Press Ctrl+C to stop" in log
            )

        _wait_for(
            connected,
            "Matrix gateway sync and startup restoration after restart",
            timeout=30,
            details=lambda: self.container.get_wrapped_container()
            .logs()
            .decode(errors="replace")[-6000:],
        )

    def log_tail(self, lines: int = 200) -> str:
        path = self.home / "logs" / "gateway.log"
        if not path.exists():
            return f"{path} does not exist"
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
