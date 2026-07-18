"""Run the local-only IVX Edge pilot demonstration."""

from __future__ import annotations

import json
import time

from .runtime import Command, DeviceRuntime, EnrollmentAuthority, RuntimePolicy, sign_command


def main() -> None:
    authority = EnrollmentAuthority()
    identity = authority.enroll(
        authority.issue_code("demo-workspace", "demo-app"),
        "demo-endpoint",
    )
    runtime = DeviceRuntime(
        identity,
        RuntimePolicy(frozenset({"sensor.temperature.read", "screen.content.set"})),
        confirm=lambda command: command.capability == "screen.content.set",
    )
    now = int(time.time())
    command = sign_command(
        Command(
            command_id="demo-command-001",
            device_id=identity.device_id,
            workspace_id=identity.workspace_id,
            app_id=identity.app_id,
            capability="sensor.temperature.read",
            payload={},
            issued_at=now,
            expires_at=now + 60,
            nonce="demo-nonce-001",
        ),
        identity.key,
    )
    result = runtime.submit(command)
    print(
        json.dumps(
            {
                "mode": "simulation",
                "hardware_access": False,
                "result": result.__dict__,
                "audit": [event.__dict__ for event in runtime.audit],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
