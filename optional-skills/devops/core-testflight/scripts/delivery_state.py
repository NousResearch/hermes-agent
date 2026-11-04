"""Classify TestFlight readiness separately from device-plane evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any


TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE = (
    "TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE"
)


@dataclass(frozen=True)
class DeliveryEvidence:
    """The independent evidence gates used for a TestFlight outcome."""

    apple_ready: bool
    tester_eligible: bool
    device_paired: bool
    device_tunnel_available: bool
    target_build: str
    installed_build: str | None = None

    @property
    def device_control_available(self) -> bool:
        return self.device_paired and self.device_tunnel_available

    @property
    def outcome(self) -> str:
        if not self.apple_ready or not self.tester_eligible:
            return "TESTFLIGHT_NOT_READY"
        if self.installed_build == self.target_build:
            return "DONE"
        if not self.device_control_available:
            return TESTFLIGHT_READY_DEVICE_CONTROL_UNAVAILABLE
        return "TESTFLIGHT_INSTALL_UNVERIFIED"

    def report(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "device_control_available": self.device_control_available,
                "outcome": self.outcome,
                "installation_verified": self.outcome == "DONE",
            }
        )
        return result


def classify_delivery(**kwargs: Any) -> dict[str, Any]:
    """Return a redacted, machine-readable delivery classification."""

    return DeliveryEvidence(**kwargs).report()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="read evidence JSON from stdin"
    )
    args = parser.parse_args(argv)
    if not args.json:
        parser.error("--json is required")
    evidence = json.load(sys.stdin)
    print(json.dumps(classify_delivery(**evidence), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
