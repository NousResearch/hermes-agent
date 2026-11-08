"""Client for a sealed evaluation battery that never returns task contents."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class BatteryScore:
    passed: bool
    cost: float

    def __post_init__(self) -> None:
        if type(self.passed) is not bool:
            raise ValueError("battery score passed must be a boolean")
        if isinstance(self.cost, bool) or not isinstance(self.cost, (int, float)) or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("battery score cost must be a finite non-negative number")


class SealedBatteryClient:
    """Submit opaque candidate references; receive only pass/fail and cost."""

    def __init__(self, endpoint: str, *, timeout: float = 60.0):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def score(self, *, battery_version: str, harness_version: str, task_id: str, candidate_ref: str) -> BatteryScore:
        payload = {
            "battery_version": battery_version,
            "harness_version": harness_version,
            "task_id": task_id,
            "candidate_ref": candidate_ref,
        }
        request = Request(
            f"{self.endpoint}/v1/score",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            raw: Any = json.load(response)
        if not isinstance(raw, dict) or set(raw) != {"passed", "cost"}:
            raise ValueError("sealed scorer response must contain only passed and cost")
        if type(raw["passed"]) is not bool or isinstance(raw["cost"], bool) or not isinstance(raw["cost"], (int, float)):
            raise ValueError("sealed scorer returned invalid passed/cost values")
        return BatteryScore(raw["passed"], float(raw["cost"]))
