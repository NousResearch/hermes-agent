"""Optional local LAYA decision sidecar for engineering-memory advice.

The main Hermes process never imports a model runtime on ordinary startup. The
adapter accepts a small typed JSON protocol and degrades to deterministic-only
behavior whenever the sidecar is unavailable or uncalibrated.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

LAYA_ADAPTER_VERSION = "engineering-memory-laya-v1"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class LayaRuntimeConfig:
    model_repo: str = "convaiinnovations/laya"
    revision: str = ""
    local_path: str = ""
    runtime: str = "subprocess"
    device: str = "auto"
    min_confidence: float = 0.8
    calibration_path: str = ""


@dataclass(frozen=True)
class LayaQuestion:
    question_id: str
    choices: tuple[str, ...]


@dataclass(frozen=True)
class LayaDecision:
    question_id: str
    choice: str
    probability: float


@dataclass(frozen=True)
class LayaEvaluation:
    decisions: tuple[LayaDecision, ...]
    diagnostic: str
    metadata: dict[str, Any]


class LayaRuntime(Protocol):
    model_revision: str
    device: str

    def decide(self, state: dict[str, Any], questions: Sequence[dict[str, Any]]) -> dict[str, Any]: ...


class SubprocessLayaRuntime:
    def __init__(self, command: Sequence[str], *, model_revision: str, device: str = "auto", timeout: float = 15.0):
        self.command = tuple(command)
        self.model_revision = model_revision
        self.device = device
        self.timeout = timeout

    def decide(self, state: dict[str, Any], questions: Sequence[dict[str, Any]]) -> dict[str, Any]:
        request = {"protocol": "laya.decide.v1", "state": state, "questions": list(questions)}
        result = subprocess.run(
            self.command,
            input=json.dumps(request, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=self.timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"LAYA sidecar exited {result.returncode}")
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
            raise ValueError("invalid LAYA sidecar response")
        return payload


class LayaDecisionEngine:
    def __init__(self, runtime: LayaRuntime | None, config: LayaRuntimeConfig):
        self.runtime = runtime
        self.config = config

    def evaluate(self, state: dict[str, Any], questions: Sequence[dict[str, Any]]) -> LayaEvaluation:
        metadata = {
            "adapter_version": LAYA_ADAPTER_VERSION,
            "model_repo": self.config.model_repo,
            "model_revision": getattr(self.runtime, "model_revision", self.config.revision),
            "device": getattr(self.runtime, "device", self.config.device),
            "calibration_path": self.config.calibration_path,
        }
        if self.runtime is None:
            return LayaEvaluation((), "laya_unavailable", metadata)
        if self.config.calibration_path and not Path(self.config.calibration_path).is_file():
            return LayaEvaluation((), "laya_uncalibrated", metadata)
        try:
            payload = self.runtime.decide(state, questions)
            decisions: list[LayaDecision] = []
            for raw in payload.get("decisions", []):
                if not isinstance(raw, dict):
                    raise ValueError("invalid LAYA decision")
                question_id = raw.get("question_id")
                choice = raw.get("choice")
                probability = raw.get("probability")
                if not isinstance(question_id, str) or not isinstance(choice, str) or not isinstance(probability, (int, float)):
                    raise ValueError("invalid LAYA decision fields")
                probability = float(probability)
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("invalid LAYA probability")
                if probability < self.config.min_confidence:
                    return LayaEvaluation((), "laya_uncalibrated", metadata)
                decisions.append(LayaDecision(question_id, choice, probability))
            return LayaEvaluation(tuple(decisions), "ok", metadata)
        except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError):
            return LayaEvaluation((), "laya_unavailable", metadata)


def download_laya(repo: str, revision: str, destination: Path) -> Path:
    if not repo.strip() or not _IMMUTABLE_REVISION.fullmatch(revision.strip()):
        raise ValueError("LAYA download requires an immutable revision")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required only for explicit LAYA download") from exc
    path = snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination))
    return Path(path)


def calibrate_laya(runtime: LayaRuntime, fixtures_path: Path, output_path: Path, *, model_revision: str) -> dict[str, Any]:
    raw_lines = [line for line in fixtures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fixture_digest = hashlib.sha256(("\n".join(raw_lines) + "\n").encode("utf-8")).hexdigest()
    total = 0
    valid = 0
    for line in raw_lines:
        fixture = json.loads(line)
        total += 1
        result = runtime.decide(fixture.get("state", {}), fixture.get("questions", []))
        if isinstance(result, dict) and isinstance(result.get("decisions"), list):
            valid += 1
    artifact = {
        "artifact": "hermes.engineering_memory.laya_calibration.v1",
        "adapter_version": LAYA_ADAPTER_VERSION,
        "model_revision": model_revision,
        "fixture_digest": fixture_digest,
        "fixture_count": total,
        "valid_response_count": valid,
        "held_out": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact
