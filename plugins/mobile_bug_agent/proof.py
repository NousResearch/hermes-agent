import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import MonicaConfig, runtime_root

_MANIFEST_NAME = "monica-proof-manifest.json"


@dataclass(frozen=True)
class ProofResult:
    passed: bool
    summary: str
    output: str
    artifacts: tuple[str, ...]
    platforms: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "output": self.output,
            "artifacts": list(self.artifacts),
            "platforms": list(self.platforms),
        }


RunCommand = Callable[[str, Path, int, dict[str, str]], tuple[int, str, str]]


class ProofRunner:
    def __init__(
        self,
        *,
        config: MonicaConfig,
        run_command: RunCommand | None = None,
    ) -> None:
        self.config = config
        self._run_command = run_command or self._default_run

    def run(
        self,
        *,
        run: Any,
        worktree: str | Path,
        verification: dict[str, Any] | None = None,
    ) -> ProofResult:
        worktree_path = Path(worktree)
        command_list = tuple(command for command in self.config.proof.commands if command.strip())
        platforms = tuple(platform for platform in self.config.proof.platform_order if platform.strip())
        proof_dir = self._proof_dir(run)
        proof_dir.mkdir(parents=True, exist_ok=True)

        if not worktree_path.is_dir():
            return ProofResult(
                passed=False,
                summary="Proof blocked: worktree does not exist.",
                output=f"Worktree does not exist: {worktree_path}",
                artifacts=(),
                platforms=platforms,
            )
        if not (worktree_path / ".git").exists():
            return ProofResult(
                passed=False,
                summary="Proof blocked: worktree is not a git worktree.",
                output=f"Worktree is not a git worktree: {worktree_path}",
                artifacts=(),
                platforms=platforms,
            )
        if not command_list:
            return ProofResult(
                passed=False,
                summary="Proof blocked: proof.commands is empty.",
                output="mobile_bug_agent.proof.commands is empty.",
                artifacts=(),
                platforms=platforms,
            )

        output_parts: list[str] = []
        timeout = self.config.proof.timeout_minutes * 60
        env = self._proof_env(run=run, worktree=worktree_path, proof_dir=proof_dir)
        for command in command_list:
            code, stdout, stderr = self._run_command(command, worktree_path, timeout, env)
            output_parts.append(
                "\n".join(
                    [
                        f"$ {command}",
                        stdout.strip(),
                        stderr.strip(),
                    ]
                ).strip()
            )
            if code != 0:
                return ProofResult(
                    passed=False,
                    summary=f"Proof blocked: {command}",
                    output="\n\n".join(output_parts),
                    artifacts=self._collect_artifacts(proof_dir),
                    platforms=platforms,
                )

        artifacts = self._collect_artifacts(proof_dir)
        if not artifacts:
            return ProofResult(
                passed=False,
                summary="Proof blocked: proof commands produced no artifacts.",
                output="\n\n".join(output_parts),
                artifacts=(),
                platforms=platforms,
            )
        missing_platforms = _missing_platform_artifacts(artifacts=artifacts, platforms=platforms)
        if missing_platforms:
            missing = ", ".join(missing_platforms)
            return ProofResult(
                passed=False,
                summary=f"Proof blocked: missing required platform artifacts: {missing}.",
                output="\n\n".join(output_parts),
                artifacts=artifacts,
                platforms=platforms,
            )

        manifest_path = self._write_manifest(
            run=run,
            worktree=worktree_path,
            proof_dir=proof_dir,
            artifacts=artifacts,
            platforms=platforms,
        )
        artifacts = (str(manifest_path), *artifacts)
        return ProofResult(
            passed=True,
            summary="Proof captured.",
            output="\n\n".join(output_parts),
            artifacts=artifacts,
            platforms=platforms,
        )

    def _proof_dir(self, run: Any) -> Path:
        raw_artifact_dir = self.config.proof.artifact_dir.strip() or "proof"
        artifact_dir = Path(raw_artifact_dir).expanduser()
        if artifact_dir.is_absolute():
            root = artifact_dir
        else:
            root = runtime_root(self.config) / artifact_dir
        return root / str(getattr(run, "id", "") or "unknown-run")

    def _proof_env(self, *, run: Any, worktree: Path, proof_dir: Path) -> dict[str, str]:
        env = dict(os.environ)
        hermes_agent_root = Path(__file__).resolve().parents[2]
        pythonpath_parts = [str(hermes_agent_root)]
        if env.get("PYTHONPATH"):
            pythonpath_parts.append(str(env["PYTHONPATH"]))
        env.update(
            {
                "MONICA_HERMES_AGENT_ROOT": str(hermes_agent_root),
                "MONICA_PROOF_DIR": str(proof_dir),
                "MONICA_WORKTREE": str(worktree),
                "MONICA_RUN_ID": str(getattr(run, "id", "") or ""),
                "MONICA_LINEAR_IDENTIFIER": str(getattr(run, "linear_identifier", "") or ""),
                "MONICA_PROOF_PLATFORM_ORDER": ",".join(self.config.proof.platform_order),
                "MONICA_DEV_CLIENT_SCHEME": self.config.proof.dev_client_scheme,
                "MONICA_IOS_SIMULATOR_UDID": self.config.proof.ios_simulator_udid,
                "MONICA_IOS_BUNDLE_ID": self.config.proof.ios_bundle_id,
                "MONICA_ANDROID_SERIAL": self.config.proof.android_serial,
                "MONICA_ANDROID_AVD": self.config.proof.android_avd,
                "MONICA_ANDROID_PACKAGE": self.config.proof.android_package,
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            }
        )
        return env

    @staticmethod
    def _collect_artifacts(proof_dir: Path) -> tuple[str, ...]:
        if not proof_dir.is_dir():
            return ()
        return tuple(
            str(path)
            for path in sorted(proof_dir.rglob("*"))
            if path.is_file() and path.name != _MANIFEST_NAME
        )

    def _write_manifest(
        self,
        *,
        run: Any,
        worktree: Path,
        proof_dir: Path,
        artifacts: tuple[str, ...],
        platforms: tuple[str, ...],
    ) -> Path:
        manifest_path = proof_dir / _MANIFEST_NAME
        payload = {
            "run_id": str(getattr(run, "id", "") or ""),
            "linear_identifier": str(getattr(run, "linear_identifier", "") or ""),
            "branch_name": str(getattr(run, "branch_name", "") or ""),
            "worktree": str(worktree),
            "platforms": list(platforms),
            "proof_artifacts": list(artifacts),
        }
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _default_run(
        command: str,
        cwd: Path,
        timeout: int,
        env: dict[str, str],
    ) -> tuple[int, str, str]:
        try:
            proc = subprocess.run(
                command,
                cwd=str(cwd),
                shell=True,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return 124, "", f"command timed out after {timeout}s"
        return proc.returncode, proc.stdout, proc.stderr


def _missing_platform_artifacts(*, artifacts: tuple[str, ...], platforms: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for platform in platforms:
        normalized = _normalize_platform_name(platform)
        if not normalized:
            continue
        if not any(_artifact_matches_platform(artifact, normalized) for artifact in artifacts):
            missing.append(normalized)
    return tuple(missing)


def _artifact_matches_platform(artifact: str, platform: str) -> bool:
    path = Path(artifact)
    haystack = " ".join((path.name, path.stem, *path.parts[-3:])).lower()
    if platform == "ios":
        return any(token in haystack for token in ("ios", "iphone", "ipad"))
    return platform in haystack


def _normalize_platform_name(platform: str) -> str:
    value = str(platform or "").strip().lower()
    if value in {"iphone", "ipad", "ios-simulator"}:
        return "ios"
    if value in {"android-emulator"}:
        return "android"
    return value
