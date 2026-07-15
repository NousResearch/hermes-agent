"""Transactional provider/model switching core.

The pure engine is side-effect agnostic: production adapters supply profile
preflight, config I/O, gateway restart and an active-interpreter API smoke.
No credential or command values are included in result payloads.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from hermes_cli.model_preflight import ModelPreflightResult


class ConfigRevisionConflict(RuntimeError):
    """The on-disk config changed after the transaction snapshot."""


@dataclass(frozen=True)
class ModelPair:
    provider: str
    model: str

    def normalized(self) -> "ModelPair":
        return ModelPair(self.provider.strip(), self.model.strip())


@dataclass(frozen=True)
class ModelSmokeEvidence:
    provider: str
    model: str
    api_calls: int
    completed: bool


@dataclass
class TransactionHooks:
    collect_preflight: Callable[[], ModelPreflightResult]
    read_config: Callable[[], dict[str, Any]]
    write_config: Callable[[dict[str, Any], dict[str, Any]], None]
    restart_gateway: Callable[[int | None], int]
    smoke_model: Callable[[ModelPair, str, str], ModelSmokeEvidence]
    restart_allowed: Callable[[], bool] = field(
        default_factory=lambda: (lambda: True)
    )


@dataclass(frozen=True)
class ModelTransactionResult:
    status: str
    profile: str
    previous: ModelPair
    target: ModelPair
    previous_pid: int | None = None
    new_pid: int | None = None
    smoke: ModelSmokeEvidence | None = None
    rollback_verified: bool | None = None
    blockers: tuple[str, ...] = field(default_factory=tuple)


def _pair_from_config(config: Mapping[str, Any]) -> ModelPair:
    model_cfg = config.get("model")
    if isinstance(model_cfg, Mapping):
        return ModelPair(
            str(model_cfg.get("provider") or "").strip(),
            str(
                model_cfg.get("default")
                or model_cfg.get("model")
                or model_cfg.get("name")
                or ""
            ).strip(),
        )
    if isinstance(model_cfg, str):
        return ModelPair("", model_cfg.strip())
    return ModelPair("", "")


def _candidate_config(config: Mapping[str, Any], target: ModelPair) -> dict[str, Any]:
    candidate = copy.deepcopy(dict(config))
    existing = candidate.get("model")
    model_cfg = dict(existing) if isinstance(existing, Mapping) else {}
    model_cfg["provider"] = target.provider
    model_cfg["default"] = target.model
    candidate["model"] = model_cfg
    return candidate


def _smoke_matches(smoke: ModelSmokeEvidence, pair: ModelPair) -> bool:
    return bool(
        smoke.completed
        and smoke.api_calls > 0
        and smoke.provider == pair.provider
        and smoke.model == pair.model
    )


def execute_model_transaction(
    target: ModelPair,
    *,
    confirm_profile: str,
    apply: bool,
    hooks: TransactionHooks,
) -> ModelTransactionResult:
    """Preview or execute a provider/model transaction with verified rollback."""

    target = target.normalized()
    preflight = hooks.collect_preflight()
    profile = preflight.profile_id or Path(preflight.profile_home).name

    blockers: list[str] = []
    if not target.provider:
        blockers.append("target_provider_missing")
    if not target.model:
        blockers.append("target_model_missing")
    if preflight.status != "PASS":
        blockers.append("preflight_not_pass")
    if not confirm_profile or confirm_profile != profile:
        blockers.append("confirmed_profile_mismatch")
    if not preflight.gateway_interpreter:
        blockers.append("gateway_interpreter_missing")

    if blockers:
        return ModelTransactionResult(
            status="BLOCKED",
            profile=profile,
            previous=ModelPair(
                preflight.configured_provider, preflight.configured_model
            ),
            target=target,
            previous_pid=preflight.gateway_pid,
            blockers=tuple(blockers),
        )

    original_config = hooks.read_config()
    previous = _pair_from_config(original_config)
    preflight_pair = ModelPair(
        preflight.configured_provider, preflight.configured_model
    ).normalized()
    if previous != preflight_pair:
        return ModelTransactionResult(
            status="BLOCKED",
            profile=profile,
            previous=previous,
            target=target,
            previous_pid=preflight.gateway_pid,
            blockers=("preflight_config_drift",),
        )
    if not previous.provider or not previous.model:
        return ModelTransactionResult(
            status="BLOCKED",
            profile=profile,
            previous=previous,
            target=target,
            previous_pid=preflight.gateway_pid,
            blockers=("previous_pair_missing",),
        )

    if not apply:
        return ModelTransactionResult(
            status="PREVIEW",
            profile=profile,
            previous=previous,
            target=target,
            previous_pid=preflight.gateway_pid,
        )

    if target == previous:
        return ModelTransactionResult(
            status="NOOP",
            profile=profile,
            previous=previous,
            target=target,
            previous_pid=preflight.gateway_pid,
        )

    if not hooks.restart_allowed():
        return ModelTransactionResult(
            status="BLOCKED",
            profile=profile,
            previous=previous,
            target=target,
            previous_pid=preflight.gateway_pid,
            blockers=("transaction_restart_not_supervisor_safe",),
        )

    write_committed = False
    new_pid: int | None = None
    target_smoke: ModelSmokeEvidence | None = None
    failure_code = "transaction_verification_failed"
    try:
        candidate = _candidate_config(original_config, target)
        hooks.write_config(original_config, candidate)
        write_committed = True
        if _pair_from_config(hooks.read_config()) != target:
            failure_code = "target_config_readback_mismatch"
            raise RuntimeError(failure_code)

        new_pid = hooks.restart_gateway(preflight.gateway_pid)
        if not new_pid or new_pid == preflight.gateway_pid:
            failure_code = "gateway_pid_not_replaced"
            raise RuntimeError(failure_code)

        target_smoke = hooks.smoke_model(
            target, preflight.gateway_interpreter, profile
        )
        if not _smoke_matches(target_smoke, target):
            failure_code = "target_smoke_mismatch"
            raise RuntimeError(failure_code)

        return ModelTransactionResult(
            status="PASS",
            profile=profile,
            previous=previous,
            target=target,
            previous_pid=preflight.gateway_pid,
            new_pid=new_pid,
            smoke=target_smoke,
            rollback_verified=None,
        )
    except Exception as exc:
        safe_failure_codes = {
            "config_lock_timeout",
            "gateway_restart_failed",
            "gateway_restart_timeout",
            "model_smoke_failed",
            "model_smoke_timeout",
        }
        if isinstance(exc, ConfigRevisionConflict):
            failure_code = "config_revision_conflict"
        elif str(exc) in safe_failure_codes:
            failure_code = str(exc)
        elif failure_code == "transaction_verification_failed":
            failure_code = f"transaction_exception:{type(exc).__name__}"

    if not write_committed:
        return ModelTransactionResult(
            status="FAILED",
            profile=profile,
            previous=previous,
            target=target,
            previous_pid=preflight.gateway_pid,
            new_pid=new_pid,
            smoke=target_smoke,
            blockers=(failure_code,),
        )

    rollback_blockers = [failure_code]
    rollback_verified = False
    try:
        current_config = hooks.read_config()
        current_pair = _pair_from_config(current_config)
        if current_pair not in {target, previous}:
            raise RuntimeError("rollback_conflict")
        rollback_config = _candidate_config(current_config, previous)
        hooks.write_config(current_config, rollback_config)
        if _pair_from_config(hooks.read_config()) != previous:
            raise RuntimeError("rollback_config_readback_mismatch")
        rollback_pid = hooks.restart_gateway(new_pid or preflight.gateway_pid)
        if not rollback_pid or rollback_pid == (new_pid or preflight.gateway_pid):
            raise RuntimeError("rollback_pid_not_replaced")
        rollback_smoke = hooks.smoke_model(
            previous, preflight.gateway_interpreter, profile
        )
        if not _smoke_matches(rollback_smoke, previous):
            raise RuntimeError("rollback_smoke_mismatch")
        rollback_verified = True
    except RuntimeError as exc:
        if str(exc) == "rollback_conflict":
            rollback_blockers.append("rollback_conflict")
        rollback_blockers.append("rollback_verification_failed")
    except Exception:
        rollback_blockers.append("rollback_verification_failed")

    return ModelTransactionResult(
        status="ROLLED_BACK" if rollback_verified else "ROLLBACK_FAILED",
        profile=profile,
        previous=previous,
        target=target,
        previous_pid=preflight.gateway_pid,
        new_pid=new_pid,
        smoke=target_smoke,
        rollback_verified=rollback_verified,
        blockers=tuple(rollback_blockers),
    )


def build_production_hooks(
    *,
    config_lock_timeout: float = 10.0,
    restart_timeout: float = 90.0,
    smoke_timeout: float = 180.0,
) -> TransactionHooks:
    """Bind the pure engine to the active profile's real runtime."""

    from gateway.status import (
        get_runtime_status_running_pid,
        read_runtime_status,
    )
    from hermes_cli.config import read_raw_config, save_config
    from hermes_cli.model_preflight import collect_model_preflight

    preflight = collect_model_preflight()
    home = Path(preflight.profile_home).expanduser().resolve(strict=False)
    profile = preflight.profile_id or home.name
    interpreter = preflight.gateway_interpreter
    source_root = preflight.gateway_source_root

    if preflight.status == "PASS" and interpreter and source_root:
        probe_env = os.environ.copy()
        probe_env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            probe = subprocess.run(
                [
                    interpreter,
                    "-c",
                    (
                        "import pathlib, hermes_cli; "
                        "print(pathlib.Path(hermes_cli.__file__).resolve().parents[1])"
                    ),
                ],
                cwd=source_root,
                env=probe_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            imported_source = str(Path(probe.stdout.strip()).resolve(strict=False))
            source_bound = probe.returncode == 0 and imported_source == str(
                Path(source_root).resolve(strict=False)
            )
        except (OSError, subprocess.SubprocessError):
            source_bound = False
        if not source_bound:
            preflight = replace(
                preflight,
                status="FAIL",
                blockers=preflight.blockers + ("gateway_interpreter_source_mismatch",),
            )

    @contextmanager
    def config_write_lock():
        lock_path = home / ".model-transaction.lock"
        handle = open(lock_path, "a+", encoding="utf-8")
        acquired = False
        deadline = time.monotonic() + config_lock_timeout
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write("\0")
                    handle.flush()
                while not acquired:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("config_lock_timeout")
                        time.sleep(0.05)
            else:
                import fcntl

                while not acquired:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                    except (BlockingIOError, OSError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("config_lock_timeout")
                        time.sleep(0.05)
            yield
        finally:
            try:
                if acquired and os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif acquired:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def write_config(
        expected: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        with config_write_lock():
            if read_raw_config() != expected:
                raise ConfigRevisionConflict("config_revision_conflict")
            save_config(
                copy.deepcopy(config),
                strip_defaults=False,
                preserve_keys={("model", "provider"), ("model", "default")},
            )

    def running_pid() -> int | None:
        runtime = read_runtime_status()
        return get_runtime_status_running_pid(runtime, expected_home=home)

    def restart_gateway(previous_pid: int | None) -> int:
        command = [
            interpreter,
            "-m",
            "hermes_cli.main",
            "--profile",
            profile,
            "gateway",
            "restart",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=source_root or None,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=restart_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("gateway_restart_timeout") from exc
        if completed.returncode != 0:
            raise RuntimeError("gateway_restart_failed")
        deadline = time.monotonic() + restart_timeout
        while time.monotonic() < deadline:
            pid = running_pid()
            if pid and pid != previous_pid:
                post = collect_model_preflight()
                post_profile = post.profile_id or Path(post.profile_home).name
                if (
                    post.status == "PASS"
                    and post.gateway_pid == pid
                    and post_profile == profile
                    and post.gateway_source_root == source_root
                    and post.gateway_interpreter == interpreter
                ):
                    return pid
            time.sleep(0.25)
        raise RuntimeError("gateway_restart_timeout")

    def smoke_model(
        expected: ModelPair,
        active_interpreter: str,
        confirmed_profile: str,
    ) -> ModelSmokeEvidence:
        fd, usage_path_raw = tempfile.mkstemp(prefix="hermes-model-transaction-", suffix=".json")
        os.close(fd)
        usage_path = Path(usage_path_raw)
        try:
            command = [
                active_interpreter,
                "-m",
                "hermes_cli.main",
                "--profile",
                confirmed_profile,
                "--oneshot",
                "Reply with exactly HERMES_MODEL_TRANSACTION_OK and nothing else.",
                "--usage-file",
                str(usage_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=source_root or None,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=smoke_timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("model_smoke_timeout") from exc
            if completed.returncode != 0:
                raise RuntimeError("model_smoke_failed")
            payload = json.loads(usage_path.read_text(encoding="utf-8"))
            return ModelSmokeEvidence(
                provider=str(payload.get("provider") or ""),
                model=str(payload.get("model") or ""),
                api_calls=int(payload.get("api_calls") or 0),
                completed=bool(payload.get("completed")) and not bool(payload.get("failed")),
            )
        finally:
            try:
                usage_path.unlink()
            except OSError:
                pass

    def restart_allowed() -> bool:
        if preflight.gateway_pid is None:
            return False
        try:
            import psutil

            if preflight.gateway_pid in {
                process.pid for process in psutil.Process().parents()
            }:
                return False
        except Exception:
            return False

        try:
            from hermes_cli.service_manager import (
                detect_service_manager,
                get_service_manager,
            )

            manager_kind = detect_service_manager()
            supervised_pid: int | None = None

            if manager_kind == "launchd":
                from hermes_cli.gateway import (
                    _parse_launchd_pid_from_list_output,
                    get_launchd_label,
                    get_launchd_plist_path,
                )

                if not get_launchd_plist_path().exists():
                    return False
                probe = subprocess.run(
                    ["launchctl", "list", get_launchd_label()],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if probe.returncode == 0:
                    supervised_pid = _parse_launchd_pid_from_list_output(
                        probe.stdout
                    )

            elif manager_kind == "systemd":
                from hermes_cli.gateway import (
                    _probe_systemd_service_running,
                    _systemd_main_pid,
                    get_systemd_unit_path,
                )

                selected_system, service_running = _probe_systemd_service_running()
                if (
                    service_running
                    and get_systemd_unit_path(system=selected_system).exists()
                ):
                    supervised_pid = _systemd_main_pid(system=selected_system)

            elif manager_kind == "s6":
                manager = get_service_manager()
                service_name = f"gateway-{profile}"
                supervised_pid_reader = getattr(manager, "_supervised_pid", None)
                if callable(supervised_pid_reader):
                    candidate_pid = supervised_pid_reader(service_name)
                    if isinstance(candidate_pid, int) and candidate_pid > 0:
                        supervised_pid = candidate_pid

            # Windows scheduled tasks do not currently expose the child gateway
            # PID. Fail closed until an exact task-to-process binding is available.
            return supervised_pid == preflight.gateway_pid
        except Exception:
            return False

    return TransactionHooks(
        collect_preflight=lambda: preflight,
        read_config=read_raw_config,
        write_config=write_config,
        restart_gateway=restart_gateway,
        smoke_model=smoke_model,
        restart_allowed=restart_allowed,
    )


def format_model_transaction(
    result: ModelTransactionResult,
    *,
    as_json: bool = False,
) -> str:
    payload = asdict(result)
    if as_json:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    blockers = ",".join(result.blockers) if result.blockers else "none"
    return "\n".join(
        (
            f"MODEL_TRANSACTION_RESULT: {result.status}",
            f"profile: {result.profile}",
            f"previous: {result.previous.provider}/{result.previous.model}",
            f"target: {result.target.provider}/{result.target.model}",
            f"previous_pid: {result.previous_pid or 'none'}",
            f"new_pid: {result.new_pid or 'none'}",
            f"rollback_verified: {result.rollback_verified}",
            f"blockers: {blockers}",
        )
    )
