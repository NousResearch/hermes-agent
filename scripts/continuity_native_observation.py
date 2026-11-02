#!/usr/bin/env python3
"""Run exact-host continuity hook observations without persisting payload data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from continuity_common import (
    ContinuityError,
    _spawn_contained_process,
    confined_file_checkpoint,
    confined_read_range,
    load_json,
    minimal_child_env,
    run_command,
    sha256_file,
    validate_state_storage,
)
from install_continuity_adapters import (
    HERMES_MANIFEST,
    _managed_hermes_hooks,
    render_project_adapters,
    validate_hermes_manifest,
)


def _host_executable(config: dict, surface: str) -> Path:
    policy = config.get("native_observation_policy")
    hosts = policy.get("hosts") if isinstance(policy, dict) else None
    identity = hosts.get(surface) if isinstance(hosts, dict) else None
    if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
        raise ContinuityError(f"{surface} native host identity is missing")
    executable = Path(str(identity["path"])).resolve()
    if not executable.is_file() or sha256_file(executable) != identity.get("sha256"):
        raise ContinuityError(f"{surface} native host identity is stale")
    return executable


def _require_codex_discovery_checkout(repo_root: Path, config: dict) -> None:
    result = run_command(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=repo_root,
        timeout=30,
        env=minimal_child_env(
            state_root=config["state_root"],
            external_volume=config["external_volume"],
        ),
        required_mount=(config["external_volume"], config["state_root"]),
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ContinuityError("cannot resolve Codex hook discovery checkout")
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = repo_root / common
    discovery_root = common.resolve().parent
    if discovery_root != repo_root.resolve():
        raise ContinuityError(
            "Codex native observation requires the canonical common checkout; "
            "linked worktree hook provenance is not certifiable"
        )


MAX_EVENT_EVIDENCE_BYTES = 1_048_576


def _event_log_checkpoint(
    config: dict,
) -> tuple[Path, int, tuple[int, int] | None, dict]:
    path = Path(config["event_log"])
    try:
        size, identity = confined_file_checkpoint(config, path)
    except FileNotFoundError:
        return path, 0, None, config
    return path, size, identity, config


def _session_ids(output: str) -> set[str]:
    identities: set[str] = set()
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        for key in ("session_id", "thread_id"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                identities.add(candidate)
    return identities


def _require_native_event(
    checkpoint: tuple[Path, int, tuple[int, int] | None, dict],
    surface: str,
    output: str,
) -> None:
    path, offset, prior_identity = checkpoint[:3]
    config = checkpoint[3]
    try:
        appended_bytes, _size, identity = confined_read_range(
            config,
            path,
            offset=offset,
            max_bytes=MAX_EVENT_EVIDENCE_BYTES,
        )
    except FileNotFoundError:
        raise ContinuityError(f"{surface} host produced no continuity event log")
    if prior_identity is not None and identity != prior_identity:
        raise ContinuityError("continuity event log rotated during native observation")
    identities = _session_ids(output)
    fingerprints = {
        hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        for identity in identities
    }
    if not fingerprints:
        raise ContinuityError(f"{surface} host exposed no session identity")
    try:
        appended = appended_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContinuityError("native observation event evidence is not UTF-8") from exc
    values = []
    for line in appended.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContinuityError(
                "native observation found malformed event evidence"
            ) from exc
        if isinstance(value, dict):
            values.append(value)
    matched = any(
        value.get("surface") == surface
        and value.get("event") == "session_start"
        and value.get("session") in fingerprints
        and value.get("completion_allowed") is True
        for value in values
    )
    if not matched:
        raise ContinuityError(
            f"{surface} host did not invoke an allowed project SessionStart hook"
        )


def _run_host_until_native_event(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    checkpoint: tuple[Path, int, tuple[int, int] | None, dict],
    surface: str,
    output_dir: Path,
    timeout: float,
    required_mount: tuple[str | Path, str | Path] | None = None,
) -> str:
    if required_mount is not None:
        validate_state_storage(*required_mount)
    stdout_path = output_dir / "host.stdout"
    stderr_path = output_dir / "host.stderr"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        handled_signals = (signal.SIGTERM, signal.SIGINT)
        prior_handlers = {sig: signal.getsignal(sig) for sig in handled_signals}
        lifecycle: dict[str, object] = {
            "contained": None,
            "pending_signal": None,
            "signal_consumed": False,
        }

        def request_nested_host_termination(signum: int, _frame: object) -> None:
            if lifecycle["pending_signal"] is None:
                lifecycle["pending_signal"] = signum

        def terminate_for_pending_signal(contained: Any | None) -> None:
            pending_signal = lifecycle["pending_signal"]
            if not isinstance(pending_signal, int) or lifecycle["signal_consumed"]:
                return
            lifecycle["signal_consumed"] = True
            if contained is not None:
                contained.terminate_tree()
            raise SystemExit(128 + pending_signal)

        prior_mask: set[signal.Signals] | None = None
        if os.name == "posix":
            prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, handled_signals)
        for sig in handled_signals:
            signal.signal(sig, request_nested_host_termination)
        mask_is_blocked = prior_mask is not None
        try:
            contained = _spawn_contained_process(
                command,
                popen_factory=subprocess.Popen,
                require_native_containment=True,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            lifecycle["contained"] = contained
            process = contained.process
            if prior_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
                mask_is_blocked = False
            terminate_for_pending_signal(contained)
            deadline = time.monotonic() + timeout
            last_error: ContinuityError | None = None
            while time.monotonic() < deadline:
                terminate_for_pending_signal(contained)
                if required_mount is not None:
                    try:
                        validate_state_storage(*required_mount)
                    except ContinuityError as exc:
                        contained.terminate_tree()
                        raise ContinuityError(
                            "required external storage became unavailable during "
                            f"{surface} native observation"
                        ) from exc
                stdout_handle.flush()
                stderr_handle.flush()
                output = stdout_path.read_text(
                    encoding="utf-8"
                ) + stderr_path.read_text(encoding="utf-8")
                try:
                    _require_native_event(checkpoint, surface, output)
                except ContinuityError as exc:
                    last_error = exc
                else:
                    contained.terminate_tree()
                    terminate_for_pending_signal(contained)
                    return output
                process_returncode = process.poll()
                terminate_for_pending_signal(contained)
                if process_returncode is not None:
                    break
                time.sleep(0.1)
            contained.terminate_tree()
            terminate_for_pending_signal(contained)
            stdout_handle.flush()
            stderr_handle.flush()
        finally:
            if mask_is_blocked and prior_mask is not None:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
            for sig, handler in prior_handlers.items():
                signal.signal(sig, handler)
            # A signal delivered while handlers were being restored was still
            # owned by our passive handler. Consume that recorded intent after
            # the complete handoff so it cannot be silently returned past.
            terminate_for_pending_signal(lifecycle["contained"])
    output = stdout_path.read_text(encoding="utf-8") + stderr_path.read_text(
        encoding="utf-8"
    )
    try:
        _require_native_event(checkpoint, surface, output)
    except ContinuityError as exc:
        detail = str(last_error or exc)
        raise ContinuityError(
            f"{surface} native host exited or timed out before evidence: {detail}"
        ) from exc
    return output


def _project_host_command(executable: Path, surface: str) -> list[str]:
    prompt = "Return only NATIVE_OBSERVATION_OK. Do not use tools."
    if surface == "claude":
        return [
            str(executable),
            "-p",
            "--verbose",
            "--setting-sources",
            "project",
            "--strict-mcp-config",
            "--tools",
            "",
            "--permission-mode",
            "dontAsk",
            "--model",
            "haiku",
            "--max-budget-usd",
            "0.01",
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--include-hook-events",
            prompt,
        ]
    return [
        str(executable),
        "exec",
        "--ephemeral",
        "--dangerously-bypass-hook-trust",
        "-s",
        "read-only",
        "-c",
        'approval_policy="never"',
        "--json",
        prompt,
    ]


def observe_project_hook(repo_root: Path, config_path: Path, surface: str) -> None:
    config = load_json(config_path)
    rendered = render_project_adapters(repo_root)
    suffix = ".claude/settings.json" if surface == "claude" else ".codex/hooks.json"
    target = repo_root / suffix
    if target.read_text(encoding="utf-8") != rendered[target]:
        raise ContinuityError(f"{surface} project hook adapter is stale")
    if surface == "codex":
        _require_codex_discovery_checkout(repo_root, config)
        codex_config = repo_root / ".codex/config.toml"
        if codex_config.read_text(encoding="utf-8") != rendered[codex_config]:
            raise ContinuityError("codex project config layer is stale")
    executable = _host_executable(config, surface)
    checkpoint = _event_log_checkpoint(config)
    confined_env = minimal_child_env(
        state_root=config["state_root"], external_volume=config["external_volume"]
    )
    state_root = Path(confined_env["HOME"])
    temp_root = Path(confined_env["TMPDIR"])
    with tempfile.TemporaryDirectory(
        prefix=f"continuity-{surface}-host-", dir=temp_root
    ) as temp:
        extra_env = {
            "ANTHROPIC_API_KEY": "native-observation-not-a-credential",
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:1",
            "OPENAI_API_KEY": "native-observation-not-a-credential",
            "OPENAI_BASE_URL": "http://127.0.0.1:1",
            "CONTINUITY_OBSERVATION_CONFIG": str(config_path),
            "HOME": temp,
            "TMPDIR": temp,
            "TEMP": temp,
            "TMP": temp,
        }
        if surface == "codex":
            codex_home = Path(temp) / ".codex"
            codex_home.mkdir()
            (codex_home / "config.toml").write_text(
                "[features]\n"
                "hooks = true\n"
                f"[projects.{json.dumps(str(repo_root))}]\n"
                'trust_level = "trusted"\n',
                encoding="utf-8",
            )
            extra_env["CODEX_HOME"] = str(codex_home)
        _run_host_until_native_event(
            _project_host_command(executable, surface),
            cwd=repo_root,
            env=minimal_child_env(
                extra_env,
                state_root=state_root,
                external_volume=config["external_volume"],
            ),
            checkpoint=checkpoint,
            surface=surface,
            output_dir=Path(temp),
            timeout=120,
            required_mount=(config["external_volume"], config["state_root"]),
        )
    if (
        sha256_file(executable)
        != (config["native_observation_policy"]["hosts"][surface]["sha256"])
    ):
        raise ContinuityError(f"{surface} native host changed during observation")
    if target.read_text(encoding="utf-8") != rendered[target]:
        raise ContinuityError(f"{surface} project hook changed during observation")
    if (
        surface == "codex"
        and (repo_root / ".codex/config.toml").read_text(encoding="utf-8")
        != rendered[repo_root / ".codex/config.toml"]
    ):
        raise ContinuityError("codex project config changed during observation")


def observe_hermes_hook(repo_root: Path, config_path: Path, hermes_home: Path) -> None:
    target = hermes_home / "config.yaml"
    config = load_json(config_path)
    hermes_runtime = next(
        (
            item
            for item in (config.get("evidence_policy") or {}).get("runtime_checks", [])
            if isinstance(item, dict) and item.get("surface") == "hermes"
        ),
        None,
    )
    if (
        not isinstance(hermes_runtime, dict)
        or Path(str(hermes_runtime.get("adapter_path"))).resolve() != target
    ):
        raise ContinuityError("Hermes runtime policy does not bind installed config")
    manifest = load_json(hermes_home / HERMES_MANIFEST)
    validate_hermes_manifest(repo_root, hermes_home, manifest)
    manifest_digest = sha256_file(hermes_home / HERMES_MANIFEST)
    target_digest = sha256_file(target)
    installed_hooks = manifest.get("installed_hooks")
    expected_hooks = _managed_hermes_hooks(repo_root)
    if (
        manifest.get("status") != "INSTALLED"
        or manifest.get("repo_root") != str(repo_root)
        or manifest.get("target") != str(target)
        or not isinstance(installed_hooks, dict)
        or installed_hooks != expected_hooks
    ):
        raise ContinuityError("Hermes adapter ownership manifest is stale")
    loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    hooks = loaded.get("hooks") if isinstance(loaded, dict) else None
    if (
        not isinstance(hooks, dict)
        or {name: hooks.get(name) for name in installed_hooks} != installed_hooks
    ):
        raise ContinuityError("installed Hermes hook config drifted from its manifest")
    executable = _host_executable(config, "hermes")
    env = minimal_child_env(
        {"HERMES_HOME": str(hermes_home)},
        state_root=config["state_root"],
        external_volume=config["external_volume"],
    )
    commands = (
        ("list", ["hooks", "list"]),
        ("doctor", ["hooks", "doctor"]),
        ("test", ["hooks", "test", "pre_llm_call"]),
    )
    for action, arguments in commands:
        result = run_command(
            [str(executable), str(repo_root / "main.py"), *arguments],
            cwd=repo_root,
            timeout=120,
            env=env,
            required_mount=(config["external_volume"], config["state_root"]),
            require_native_containment=True,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            raise ContinuityError(f"hermes hooks {action} exited nonzero")
        if action == "list" and "Configured shell hooks" not in output:
            raise ContinuityError("hermes hooks list found no configured shell hooks")
        if action == "doctor" and "All shell hooks look healthy." not in output:
            raise ContinuityError("hermes hooks doctor did not report healthy hooks")
        if action == "test" and (
            "Firing" not in output or '"completion_allowed": true' not in output
        ):
            raise ContinuityError(
                "hermes hooks test did not produce an allowed fresh-session admission"
            )
    if (
        sha256_file(executable)
        != (config["native_observation_policy"]["hosts"]["hermes"]["sha256"])
    ):
        raise ContinuityError("hermes native host changed during observation")
    if sha256_file(target) != target_digest:
        raise ContinuityError("installed Hermes config changed during observation")
    if sha256_file(hermes_home / HERMES_MANIFEST) != manifest_digest:
        raise ContinuityError(
            "Hermes adapter ownership manifest changed during observation"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface", choices=("claude", "codex", "hermes"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_json(config_path)
    repo_root = Path(config["expected_repo_root"]).resolve()
    try:
        if args.surface == "hermes":
            if args.hermes_home is None:
                raise ContinuityError(
                    "--hermes-home is required for Hermes observation"
                )
            observe_hermes_hook(repo_root, config_path, args.hermes_home.resolve())
            checks = ["hooks-list", "hooks-doctor", "fresh-session-admission"]
        else:
            observe_project_hook(repo_root, config_path, args.surface)
            checks = ["generated-adapter", "native-host-session-start"]
        print(json.dumps({"surface": args.surface, "checks": checks}, sort_keys=True))
        return 0
    except (ContinuityError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"native observation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
