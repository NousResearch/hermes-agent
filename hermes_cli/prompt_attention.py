"""Cross-process attention arbitration for blocking CLI prompts.

The coordinator deliberately stores only routing metadata. Prompt text, commands,
secret names and answers never leave the owning process.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import struct
import subprocess
import threading
import time
import uuid
import wave
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_default_hermes_root
from utils import atomic_json_write

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptAttentionRequest:
    request_id: str
    pid: int
    process_start_time: float | None
    session_id: str
    tmux_session: str = ""
    tmux_pane: str = ""
    tmux_client_tty: str = ""
    sway_container_id: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "PromptAttentionRequest":
        return cls(
            request_id=str(raw["request_id"]),
            pid=int(raw["pid"]),
            process_start_time=(
                None
                if raw.get("process_start_time") is None
                else float(raw["process_start_time"])
            ),
            session_id=str(raw.get("session_id") or ""),
            tmux_session=str(raw.get("tmux_session") or ""),
            tmux_pane=str(raw.get("tmux_pane") or ""),
            tmux_client_tty=str(raw.get("tmux_client_tty") or ""),
            sway_container_id=(
                None
                if raw.get("sway_container_id") is None
                else int(raw["sway_container_id"])
            ),
        )


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError("prompt attention file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        fh, self._fh = self._fh, None
        if fh is None:
            return
        with suppress(Exception):
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def _process_start_time(pid: int) -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _process_alive(pid: int, expected_start: float | None) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    if expected_start is None:
        return True
    current = _process_start_time(pid)
    return current is not None and abs(current - expected_start) < 0.01


class PromptAttentionLease:
    def __init__(
        self, arbiter: "PromptAttentionArbiter", request: PromptAttentionRequest
    ):
        self._arbiter = arbiter
        self.request = request
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._arbiter.release(self.request.request_id)
            self._released = True

    def __enter__(self) -> "PromptAttentionLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class PromptAttentionArbiter:
    """FIFO lease shared by every Hermes profile under one default root."""

    def __init__(
        self,
        runtime_dir: str | Path | None = None,
        *,
        process_alive: Callable[[int, float | None], bool] = _process_alive,
        poll_interval: float = 0.1,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.runtime_dir = Path(runtime_dir or (get_default_hermes_root() / "runtime"))
        self.state_path = self.runtime_dir / "prompt_attention.json"
        self.lock_path = self.runtime_dir / "prompt_attention.lock"
        self.process_alive = process_alive
        self.poll_interval = poll_interval
        self.monotonic = monotonic

    def _read_state(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"next_sequence": 1, "owner": None, "requests": []}
        except Exception as exc:
            raise RuntimeError(
                f"prompt attention coordination state unreadable: {self.state_path}"
            ) from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("requests"), list):
            raise RuntimeError(
                f"prompt attention coordination state invalid: {self.state_path}"
            )
        return raw

    def _write_state(self, state: dict[str, Any]) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_write(self.state_path, state, indent=2, sort_keys=True)
        with suppress(OSError):
            os.chmod(self.state_path, 0o600)

    def _prune_dead(self, state: dict[str, Any]) -> bool:
        before = list(state["requests"])
        live = []
        for entry in before:
            try:
                request = PromptAttentionRequest.from_json(entry)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"prompt attention coordination state invalid: {self.state_path}"
                ) from exc
            if self.process_alive(request.pid, request.process_start_time):
                live.append(entry)
        state["requests"] = live
        ids = {str(entry.get("request_id")) for entry in live}
        if state.get("owner") not in ids:
            state["owner"] = None
        return live != before

    def acquire(
        self, request: PromptAttentionRequest, *, timeout: float | None = None
    ) -> PromptAttentionLease:
        deadline = None if timeout is None else self.monotonic() + max(0.0, timeout)
        enqueued = False
        while True:
            with _FileLock(self.lock_path):
                state = self._read_state()
                changed = self._prune_dead(state)
                existing = next(
                    (
                        e
                        for e in state["requests"]
                        if e.get("request_id") == request.request_id
                    ),
                    None,
                )
                if existing is None:
                    sequence = int(state.get("next_sequence") or 1)
                    state["next_sequence"] = sequence + 1
                    state["requests"].append({
                        **request.to_json(),
                        "sequence": sequence,
                    })
                    enqueued = True
                    changed = True
                if state.get("owner") is None and state["requests"]:
                    state["requests"].sort(
                        key=lambda entry: int(entry.get("sequence") or 0)
                    )
                    state["owner"] = state["requests"][0]["request_id"]
                    changed = True
                owner = state.get("owner")
                if changed:
                    self._write_state(state)
                if owner == request.request_id:
                    return PromptAttentionLease(self, request)
            if deadline is not None and self.monotonic() >= deadline:
                if enqueued:
                    self.release(request.request_id)
                raise TimeoutError("timed out waiting for prompt attention")
            time.sleep(self.poll_interval)

    def release(self, request_id: str) -> None:
        with _FileLock(self.lock_path):
            state = self._read_state()
            self._prune_dead(state)
            state["requests"] = [
                entry
                for entry in state["requests"]
                if entry.get("request_id") != request_id
            ]
            if state.get("owner") == request_id:
                state["owner"] = None
            if state.get("owner") is None and state["requests"]:
                state["requests"].sort(
                    key=lambda entry: int(entry.get("sequence") or 0)
                )
                state["owner"] = state["requests"][0]["request_id"]
            self._write_state(state)


def _ancestor_pids(pid: int) -> set[int]:
    ancestors: set[int] = set()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                fields = fh.read().rsplit(b")", 1)[1].split()
            pid = int(fields[1])
        except (OSError, ValueError, IndexError):
            break
    return ancestors


def _sway_container_for_pid(client_pid: int, *, runner=subprocess.run) -> int | None:
    if not client_pid or not os.environ.get("SWAYSOCK"):
        return None
    try:
        result = runner(
            ["swaymsg", "-t", "get_tree", "-r"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        tree = json.loads(result.stdout)
    except Exception:
        return None
    ancestors = _ancestor_pids(client_pid)
    matches: set[int] = set()
    stack = [tree]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("pid") in ancestors and node.get("type") == "con":
            try:
                matches.add(int(node["id"]))
            except (KeyError, TypeError, ValueError):
                pass
        stack.extend(node.get("nodes") or ())
        stack.extend(node.get("floating_nodes") or ())
    return next(iter(matches)) if len(matches) == 1 else None


def capture_prompt_attention_request(
    session_id: str, *, runner=subprocess.run
) -> PromptAttentionRequest:
    pane = os.environ.get("TMUX_PANE", "")
    tmux_session = ""
    client_tty = ""
    client_pid = os.getpid() if not pane else 0
    if pane:
        try:
            result = runner(
                ["tmux", "display-message", "-p", "-t", pane, "#{session_name}"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            tmux_session = result.stdout.strip() if result.returncode == 0 else ""
            clients = runner(
                [
                    "tmux",
                    "list-clients",
                    "-F",
                    "#{client_tty}|#{client_pid}|#{session_name}",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            matches = []
            for line in clients.stdout.splitlines() if clients.returncode == 0 else ():
                tty, pid, session = (line.split("|", 2) + ["", "", ""])[:3]
                if session == tmux_session:
                    matches.append((tty, int(pid)))
            if len(matches) == 1:
                client_tty, client_pid = matches[0]
        except Exception:
            pass
    return PromptAttentionRequest(
        request_id=uuid.uuid4().hex,
        pid=os.getpid(),
        process_start_time=_process_start_time(os.getpid()),
        session_id=session_id,
        tmux_session=tmux_session,
        tmux_pane=pane,
        tmux_client_tty=client_tty,
        sway_container_id=_sway_container_for_pid(client_pid, runner=runner),
    )


def _write_thunder(path: Path) -> None:
    """Generate a short low-frequency thunder roll without shipping a binary asset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 22_050
    duration = 2.8
    rng = random.Random(0xA17E)
    low = 0.0
    frames = bytearray()
    for index in range(int(sample_rate * duration)):
        t = index / sample_rate
        noise = rng.uniform(-1.0, 1.0)
        low = 0.985 * low + 0.015 * noise
        crack = math.sin(2 * math.pi * 58 * t) * math.exp(-7 * t)
        rumble = low * (0.75 * math.exp(-0.65 * t))
        secondary = (
            math.sin(2 * math.pi * 43 * t) * math.exp(-3.2 * max(0.0, t - 0.7))
            if t > 0.7
            else 0.0
        )
        sample = max(-1.0, min(1.0, 2.4 * rumble + 0.55 * crack + 0.22 * secondary))
        frames.extend(struct.pack("<h", int(sample * 30_000)))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


class PromptAttentionController:
    def __init__(
        self,
        *,
        runtime_dir: str | Path | None = None,
        runner=subprocess.run,
        popen=subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.runtime_dir = Path(runtime_dir or (get_default_hermes_root() / "runtime"))
        self.runner = runner
        self.popen = popen
        self.monotonic = monotonic

    def _run(self, argv: list[str], *, timeout: float = 2.0) -> None:
        try:
            self.runner(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception as exc:
            logger.debug("prompt attention command failed (%s): %s", argv[0], exc)

    @staticmethod
    def _configured_command(
        config: dict[str, Any], key: str, request: PromptAttentionRequest | None = None
    ) -> list[str]:
        raw = config.get(key)
        if (
            not isinstance(raw, (list, tuple))
            or not raw
            or not all(isinstance(arg, str) for arg in raw)
        ):
            return []
        if request is None:
            return list(raw)
        values = {
            "session_id": request.session_id,
            "tmux_session": request.tmux_session,
            "tmux_pane": request.tmux_pane,
            "tmux_client_tty": request.tmux_client_tty,
            "sway_container_id": ""
            if request.sway_container_id is None
            else str(request.sway_container_id),
        }
        try:
            return [arg.format_map(values) for arg in raw]
        except (KeyError, ValueError):
            return []

    @staticmethod
    def _command_timeout(config: dict[str, Any]) -> float:
        try:
            return max(0.1, min(30.0, float(config.get("command_timeout", 2.0))))
        except (TypeError, ValueError):
            return 2.0

    def _focus(self, request: PromptAttentionRequest, config: dict[str, Any]) -> None:
        timeout = self._command_timeout(config)
        custom = self._configured_command(config, "focus_command", request)
        if custom:
            self._run(custom, timeout=timeout)
            return
        if request.tmux_client_tty and request.tmux_session:
            # tmux explicitly permits a pane target here and switches only the named client
            # to that pane's session, window and active pane in one operation.
            target = request.tmux_pane or request.tmux_session
            self._run(
                [
                    "tmux",
                    "switch-client",
                    "-c",
                    request.tmux_client_tty,
                    "-t",
                    target,
                ],
                timeout=timeout,
            )
        if request.sway_container_id is not None:
            self._run(
                ["swaymsg", f"[con_id={request.sway_container_id}]", "focus"],
                timeout=timeout,
            )

    def _sound(self, config: dict[str, Any]) -> None:
        argv = self._configured_command(config, "sound_command")
        path = self.runtime_dir / "prompt-attention-thunder.wav"
        if not argv:
            if not path.exists():
                _write_thunder(path)
            argv = ["pw-play", str(path)]
        try:
            process = self.popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                shell=False,
            )
            if hasattr(process, "wait"):
                timeout = self._command_timeout(config)

                def _reap() -> None:
                    try:
                        process.wait(timeout=timeout)
                        return
                    except subprocess.TimeoutExpired:
                        with suppress(Exception):
                            process.terminate()
                    try:
                        process.wait(timeout=min(1.0, timeout))
                        return
                    except subprocess.TimeoutExpired:
                        with suppress(Exception):
                            process.kill()
                    with suppress(Exception):
                        process.wait(timeout=min(1.0, timeout))

                threading.Thread(
                    target=_reap, daemon=True, name="prompt-attention-sound"
                ).start()
        except Exception as exc:
            logger.debug("prompt attention sound failed: %s", exc)

    def activate(
        self,
        request: PromptAttentionRequest,
        config: dict[str, Any],
        *,
        voice_enabled_at: float | None = None,
        voice_callback: Callable[[], None] | None = None,
        voice_allowed: bool = True,
    ) -> None:
        if config.get("focus", True):
            self._focus(request, config)
        if config.get("sound", True):
            self._sound(config)
        try:
            recent_seconds = min(
                300.0, max(0.0, float(config.get("voice_recent_seconds", 300)))
            )
        except (TypeError, ValueError):
            recent_seconds = 300.0
        if (
            voice_allowed
            and voice_callback is not None
            and voice_enabled_at is not None
            and recent_seconds > 0
            and 0 <= self.monotonic() - voice_enabled_at < recent_seconds
        ):
            voice_callback()
