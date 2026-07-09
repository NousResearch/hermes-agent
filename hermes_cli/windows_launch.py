"""Windows-safe Hermes launch helpers.

The helpers in this module intentionally build argv arrays instead of shell
strings.  This keeps query-file paths with spaces atomic through PowerShell,
``Start-Process``, and Win32 ``CreateProcess`` boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Iterable


_PARSER_OUTPUT_RE = re.compile(
    r"(?im)(^usage:\s+hermes\b|\bunrecognized arguments\b|\berror:\s+argument\b)"
)


@dataclass(frozen=True)
class WindowsQueryLaunch:
    """A direct-argv Hermes launch description."""

    executable: str
    args: list[str]
    cwd: str | None = None
    shell: bool = False

    @property
    def popen_argv(self) -> list[str]:
        return [self.executable, *self.args]


@dataclass(frozen=True)
class LaunchSmokeResult:
    """Value-free classification of early process launch output."""

    status: str
    reason: str
    parser_output_detected: bool


def _as_atomic_arg(value: str | Path, *, name: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{name} is required")
    if "\n" in text or "\r" in text:
        raise ValueError(f"{name} must be a single argv atom, not multiline text")
    return text


def build_query_file_argv(
    *,
    session_id: str,
    prompt_path: str | Path,
    model: str = "gpt-5.5",
    module: str = "hermes_cli.main",
) -> list[str]:
    """Build ``python -m hermes_cli.main chat --resume ... --query-file ...`` args.

    The returned list is meant to be passed as an argv array.  Do not join it
    into a shell string; preserving ``prompt_path`` as a single list element is
    what keeps paths such as ``Hermes monitoring\\runs\\prompt.txt`` intact.
    """

    safe_session_id = _as_atomic_arg(session_id, name="session_id")
    safe_prompt_path = _as_atomic_arg(prompt_path, name="prompt_path")
    safe_model = _as_atomic_arg(model, name="model")
    safe_module = _as_atomic_arg(module, name="module")
    return [
        "-m",
        safe_module,
        "chat",
        "--resume",
        safe_session_id,
        "--model",
        safe_model,
        "--query-file",
        safe_prompt_path,
    ]


def build_query_file_launch(
    *,
    python_exe: str | Path,
    session_id: str,
    prompt_path: str | Path,
    model: str = "gpt-5.5",
    cwd: str | Path | None = None,
) -> WindowsQueryLaunch:
    """Return a shell-free launch object for a pre-existing prompt file."""

    return WindowsQueryLaunch(
        executable=_as_atomic_arg(python_exe, name="python_exe"),
        args=build_query_file_argv(
            session_id=session_id,
            prompt_path=prompt_path,
            model=model,
        ),
        cwd=str(cwd) if cwd is not None else None,
        shell=False,
    )


def build_prompt_file_launch(
    *,
    python_exe: str | Path,
    session_id: str,
    prompt_text: str,
    prompt_path: str | Path,
    model: str = "gpt-5.5",
    cwd: str | Path | None = None,
) -> WindowsQueryLaunch:
    """Write a prompt file and return a direct-argv launch for it.

    Multiline prompts must travel through the file, not through a shell-joined
    ``--query`` string.  The resulting argv contains only the prompt-file path.
    """

    path = Path(prompt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt_text, encoding="utf-8")
    return build_query_file_launch(
        python_exe=python_exe,
        session_id=session_id,
        prompt_path=path,
        model=model,
        cwd=cwd,
    )


def _has_parser_output(chunks: Iterable[str | None]) -> bool:
    text = "\n".join(chunk or "" for chunk in chunks)
    return bool(_PARSER_OUTPUT_RE.search(text))


def classify_launch_smoke(
    *,
    stdout: str = "",
    stderr: str = "",
    db_message_created: bool,
    returncode: int | None = None,
) -> LaunchSmokeResult:
    """Classify early launch output before blaming Hermes runtime/tool logic.

    If argparse usage/errors appear and no session DB message was created, the
    process failed at launch/parser time.  Once a DB message exists, parser-like
    words in later output may be user/model text and must not override progress.
    """

    parser_output = _has_parser_output((stdout, stderr))
    if db_message_created:
        return LaunchSmokeResult(
            status="ok",
            reason="db_message_created",
            parser_output_detected=parser_output,
        )
    if parser_output:
        return LaunchSmokeResult(
            status="launch_failure",
            reason="parser_output_without_db_message",
            parser_output_detected=True,
        )
    if returncode not in (None, 0):
        return LaunchSmokeResult(
            status="launch_failure",
            reason="process_exited_without_db_message",
            parser_output_detected=False,
        )
    return LaunchSmokeResult(
        status="ok",
        reason="no_parser_output",
        parser_output_detected=False,
    )


def classify_launch_smoke_files(
    *,
    stdout_path: str | Path,
    stderr_path: str | Path,
    db_message_created: bool,
    returncode: int | None = None,
) -> LaunchSmokeResult:
    """Read launch logs and apply :func:`classify_launch_smoke`."""

    def _read(path: str | Path) -> str:
        try:
            return Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    return classify_launch_smoke(
        stdout=_read(stdout_path),
        stderr=_read(stderr_path),
        db_message_created=db_message_created,
        returncode=returncode,
    )


def launch_query_file_background(
    launch: WindowsQueryLaunch,
    *,
    stdout_path: str | Path,
    stderr_path: str | Path,
) -> subprocess.Popen:
    """Start a query-file launch in the background without shell joining."""

    from hermes_cli._subprocess_compat import windows_hide_flags

    stdout = Path(stdout_path)
    stderr = Path(stderr_path)
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stderr.parent.mkdir(parents=True, exist_ok=True)
    stdout_fh = stdout.open("w", encoding="utf-8", errors="replace")
    stderr_fh = stderr.open("w", encoding="utf-8", errors="replace")
    try:
        return subprocess.Popen(
            launch.popen_argv,
            cwd=launch.cwd,
            stdin=subprocess.DEVNULL,
            stdout=stdout_fh,
            stderr=stderr_fh,
            shell=False,
            creationflags=windows_hide_flags(),
        )
    finally:
        stdout_fh.close()
        stderr_fh.close()
