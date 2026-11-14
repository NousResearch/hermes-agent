"""Killable auto-TTS worker. Provider calls must never run in gateway executor threads."""

import asyncio
import contextlib
import json
import os
from pathlib import Path
import signal
import sys

from hermes_platform.host.facts import os_family
from hermes_cli._startup_fast import project_root_str


async def _reap(proc):
    # Command providers may start their own session, so kill descendants as well
    # as our process group. Stop the worker first to prevent further spawns.
    import psutil

    children = []
    if proc.returncode is None:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            parent = psutil.Process(proc.pid)
            parent.suspend()
            children = parent.children(recursive=True)
            for child in reversed(children):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    child.kill()
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
    if os_family() != "win32":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok -- POSIX-only own session
    await proc.wait()
    if children:
        await asyncio.to_thread(psutil.wait_procs, children, timeout=1)


def _read_result(directory):
    result_path = directory / "result.json"
    if result_path.stat().st_size > 1024 * 1024:
        raise ValueError("Oversized auto-TTS result")
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    paths = result.get("file_paths") or [result.get("file_path")]
    if not result.get("success", False) or any(not path or not Path(path).is_file() for path in paths):
        raise RuntimeError("Incomplete auto-TTS output")
    return result


async def synthesize(text: str, output_path: str) -> dict:
    from hermes_constants import get_hermes_home
    from tools.environments.local import served_profile_child_env

    directory = Path(output_path).parent
    env = served_profile_child_env(target_home=get_hermes_home(), inherit_credentials=True)
    # Contain WAV sidecars, ffmpeg inputs and command-provider text files too.
    env.update(TMPDIR=str(directory), TEMP=str(directory), TMP=str(directory),
               HERMES_SESSION_PLATFORM="discord")
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        sys.executable, "-m", "gateway.platforms.base_auto_tts_worker",
        cwd=project_root_str(), env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
        # detached-spawn: ok -- awaited worker; cancellation/deadline reaps its group
        stderr=asyncio.subprocess.DEVNULL, start_new_session=(os_family() != "win32")))
    cancelled = False
    try:
        proc = await asyncio.shield(spawn)
        await proc.communicate(json.dumps({"text": text, "output_path": output_path}).encode())
        if proc.returncode:
            raise RuntimeError("Auto-TTS worker failed")
        return await asyncio.to_thread(_read_result, directory)
    except asyncio.CancelledError:
        cancelled = True
        raise
    finally:
        # Cancellation during spawn still owns the eventual child. A second
        # cancellation (runner + adapter shutdown) cannot interrupt reaping.
        async def cleanup():
            proc = await spawn
            await _reap(proc)

        cleanup_task = asyncio.create_task(cleanup())
        try:
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup_task.result()
        finally:
            # Spawn/reap failures must not turn cancellation into a late notice.
            if cancelled:
                raise asyncio.CancelledError


def main():
    from tools.tts_tool import check_tts_requirements, text_to_speech_tool

    request = json.load(sys.stdin)
    path = Path(request["output_path"])
    if not check_tts_requirements():
        result = {"success": False}
    else:
        result = json.loads(text_to_speech_tool(text=request["text"], output_path=str(path)))
    (path.parent / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
