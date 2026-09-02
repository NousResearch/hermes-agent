"""``hermes realtime`` — a terminal speech-to-speech session with a registered provider.

The smallest surface that exercises the realtime voice stack end to end
with zero user plugins: microphone → provider → speaker, transcripts on
stdout, and tool calls executed through Hermes' own registry (with its
approval prompts) on a daemon thread.

Layout:

* :class:`DuplexAudio` — PortAudio capture + playback at the rates the
  session negotiated. ``sounddevice`` is imported lazily, the same way
  :mod:`tools.voice_mode` does, so this module imports cleanly on a
  headless box.
* :class:`TerminalVoiceHost` — the :class:`RealtimeVoiceHost` for a
  terminal: prints transcripts, feeds playback, and answers barge-ins with
  the milliseconds actually played so the provider can truncate.
* :func:`build_tool_catalog` / :func:`make_tool_executor` — the bridge from
  Hermes' OpenAI-format tool schemas and ``handle_function_call`` to the
  provider-neutral contract.
* :func:`run_session` / :func:`cmd_realtime` — the async session runner and
  the argparse entry point.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import sys
import threading
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from agent.realtime_voice_orchestrator import (
    RealtimeVoiceError,
    RealtimeVoiceHost,
    RealtimeVoiceOrchestrator,
    run_blocking_tool,
)
from agent.realtime_voice_provider import (
    RealtimeAudioFormat,
    RealtimeTool,
    RealtimeVoiceSetup,
    ToolCall,
)

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "openai"
DEFAULT_TOOLSET = "hermes-cli"
DEFAULT_TOOL_TIMEOUT_S = 60.0
IDLE_POLL_S = 0.01
BLOCK_SECONDS = 0.1
#: Bounded so a stalled reader cannot grow the process without limit. Input
#: is the tighter of the two: stale microphone audio is worse than dropped.
MAX_INPUT_BLOCKS = 50
MAX_PLAYBACK_BLOCKS = 200

DEFAULT_INSTRUCTIONS = (
    "You are Hermes, a helpful voice assistant talking with the user out loud. "
    "Keep replies short and conversational: one or two sentences unless the "
    "user asks for detail. Use tools when they help, and say what you are "
    "about to do before running a slow tool."
)


class RealtimeAudioError(RuntimeError):
    """Audio devices are unusable."""


def _install_hint() -> str:
    from tools.voice_mode import _voice_capture_install_hint

    return _voice_capture_install_hint()


def import_sounddevice():
    """Lazy-import sounddevice with an actionable failure."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RealtimeAudioError(
            f"audio support is not installed — run: {_install_hint()} ({exc})"
        ) from exc
    return sd


# -- audio -------------------------------------------------------------------


class DuplexAudio:
    """Full-duplex pcm16 capture and playback over PortAudio.

    The PortAudio callbacks run on their own thread and touch nothing but
    plain queues — no asyncio, no locks held across a device call. The
    session polls :meth:`read_input_chunk` and pushes :meth:`queue_playback`.
    :attr:`played_ms` counts audio actually handed to the speaker, which is
    the number a barge-in truncation has to be measured in.
    """

    def __init__(self) -> None:
        self._input: queue.Queue[bytes] = queue.Queue(maxsize=MAX_INPUT_BLOCKS)
        self._playback: queue.Queue[bytes] = queue.Queue(maxsize=MAX_PLAYBACK_BLOCKS)
        self._residual = b""
        self._lock = threading.Lock()
        self._played_frames = 0
        self._output_rate = 24_000
        self._frame_bytes = 2
        self._in_stream = None
        self._out_stream = None

    def start(self, input_format: RealtimeAudioFormat, output_format: RealtimeAudioFormat) -> None:
        """Open both streams at the negotiated rates. Raises :class:`RealtimeAudioError`."""
        sd = import_sounddevice()
        self._output_rate = output_format.sample_rate_hz
        self._frame_bytes = 2 * output_format.channels
        try:
            self._in_stream = sd.RawInputStream(
                samplerate=input_format.sample_rate_hz,
                blocksize=int(input_format.sample_rate_hz * BLOCK_SECONDS),
                channels=input_format.channels,
                dtype="int16",
                callback=self._input_callback,
            )
            self._out_stream = sd.RawOutputStream(
                samplerate=output_format.sample_rate_hz,
                blocksize=int(output_format.sample_rate_hz * BLOCK_SECONDS),
                channels=output_format.channels,
                dtype="int16",
                callback=self._output_callback,
            )
            self._in_stream.start()
            self._out_stream.start()
        except Exception as exc:  # PortAudio raises its own exception types
            self.stop()
            raise RealtimeAudioError(f"could not open audio devices: {exc}") from exc

    def stop(self) -> None:
        """Close both streams. Safe to call twice, and after a failed start."""
        for attr in ("_in_stream", "_out_stream"):
            stream = getattr(self, attr, None)
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 — teardown is best-effort
                logger.debug("audio stream teardown failed", exc_info=True)
            setattr(self, attr, None)

    # PortAudio thread ------------------------------------------------------

    def _input_callback(self, indata, _frames, _time, _status) -> None:
        # Drop the block rather than stall the device: a blocked PortAudio
        # callback is a glitch the operator hears.
        with contextlib.suppress(queue.Full):
            self._input.put_nowait(bytes(indata))

    def _output_callback(self, outdata, frames, _time, _status) -> None:
        wanted = frames * self._frame_bytes
        chunk = self._take_playback(wanted)
        outdata[: len(chunk)] = chunk
        if len(chunk) < wanted:
            outdata[len(chunk):] = b"\x00" * (wanted - len(chunk))
        with self._lock:
            self._played_frames += len(chunk) // self._frame_bytes

    def _take_playback(self, wanted: int) -> bytes:
        buf = self._residual
        while len(buf) < wanted:
            try:
                buf += self._playback.get_nowait()
            except queue.Empty:
                break
        self._residual = buf[wanted:]
        return buf[:wanted]

    # session side ----------------------------------------------------------

    def read_input_chunk(self) -> bytes | None:
        """One captured block, or ``None`` when the microphone has nothing yet."""
        try:
            return self._input.get_nowait()
        except queue.Empty:
            return None

    def queue_playback(self, pcm: bytes) -> None:
        """Queue model audio for the speaker. Drops on overflow, never blocks."""
        if not pcm:
            return
        with contextlib.suppress(queue.Full):
            self._playback.put_nowait(pcm)

    def drain_playback(self) -> bool:
        """Barge-in: discard everything not yet played. ``True`` if anything was."""
        dropped = False
        while True:
            try:
                self._playback.get_nowait()
                dropped = True
            except queue.Empty:
                break
        if self._residual:
            dropped = True
        self._residual = b""
        return dropped

    @property
    def played_ms(self) -> int:
        """Milliseconds of the current item actually sent to the speaker."""
        with self._lock:
            return int(self._played_frames * 1000 / self._output_rate)

    def reset_played_ms(self) -> None:
        with self._lock:
            self._played_frames = 0


# -- host --------------------------------------------------------------------


class TerminalVoiceHost(RealtimeVoiceHost):
    """Transcripts to stdout, audio to the speaker, honest barge-in receipts."""

    def __init__(self, audio: DuplexAudio, *, out: Callable[[str], None] | None = None) -> None:
        self._audio = audio
        self._out = out or _print_line
        self._speaking_item: str | None = None
        self._response_open = False
        self._assistant_line_open = False

    def on_session_ready(self, session_id: str) -> None:
        self._out(f"realtime: connected (session {session_id}). Speak; Ctrl+C hangs up.")

    def on_session_closed(self, reason: str) -> None:
        self._end_assistant_line()
        self._out(f"realtime: session closed ({reason or 'no reason given'})")

    def on_error(self, message: str, *, terminal: bool) -> None:
        self._end_assistant_line()
        self._out(f"realtime: {'error' if terminal else 'warning'}: {message}")

    def on_input_transcript(self, text: str, final: bool) -> None:
        if final:
            self._end_assistant_line()
            self._out(f"You: {text}")

    def on_output_transcript(self, text: str, final: bool) -> None:
        if final:
            if not self._assistant_line_open and text:
                self._out(f"Hermes: {text}")
            self._end_assistant_line()
            return
        if not self._assistant_line_open:
            sys.stdout.write("Hermes: ")
            self._assistant_line_open = True
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_output_item_started(self, item_id: str) -> None:
        self._audio.reset_played_ms()
        self._speaking_item = item_id

    def on_output_audio(self, pcm: bytes) -> None:
        self._audio.queue_playback(pcm)

    def on_response_started(self, response_id: str | None) -> None:
        self._response_open = True

    def on_response_completed(self, response_id: str | None) -> None:
        self._response_open = False

    def on_barge_in(self) -> int | None:
        played = self._audio.played_ms
        dropped = self._audio.drain_playback()
        item = self._speaking_item
        self._speaking_item = None
        if item is None or not (dropped or self._response_open):
            return None  # nothing was mid-playback: nothing to truncate
        self._end_assistant_line()
        return played

    def on_tool_call(self, call: ToolCall) -> None:
        self._end_assistant_line()
        self._out(f"→ tool: {call.name}")

    def on_tool_result(self, call: ToolCall, output: str, outcome: str) -> None:
        if outcome != "ok":
            self._out(f"← tool {call.name}: {outcome}")

    def _end_assistant_line(self) -> None:
        if self._assistant_line_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._assistant_line_open = False


def _print_line(text: str) -> None:
    print(text, flush=True)


# -- tools -------------------------------------------------------------------


def to_realtime_tools(definitions: list[Mapping[str, Any]]) -> tuple[RealtimeTool, ...]:
    """Convert OpenAI chat-format tool schemas into the provider-neutral shape."""
    tools: list[RealtimeTool] = []
    for definition in definitions:
        function = definition.get("function") if isinstance(definition, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        parameters = function.get("parameters")
        tools.append(
            RealtimeTool(
                name=name.strip(),
                description=str(function.get("description") or ""),
                parameters=parameters if isinstance(parameters, Mapping) else {"type": "object"},
            )
        )
    return tuple(tools)


def build_tool_catalog(toolset: str) -> tuple[RealtimeTool, ...]:
    """The tools Hermes would give a chat session for *toolset*, as RealtimeTools."""
    from model_tools import get_tool_definitions

    return to_realtime_tools(get_tool_definitions(enabled_toolsets=[toolset], quiet_mode=True))


def make_tool_executor(*, task_id: str, toolset: str):
    """Run tools through Hermes' registry (hooks, approvals, budgets) off the loop."""
    from model_tools import handle_function_call

    async def execute(name: str, arguments: Mapping[str, Any]) -> Any:
        return await run_blocking_tool(
            handle_function_call,
            name,
            dict(arguments),
            task_id=task_id,
            tool_call_id=f"{task_id}-{uuid.uuid4().hex[:8]}",
            enabled_toolsets=[toolset],
        )

    return execute


# -- session runner ----------------------------------------------------------


def resolve_provider(name: str):
    """Load plugins (bundled backends included) and look the provider up."""
    from hermes_cli.plugins import discover_plugins

    from agent import realtime_voice_registry

    discover_plugins()
    return realtime_voice_registry.get_provider(name)


def list_provider_lines() -> list[str]:
    from hermes_cli.plugins import discover_plugins

    from agent import realtime_voice_registry

    discover_plugins()
    lines = []
    for provider in realtime_voice_registry.list_providers():
        state = "ready" if provider.is_available() else "needs setup"
        model = provider.default_model() or "-"
        lines.append(f"{provider.name:<16} {state:<12} {provider.display_name} (default model {model})")
    return lines


async def pump_microphone(audio: DuplexAudio, orchestrator: RealtimeVoiceOrchestrator) -> None:
    while True:
        chunk = audio.read_input_chunk()
        if chunk is None:
            await asyncio.sleep(IDLE_POLL_S)
            continue
        await orchestrator.send_audio(chunk)


async def run_session(
    *,
    provider_name: str = DEFAULT_PROVIDER,
    model: str | None = None,
    voice: str | None = None,
    toolset: str = DEFAULT_TOOLSET,
    tools_enabled: bool = True,
    tool_timeout_s: float = DEFAULT_TOOL_TIMEOUT_S,
    instructions: str = DEFAULT_INSTRUCTIONS,
    audio_factory: Callable[[], DuplexAudio] | None = None,
    out: Callable[[str], None] | None = None,
) -> int:
    """Run one terminal session; returns the process exit code."""
    out = out or _print_line
    provider = resolve_provider(provider_name)
    if provider is None:
        out(f"realtime: unknown provider '{provider_name}'. Registered providers:")
        for line in list_provider_lines():
            out(f"  {line}")
        return 2
    if not provider.is_available():
        schema = provider.get_setup_schema()
        env_vars = ", ".join(str(item.get("key")) for item in schema.get("env_vars", ()))
        out(
            f"realtime: provider '{provider.name}' is not configured"
            + (f" (set {env_vars})" if env_vars else "")
        )
        return 2

    # A missing microphone must fail here, not after a session was opened.
    audio = (audio_factory or DuplexAudio)()
    if audio_factory is None:
        import_sounddevice()

    tools = build_tool_catalog(toolset) if tools_enabled else ()
    task_id = f"realtime-{uuid.uuid4().hex[:12]}"
    executor = make_tool_executor(task_id=task_id, toolset=toolset) if tools_enabled else None
    setup = RealtimeVoiceSetup(
        model=model or provider.default_model(),
        voice=voice or provider.default_voice(),
        instructions=instructions,
        tools=tools,
    )
    out(
        f"realtime: opening {provider.display_name} ({setup.model or 'default model'}, "
        f"voice {setup.voice or 'default'}, {len(tools)} tool(s))"
    )
    session = await provider.open_session(setup)
    host = TerminalVoiceHost(audio, out=out)
    orchestrator = RealtimeVoiceOrchestrator(
        session, host, tool_executor=executor, tool_timeout_s=tool_timeout_s
    )
    try:
        audio.start(session.input_audio_format, session.output_audio_format)
    except RealtimeAudioError as exc:
        await session.close()
        out(f"realtime: {exc}")
        return 1

    run_task = asyncio.create_task(orchestrator.run(), name="realtime-voice-run")
    pump = asyncio.create_task(pump_microphone(audio, orchestrator), name="realtime-voice-mic")
    try:
        done, _pending = await asyncio.wait({run_task, pump}, return_when=asyncio.FIRST_COMPLETED)
        if pump in done and run_task not in done:
            pump.result()  # surfaces the microphone failure
            raise RuntimeError("microphone pump stopped unexpectedly")
        await run_task
    except RealtimeVoiceError as exc:
        out(f"realtime: session failed: {exc}")
        return 1
    finally:
        # Ctrl+C cancels this coroutine; the orchestrator still owns the
        # session close, so it is cancelled and awaited here, not abandoned.
        for task in (pump, run_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(pump, run_task, return_exceptions=True)
        audio.stop()
    return 0


def cmd_realtime(args) -> int:
    """``hermes realtime`` entry point."""
    if getattr(args, "list", False):
        lines = list_provider_lines()
        if not lines:
            print("No realtime voice providers are registered.")
        for line in lines:
            print(line)
        return 0
    try:
        return asyncio.run(
            run_session(
                provider_name=(getattr(args, "provider", None) or DEFAULT_PROVIDER).strip().lower(),
                model=getattr(args, "model", None),
                voice=getattr(args, "voice", None),
                toolset=getattr(args, "toolset", None) or DEFAULT_TOOLSET,
                tools_enabled=not getattr(args, "no_tools", False),
                tool_timeout_s=float(getattr(args, "tool_timeout", None) or DEFAULT_TOOL_TIMEOUT_S),
            )
        )
    except KeyboardInterrupt:
        print("\nrealtime: hung up.")
        return 0
    except RealtimeAudioError as exc:
        print(f"realtime: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI boundary: one line, not a traceback
        logger.warning("realtime session failed", exc_info=True)
        print(f"realtime: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


__all__ = [
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_PROVIDER",
    "DEFAULT_TOOLSET",
    "DEFAULT_TOOL_TIMEOUT_S",
    "DuplexAudio",
    "RealtimeAudioError",
    "TerminalVoiceHost",
    "build_tool_catalog",
    "cmd_realtime",
    "list_provider_lines",
    "make_tool_executor",
    "resolve_provider",
    "run_session",
    "to_realtime_tools",
]
