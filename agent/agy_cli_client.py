"""Connect-RPC client that drives the Antigravity ``language_server`` daemon.

This replaces the previous ``agy --print`` subprocess shim with a real
in-process Connect-RPC client (the same wire protocol the Antigravity IDE
itself uses), modeled after the proven TypeScript implementation in
``vscode-ai-extensions/packages/nous-agy-chat/src/agy-backend.ts``.

Behavior summary
================
* Lazily spawns the bundled ``language_server_linux_arm`` Go binary as a
  long-lived daemon (singleton). The daemon listens on a random localhost
  HTTPS port for Connect-RPC traffic and a random HTTP port for /healthz.
* Reads the discovery JSON the daemon writes to
  ``$gemini_dir/<app_data_dir>/daemon/ls_*.json`` to learn the port and
  CSRF token.
* Talks Connect-RPC v1 (``connect-protocol-version: 1`` +
  ``x-codeium-csrf-token``) over the self-signed HTTPS port (verify=False is
  safe, 127.0.0.1 only).
* Translates Hermes' chat-completion requests into the LS's
  ``StartCascade`` + ``SendUserCascadeMessage`` + ``GetCascadeTrajectorySteps``
  poll loop. ``StreamCascadeReactiveUpdates`` is documented in the proto
  catalog but the server returns ``reactive state is deprecated`` for the
  ``language_server_pb`` endpoint, so we use polled trajectory steps and
  yield deltas as new assistant tokens appear.
* Exposes an OpenAI-shaped ``client.chat.completions.create(...)`` surface
  so the rest of Hermes (conversation_loop, tool_executor, display) sees
  the same interface as openai/anthropic/etc.

Auth note
=========
The daemon proxies all model traffic to ``cloudcode-pa.googleapis.com``
using an OAuth token it manages itself under ``$gemini_dir/<app_data_dir>``.
We never touch tokens; the binary is responsible. If the user hasn't
authenticated yet, ``GetCascadeModelConfigData`` and the cascade calls
will surface ``UNAUTHENTICATED`` errors which we propagate.

Environment overrides
=====================
* ``HERMES_AGY_LANGUAGE_SERVER``: full path to the LS binary. Default
  ``/tmp/ag-ide/Antigravity IDE/resources/app/extensions/antigravity/bin/language_server_linux_arm``.
* ``HERMES_AGY_GEMINI_DIR``: gemini config dir. Default ``~/.gemini``.
* ``HERMES_AGY_APP_DATA_DIR``: subfolder for state/discovery. Default
  ``hermes-agy``.
* ``HERMES_AGY_TIMEOUT_SECONDS``: total per-call wall clock for the
  ``chat.completions.create`` cascade (default 120s).
* ``HERMES_AGY_REQUEST_TIMEOUT_SECONDS``: per-RPC HTTP timeout
  (default 30s).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

logger = logging.getLogger(__name__)

AGY_MARKER_BASE_URL = "agy://antigravity"

_DEFAULT_BINARY = (
    "/tmp/ag-ide/Antigravity IDE/resources/app/extensions/"
    "antigravity/bin/language_server_linux_arm"
)
_DEFAULT_APP_DATA_DIR = "hermes-agy"
# Discovery file write timing on aarch64 OL8 with memory pressure: cold start
# of the LS takes 4-7s. Old default 8.0 was too tight; bump to 25 and let the
# stderr-on-failure path catch real crashes early. Set HERMES_AGY_DISCOVERY_TIMEOUT
# in env to override for slow VMs.
try:
    _DISCOVERY_TIMEOUT_SECONDS = float(os.environ.get("HERMES_AGY_DISCOVERY_TIMEOUT", "25"))
except (TypeError, ValueError):
    _DISCOVERY_TIMEOUT_SECONDS = 25.0
_DEFAULT_TIMEOUT_SECONDS = 120.0
_DEFAULT_REQUEST_TIMEOUT = 30.0
# (Discovery timeout defined above with env override; do NOT add a second
# unconditional assignment here; it overrode the env-honoring value.)
_DISCOVERY_POLL_INTERVAL = 0.15

# Connect RPC service prefix
_SVC = "/exa.language_server_pb.LanguageServerService"

# Default model id sent to the LS when a Hermes slug maps to nothing.
_DEFAULT_LS_MODEL = "MODEL_GOOGLE_GEMINI_2_5_FLASH"

# Mapping from Hermes slug -> language_server enum id.
# Probed from the LS binary's enum table (`strings ... | grep MODEL_GOOGLE_`).
# Anything not listed here falls through to the raw slug; if the LS rejects
# it the caller gets a clear error step.
_HERMES_SLUG_TO_LS_MODEL: dict[str, str] = {
    "default":                  _DEFAULT_LS_MODEL,
    "gemini-3.5-flash-low":     "MODEL_GOOGLE_GEMINI_2_5_FLASH",
    "gemini-3.5-flash-medium":  "MODEL_GOOGLE_GEMINI_2_5_FLASH",
    "gemini-3.5-flash-high":    "MODEL_GOOGLE_GEMINI_2_5_FLASH_THINKING",
    "gemini-3.1-pro-low":       "MODEL_GOOGLE_GEMINI_2_5_PRO",
    "gemini-3.1-pro-high":      "MODEL_GOOGLE_GEMINI_2_5_PRO",
    "gemini-2.5-flash":         "MODEL_GOOGLE_GEMINI_2_5_FLASH",
    "gemini-2.5-pro":           "MODEL_GOOGLE_GEMINI_2_5_PRO",
    "claude-sonnet-4.6-thinking": "MODEL_CLAUDE_4_5_SONNET_THINKING",
    "claude-opus-4.6-thinking":   "MODEL_CLAUDE_4_OPUS_THINKING",
    "gpt-oss-120b":             "MODEL_OPENAI_GPT_OSS_120B_MEDIUM",
}


def _slug_to_ls_model(slug: str) -> str:
    if not slug:
        return _DEFAULT_LS_MODEL
    return _HERMES_SLUG_TO_LS_MODEL.get(slug, slug)


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------

class LanguageServerDaemon:
    """Singleton supervisor for the language_server child process.

    Instantiated lazily by AgyCliClient. ``start()`` is idempotent. ``stop()``
    SIGTERMs + SIGKILLs the child.
    """

    _instance_lock = threading.Lock()
    _instance: "LanguageServerDaemon | None" = None

    @classmethod
    def shared(cls) -> "LanguageServerDaemon":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def shutdown_shared(cls) -> None:
        with cls._instance_lock:
            if cls._instance is not None:
                try:
                    cls._instance.stop()
                except Exception:
                    logger.exception("agy: error stopping daemon")
                cls._instance = None

    def __init__(self) -> None:
        self.binary = os.environ.get("HERMES_AGY_LANGUAGE_SERVER", _DEFAULT_BINARY)
        self.gemini_dir = Path(
            os.environ.get("HERMES_AGY_GEMINI_DIR")
            or (Path.home() / ".gemini")
        ).expanduser()
        # IMPORTANT: the Antigravity language_server requires a RELATIVE
        # app_data_dir (relative to -gemini_dir). It rejects absolute paths
        # with a fatal startup error:
        #   "Language server failed - must not be absolute: /home/.../hermes-agy"
        # If a caller (or sloppy env) supplied an absolute path, derive the
        # basename so the daemon can actually start. Verified 2026-06-05 by
        # spawning the LS manually with both shapes.
        raw_app_data = os.environ.get(
            "HERMES_AGY_APP_DATA_DIR", _DEFAULT_APP_DATA_DIR
        ) or _DEFAULT_APP_DATA_DIR
        if os.path.isabs(raw_app_data):
            normalized = os.path.basename(os.path.normpath(raw_app_data)) or _DEFAULT_APP_DATA_DIR
            logger.warning(
                "agy: HERMES_AGY_APP_DATA_DIR=%r is absolute; LS requires "
                "relative path. Using basename %r instead.",
                raw_app_data, normalized,
            )
            self.app_data_dir = normalized
        else:
            self.app_data_dir = raw_app_data
        self.daemon_dir = self.gemini_dir / self.app_data_dir / "daemon"
        self.proc: subprocess.Popen | None = None
        self.discovery: dict[str, Any] | None = None
        self._start_lock = threading.Lock()
        self._csrf = secrets.token_hex(16)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> dict[str, Any]:
        with self._start_lock:
            if self.discovery and self._is_alive():
                return self.discovery
            if not Path(self.binary).is_file() or not os.access(self.binary, os.X_OK):
                raise FileNotFoundError(
                    f"Antigravity language_server binary not found or not "
                    f"executable: {self.binary}. Install the Antigravity IDE "
                    f"or set HERMES_AGY_LANGUAGE_SERVER."
                )

            # Wipe stale discovery file so we don't pick up a previous run.
            self.daemon_dir.mkdir(parents=True, exist_ok=True)
            for stale in self.daemon_dir.glob("ls_*.json"):
                try:
                    stale.unlink()
                except OSError:
                    pass

            args = [
                self.binary,
                "-standalone=true",
                "-persistent_mode=true",
                "-disable_telemetry=true",
                f"-gemini_dir={self.gemini_dir}",
                f"-app_data_dir={self.app_data_dir}",
                "-override_ide_name=hermes",
                "-subclient_type=sdk",
                "-model_api_client_type=ccpa",
                "-limit_go_max_procs=2",
                f"-csrf_token={self._csrf}",
            ]
            cloud_ep = os.environ.get(
                "HERMES_AGY_CLOUD_CODE_ENDPOINT",
                "https://cloudcode-pa.googleapis.com",
            )
            if cloud_ep:
                args.append(f"-cloud_code_endpoint={cloud_ep}")
                args.append(f"-inference_api_server_url={cloud_ep}")

            logger.info("agy: spawning language_server: %s", " ".join(args))
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env={**os.environ, "HOME": os.environ.get("HOME", str(Path.home()))},
                start_new_session=True,
            )
            try:
                disco = self._wait_for_discovery(_DISCOVERY_TIMEOUT_SECONDS)
            except Exception:
                # Best-effort teardown on failed start.
                self._terminate_child()
                raise
            self.discovery = disco
            logger.info(
                "agy: language_server ready pid=%s https=%s http=%s",
                disco.get("pid"), disco.get("httpsPort"), disco.get("httpPort"),
            )
            return disco

    def stop(self) -> None:
        with self._start_lock:
            self._terminate_child()
            self.discovery = None

    def _terminate_child(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        self.proc = None

    def _is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # -- discovery ----------------------------------------------------------

    def _wait_for_discovery(self, timeout: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_err: str | None = None
        while time.time() < deadline:
            try:
                files = sorted(
                    self.daemon_dir.glob("ls_*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for f in files:
                    try:
                        data = json.loads(f.read_text())
                    except (OSError, json.JSONDecodeError) as e:
                        last_err = f"{f.name}: {e}"
                        continue
                    if (
                        data.get("httpsPort")
                        and data.get("csrfToken")
                        and (self.proc is None or data.get("pid") == self.proc.pid)
                    ):
                        return data
            except OSError as e:
                last_err = str(e)
            if self.proc is not None and self.proc.poll() is not None:
                stderr = (self.proc.stderr.read().decode(errors="replace")
                          if self.proc.stderr else "")
                raise RuntimeError(
                    f"language_server exited with code {self.proc.returncode} "
                    f"before writing discovery file. Tail: {stderr[-500:]!r}"
                )
            time.sleep(_DISCOVERY_POLL_INTERVAL)
        raise TimeoutError(
            f"Timed out waiting for language_server discovery file in "
            f"{self.daemon_dir} after {timeout}s. last_err={last_err}"
        )

    # -- HTTP plumbing ------------------------------------------------------

    @property
    def base_https(self) -> str:
        assert self.discovery, "daemon not started"
        return f"https://127.0.0.1:{self.discovery['httpsPort']}"

    @property
    def base_http(self) -> str:
        assert self.discovery, "daemon not started"
        return f"http://127.0.0.1:{self.discovery['httpPort']}"

    @property
    def csrf_token(self) -> str:
        assert self.discovery, "daemon not started"
        return self.discovery["csrfToken"]

    def headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "content-type": content_type,
            "connect-protocol-version": "1",
            "x-codeium-csrf-token": self.csrf_token,
        }


# ---------------------------------------------------------------------------
# OpenAI-shaped client
# ---------------------------------------------------------------------------

def _render_messages_to_text(messages: list[dict]) -> str:
    """Flatten Hermes' chat messages into a single user-turn string.

    The cascade RPC only takes a single ``message.text`` per turn; there's
    no per-message role channel like OpenAI's chat API. We render history
    in a [ROLE] block format which most LLMs handle gracefully.
    """
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content") or ""
        if isinstance(content, list):
            content = "\n".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") in {"text", "input_text"}
            )
        if not content:
            continue
        if role in {"system", "user", "assistant"}:
            parts.append(f"[{role.upper()}]\n{content}")
        else:
            parts.append(f"[{role.upper() or 'CONTEXT'}]\n{content}")
    return "\n\n".join(parts).strip() or "Hi."


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _extract_assistant_text(step: dict) -> str:
    """Pull the visible assistant text out of one CortexStep dict.

    Probed shapes (see ``probes/full_run_*.json``):
      - {"assistantMessage": {"text": "...", "messageMarkdown": "..."}}
      - {"chatResponse": {"text": "..."}}
      - {"finishedResponse": {"text": "..."}}
      - {"finalResponse": {"messageMarkdown": "..."}}
    We try the most-specific known fields, then fall back to any
    string-valued ``text`` / ``messageMarkdown`` field anywhere inside.
    """
    for key in ("assistantMessage", "chatResponse", "finishedResponse",
                "finalResponse", "assistantTurnCompleted"):
        block = step.get(key)
        if isinstance(block, dict):
            for sub in ("text", "messageMarkdown", "content", "delta"):
                v = block.get(sub)
                if isinstance(v, str) and v:
                    return v
    # Last resort: scan any nested string under common names.
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in {"text", "messageMarkdown"} and isinstance(v, str) and v:
                    return v
                r = walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for v in obj:
                r = walk(v)
                if r:
                    return r
        return None
    return walk(step) or ""


def _is_user_step(step: dict) -> bool:
    t = step.get("type") or ""
    return t in ("CORTEX_STEP_TYPE_USER_MESSAGE",)


def _is_terminal_step(step: dict, status: str) -> bool:
    t = step.get("type") or ""
    if t == "CORTEX_STEP_TYPE_ERROR_MESSAGE":
        return True
    if status not in ("CORTEX_STEP_STATUS_DONE", "CORTEX_STEP_STATUS_FAILED"):
        return False
    return t in (
        "CORTEX_STEP_TYPE_FINISHED_RESPONSE",
        "CORTEX_STEP_TYPE_ASSISTANT_TURN_COMPLETED",
        "CORTEX_STEP_TYPE_ASSISTANT_MESSAGE",
    )


def _step_error_message(step: dict) -> str | None:
    em = step.get("errorMessage")
    if isinstance(em, dict):
        err = em.get("error") or {}
        return (err.get("userErrorMessage")
                or err.get("shortError")
                or err.get("modelErrorMessage")
                or json.dumps(err)[:300])
    return None


class AgyCliClient:
    """Hermes-facing client that speaks to the language_server daemon.

    Exposes ``client.chat.completions.create(model=..., messages=[...], stream=False)``
    so existing Hermes plumbing works unchanged.

    ``base_url``, ``api_key`` and friends are accepted but ignored; auth is
    fully handled by the daemon.
    """

    class _ChatCompletions:
        def __init__(self, client: "AgyCliClient"):
            self._client = client

        def create(self, **kwargs: Any) -> Any:
            return self._client._run(kwargs)

    class _Chat:
        def __init__(self, client: "AgyCliClient"):
            self.completions = AgyCliClient._ChatCompletions(client)

    def __init__(self, **kwargs: Any) -> None:
        # Tolerated kwargs: base_url, api_key, default_headers, http_client...
        self._kwargs = kwargs
        self._timeout = float(
            os.environ.get("HERMES_AGY_TIMEOUT_SECONDS", _DEFAULT_TIMEOUT_SECONDS)
        )
        self._req_timeout = float(
            os.environ.get("HERMES_AGY_REQUEST_TIMEOUT_SECONDS", _DEFAULT_REQUEST_TIMEOUT)
        )
        self.chat = AgyCliClient._Chat(self)
        self._http = None  # lazy httpx client

    # -- httpx helpers ------------------------------------------------------

    def _httpx(self):
        if self._http is None:
            import httpx  # local import: keep cold path cheap
            self._http = httpx.Client(
                verify=False,           # self-signed CN=localhost on 127.0.0.1
                http2=False,
                timeout=self._req_timeout,
                trust_env=False,        # don't honor proxies for localhost
            )
        return self._http

    def close(self) -> None:
        if self._http is not None:
            try:
                self._http.close()
            except Exception:
                pass
            self._http = None

    @property
    def default_headers(self) -> dict[str, str]:
        return {}

    # -- low-level RPC ------------------------------------------------------

    def _rpc(self, method: str, body: dict | None = None) -> Any:
        daemon = LanguageServerDaemon.shared()
        daemon.start()
        url = f"{daemon.base_https}{_SVC}/{method}"
        r = self._httpx().post(url, headers=daemon.headers(), json=body or {})
        if r.status_code >= 400:
            raise RuntimeError(
                f"agy RPC {method} failed {r.status_code}: {r.text[:500]}"
            )
        if not r.content:
            return {}
        return r.json()

    def healthz(self) -> bool:
        """Hit /healthz on the HTTP port (no TLS, no CSRF)."""
        daemon = LanguageServerDaemon.shared()
        daemon.start()
        r = self._httpx().get(f"{daemon.base_http}/healthz")
        return r.status_code == 200

    # -- high-level chat ----------------------------------------------------

    def _run(self, request: dict[str, Any]) -> Any:
        messages = request.get("messages") or []
        model_slug = (request.get("model") or "").strip()
        ls_model = _slug_to_ls_model(model_slug)
        prompt_text = _render_messages_to_text(messages)
        stream_requested = bool(request.get("stream", False))

        in_tok = sum(_approx_tokens(str(m.get("content") or "")) for m in messages)

        t0 = time.time()
        if stream_requested:
            return _AgyStreamingResult(
                client=self,
                model_slug=model_slug,
                ls_model=ls_model,
                prompt=prompt_text,
                in_tok=in_tok,
                started_at=t0,
            )

        # Non-streaming: drive the cascade then return a fully populated
        # ChatCompletion-shaped object.
        content = "".join(self._drive_cascade(ls_model, prompt_text))
        out_tok = _approx_tokens(content)
        rid = f"agy-{int(time.time() * 1000)}"
        message = SimpleNamespace(
            role="assistant", content=content, tool_calls=None, function_call=None
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        usage = SimpleNamespace(
            prompt_tokens=in_tok, completion_tokens=out_tok,
            total_tokens=in_tok + out_tok,
        )
        elapsed = round(time.time() - t0, 2)
        logger.info("agy: chat completed in %ss model=%s in~=%d out~=%d",
                    elapsed, model_slug, in_tok, out_tok)
        return SimpleNamespace(
            id=rid, object="chat.completion", created=int(time.time()),
            model=model_slug, choices=[choice], usage=usage,
            system_fingerprint=None,
        )

    def _drive_cascade(self, ls_model: str, prompt: str) -> Iterator[str]:
        """Run a cascade and yield assistant-text deltas as they appear.

        The LS does support a server-streaming RPC
        (``StreamCascadeReactiveUpdates``) but on the current binary the
        ``language_server_pb`` variant returns ``reactive state is
        deprecated`` (see probes/). So we use the poll-based
        ``GetCascadeTrajectorySteps`` path, which is what the IDE itself
        falls back to anyway. We yield text **incrementally** by tracking
        how much of the assistant message has already been emitted.
        """
        # 1) Start a cascade
        start = self._rpc("StartCascade", {"source": "CORTEX_TRAJECTORY_SOURCE_SDK"})
        cid = start.get("cascadeId")
        if not cid:
            raise RuntimeError(f"StartCascade returned no cascadeId: {start!r}")

        # 2) Send the user turn. Per probe runs against the live daemon
        # (see ``probes/``), the top-level ``requestedModelId`` field on
        # SendUserCascadeMessageRequest is the path the JSON codec
        # actually accepts as an enum string; the nested
        # ``cascadeConfig.plannerConfig.{plan,requested}Model`` shape
        # 400s with ``unexpected token "MODEL_..."``. The server still
        # complains "neither PlanModel nor RequestedModel specified"
        # until ``LoadCodeAssist`` (which requires OAuth) succeeds, at
        # which point the daemon resolves the model itself.
        send_body = {
            "cascadeId": cid,
            "message": {"text": prompt},
            "requestedModelId": ls_model,
        }
        self._rpc("SendUserCascadeMessage", send_body)

        # 3) Poll trajectory steps until terminal or timeout.
        deadline = time.time() + self._timeout
        emitted = 0
        last_step_count = -1
        # Adaptive poll: fast initially, back off if nothing is happening.
        poll = 0.25
        while time.time() < deadline:
            doc = self._rpc("GetCascadeTrajectorySteps", {"cascadeId": cid})
            steps = doc.get("steps") or []
            if len(steps) == last_step_count:
                poll = min(poll * 1.4, 1.5)
            else:
                last_step_count = len(steps)
                poll = 0.25
            terminal = False
            full_text = ""
            for s in steps:
                if _is_user_step(s):
                    continue
                err = _step_error_message(s)
                if err:
                    raise RuntimeError(f"agy cascade error: {err}")
                text = _extract_assistant_text(s)
                if text:
                    # Concatenate assistant text across steps; an assistant
                    # turn may be split into several steps.
                    if not full_text or text.startswith(full_text):
                        full_text = text
                    else:
                        full_text = full_text + text
                if _is_terminal_step(s, s.get("status") or ""):
                    terminal = True
            if full_text and len(full_text) > emitted:
                delta = full_text[emitted:]
                emitted = len(full_text)
                yield delta
            if terminal:
                return
            time.sleep(poll)
        # Cooperative cancel on timeout
        try:
            self._rpc("CancelCascadeInvocation", {"cascadeId": cid})
        except Exception:
            pass
        raise TimeoutError(
            f"agy: cascade timed out after {self._timeout}s "
            f"(cascadeId={cid}, model={ls_model})"
        )


# ---------------------------------------------------------------------------
# Streaming wrapper
# ---------------------------------------------------------------------------

class _AgyStreamingResult:
    """OpenAI-Stream-shaped iterable that drives the cascade lazily.

    Iterating yields ChatCompletionChunk-shaped SimpleNamespace objects
    with one ``choices[0].delta.content`` per assistant text increment.
    """

    def __init__(self, *, client: AgyCliClient, model_slug: str, ls_model: str,
                 prompt: str, in_tok: int, started_at: float):
        self._client = client
        self._model_slug = model_slug
        self._ls_model = ls_model
        self._prompt = prompt
        self._in_tok = in_tok
        self._t0 = started_at
        self._rid = f"agy-{int(time.time() * 1000)}"
        self.response = None  # No underlying httpx response surfaced.

    def __iter__(self):
        out_tok = 0
        first = True
        try:
            for delta_text in self._client._drive_cascade(self._ls_model, self._prompt):
                out_tok += _approx_tokens(delta_text)
                yield self._chunk(delta_text, finish_reason=None, role_only=first)
                first = False
        except Exception as e:
            # Surface as an error chunk + raise; Hermes' helpers handle this.
            logger.exception("agy: cascade streaming failed: %s", e)
            raise
        # Final chunk: empty delta + finish_reason + usage
        usage = SimpleNamespace(
            prompt_tokens=self._in_tok,
            completion_tokens=out_tok,
            total_tokens=self._in_tok + out_tok,
        )
        delta = SimpleNamespace(role=None, content="", tool_calls=None,
                                function_call=None, reasoning=None,
                                reasoning_content=None)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason="stop",
                                 logprobs=None)
        yield SimpleNamespace(
            id=self._rid, object="chat.completion.chunk",
            created=int(time.time()), model=self._model_slug,
            choices=[choice], usage=usage, system_fingerprint=None,
        )

    def _chunk(self, text: str, *, finish_reason: str | None, role_only: bool):
        delta = SimpleNamespace(
            role="assistant" if role_only else None,
            content=text,
            tool_calls=None, function_call=None,
            reasoning=None, reasoning_content=None,
        )
        choice = SimpleNamespace(index=0, delta=delta,
                                 finish_reason=finish_reason, logprobs=None)
        return SimpleNamespace(
            id=self._rid, object="chat.completion.chunk",
            created=int(time.time()), model=self._model_slug,
            choices=[choice], usage=None, system_fingerprint=None,
        )

    def close(self):  # pragma: no cover, called by Hermes on early exit
        pass
