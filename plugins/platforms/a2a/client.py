"""Strict outbound client for configured named A2A peers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx
from a2a.client import A2ACardResolver, ClientCallContext, ClientConfig, create_client
from a2a.client.auth import AuthInterceptor
from a2a.client.auth.credentials import CredentialService
from a2a.types.a2a_pb2 import (
    ROLE_USER,
    TASK_STATE_COMPLETED,
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    Message,
    Part,
    SendMessageRequest,
)

from . import auth, config
from . import client_state
from .client_state import abort_request, complete_request, try_begin_request

_TOTAL_TIMEOUT = 120.0
_CLOSE_TIMEOUT = 2.0
_MAX_BODY_BYTES = 2 * 1024 * 1024
_MAX_ID_CHARS = 256

if client_state._LEASE_SECONDS <= _TOTAL_TIMEOUT:
    raise RuntimeError("A2A client lease must exceed the total request timeout")


class A2AClientError(RuntimeError):
    pass


class _SanitizingClientLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = "A2A client protocol event"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        return True


def _install_log_filter() -> None:
    names = {"a2a.client", "a2a.client.card_resolver", "a2a.client.transports.jsonrpc"}
    names.update(name for name in logging.Logger.manager.loggerDict if name.startswith("a2a.client"))
    for name in names:
        logger = logging.getLogger(name)
        if not any(isinstance(item, _SanitizingClientLogFilter) for item in logger.filters):
            logger.addFilter(_SanitizingClientLogFilter())


class _BoundedTransport(httpx.AsyncBaseTransport):
    def __init__(self, delegate: httpx.AsyncBaseTransport):
        self.delegate = delegate

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.delegate.handle_async_request(request)
        length = response.headers.get("content-length")
        if length:
            try:
                parsed_length = int(length)
                if parsed_length < 0 or parsed_length > _MAX_BODY_BYTES:
                    await response.aclose()
                    raise A2AClientError("A2A peer response is too large")
            except ValueError:
                await response.aclose()
                raise A2AClientError("A2A peer response has invalid framing") from None
        content = bytearray()
        try:
            if response.is_stream_consumed:
                content.extend(response.content)
            else:
                async for chunk in response.aiter_raw():
                    if len(content) + len(chunk) > _MAX_BODY_BYTES:
                        raise A2AClientError("A2A peer response is too large")
                    content.extend(chunk)
            if len(content) > _MAX_BODY_BYTES:
                raise A2AClientError("A2A peer response is too large")
        finally:
            await response.aclose()
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=bytes(content),
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self.delegate.aclose()


class _CloseSerializedAsyncClient(httpx.AsyncClient):
    """Coalesce SDK and fallback close calls without concurrent double-close."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._a2a_close_lock = asyncio.Lock()
        self._a2a_closed = False

    async def aclose(self) -> None:
        async with self._a2a_close_lock:
            if self._a2a_closed:
                return
            await super().aclose()
            self._a2a_closed = True


class _CredentialService(CredentialService):
    def __init__(self, token: str):
        self._token = token

    async def get_credentials(self, security_scheme_name, context):  # noqa: ARG002
        return self._token if security_scheme_name == "bearer" else None


def _peer(name: str) -> tuple[str, auth.SecretToken, str]:
    name = config.validate_name(name, label="peer")
    # Peer setup/removal uses setup -> credential locking.  Resolve the
    # non-secret authority and its bearer under that same lock order so a
    # remove/re-add cannot pair an old URL with the replacement credential.
    from . import setup

    try:
        with setup._setup_transaction(), auth._locked_credential_mutation() as directory_fd:
            entry = config.load_a2a_settings().peers.get(name)
            if not entry:
                raise A2AClientError("A2A peer is not configured")
            url = config.validate_peer_url(entry.get("url", ""))
            generation = _identifier(
                entry.get("generation", ""), label="peer generation"
            )
            token = auth._load_outbound_token_unlocked(
                entry.get("credential_ref", ""), directory_fd
            )
    except A2AClientError:
        raise
    except (auth.CredentialStoreError, KeyError, OSError, TypeError, ValueError):
        raise A2AClientError("A2A peer configuration is invalid") from None
    if token is None:
        raise A2AClientError("A2A peer credential is unavailable")
    return url, token, generation


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _identifier(value: str, *, label: str) -> str:
    value = str(value or "").strip()
    if not value or len(value) > _MAX_ID_CHARS:
        raise ValueError(f"A2A {label} is invalid")
    return value


def _validate_card(card, expected_url: str) -> None:
    if card.capabilities.streaming or card.capabilities.push_notifications or card.capabilities.extended_agent_card:
        raise A2AClientError("A2A peer card enables unsupported capabilities")
    if list(card.default_input_modes) != ["text/plain"] or list(card.default_output_modes) != ["text/plain"]:
        raise A2AClientError("A2A peer card modes are unsupported")
    interfaces = list(card.supported_interfaces)
    if (
        len(interfaces) != 1
        or interfaces[0].protocol_binding.upper() != "JSONRPC"
        or interfaces[0].protocol_version != "1.0"
        or config.validate_peer_url(interfaces[0].url) != expected_url
    ):
        raise A2AClientError("A2A peer card interface does not match configuration")
    scheme = card.security_schemes.get("bearer")
    if scheme is None or not scheme.HasField("http_auth_security_scheme") or scheme.http_auth_security_scheme.scheme.lower() != "bearer":
        raise A2AClientError("A2A peer card requires unsupported authentication")
    requirements = list(card.security_requirements)
    if len(requirements) != 1 or set(requirements[0].schemes) != {"bearer"}:
        raise A2AClientError("A2A peer card requires unsupported authentication")


@dataclass
class _PeerLock:
    lock: asyncio.Lock
    references: int = 0


class NamedPeerClient:
    def __init__(self, *, transport=None):
        self._transport = transport
        self._locks_guard = asyncio.Lock()
        self._locks: dict[str, _PeerLock] = {}
        self._owned_tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def _peer_lock(self, peer: str):
        async with self._locks_guard:
            entry = self._locks.setdefault(peer, _PeerLock(asyncio.Lock()))
            entry.references += 1
        try:
            await entry.lock.acquire()
            try:
                yield
            finally:
                entry.lock.release()
        finally:
            async with self._locks_guard:
                entry.references -= 1
                if entry.references == 0:
                    self._locks.pop(peer, None)

    def _http(self) -> httpx.AsyncClient:
        delegate = self._transport or httpx.AsyncHTTPTransport(verify=True, retries=0)
        return _CloseSerializedAsyncClient(
            transport=_BoundedTransport(delegate),
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def _consume_task(self, task: asyncio.Task) -> None:
        self._owned_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _track(self, awaitable, *, name: str) -> asyncio.Task:
        task = asyncio.create_task(awaitable, name=name)
        self._owned_tasks.add(task)
        task.add_done_callback(self._consume_task)
        return task

    async def _gate_owned_cleanup(self) -> None:
        self._owned_tasks = {task for task in self._owned_tasks if not task.done()}
        if not self._owned_tasks:
            return
        await asyncio.wait(set(self._owned_tasks), timeout=min(_CLOSE_TIMEOUT, 0.05))
        self._owned_tasks = {task for task in self._owned_tasks if not task.done()}
        if self._owned_tasks:
            raise A2AClientError("A2A client cleanup is still in progress")

    async def _observe_owned(self, awaitable, *, timeout: float):
        task = self._track(awaitable, name="a2a-client-operation")
        try:
            done, _pending = await asyncio.wait({task}, timeout=timeout)
        except asyncio.CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            raise TimeoutError
        return task.result()

    async def _close_sequence(self, http, sdk_client) -> None:
        deadline = asyncio.get_running_loop().time() + _CLOSE_TIMEOUT
        sdk_task = None
        if sdk_client is not None:
            sdk_task = self._track(sdk_client.close(), name="a2a-sdk-close")
            await asyncio.wait(
                {sdk_task},
                timeout=min(_CLOSE_TIMEOUT / 2, max(0.0, deadline - asyncio.get_running_loop().time())),
            )
        http_task = None
        if http is not None:
            http_task = self._track(http.aclose(), name="a2a-http-close")
        pending = {task for task in (sdk_task, http_task) if task is not None and not task.done()}
        if pending:
            await asyncio.wait(
                pending,
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )

    async def _close_owned(self, http, sdk_client) -> None:
        if http is None and sdk_client is None:
            return
        cleanup = self._track(
            self._close_sequence(http, sdk_client), name="a2a-client-close"
        )
        cancellation = None
        try:
            await asyncio.wait({cleanup}, timeout=_CLOSE_TIMEOUT)
        except asyncio.CancelledError as exc:
            cancellation = exc
        if cancellation is not None:
            raise cancellation

    async def aclose(self) -> None:
        tasks = {task for task in self._owned_tasks if not task.done()}
        if tasks:
            await asyncio.wait(tasks, timeout=_CLOSE_TIMEOUT)
        self._owned_tasks = {task for task in self._owned_tasks if not task.done()}

    async def fetch_card(self, peer: str):
        peer = config.validate_name(peer, label="peer")
        await self._gate_owned_cleanup()
        async with self._peer_lock(peer):
            _url, _token, generation = _peer(peer)
            url, _token, current_generation = _peer(peer)
            if current_generation != generation:
                raise A2AClientError("A2A peer authority changed")
            http = self._http()
            try:
                async with asyncio.timeout(_TOTAL_TIMEOUT):
                    card = await self._observe_owned(
                        A2ACardResolver(http, _origin(url)).get_agent_card(),
                        timeout=_TOTAL_TIMEOUT,
                    )
                _validate_card(card, url)
                return card
            except asyncio.CancelledError:
                raise
            except Exception:
                raise A2AClientError("A2A peer card request failed") from None
            finally:
                await self._close_owned(http, None)

    async def _client(self, peer: str, generation: str):
        url, token, current_generation = _peer(peer)
        if current_generation != generation:
            raise A2AClientError("A2A peer authority changed")
        http = self._http()
        try:
            async with asyncio.timeout(_TOTAL_TIMEOUT):
                card = await self._observe_owned(
                    A2ACardResolver(http, _origin(url)).get_agent_card(),
                    timeout=_TOTAL_TIMEOUT,
                )
            _validate_card(card, url)
            sdk_client = await create_client(
                card,
                ClientConfig(streaming=False, polling=False, httpx_client=http, supported_protocol_bindings=["JSONRPC"]),
                interceptors=[AuthInterceptor(_CredentialService(token))],
            )
            return http, sdk_client
        except BaseException:
            await self._close_owned(http, None)
            raise

    @staticmethod
    def _successful_task(task) -> list[str]:
        try:
            _identifier(task.id, label="task id")
            _identifier(task.context_id, label="context id")
        except ValueError:
            raise A2AClientError("A2A peer returned invalid task identifiers") from None
        if task.status.state != TASK_STATE_COMPLETED:
            raise A2AClientError("A2A peer task did not complete successfully")
        texts = []
        for artifact in task.artifacts:
            for part in artifact.parts:
                if part.WhichOneof("content") != "text":
                    raise A2AClientError("A2A peer returned a non-text artifact")
                if not part.text.strip():
                    raise A2AClientError("A2A peer returned an empty text artifact")
                texts.append(part.text)
        if not texts:
            raise A2AClientError("A2A peer returned no text artifact")
        return texts

    async def ask(self, peer: str, text: str, *, new_context: bool = False, context_id: str | None = None):
        peer = config.validate_name(peer, label="peer")
        await self._gate_owned_cleanup()
        text = str(text or "").strip()
        if not text or text.startswith("/"):
            raise ValueError("A2A request must be nonempty text without slash commands")
        if new_context and context_id is not None:
            raise ValueError("A2A new_context cannot be combined with context_id")
        async with self._peer_lock(peer):
            _url, _token, generation = _peer(peer)
            owner = uuid.uuid4().hex
            deadline = asyncio.get_running_loop().time() + _TOTAL_TIMEOUT
            delay = 0.01
            claim = None
            while claim is None:
                try:
                    claim = try_begin_request(
                        peer, generation, owner, new_context=new_context
                    )
                except RuntimeError:
                    raise A2AClientError("A2A peer authority changed") from None
                if claim is not None:
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise A2AClientError("A2A peer request queue timed out")
                await asyncio.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.25)
            http = None
            sdk_client = None
            lease_completed = False
            try:
                selected_context = context_id or claim.context_id
                if selected_context:
                    selected_context = _identifier(selected_context, label="context id")
                message = Message(
                    role=ROLE_USER,
                    message_id=uuid.uuid4().hex,
                    parts=[Part(text=text)],
                )
                if selected_context:
                    message.context_id = selected_context
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise A2AClientError("A2A peer request queue timed out")
                async with asyncio.timeout(remaining):
                    http, sdk_client = await self._client(peer, generation)
                    final = None
                    async for response in sdk_client.send_message(
                        SendMessageRequest(message=message), context=ClientCallContext()
                    ):
                        if response.HasField("task"):
                            final = response.task
                if final is None:
                    raise A2AClientError("A2A peer returned no task")
                texts = self._successful_task(final)
                lease_completed = complete_request(
                    peer,
                    generation,
                    claim,
                    context_id=final.context_id,
                    task_id=final.id,
                )
                if not lease_completed:
                    raise A2AClientError("A2A peer authority changed before completion")
                return final, texts
            except A2AClientError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception:
                raise A2AClientError("A2A peer request failed") from None
            finally:
                try:
                    await self._close_owned(http, sdk_client)
                finally:
                    if not lease_completed:
                        try:
                            abort_request(peer, generation, claim)
                        except Exception:
                            pass

    async def _task_op(self, peer, method, request):
        peer = config.validate_name(peer, label="peer")
        await self._gate_owned_cleanup()
        async with self._peer_lock(peer):
            _url, _token, generation = _peer(peer)
            http = None
            sdk_client = None
            try:
                async with asyncio.timeout(_TOTAL_TIMEOUT):
                    http, sdk_client = await self._client(peer, generation)
                    return await getattr(sdk_client, method)(request, context=ClientCallContext())
            except asyncio.CancelledError:
                raise
            except Exception:
                raise A2AClientError("A2A peer task request failed") from None
            finally:
                await self._close_owned(http, sdk_client)

    async def get_task(self, peer: str, task_id: str):
        return await self._task_op(peer, "get_task", GetTaskRequest(id=_identifier(task_id, label="task id")))

    async def list_tasks(self, peer: str):
        return await self._task_op(peer, "list_tasks", ListTasksRequest())

    async def cancel(self, peer: str, task_id: str):
        return await self._task_op(peer, "cancel_task", CancelTaskRequest(id=_identifier(task_id, label="task id")))


_install_log_filter()
