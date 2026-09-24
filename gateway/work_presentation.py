"""Trusted, route-bound publication for Telegram task and proposal briefs.

Presentation is deliberately authored data.  This module never derives a brief
from a task body, plan contents, comments, transcripts, or model results.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import base64
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import unicodedata
from uuid import uuid4
from urllib.parse import urlsplit


_ID_RE = re.compile(r"p_[a-f0-9]{16,64}")
_SECRET_RE = re.compile(
    r"(?i)(?:https?://[^\s/@]+:[^\s/@]+@|(?:token|password|secret|api[_-]?key)\s*[:=]\s*\S+|"
    r"(?<!\w)(?:sk-|ghp_|github_pat_)[\w-]+|\b\d{6,}:[\w-]{20,})"
)
_PATH_RE = re.compile(r"(?<!\w)(?:~?/[^\s]+|[A-Za-z]:[\\/][^\s]+)")
_STEP_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})
_PROPOSAL_STATUSES = frozenset({"proposed", "revised", "approved", "rejected", "discarded"})


@dataclass(frozen=True)
class TrustedWorkAudience:
    profile: str
    platform: str
    bot_id: int
    chat_id: str
    thread_id: str | None
    actor_id: int

    def key(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class WorkRef:
    proposal_id: str
    incarnation: int
    revision: int
    source_sha256: str
    status: str
    profile: str
    board: str = "proposals"
    kind: str = "proposal"

    @property
    def selector(self) -> str:
        raw = json.dumps(
            [self.profile, self.board, self.proposal_id, self.incarnation],
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")


@dataclass(frozen=True)
class ProposalCardSnapshot:
    profile: str
    board: str
    task_id: str
    proposal_id: str
    incarnation: int
    revision: int
    status: str
    title: str
    plan_mode: str
    presentation: dict
    steps: tuple
    published_at: int
    selector: str


@dataclass(frozen=True)
class _ProposalTransportBinding:
    audience_key: str
    profile_home: str
    profile: str
    transport_profile: str
    chat_id: str
    thread_id: str | None
    bot_id: int
    adapter: object
    client: object
    epoch: str
    loop: object


class _ProposalLane:
    """One proposal message on an ingress-bound fenced Telegram transport."""

    durable_card = True

    def __init__(self, service, transport, snapshot, renderer):
        self.service = self.registration = service
        self.transport, self.snapshot = transport, snapshot
        self.adapter, self.client, self.binding = transport.adapter, transport.client, transport
        self.epoch, self.loop = transport.epoch, transport.loop
        self.lock = threading.RLock()
        self.message_id = None
        self.reply_markup = None
        self.last_outcome = None
        self.task = None
        self.active, self.inflight, self.unknown = True, False, False
        self._scheduled = False
        self._detached = False
        self._attempt_snapshot = None
        self._attempt_id = None
        self.renderer = renderer
        self.adapter._live_todo_sources.add(self)

    def admitted(self):
        if not self.active:
            return False
        with self.service.lock, self.lock:
            snapshot = self._attempt_snapshot
            return bool(self.active and self.service.active and not self.unknown
                        and snapshot is not None
                        and self.service._transport_current(self.transport)
                        and self.service._card_current(snapshot))

    def publish(self, snapshot):
        with self.service.lock, self.lock:
            if (not self.active or snapshot.proposal_id != self.snapshot.proposal_id
                    or snapshot.incarnation != self.snapshot.incarnation
                    or snapshot.revision < self.snapshot.revision):
                return False
            self.snapshot = snapshot
            if self._scheduled or (self.task is not None and not self.task.done()):
                return True
            self._scheduled = True

        def start():
            with self.service.lock, self.lock:
                self._scheduled = False
                if self.active and (self.task is None or self.task.done()):
                    self.task = self.loop.create_task(self._run())
        try:
            self.loop.call_soon_threadsafe(start)
        except RuntimeError:
            self.close()
            return False
        return True

    async def _run(self):
        processed = -1
        try:
            while True:
                with self.service.lock, self.lock:
                    if not self.active or self.unknown or self.snapshot.revision <= processed:
                        return
                    snapshot = self.snapshot
                    self._attempt_snapshot = snapshot
                rendered = self.renderer(snapshot)
                if type(rendered) is not dict or set(rendered) != {"text", "links"}:
                    return
                await self.deliver(snapshot, rendered["text"], links=rendered["links"])
                with self.service.lock, self.lock:
                    processed = max(processed, snapshot.revision)
                    self._attempt_snapshot = None
                    self._attempt_id = None
                    if self.unknown or self.snapshot.revision <= processed:
                        return
        finally:
            self._detach()

    async def deliver(self, snapshot, text, *, links=(), controls=()):
        from gateway.live_todo import DeliveryOutcome, DeliveryStatus
        if controls or snapshot != self._attempt_snapshot or not self.admitted():
            return DeliveryOutcome(DeliveryStatus.REJECTED, reason="stale proposal card")
        if not isinstance(text, str) or not text or len(text) > 4096:
            return DeliveryOutcome(DeliveryStatus.REJECTED, reason="invalid proposal card")
        buttons = []
        if type(links) not in {tuple, list} or len(links) > 4:
            return DeliveryOutcome(DeliveryStatus.REJECTED, reason="invalid proposal links")
        for link in links:
            try:
                parsed = urlsplit(link.get("url")) if isinstance(link, dict) else None
            except (TypeError, ValueError):
                parsed = None
            if (type(link) is not dict or set(link) != {"label", "url"}
                    or not isinstance(link["label"], str) or not 1 <= len(link["label"]) <= 80
                    or not isinstance(link["url"], str) or len(link["url"]) > 2048
                    or parsed is None or parsed.scheme != "https" or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or any(c.isspace() for c in link["url"])):
                return DeliveryOutcome(DeliveryStatus.REJECTED, reason="invalid proposal links")
            buttons.append(dict(text=link["label"], url=link["url"]))
        from gateway.live_todo import rendered_payload_hash
        renderer_hash = rendered_payload_hash(text, [buttons] if buttons else [])
        attempt = self.service._begin_card(snapshot, renderer_hash)
        if attempt is None:
            return DeliveryOutcome(DeliveryStatus.SKIPPED, reason="proposal card already attempted")
        if attempt.get("equivalent"):
            with self.service.lock, self.lock:
                self.message_id = attempt["message_id"]
                self.last_outcome = DeliveryOutcome(
                    DeliveryStatus.SKIPPED, reason="confirmed proposal payload unchanged")
            return self.last_outcome
        with self.service.lock, self.lock:
            self._attempt_id = attempt["attempt_id"]
            self.message_id = attempt.get("message_id")
            self.reply_markup = [buttons] if buttons else None
            self.last_outcome = None
        try:
            outcome = await self.adapter.deliver_live_todo(self, text)
        except asyncio.CancelledError:
            outcome = self.last_outcome or DeliveryOutcome(
                DeliveryStatus.UNKNOWN, reason="cancelled proposal transport")
            self.service._settle_card(
                snapshot, attempt["attempt_id"], outcome, renderer_hash=renderer_hash)
            raise
        except Exception:
            outcome = DeliveryOutcome(DeliveryStatus.UNKNOWN, reason="transport exception")
            self.service._settle_card(
                snapshot, attempt["attempt_id"], outcome, renderer_hash=renderer_hash)
            raise
        else:
            self.service._settle_card(
                snapshot, attempt["attempt_id"], outcome, renderer_hash=renderer_hash)
            self.last_outcome = outcome
            return outcome

    def close(self):
        with self.lock:
            self.active = False

    def _detach(self):
        retry = False
        with self.service.lock, self.lock:
            if self._detached:
                return
            self._detached = True
            self.active = False
            key = (self.snapshot.proposal_id, self.snapshot.incarnation)
            if self.service._card_lanes.get(key) is self:
                self.service._card_lanes.pop(key, None)
            retry = not self.service._transport_current(self.transport) and not self.unknown
            self.adapter._live_todo_sources.discard(self)
        if retry:
            self.service._publish_current_card(self.transport.audience_key)

    async def finish(self):
        self.close()
        task = self.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            done, _ = await asyncio.wait({task}, timeout=2)
            if not done:
                with self.service.lock, self.lock:
                    self.unknown = bool(self.inflight)
                    snapshot, attempt_id = self._attempt_snapshot, self._attempt_id
                if self.unknown and snapshot is not None and attempt_id is not None:
                    from gateway.live_todo import DeliveryOutcome, DeliveryStatus
                    self.service._settle_card(snapshot, attempt_id, DeliveryOutcome(
                        DeliveryStatus.UNKNOWN, reason="proposal transport stopped during dispatch"))
                task.cancel()
                await asyncio.wait({task}, timeout=0.1)
        if task is None or task.done():
            self._detach()


def trusted_audience_for_source(source) -> TrustedWorkAudience | None:
    """Return ingress provenance only for a live Telegram group event."""
    from gateway.config import Platform
    from gateway.session_identity import identity_of

    identity = identity_of(source)
    if (identity is None or getattr(source, "platform", None) != Platform.TELEGRAM
            or getattr(source, "chat_type", None) not in {"group", "thread"}):
        return None
    adapter = identity.adapter()
    bot = getattr(adapter, "_bot", None) if adapter is not None else None
    bot_id = getattr(bot, "id", None)
    chat_id = str(getattr(source, "chat_id", "") or "")
    actor = str(getattr(source, "user_id", "") or "")
    if (type(bot_id) is not int or bot_id <= 0 or not chat_id
            or re.fullmatch(r"[1-9][0-9]{0,15}", actor) is None):
        return None
    thread = getattr(source, "thread_id", None)
    return TrustedWorkAudience(
        profile=identity.runtime_profile, platform="telegram", bot_id=bot_id,
        chat_id=chat_id, thread_id=str(thread) if thread else None, actor_id=int(actor),
    )


def _plain(value, *, limit: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("presentation text must be a string")
    text = " ".join(
        "".join(c if not unicodedata.category(c).startswith("C") else " " for c in value).split()
    )
    if required and not text:
        raise ValueError("presentation summary is required")
    if len(text) > limit or _SECRET_RE.search(text) or _PATH_RE.search(text):
        raise ValueError("presentation text is unsafe or too long")
    return text


def _sources(value) -> list[dict]:
    if value is None:
        return []
    if type(value) is not list or len(value) > 12:
        raise ValueError("presentation sources must be a list of at most 12 entries")
    result = []
    for item in value:
        if type(item) is not dict or set(item) != {"label", "url"}:
            raise ValueError("presentation source has an invalid shape")
        label = _plain(item["label"], limit=160, required=True)
        url = item["url"]
        try:
            parsed = urlsplit(url)
        except (TypeError, ValueError):
            parsed = None
        if (not isinstance(url, str) or len(url) > 2048 or any(c.isspace() for c in url)
                or parsed is None or parsed.scheme != "https" or not parsed.hostname
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("presentation sources require an HTTPS URL")
        result.append({"label": label, "url": url})
    return result


def validate_presentation(value) -> dict:
    if type(value) is not dict:
        raise ValueError("presentation must be an object")
    allowed = {"title", "summary", "current_step", "blocker", "input_acknowledgement", "findings", "result"}
    if not set(value) <= allowed or "summary" not in value:
        raise ValueError("presentation has an invalid shape")
    result = {"summary": _plain(value["summary"], limit=4000, required=True)}
    if "title" in value:
        title = _plain(value["title"], limit=120)
        if title:
            result["title"] = title
    for key in ("current_step", "blocker", "input_acknowledgement"):
        if key in value:
            text = _plain(value[key], limit=4000)
            if text:
                result[key] = text
    findings = value.get("findings")
    if findings is not None:
        if type(findings) is not list or len(findings) > 12:
            raise ValueError("presentation findings must be a list of at most 12 entries")
        clean = []
        for item in findings:
            if type(item) is not dict or not set(item) <= {"title", "detail", "sources"} \
                    or not {"title", "detail"} <= set(item):
                raise ValueError("presentation finding has an invalid shape")
            finding = {"title": _plain(item["title"], limit=240, required=True),
                       "detail": _plain(item["detail"], limit=4000, required=True)}
            sources = _sources(item.get("sources"))
            if sources:
                finding["sources"] = sources
            clean.append(finding)
        if clean:
            result["findings"] = clean
    final = value.get("result")
    if final is not None:
        if type(final) is not dict or not set(final) <= {"summary", "sources"} or "summary" not in final:
            raise ValueError("presentation result has an invalid shape")
        clean_final = {"summary": _plain(final["summary"], limit=4000, required=True)}
        sources = _sources(final.get("sources"))
        if sources:
            clean_final["sources"] = sources
        result["result"] = clean_final
    if len(json.dumps(result, ensure_ascii=False).encode()) > 48 * 1024:
        raise ValueError("presentation is too large")
    return result


def validate_steps(value) -> list[dict]:
    if value is None:
        return []
    if type(value) not in {list, tuple} or len(value) > 32:
        raise ValueError("steps must contain at most 32 entries")
    result = []
    for item in value:
        if type(item) is not dict or set(item) != {"content", "status"}:
            raise ValueError("step has an invalid shape")
        status = item["status"]
        if status not in _STEP_STATUSES:
            raise ValueError("step status is invalid")
        result.append({"content": _plain(item["content"], limit=500, required=True), "status": status})
    return result


def audience_dict(audience: TrustedWorkAudience) -> dict:
    return asdict(audience)


def audience_from_dict(value) -> TrustedWorkAudience:
    if type(value) is not dict or set(value) != {
            "profile", "platform", "bot_id", "chat_id", "thread_id", "actor_id"}:
        raise ValueError("invalid publication audience")
    audience = TrustedWorkAudience(**value)
    if (not audience.profile or audience.platform != "telegram" or type(audience.bot_id) is not int
            or audience.bot_id <= 0 or not audience.chat_id
            or type(audience.actor_id) is not int or audience.actor_id <= 0
            or (audience.thread_id is not None and not isinstance(audience.thread_id, str))):
        raise ValueError("invalid publication audience")
    return audience


class WorkPresentationService:
    """One profile-scoped service backed by the registering plugin's state."""

    STATE_KEY = "work_presentation.v1"

    def __init__(self, ctx, scope):
        from hermes_constants import get_hermes_home
        from hermes_cli.profiles import profile_matches_home
        from gateway.task_read import TaskReadService

        profiles = {route.profile for route in scope.routes}
        if len(profiles) != 1:
            raise ValueError("work presentation scope must name exactly one runtime profile")
        self.profile = next(iter(profiles))
        self.home = Path(get_hermes_home()).resolve()
        if not profile_matches_home(self.profile, self.home):
            raise ValueError("work presentation scope profile must match its runtime profile")
        self.plugin_id, self.scope, self.state = ctx.plugin_id, scope, ctx.state
        self.active, self.binding = True, None
        self.lock = threading.RLock()
        self._plan_mode: dict[str, str] = {}
        self._begin_new: set[str] = set()
        self._card_renderer = None
        self._card_lanes = {}
        self._transport_bindings = {}
        self.reader = TaskReadService(ctx, scope, proposal_provider=self)

    def close(self):
        lanes = ()
        with self.lock:
            self.active = False
            self.binding = None
            self._plan_mode.clear()
            self._begin_new.clear()
            lanes = tuple(self._card_lanes.values())
            for lane in lanes:
                lane.close()
            self._card_lanes.clear()
            self._transport_bindings.clear()
            self.reader.close()
        for lane in lanes:
            try:
                lane.loop.call_soon_threadsafe(lambda item=lane: item.loop.create_task(item.finish()))
            except RuntimeError:
                pass

    def bind(self, app, adapter):
        self.reader.bind(app, adapter)
        with self.lock:
            self.binding = (app, adapter)

    def available(self, app):
        return self.reader.available(app)

    async def detail(self, app, raw, selector):
        return await self.reader.detail(app, raw, selector)

    def _policy(self):
        from gateway.session_context import current_work_audience
        if not self.active:
            raise ValueError("publication unavailable")
        audience = current_work_audience()
        if audience is None or audience.profile != self.profile or not self.scope.allows_route(
                audience.profile, audience.platform, audience.chat_id, audience.thread_id):
            raise ValueError("publication unavailable")
        self._enabled()
        return audience

    def _enabled(self):
        from hermes_cli.config_effective import load_user_config_effective
        from yaml import YAMLError
        try:
            cfg = load_user_config_effective(fail_closed=True, side_effect_free=True) or {}
        except YAMLError:
            raise ValueError("publication unavailable") from None
        plugins = cfg.get("plugins", {})
        settings = plugins.get("entries", {}).get(self.plugin_id, {}).get("settings", {})
        if (self.plugin_id not in plugins.get("enabled", []) or self.plugin_id in plugins.get("disabled", [])
                or settings.get("enabled") is not True or settings.get("work_briefs") is not True):
            raise ValueError("publication unavailable")

    def accepts_audience(self, audience) -> bool:
        try:
            if (not self.active or audience.profile != self.profile
                    or not self.scope.allows_route(audience.profile, audience.platform,
                                                   audience.chat_id, audience.thread_id)):
                return False
            self._enabled()
            return True
        except (ValueError, TypeError, OSError):
            return False

    def available_current_route(self) -> bool:
        try:
            self._policy()
            return True
        except (ValueError, TypeError, OSError):
            return False

    def register_proposal_cards(self, factory):
        """Register plugin-owned rendering over host-owned proposal delivery."""
        if not callable(factory):
            raise ValueError("proposal card renderer must be callable")
        with self.lock:
            if self._card_renderer is not None and self._card_renderer is not factory:
                raise ValueError("proposal cards are already registered")
            self._card_renderer = factory
        return self

    def bind_transport_from_ingress(self, source, adapter) -> bool:
        """Pin the current fenced Telegram transport from one admitted native event."""
        from gateway.session_identity import identity_of

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        identity = identity_of(source)
        audience = trusted_audience_for_source(source)
        if (identity is None or audience is None or adapter is not identity.adapter()
                or Path(identity.runtime_home).resolve() != self.home
                or identity.runtime_profile != self.profile
                or getattr(adapter, "live_todo_transport", None) != 1
                or getattr(adapter, "_bot", None) is None
                or getattr(adapter._bot, "id", None) != audience.bot_id
                or getattr(adapter, "_send_path_degraded", False)
                or not self.scope.allows_route(audience.profile, audience.platform,
                                               audience.chat_id, audience.thread_id)):
            return False
        try:
            self._enabled()
        except (ValueError, TypeError, OSError):
            return False
        with self.lock:
            if not self.active:
                return False
            if getattr(adapter, "_live_todo_client", None) is not adapter._bot:
                adapter._live_todo_epoch = uuid4().hex
                adapter._live_todo_client = adapter._bot
            key = audience.key()
            previous = self._transport_bindings.get(key)
            if (previous is not None and previous.adapter is adapter
                    and previous.client is adapter._bot
                    and previous.epoch == adapter._live_todo_epoch
                    and previous.loop is loop):
                binding = previous
            else:
                binding = _ProposalTransportBinding(
                    audience_key=key, profile_home=str(self.home), profile=audience.profile,
                    transport_profile=identity.transport_profile, chat_id=audience.chat_id,
                    thread_id=audience.thread_id, bot_id=audience.bot_id,
                    adapter=adapter, client=adapter._bot, epoch=adapter._live_todo_epoch,
                    loop=loop,
                )
                self._transport_bindings[key] = binding
                for lane in tuple(self._card_lanes.values()):
                    if lane.transport.audience_key == key and lane.transport is not binding:
                        lane.close()
        self._publish_current_card(key)
        return True

    def _load(self) -> dict:
        value = self.state.get(self.STATE_KEY, {"proposals": {}, "current": {}})
        if (type(value) is not dict or set(value) != {"proposals", "current"}
                or type(value["proposals"]) is not dict or type(value["current"]) is not dict):
            raise ValueError("publication unavailable")
        return value

    @staticmethod
    def _ref(record: dict) -> WorkRef:
        return WorkRef(
            proposal_id=record["proposal_id"], incarnation=record["incarnation"],
            revision=record["revision"], source_sha256=record["source_sha256"],
            status=record["status"], profile=record["audience"]["profile"],
        )

    @staticmethod
    def _source(source_path, asserted_digest=None, *, expected_root=None) -> tuple[str, str, str]:
        from agent.runtime_cwd import resolve_agent_cwd
        root = Path(expected_root).resolve() if expected_root else resolve_agent_cwd().resolve()
        plans = (root / ".hermes" / "plans").resolve()
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("source_path is required")
        supplied = Path(source_path).expanduser()
        if supplied.is_symlink():
            raise ValueError("source must be a regular file under .hermes/plans")
        source = supplied.resolve(strict=True)
        if plans not in source.parents or not source.is_file():
            raise ValueError("source must be a regular file under .hermes/plans")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if asserted_digest is not None and asserted_digest != digest:
            raise ValueError("source digest mismatch")
        return str(root), source.relative_to(root).as_posix(), digest

    def publish_proposal(self, *, source_path: str, presentation: dict,
                         source_sha256: str | None = None, steps=()) -> WorkRef:
        audience = self._policy()
        source_root, locator, digest = self._source(source_path, source_sha256)
        clean, clean_steps = validate_presentation(presentation), validate_steps(steps)
        now = int(time.time())
        with self.lock:
            state = self._load()
            key = audience.key()
            current_id = state["current"].get(key)
            old = None if key in self._begin_new else (
                state["proposals"].get(current_id) if current_id else None)
            if isinstance(old, dict) and old.get("status") in {"approved", "discarded"}:
                raise ValueError("start a new plan before publishing another proposal")
            proposal_id = old["proposal_id"] if isinstance(old, dict) else "p_" + uuid4().hex
            incarnation = (old["incarnation"] if isinstance(old, dict)
                           else uuid4().int % ((1 << 53) - 1) + 1)
            revision = int(old["revision"]) + 1 if isinstance(old, dict) else 1
            record = {
                "proposal_id": proposal_id, "incarnation": incarnation, "revision": revision,
                "source_root": source_root, "source_locator": locator, "source_sha256": digest,
                "status": "revised" if old else "proposed",
                "plan_mode": "enforced" if self._plan_mode.get(key) == self._session_id() else "prompt",
                "presentation": clean, "steps": clean_steps, "audience": audience_dict(audience),
                "linked_tasks": list(old.get("linked_tasks", [])) if isinstance(old, dict) else [],
                "published_at": now,
            }
            if isinstance(old, dict) and isinstance(old.get("delivery"), dict):
                record["delivery"] = dict(old["delivery"])
            state["proposals"][proposal_id] = record
            state["current"][key] = proposal_id
            self.state.set(self.STATE_KEY, state)
            self._begin_new.discard(key)
            ref = self._ref(record)
        self._publish_card(record)
        return ref

    @staticmethod
    def _snapshot(record):
        from gateway.task_read import _clean_title
        presentation = dict(record["presentation"])
        selector = base64.urlsafe_b64encode(json.dumps([
            record["audience"]["profile"], "proposals", record["proposal_id"],
            record["incarnation"],
        ], separators=(",", ":")).encode()).decode().rstrip("=")
        return ProposalCardSnapshot(
            profile=record["audience"]["profile"], board="proposals",
            task_id=record["proposal_id"], proposal_id=record["proposal_id"],
            incarnation=record["incarnation"], revision=record["revision"],
            status=record["status"], title=_clean_title(presentation.get("title") or "Plan"),
            plan_mode=record["plan_mode"], presentation=presentation,
            steps=tuple(dict(step) for step in record.get("steps", [])),
            published_at=record["published_at"],
            selector=selector,
        )

    def _publish_card(self, record):
        renderer = self._card_renderer
        if renderer is None:
            return
        audience = audience_from_dict(record["audience"])
        snapshot = self._snapshot(record)
        key = (snapshot.proposal_id, snapshot.incarnation)
        with self.lock:
            transport = self._transport_bindings.get(audience.key())
            if transport is None or not self._transport_current(transport):
                return
            lane = self._card_lanes.get(key)
            if lane is None or not lane.active or lane.transport is not transport:
                try:
                    lane = _ProposalLane(self, transport, snapshot, renderer)
                except Exception:
                    return
                self._card_lanes[key] = lane
        lane.publish(snapshot)

    def _publish_current_card(self, audience_key):
        with self.lock:
            try:
                state = self._load()
                proposal_id = state["current"].get(audience_key)
                record = state["proposals"].get(proposal_id) if proposal_id else None
                current = dict(record) if isinstance(record, dict) else None
            except (ValueError, RuntimeError, OSError):
                current = None
        if current is not None:
            self._publish_card(current)

    def _transport_current(self, transport):
        current = self._transport_bindings.get(transport.audience_key)
        return bool(current is transport and self.active
                    and transport.adapter._bot is transport.client
                    and transport.adapter._live_todo_epoch == transport.epoch
                    and not getattr(transport.adapter, "_send_path_degraded", False))

    def _card_current(self, snapshot):
        with self.lock:
            try:
                record = self._load()["proposals"].get(snapshot.proposal_id)
            except (ValueError, RuntimeError, OSError):
                return False
            return bool(isinstance(record, dict) and record.get("incarnation") == snapshot.incarnation
                        and record.get("revision") == snapshot.revision)

    def _begin_card(self, snapshot, renderer_hash):
        with self.lock:
            state = self._load()
            record = state["proposals"].get(snapshot.proposal_id)
            if (not isinstance(record, dict) or record.get("incarnation") != snapshot.incarnation
                    or record.get("revision") != snapshot.revision):
                return None
            prior = record.get("delivery") if isinstance(record.get("delivery"), dict) else {}
            if prior.get("state") in {"inflight", "unknown"}:
                return None
            if (prior.get("state") in {"sent", "failed"} and prior.get("message_id")
                    and prior.get("delivered_revision") is not None
                    and prior.get("renderer_hash") == renderer_hash):
                equivalent = {
                    "revision": snapshot.revision, "state": "sent",
                    "message_id": prior["message_id"],
                    "delivered_revision": snapshot.revision,
                    "renderer_hash": renderer_hash,
                }
                record["delivery"] = equivalent
                self.state.set(self.STATE_KEY, state)
                return {**equivalent, "equivalent": True}
            if prior.get("revision") == snapshot.revision and prior.get("state") in {"inflight", "sent", "unknown"}:
                return None
            attempt = {"attempt_id": uuid4().hex, "revision": snapshot.revision, "state": "inflight",
                       "message_id": prior.get("message_id"),
                       "delivered_revision": prior.get("delivered_revision"),
                       "renderer_hash": prior.get("renderer_hash")}
            record["delivery"] = attempt
            self.state.set(self.STATE_KEY, state)
            return dict(attempt)

    def _settle_card(self, snapshot, attempt_id, outcome, *, renderer_hash=None):
        from gateway.live_todo import DeliveryStatus
        with self.lock:
            state = self._load()
            record = state["proposals"].get(snapshot.proposal_id)
            delivery = record.get("delivery") if isinstance(record, dict) else None
            if not isinstance(delivery, dict) or delivery.get("attempt_id") != attempt_id:
                return
            delivery["state"] = ({DeliveryStatus.DELIVERED: "sent",
                                  DeliveryStatus.UNKNOWN: "unknown"}.get(outcome.status, "failed"))
            if outcome.message_id:
                delivery["message_id"] = str(outcome.message_id)
            if outcome.status == DeliveryStatus.DELIVERED:
                delivery["delivered_revision"] = snapshot.revision
                delivery["renderer_hash"] = renderer_hash
            record["delivery"] = delivery
            self.state.set(self.STATE_KEY, state)

    def current_proposal(self) -> WorkRef | None:
        audience = self._policy()
        with self.lock:
            state = self._load()
            proposal_id = state["current"].get(audience.key())
            record = state["proposals"].get(proposal_id) if proposal_id else None
            if not isinstance(record, dict) or record.get("audience") != audience_dict(audience):
                return None
            return self._ref(record)

    def set_plan_mode(self, active: bool) -> None:
        if type(active) is not bool:
            raise ValueError("active must be boolean")
        audience = self._policy()
        session_id = self._session_id()
        if active and session_id is None:
            raise ValueError("plan mode requires an exact session")
        with self.lock:
            if active:
                self._plan_mode[audience.key()] = session_id
            else:
                self._plan_mode.pop(audience.key(), None)

    def begin_proposal(self) -> None:
        """Mark the next publication as a new proposal on this trusted route."""
        audience = self._policy()
        with self.lock:
            self._begin_new.add(audience.key())

    @staticmethod
    def _session_id():
        from gateway.session_context import current_work_session_id
        return current_work_session_id()

    def transition_proposal(self, ref: WorkRef, *, expected_revision: int, status: str,
                            source_path: str | None = None, source_sha256: str | None = None,
                            presentation: dict | None = None, steps=None) -> WorkRef:
        if status not in _PROPOSAL_STATUSES - {"proposed"}:
            raise ValueError("invalid proposal transition")
        audience = self._policy()
        with self.lock:
            state = self._load()
            record = state["proposals"].get(getattr(ref, "proposal_id", ""))
            if (not isinstance(record, dict) or record.get("audience") != audience_dict(audience)
                    or record.get("incarnation") != getattr(ref, "incarnation", None)
                    or record.get("revision") != expected_revision
                    or getattr(ref, "revision", None) != expected_revision):
                raise ValueError("stale proposal reference")
            allowed_from = {
                "approved": {"proposed", "revised"},
                "rejected": {"proposed", "revised"},
                "discarded": {"proposed", "revised", "rejected", "approved"},
                "revised": {"proposed", "revised", "rejected"},
            }
            if record.get("status") not in allowed_from[status]:
                raise ValueError("proposal transition is not valid from its current status")
            if status == "approved":
                delivery = record.get("delivery")
                if (not isinstance(delivery, dict) or delivery.get("state") != "sent"
                        or delivery.get("delivered_revision") != expected_revision
                        or not delivery.get("message_id")):
                    raise ValueError("approve only the confirmed displayed proposal revision")
                if presentation is not None or steps is not None:
                    raise ValueError("approval cannot change the published brief")
            if status in {"revised", "approved"}:
                candidate = source_path or str(
                    (Path(record["source_root"]) / record["source_locator"]).resolve())
                source_root, locator, digest = self._source(
                    candidate, source_sha256, expected_root=record["source_root"])
                if status == "approved":
                    if (source_root != record["source_root"] or locator != record["source_locator"]
                            or digest != record["source_sha256"]):
                        raise ValueError("approval source does not match the published revision")
                else:
                    record["source_root"], record["source_locator"], record["source_sha256"] = (
                        source_root, locator, digest)
            if presentation is not None:
                record["presentation"] = validate_presentation(presentation)
            if steps is not None:
                record["steps"] = validate_steps(steps)
            record["status"], record["revision"], record["published_at"] = (
                status, expected_revision + 1, int(time.time()))
            state["proposals"][record["proposal_id"]] = record
            self.state.set(self.STATE_KEY, state)
            updated = dict(record)
            result = self._ref(record)
        self._publish_card(updated)
        return result

    def link_task(self, ref: WorkRef, *, expected_revision: int, board: str,
                  task_id: str, task_incarnation: int) -> WorkRef:
        """Associate one already-published task with an approved exact proposal."""
        import sqlite3
        from hermes_cli import kanban_db as kb, kanban_db_surface as source
        from hermes_cli.kanban_publication import latest_task_audience
        from gateway.task_read import _clean_title

        audience = self._policy()
        if (not isinstance(board, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", board) is None
                or not isinstance(task_id, str) or re.fullmatch(r"t_[a-f0-9]{8,64}", task_id) is None):
            raise ValueError("invalid linked task")
        path = kb.kanban_db_path(board=board).absolute()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.1) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            identity = source.get_task_source(conn, task_id, task_incarnation=task_incarnation)
            linked_audience = latest_task_audience(conn, task_id)
            row = conn.execute("SELECT title FROM tasks WHERE id=?", (task_id,)).fetchone()
        if linked_audience != audience or row is None:
            raise ValueError("linked task is not published to this audience")
        selector = base64.urlsafe_b64encode(json.dumps(
            [self.profile, board, task_id, identity.task_incarnation],
            separators=(",", ":")).encode()).decode().rstrip("=")
        linked = {"label": _clean_title(row["title"]), "selector": selector,
                  "task_id": task_id, "board": board, "incarnation": identity.task_incarnation}
        with self.lock:
            state = self._load()
            record = state["proposals"].get(getattr(ref, "proposal_id", ""))
            if (not isinstance(record, dict) or record.get("audience") != audience_dict(audience)
                    or record.get("incarnation") != getattr(ref, "incarnation", None)
                    or record.get("revision") != expected_revision
                    or getattr(ref, "revision", None) != expected_revision
                    or record.get("status") != "approved"):
                raise ValueError("stale or unapproved proposal reference")
            links = [item for item in record.get("linked_tasks", [])
                     if isinstance(item, dict) and item.get("task_id") != task_id]
            if len(links) >= 32:
                raise ValueError("proposal task link capacity reached")
            links.append(linked)
            record["linked_tasks"] = links
            record["revision"] = expected_revision + 1
            record["published_at"] = int(time.time())
            state["proposals"][record["proposal_id"]] = record
            self.state.set(self.STATE_KEY, state)
            updated, result = dict(record), self._ref(record)
        self._publish_card(updated)
        return result

    def link_current_task(self, *, board: str, task_id: str, task_incarnation: int) -> WorkRef | None:
        ref = self.current_proposal()
        if ref is None or ref.status != "approved":
            return None
        return self.link_task(
            ref, expected_revision=ref.revision, board=board, task_id=task_id,
            task_incarnation=task_incarnation)

    def project_proposal(self, selector: tuple) -> tuple[dict, TrustedWorkAudience]:
        from gateway.task_read import _clean_title
        profile, board, proposal_id, incarnation = selector
        if profile != self.profile or board != "proposals" or _ID_RE.fullmatch(proposal_id) is None:
            raise ValueError("unavailable")
        with self.lock:
            record = self._load()["proposals"].get(proposal_id)
            if not isinstance(record, dict) or record.get("incarnation") != incarnation:
                raise ValueError("unavailable")
            audience = audience_from_dict(record["audience"])
            result = {
                "kind": "proposal", "title": _clean_title(record["presentation"].get("title") or "Plan"),
                "status": record["status"], "incarnation": incarnation,
                "revision": record["revision"], "updated_at": record["published_at"],
                "published_at": record["published_at"], "publication_stale": False,
                "plan_mode": record["plan_mode"], "presentation": record["presentation"],
                "steps": record.get("steps", []), "linked_tasks": record.get("linked_tasks", []),
            }
            return result, audience

    def proposal_audience(self, selector: tuple) -> tuple[TrustedWorkAudience, tuple]:
        """Return only authorization metadata; never deserialize presentation."""
        profile, board, proposal_id, incarnation = selector
        if profile != self.profile or board != "proposals" or _ID_RE.fullmatch(proposal_id) is None:
            raise ValueError("unavailable")
        with self.lock:
            record = self._load()["proposals"].get(proposal_id)
            if not isinstance(record, dict) or record.get("incarnation") != incarnation:
                raise ValueError("unavailable")
            audience = audience_from_dict(record["audience"])
            identity = ("proposal", proposal_id, incarnation, record["revision"], record["published_at"])
            return audience, identity


def register_work_presentation(ctx, *, scope):
    from gateway.surface_scope import parse_surface_scope
    parsed = parse_surface_scope(scope, require_tasks=False)
    current = getattr(ctx._manager, "_work_presentation_registration", None)
    if current is not None and current.active:
        if current.plugin_id != ctx.plugin_id or current.scope != parsed:
            raise ValueError("work presentation is already bound for this profile")
        return current
    service = WorkPresentationService(ctx, parsed)
    ctx._manager._work_presentation_registration = service
    ctx.on_unload(service.close)
    return service


def get_work_presentation(ctx):
    service = getattr(ctx._manager, "_work_presentation_registration", None)
    return service if service is not None and service.available_current_route() else None
