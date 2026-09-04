"""Telegram routing methods; runtime dependencies remain on the adapter facade."""

import re
from typing import Any, Dict, List, Optional
from gateway.platforms.base import MessageEvent
try:
    from telegram import Message, Update
    from telegram.ext import ContextTypes
except ImportError:
    Message = Update = Any
    class ContextTypes:
        DEFAULT_TYPE = Any


class TelegramRoutingMixin:
    def _notification_kwargs(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """In "important" mode return disable_notification=True unless ``metadata["notify"]``."""
        if getattr(self, "_notifications_mode", "important") != "important" or (metadata or {}).get("notify"):
            return {}
        return {"disable_notification": True}

    @staticmethod
    def _normalize_chat_type(chat_type: Any, *, is_forum: bool) -> str:
        """Telegram chat type → gateway chat type (``private``→``dm``, ``supergroup``→forum/group)."""
        from . import adapter as _adapter

        normalized = str(chat_type or "dm").strip().lower() or "dm"
        if normalized == "private":
            return "dm"
        if normalized == "supergroup":
            return "forum" if is_forum else "group"
        return normalized

    def _legacy_runner_auth_fn(self):
        """``runner._is_user_authorized`` resolved off the bound handler (bare-adapter tests, direct
        embedding); None under multiplex where the handler is a profile closure."""
        # Resolve through the runner's full auth chain (platform + group allowlists, pairing store,
        # allow-all flags). Prefer the platform-bound callback registered via set_authorization_check: it
        # routes to GatewayRunner._is_user_authorized AND survives multiplex handler wrapping, whereas the
        # bound-handler __self__ lookup is None when the primary handler is a profile closure — which
        # silently dropped the chat allowlist and default-denied allowlisted group members under
        # multiplex_profiles (#87132). Fall back to the bound handler for setups without a registered
        # callback.
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        auth_fn = getattr(runner, "_is_user_authorized", None)
        return auth_fn if callable(auth_fn) else None

    @staticmethod
    def _env_allowlist_decision(user_id: str) -> Optional[bool]:
        """TELEGRAM_ALLOWED_USERS decision; None when no allowlist is configured."""
        from . import adapter as _adapter

        allowed_csv = _adapter._scoped_gate_env("TELEGRAM_ALLOWED_USERS").strip()
        if not allowed_csv:
            return None
        allowed_ids = {uid.strip() for uid in allowed_csv.split(",") if uid.strip()}
        return "*" in allowed_ids or user_id in allowed_ids

    def _is_callback_user_authorized(
        self, user_id: str, *, chat_id: Optional[str] = None, chat_type: Optional[str] = None,
        thread_id: Optional[str] = None, user_name: Optional[str] = None) -> bool:
        """Return whether a Telegram inline-button caller may perform gated actions."""
        from . import adapter as _adapter

        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False
        normalized_chat_type = self._normalize_chat_type(chat_type, is_forum=thread_id is not None)
        # Preferred: the auth callback GatewayRunner injects (set_authorization_check) → full
        # _is_user_authorized chain; also works for a multiplexed adapter whose _message_handler is a
        # profile closure. getattr tolerates partially-constructed adapters (object.__new__ in tests).
        if getattr(self, "_authorization_check", None) is not None:
            injected = self._is_sender_authorized(
                normalized_user_id, chat_type=normalized_chat_type, chat_id=str(chat_id or normalized_user_id),
                thread_id=str(thread_id) if thread_id is not None else None)
            if injected is not None:
                return injected
        auth_fn = self._legacy_runner_auth_fn()
        if auth_fn is not None:
            try:
                from gateway.session import SessionSource
                source = SessionSource(
                    platform=_adapter.Platform.TELEGRAM, chat_id=str(chat_id or normalized_user_id), chat_type=normalized_chat_type,
                    user_id=normalized_user_id, user_name=str(user_name).strip() if user_name else None,
                    thread_id=str(thread_id) if thread_id is not None else None)
                return bool(auth_fn(source))
            except Exception:
                _adapter.logger.debug(
                    "[Telegram] Falling back to env-only callback auth for user %s", normalized_user_id, exc_info=True)
        decision = self._env_allowlist_decision(normalized_user_id)
        if decision is None:
            # Fail-closed: no allowlist means deny unless GATEWAY_ALLOW_ALL_USERS is set.
            # The runner auth path in _is_user_authorized() handles GATEWAY_ALLOW_ALL_USERS; this fallback
            # must not silently allow everyone (fixes #24457).
            return _adapter._scoped_gate_env("GATEWAY_ALLOW_ALL_USERS").lower() in {"true", "1", "yes"}
        return decision

    def _source_from_message_for_auth(self, message: Message):
        """Build the SessionSource the gateway auth path expects; identity comes from ``from_user``,
        falling back to ``sender_chat`` for channel posts so an unauthorized channel can't inject."""
        from . import adapter as _adapter

        from gateway.session import SessionSource
        user = getattr(message, "from_user", None)
        chat = getattr(message, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        # Carry is_bot so the runner's ``*_ALLOW_BOTS`` branch is reachable, as in build_source.
        is_bot = bool(getattr(user, "is_bot", False)) if user is not None else False
        user_name = str(getattr(user, "username", "") or getattr(user, "full_name", "") or "").strip() or None
        if not user_id:  # channel post — authorize the sender chat instead
            sender_chat = getattr(message, "sender_chat", None)
            if sender_chat is not None:
                user_id = str(getattr(sender_chat, "id", "")).strip() or None
                if not user_name:
                    user_name = str(getattr(sender_chat, "title", "") or "").strip() or None
        chat_id = str(getattr(chat, "id", "")).strip() or user_id
        thread_id_raw = getattr(message, "message_thread_id", None)
        is_topic_message = bool(getattr(message, "is_topic_message", False))
        is_forum_group = getattr(chat, "is_forum", False) is True
        chat_type = self._normalize_chat_type(
            getattr(chat, "type", "dm"), is_forum=thread_id_raw is not None and (is_topic_message or is_forum_group))
        thread_id = None
        if thread_id_raw is not None and (
            (chat_type == "forum" and (is_topic_message or is_forum_group)) or (chat_type == "dm" and is_topic_message)):
            thread_id = str(thread_id_raw)
        return SessionSource(
            platform=_adapter.Platform.TELEGRAM, chat_id=chat_id or "", chat_type=chat_type, user_id=user_id,
            user_name=user_name, thread_id=thread_id, is_bot=is_bot)

    def _source_from_reaction_for_auth(self, update):
        """SessionSource for a ``message_reaction`` update's actor (``user`` or ``actor_chat``).

        Raises ``ValueError`` when actor, chat or message identity is absent so the post-auth boundary fails closed."""
        from . import adapter as _adapter

        mr = getattr(update, "message_reaction", None)
        if mr is None:
            raise ValueError("gateway_platform_event source extraction requires a message_reaction update")
        user = getattr(mr, "user", None) or getattr(mr, "actor_chat", None)
        chat = getattr(mr, "chat", None)
        user_id = str(getattr(user, "id", "")).strip() or None
        user_name = str(getattr(user, "username", "") or getattr(user, "full_name", "") or getattr(user, "title", "")).strip() or None
        chat_id = str(getattr(chat, "id", "")).strip() or None
        message_id = getattr(mr, "message_id", None)
        if not user_id or not chat_id or message_id is None or not str(message_id).strip():
            raise ValueError("gateway_platform_event reaction requires actor, chat, and message identities")
        # Reactions carry no message_thread_id; is_forum is the only forum signal.
        chat_type = self._normalize_chat_type(getattr(chat, "type", "dm"), is_forum=getattr(chat, "is_forum", False) is True)
        return self.build_source(
            chat_id=chat_id, chat_type=chat_type, user_id=user_id, user_name=user_name, thread_id=None, message_id=str(message_id))

    def _telegram_auth_env_configured(self) -> bool:
        """Return True when Telegram auth env vars make an early decision safe."""
        from . import adapter as _adapter

        keys = (
            "TELEGRAM_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS",
            "TELEGRAM_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS")
        return any(_adapter._scoped_gate_env(key).strip() for key in keys)

    def _should_pass_unauthorized_dm_for_pairing(self, source) -> bool:
        """True when an unauthorized DM must still reach gateway pairing (``unauthorized_dm_behavior``
        resolves to ``pair``, incl. an allowlist plus an explicit platform override)."""
        from . import adapter as _adapter

        if source.chat_type != "dm":
            return False
        # Bound-handler ``__self__`` is None under multiplex; ``gateway_runner`` survives that wrapping.
        runner = getattr(getattr(self, "_message_handler", None), "__self__", None) or getattr(self, "gateway_runner", None)
        behavior_fn = getattr(runner, "_get_unauthorized_dm_behavior", None)
        if callable(behavior_fn):
            try:
                profile = getattr(source, "profile", None) or getattr(self, "_owner_profile", None)
                return behavior_fn(_adapter.Platform.TELEGRAM, profile=profile) == "pair"
            except Exception:
                _adapter.logger.debug("[Telegram] Failed to resolve unauthorized DM behavior; falling back to adapter-local override", exc_info=True)
        extra = getattr(getattr(self, "config", None), "extra", None) or {}
        return str(extra.get("unauthorized_dm_behavior", "")).strip().lower() == "pair"

    def _is_user_authorized_from_message(self, message: Message) -> bool:
        """Intake auth prefilter, run BEFORE batching/event construction/group observation.

        Only rejects when it can make the same context-aware decision the runner would; unknown DMs pass through when
        there is no allowlist or pairing is the unauthorized-DM behavior."""
        from . import adapter as _adapter

        source = self._source_from_message_for_auth(message)
        user_id = source.user_id
        # No identity → service message or channel post without sender_chat; defer to message gating.
        if not user_id:
            return True
        authorized: _adapter.Optional[bool] = None
        # Adapter-level allow_from (DMs) / group_allow_from (groups) are the sole authority if set.
        adapter_allow_from = self.config.extra.get(
            "group_allow_from" if (source.chat_type or "") in ("group", "forum", "channel") else "allow_from")
        if adapter_allow_from is not None:
            allowed = _adapter._coerce_allow_set(adapter_allow_from)
            authorized = user_id in allowed or "*" in allowed
        # Instance-level override only (tests): the class method _is_callback_user_authorized is for
        # inline buttons and must not become a user-id-only shortcut for real messages.
        if authorized is None:
            callback_auth = self.__dict__.get("_is_callback_user_authorized")
            if callable(callback_auth):
                with _adapter.contextlib.suppress(Exception):
                    authorized = bool(callback_auth(
                        user_id, chat_id=source.chat_id, chat_type=source.chat_type, thread_id=source.thread_id,
                        user_name=source.user_name))
        if authorized is None:
            # Runner's full auth chain; prefer the set_authorization_check callback (survives multiplex
            # handler wrapping, unlike bound-handler __self__).
            auth_fn = self._legacy_runner_auth_fn()
            has_callback = getattr(self, "_authorization_check", None) is not None
            if has_callback or auth_fn is not None:
                # No allowlist → unknown DMs must reach pairing, not be default-denied here.
                if not self._telegram_auth_env_configured():
                    return True
                decision = self._is_sender_authorized(
                    user_id, chat_type=source.chat_type, chat_id=source.chat_id, is_bot=source.is_bot,
                    thread_id=source.thread_id) if has_callback else None
                if decision is not None:
                    authorized = decision
                elif auth_fn is not None:
                    try:
                        authorized = bool(auth_fn(source))
                    except Exception:
                        _adapter.logger.debug("[Telegram] Falling back to env-only auth for user %s", user_id, exc_info=True)
        if authorized is None:
            authorized = self._env_allowlist_decision(user_id)
            if authorized is None:
                return True
        if authorized:
            return True
        # Unauthorized DM the gateway would pair: forward so pairing can run.
        return self._should_pass_unauthorized_dm_for_pairing(source)

    async def _create_dm_topic(
        self, chat_id: int, name: str, icon_color: Optional[int] = None, icon_custom_emoji_id: Optional[str] = None) -> Optional[int]:
        """Create a forum topic in a private (DM) chat (Bot API 9.4+); message_thread_id or None."""
        from . import adapter as _adapter

        if not self._bot:
            return None
        try:
            kwargs: _adapter.Dict[str, _adapter.Any] = {"chat_id": chat_id, "name": name}
            if icon_color is not None:
                kwargs["icon_color"] = icon_color
            if icon_custom_emoji_id:
                kwargs["icon_custom_emoji_id"] = icon_custom_emoji_id
            topic = await self._bot.create_forum_topic(**kwargs)
            thread_id = topic.message_thread_id
            _adapter.logger.info("[%s] Created DM topic '%s' in chat %s -> thread_id=%s", self.name, name, chat_id, thread_id)
            return thread_id
        except Exception as e:
            error_text = str(e).lower()
            # Telegram has no "list topics" API: an existing topic is mapped from incoming messages.
            if "topic_name_duplicate" in error_text or "already" in error_text:
                _adapter.logger.info(
                    "[%s] DM topic '%s' already exists in chat %s (will be mapped from incoming messages)", self.name, name, chat_id)
            elif "not a forum" in error_text or "forums_disabled" in error_text:
                _adapter.logger.warning(
                    "[%s] Cannot create DM topic '%s' in chat %s: Topics mode is not enabled. "
                    "The user must open the DM with this bot in Telegram, tap the bot name "
                    "at the top, and enable 'Topics' in chat settings before topics can be created.",
                    self.name, name, chat_id)
            else:
                _adapter.logger.warning(
                    "[%s] Failed to create DM topic '%s' in chat %s: %s", self.name, name, chat_id, _adapter._redact_telegram_error_text(e))
            return None

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Create a forum topic for a session handoff; ``message_thread_id`` as str, or None."""
        from . import adapter as _adapter

        try:
            chat_id_int = int(parent_chat_id)
        except (TypeError, ValueError):
            return None
        thread_id = await self._create_dm_topic(chat_id_int, name=name)
        return str(thread_id) if thread_id else None

    async def ensure_dm_topic(self, chat_id: str, topic_name: str, force_create: bool = False) -> Optional[str]:
        """Return a private DM topic thread id, creating and persisting it if needed."""
        from . import adapter as _adapter

        name = str(topic_name or "").strip()
        if not name:
            return None
        try:
            chat_id_int = int(chat_id)
        except (TypeError, ValueError):
            return None
        cache_key = f"{chat_id_int}:{name}"
        cached = self._dm_topics.get(cache_key)
        if cached and not force_create:
            return str(cached)
        topic_conf: _adapter.Optional[_adapter.Dict[str, _adapter.Any]] = None
        chat_entry: _adapter.Optional[_adapter.Dict[str, _adapter.Any]] = None
        for entry in self._dm_topics_config:
            if str(entry.get("chat_id")) != str(chat_id_int):
                continue
            chat_entry = entry
            topic_conf = next((c for c in entry.get("topics", []) if c.get("name") == name), None)
            break
        if topic_conf and topic_conf.get("thread_id") and not force_create:
            thread_id = int(topic_conf["thread_id"])
            self._dm_topics[cache_key] = thread_id
            return str(thread_id)
        if chat_entry is None:
            chat_entry = {"chat_id": chat_id_int, "topics": []}
            self._dm_topics_config.append(chat_entry)
        if topic_conf is None:
            topic_conf = {"name": name}
            chat_entry.setdefault("topics", []).append(topic_conf)
        thread_id = await self._create_dm_topic(
            chat_id_int, name=name, icon_color=topic_conf.get("icon_color"), icon_custom_emoji_id=topic_conf.get("icon_custom_emoji_id"))
        if not thread_id:
            return None
        topic_conf["thread_id"] = thread_id
        self._dm_topics[cache_key] = int(thread_id)
        self._persist_dm_topic_thread_id(chat_id_int, name, int(thread_id), replace_existing=force_create)
        return str(thread_id)

    async def rename_dm_topic(self, chat_id: int, thread_id: int, name: str) -> None:
        """Rename a forum topic in a private (DM) chat."""
        from . import adapter as _adapter

        if not self._bot:
            return
        try:
            chat_id_arg = int(chat_id)
        except (TypeError, ValueError):
            chat_id_arg = chat_id
        await self._bot.edit_forum_topic(chat_id=chat_id_arg, message_thread_id=int(thread_id), name=name)
        _adapter.logger.info("[%s] Renamed DM topic in chat %s thread_id=%s -> '%s'", self.name, chat_id, thread_id, name)

    def _persist_dm_topic_thread_id(self, chat_id: int, topic_name: str, thread_id: int, replace_existing: bool = False) -> None:
        """Save a newly created thread_id back into config.yaml so it survives restarts."""
        from . import adapter as _adapter

        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                _adapter.logger.warning("[%s] Config file not found at %s, cannot persist thread_id", self.name, config_path)
                return
            from hermes_cli.config import atomic_config_write, read_user_config_raw
            config = read_user_config_raw(config_path)
            # platforms.telegram.extra.dm_topics — create the path for topics not predeclared in config.yaml.
            dm_topics = config.setdefault("platforms", {}).setdefault("telegram", {}).setdefault("extra", {}).setdefault("dm_topics", [])
            changed = False
            matching_chat_entry = None
            for chat_entry in dm_topics:
                try:
                    if int(chat_entry.get("chat_id", 0)) != int(chat_id):
                        continue
                except (TypeError, ValueError):
                    continue
                matching_chat_entry = chat_entry
                topics = chat_entry.setdefault("topics", [])
                t = next((t for t in topics if t.get("name") == topic_name), None)
                if t is None:
                    topics.append({"name": topic_name, "thread_id": thread_id})
                    changed = True
                elif (replace_existing or not t.get("thread_id")) and t.get("thread_id") != thread_id:
                    t["thread_id"] = thread_id
                    changed = True
                break
            if matching_chat_entry is None:
                dm_topics.append({"chat_id": chat_id, "topics": [{"name": topic_name, "thread_id": thread_id}]})
                changed = True
            if changed:
                atomic_config_write(config_path, config, default_flow_style=False, sort_keys=False)
                _adapter.logger.info("[%s] Persisted thread_id=%s for topic '%s' in config.yaml", self.name, thread_id, topic_name)
        except Exception as e:
            _adapter.logger.warning("[%s] Failed to persist thread_id to config: %s", self.name, e, exc_info=True)

    async def _setup_dm_topics(self) -> None:
        """Load or create configured DM topics: ``extra['dm_topics']`` is ``[{"chat_id", "topics": [{"name",
        "icon_color", "thread_id"?, "skill"?}]}]``; persisted thread_ids are cached without an API call."""
        from . import adapter as _adapter

        for chat_entry in self._dm_topics_config or ():
            chat_id = chat_entry.get("chat_id")
            topics = chat_entry.get("topics", [])
            if not chat_id or not topics:
                continue
            _adapter.logger.info("[%s] Setting up %d DM topic(s) for chat %s", self.name, len(topics), chat_id)
            for topic_conf in topics:
                topic_name = topic_conf.get("name")
                if not topic_name:
                    continue
                cache_key = f"{chat_id}:{topic_name}"
                existing_thread_id = topic_conf.get("thread_id")
                if existing_thread_id:
                    self._dm_topics[cache_key] = int(existing_thread_id)
                    _adapter.logger.info("[%s] DM topic loaded from config: %s -> thread_id=%s", self.name, cache_key, existing_thread_id)
                    continue
                thread_id = await self._create_dm_topic(
                    chat_id=_adapter.normalize_telegram_chat_id(chat_id), name=topic_name, icon_color=topic_conf.get("icon_color"),
                    icon_custom_emoji_id=topic_conf.get("icon_custom_emoji_id"))
                if not thread_id:
                    continue
                self._dm_topics[cache_key] = thread_id
                _adapter.logger.info("[%s] DM topic cached: %s -> thread_id=%s", self.name, cache_key, thread_id)
                self._persist_dm_topic_thread_id(int(chat_id), topic_name, thread_id)
                # Seed message: Telegram's client hides empty topics until they contain one.
                try:
                    await self._bot.send_message(
                        chat_id=_adapter.normalize_telegram_chat_id(chat_id), message_thread_id=thread_id, text=f"\U0001f4cc {topic_name}")
                except Exception as seed_err:
                    _adapter.logger.debug("[%s] Could not send seed message to topic '%s': %s", self.name, topic_name, seed_err)

    def _extra_bool(self, key: str, env_name: str, default: str, *fallback_keys: str) -> bool:
        """Boolean gate from ``config.extra[key]`` (then ``fallback_keys``), else env var."""
        from . import adapter as _adapter

        configured = self.config.extra.get(key)
        for alt in fallback_keys:
            if configured is None:
                configured = self.config.extra.get(alt)
        if configured is not None:
            if isinstance(configured, str):
                return configured.lower() in {"true", "1", "yes", "on"}
            return bool(configured)
        return _adapter.os.getenv(env_name, default).lower() in {"true", "1", "yes", "on"}

    def _extra_str_set(self, key: str, env_name: str) -> set[str]:
        """Comma/list allowlist from ``config.extra[key]``, else the profile-scoped env var."""
        from . import adapter as _adapter

        raw = self.config.extra.get(key)
        if raw is None:
            raw = _adapter._scoped_gate_env(env_name)
        if isinstance(raw, list):
            return {str(part).strip() for part in raw if str(part).strip()}
        return {part.strip() for part in str(raw).split(",") if part.strip()}

    def _telegram_require_mention(self) -> bool:
        """Return whether group chats should require an explicit bot trigger."""
        return self._extra_bool("require_mention", "TELEGRAM_REQUIRE_MENTION", "false")

    def _telegram_observe_unmentioned_group_messages(self) -> bool:
        """Store skipped unmentioned group messages as context (observe chatter, dispatch only when addressed)."""
        return self._extra_bool(
            "observe_unmentioned_group_messages", "TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES", "false",
            "ingest_unmentioned_group_messages")

    def _telegram_guest_mode(self) -> bool:
        """Return whether non-allowlisted groups may trigger via direct @mention."""
        return self._extra_bool("guest_mode", "TELEGRAM_GUEST_MODE", "false")

    def _telegram_exclusive_bot_mentions(self) -> bool:
        """Return whether explicit @...bot mentions exclusively route group messages."""
        return self._extra_bool("exclusive_bot_mentions", "TELEGRAM_EXCLUSIVE_BOT_MENTIONS", "true")

    def _telegram_free_response_chats(self) -> set[str]:
        return self._extra_str_set("free_response_chats", "TELEGRAM_FREE_RESPONSE_CHATS")

    def _telegram_free_response_topics(self) -> set[str]:
        """Topic-level free-response entries as ``<chat_id>:<thread_id>`` (General topic = ``1``)."""
        return self._extra_str_set("free_response_topics", "TELEGRAM_FREE_RESPONSE_TOPICS")

    def _telegram_is_free_response_topic(self, message: Message) -> bool:
        """True when the message's chat/topic pair is in ``free_response_topics``."""
        topics = self._telegram_free_response_topics()
        if not topics:
            return False
        chat_id = self._chat_id_str(message)
        if not chat_id:
            return False
        return f"{chat_id}:{self._topic_id_or_general(self._effective_message_thread_id(message))}" in topics

    def _telegram_allowed_chats(self) -> set[str]:
        """Group chat IDs the bot responds in (non-empty: others need ``guest_mode`` + @mention; DMs never
        filtered; empty = no restriction)."""
        return self._extra_str_set("allowed_chats", "TELEGRAM_ALLOWED_CHATS")

    def _telegram_group_allowed_chats(self) -> set[str]:
        """Return Telegram chats authorized at group scope."""
        return self._extra_str_set("group_allowed_chats", "TELEGRAM_GROUP_ALLOWED_CHATS")

    def _telegram_observe_allowed_chats(self) -> set[str]:
        """Chats where observed group context may use a shared source: ``group_allowed_chats`` ∩
        ``allowed_chats`` (when set)."""
        group_allowed = self._telegram_group_allowed_chats()
        if not group_allowed:
            return set()
        response_allowed = self._telegram_allowed_chats()
        return group_allowed & response_allowed if response_allowed else group_allowed

    def _telegram_allowed_topics(self) -> set[str]:
        """Forum topic IDs this bot handles (non-empty: other topics ignored; DMs never filtered; missing
        ``message_thread_id`` == General topic ``1``)."""
        return self._extra_str_set("allowed_topics", "TELEGRAM_ALLOWED_TOPICS")

    def _telegram_ignored_threads(self) -> set[int]:
        from . import adapter as _adapter

        raw = self.config.extra.get("ignored_threads")
        if raw is None:
            raw = _adapter._scoped_gate_env("TELEGRAM_IGNORED_THREADS")
        ignored: set[int] = set()
        for value in (raw if isinstance(raw, list) else str(raw).split(",")):
            text = str(value).strip()
            if not text:
                continue
            try:
                ignored.add(int(text))
            except (TypeError, ValueError):
                _adapter.logger.warning("[%s] Ignoring invalid Telegram thread id: %r", self.name, value)
        return ignored

    def _compile_mention_patterns(self) -> List[re.Pattern]:
        """Compile optional regex wake-word patterns for group triggers."""
        from . import adapter as _adapter

        patterns = self.config.extra.get("mention_patterns")
        if patterns is None:
            raw = _adapter.os.getenv("TELEGRAM_MENTION_PATTERNS", "").strip()
            if raw:
                try:
                    loaded = _adapter.json.loads(raw)
                except Exception:
                    loaded = [part.strip() for part in raw.splitlines() if part.strip()]
                    if not loaded:
                        loaded = [part.strip() for part in raw.split(",") if part.strip()]
                patterns = loaded
        if patterns is None:
            return []  # before touching ``self.name``: tests build bare adapters via object.__new__
        return _adapter.compile_mention_patterns(patterns, log_prefix=self.name, platform_label="telegram", display_label="Telegram", logger_=_adapter.logger)

    @staticmethod
    def _chat_type_str(chat) -> str:
        """PTB enum or plain-string ``chat.type`` → bare lowercase name (``supergroup``)."""
        from . import adapter as _adapter

        return str(getattr(chat, "type", "")).split(".")[-1].lower() if chat else ""

    @staticmethod
    def _chat_id_str(message) -> str:
        from . import adapter as _adapter

        return str(getattr(getattr(message, "chat", None), "id", ""))

    @classmethod
    def _topic_id_or_general(cls, thread_id) -> str:
        from . import adapter as _adapter

        return str(thread_id) if thread_id is not None else cls._GENERAL_TOPIC_THREAD_ID

    def _is_group_chat(self, message: Message) -> bool:
        from . import adapter as _adapter

        chat = getattr(message, "chat", None)
        return bool(chat) and self._chat_type_str(chat) in {"group", "supergroup"}

    @classmethod
    def _effective_message_thread_id(cls, message: Message) -> Optional[str]:
        """Routable thread id: forum General-topic messages arrive with ``message_thread_id=None`` but
        Telegram addresses that topic as ``1``; plain group/DM replies carry a reply-UI anchor that is NOT
        a routing id. Gating, skill binding and outbound routing must all agree on this value."""
        from . import adapter as _adapter

        chat = getattr(message, "chat", None)
        chat_type = cls._chat_type_str(chat)
        raw = getattr(message, "message_thread_id", None)
        is_topic_message = bool(getattr(message, "is_topic_message", False))
        is_group = chat_type in ("group", "supergroup")
        is_forum_group = is_group and getattr(chat, "is_forum", False) is True
        if raw is not None:
            if is_forum_group or (is_group and is_topic_message) or (chat_type == "private" and is_topic_message):
                return str(raw)
            return None
        return cls._GENERAL_TOPIC_THREAD_ID if is_forum_group else None

    def _current_bot_username(self) -> str:
        """This bot's live @username (lowercased, no ``@``): the last observed handle beats PTB's
        ``get_me()`` cache, which keeps a stale handle after a BotFather rename."""
        observed = getattr(self, "_bot_username_observed", None)
        if observed:
            return observed
        return (getattr(self._bot, "username", None) or "").lstrip("@").lower()

    def _note_bot_username(self, username: Optional[str]) -> None:
        """Record the bot's current @username, logging real renames."""
        from . import adapter as _adapter

        handle = (username or "").lstrip("@").lower()
        if not handle:
            return
        previous = getattr(self, "_bot_username_observed", None)
        if previous == handle:
            return
        self._bot_username_observed = handle
        self._bot_identity_checked_at = _adapter.time.monotonic()
        if previous:
            _adapter.logger.info(
                "[%s] Telegram bot username changed: @%s -> @%s (mention routing now follows the new handle)", self.name, previous, handle)

    def _observe_bot_identity_from_message(self, message: Message) -> None:
        """Learn our own handle from a message Telegram says we authored (own messages and
        ``reply_to_message``); only trusted when the user id matches this bot."""
        bot_id = getattr(self._bot, "id", None)
        if bot_id is None:
            return
        for candidate in (getattr(message, "from_user", None), getattr(getattr(message, "reply_to_message", None), "from_user", None)):
            if candidate is not None and getattr(candidate, "id", None) == bot_id:
                self._note_bot_username(getattr(candidate, "username", None))

    def _bot_identity_is_fresh(self) -> bool:
        """True when identity was re-read within the TTL. ``None`` (never checked) is always stale — do
        not fold it into ``0.0``: monotonic clocks have an arbitrary epoch."""
        from . import adapter as _adapter

        checked_at = getattr(self, "_bot_identity_checked_at", None)
        return checked_at is not None and (_adapter.time.monotonic() - checked_at) < self._BOT_IDENTITY_TTL_SECONDS

    async def _refresh_bot_identity(self, *, force: bool = False) -> None:
        """Re-read bot identity when the cache may be stale (``get_me()`` rewrites PTB's ``Bot._bot_user``
        in place). Best-effort: a failed probe keeps the last known handle."""
        from . import adapter as _adapter

        bot = self._bot
        if bot is None or not callable(getattr(bot, "get_me", None)):
            return
        if not force and self._bot_identity_is_fresh():
            return
        try:
            me = await _adapter.asyncio.wait_for(bot.get_me(), self._BOT_IDENTITY_PROBE_TIMEOUT)
        except _adapter.asyncio.CancelledError:
            raise
        except Exception as exc:
            _adapter.logger.debug(
                "[%s] Telegram identity refresh failed (keeping @%s): %s", self.name, self._current_bot_username() or "unknown", exc)
            return
        self._bot_identity_checked_at = _adapter.time.monotonic()
        self._note_bot_username(getattr(me, "username", None))

    def _is_reply_to_bot(self, message: Message) -> bool:
        from . import adapter as _adapter

        if not self._bot or not getattr(message, "reply_to_message", None):
            return False
        reply_user = getattr(message.reply_to_message, "from_user", None)
        return bool(reply_user and getattr(reply_user, "id", None) == getattr(self._bot, "id", None))

    @staticmethod
    def _entity_sources(message: Message):
        """``(text, entities)`` pairs for the message text and caption."""
        yield getattr(message, "text", None) or "", getattr(message, "entities", None) or []
        yield getattr(message, "caption", None) or "", getattr(message, "caption_entities", None) or []

    @staticmethod
    def _entity_type(entity) -> str:
        from . import adapter as _adapter

        return str(getattr(entity, "type", "")).split(".")[-1].lower()

    @classmethod
    def _entity_span(cls, source_text: str, entity) -> Optional[str]:
        """The entity's text, or None when its offsets are unusable."""
        # Telegram's official group-disambiguation form for slash commands (``/cmd@botname``) is emitted as
        # a single ``bot_command`` entity covering the whole span — there is no accompanying ``mention``
        # entity. Treat it as a direct address to this bot when the ``@botname`` suffix matches. This is the
        # form Telegram's own command menu autocomplete produces in groups, so dropping it at the mention
        # gate would break /new, /reset, /help, ... for every group that has ``require_mention`` enabled
        # (#15415).
        from . import adapter as _adapter

        offset = int(getattr(entity, "offset", -1))
        length = int(getattr(entity, "length", 0))
        if offset < 0 or length <= 0:
            return None
        return cls._telegram_entity_text(source_text, offset, length)

    @classmethod
    def _extract_bot_mention_usernames(cls, message: Message, self_username: str = "") -> set[str]:
        """Explicit bot usernames mentioned in text/captions: foreign handles count only when bot-shaped
        (``...bot``), ``self_username`` opts our OWN handle in regardless of shape. Entity mentions are
        authoritative; the raw-text fallback is deliberately narrow."""
        from . import adapter as _adapter

        mentioned_bot_usernames: set[str] = set()
        own = (self_username or "").lstrip("@").lower()

        def _is_bot_handle(handle: str) -> bool:
            if not handle:
                return False
            if own and handle == own:
                return True
            return bool(cls._FOREIGN_BOT_HANDLE_RE.fullmatch(handle))

        for source_text, entities in cls._entity_sources(message):
            for entity in entities:
                entity_type = cls._entity_type(entity)
                if entity_type not in {"mention", "bot_command"}:
                    continue
                entity_text = cls._entity_span(source_text, entity)
                if entity_text is None:
                    continue
                entity_text = entity_text.strip()
                if entity_type == "mention":
                    handle = entity_text.lstrip("@").lower()
                    if _is_bot_handle(handle):
                        mentioned_bot_usernames.add(handle)
                    continue
                # /cmd@botname is one bot_command entity; its suffix is an explicit bot address.
                at_index = entity_text.find("@")
                if at_index < 0:
                    continue
                command_target = entity_text[at_index + 1:].strip().lower()
                if _is_bot_handle(command_target):
                    mentioned_bot_usernames.add(command_target)
        # Entity-less fallback only: if Telegram supplied entities, trust them (no URL/code rescue).
        for raw_text, entities in cls._entity_sources(message):
            if not raw_text or entities:
                continue
            for match in _adapter.re.finditer(r"(?i)(?<![A-Za-z0-9_`/])@([A-Za-z0-9_]{2,31})\b", raw_text):
                handle = match.group(1).lower()
                if _is_bot_handle(handle):
                    mentioned_bot_usernames.add(handle)
        return mentioned_bot_usernames

    @staticmethod
    def _telegram_entity_text(source_text: str, offset: int, length: int) -> str:
        """Return a Telegram entity span using UTF-16 code-unit offsets."""
        if offset < 0 or length <= 0:
            return ""
        try:
            return source_text.encode("utf-16-le")[offset * 2:(offset + length) * 2].decode("utf-16-le")
        except UnicodeDecodeError:
            return ""

    def _message_mentions_bot(self, message: Message) -> bool:
        if not self._bot:
            return False
        bot_username = self._current_bot_username()
        bot_id = getattr(self._bot, "id", None)
        expected = f"@{bot_username}" if bot_username else None
        # Server-side MessageEntity values are authoritative: raw substrings like "foo@hermes_bot.example"
        # or handles inside URLs/code are not mentions.
        for source_text, entities in self._entity_sources(message):
            for entity in entities:
                entity_type = self._entity_type(entity)
                if entity_type == "mention" and expected:
                    span = self._entity_span(source_text, entity)
                    if span is not None and span.strip().lower() == expected:
                        return True
                elif entity_type == "text_mention":
                    user = getattr(entity, "user", None)
                    if user and getattr(user, "id", None) == bot_id:
                        return True
                elif entity_type == "bot_command" and expected:
                    # ``/cmd@botname`` (what the group command menu produces) must count as a direct address.
                    command_text = self._entity_span(source_text, entity)
                    if command_text is None:
                        continue
                    at_index = command_text.find("@")
                    if at_index >= 0 and command_text[at_index:].strip().lower() == expected:
                        return True
        if bot_username:
            return bot_username in self._extract_bot_mention_usernames(message, bot_username)
        return False

    def _schedule_bot_identity_recheck(self) -> None:
        """Fire a TTL-guarded identity refresh in the background when routing is about to discard a
        message naming other bots but not us (the symptom of a stale handle after a rename)."""
        from . import adapter as _adapter

        existing = getattr(self, "_bot_identity_refresh_task", None)
        if (existing is not None and not existing.done()) or self._bot_identity_is_fresh():
            return
        try:
            loop = _adapter.asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._refresh_bot_identity())
        self._bot_identity_refresh_task = task
        tracked = getattr(self, "_background_tasks", None)
        if isinstance(tracked, set):
            tracked.add(task)
            task.add_done_callback(tracked.discard)

    def _explicit_bot_mentions_exclude_self(self, message: Message) -> bool:
        """True when explicit bot handles target other bots, not this one (``@bot3 hi @bot4`` must not
        wake ``@bot1`` via reply/wake-word fallbacks)."""
        from . import adapter as _adapter

        if not self._bot:
            return False
        bot_username = self._current_bot_username()
        if not bot_username:
            return False
        mentioned_bot_usernames = self._extract_bot_mention_usernames(message, bot_username)
        excludes_self = bool(mentioned_bot_usernames) and bot_username not in mentioned_bot_usernames
        if excludes_self:
            # Either truly for another bot, or our handle is stale after a rename — re-check out of band.
            self._schedule_bot_identity_recheck()
        return excludes_self

    def _message_matches_mention_patterns(self, message: Message) -> bool:
        if not self._mention_patterns:
            return False
        return any(
            pattern.search(candidate)
            for candidate in (getattr(message, "text", None), getattr(message, "caption", None)) if candidate
            for pattern in self._mention_patterns)

    def _is_guest_mention(self, message: Message) -> bool:
        """Guest-mode bypass: explicit bot mention (caller already verified group chat)."""
        return self._telegram_guest_mode() and self._message_mentions_bot(message)

    def _clean_bot_trigger_text(self, text: Optional[str]) -> Optional[str]:
        from . import adapter as _adapter

        bot_username = self._current_bot_username()
        if not text or not bot_username:
            return text
        cleaned = _adapter.re.sub(rf"(?i)@{_adapter.re.escape(bot_username)}\b[,:\-]*\s*", "", text).strip()
        return cleaned or text

    def _topic_gates_pass(self, thread_id, *, warn_non_numeric: bool) -> Optional[bool]:
        """``allowed_topics`` / ``ignored_threads`` gates; False = blocked, None = undecided."""
        from . import adapter as _adapter

        allowed_topics = self._telegram_allowed_topics()
        if allowed_topics and self._topic_id_or_general(thread_id) not in allowed_topics:
            return False
        if thread_id is not None:
            try:
                if int(thread_id) in self._telegram_ignored_threads():
                    return False
            except (TypeError, ValueError):
                if not warn_non_numeric:
                    return False
                _adapter.logger.warning("[%s] Ignoring non-numeric Telegram message_thread_id: %r", self.name, thread_id)
        return None

    def _should_observe_unmentioned_group_message(self, message: Message) -> bool:
        """Return True when a group message should be stored but not dispatched."""
        if self._is_own_message(message) or not self._telegram_observe_unmentioned_group_messages() or not self._is_group_chat(message):
            return False
        if self._topic_gates_pass(getattr(message, "message_thread_id", None), warn_non_numeric=False) is False:
            return False
        chat_id_str = self._chat_id_str(message)
        if self._telegram_exclusive_bot_mentions() and self._explicit_bot_mentions_exclude_self(message):
            return False
        # Observed context is shared at chat/topic scope, so require an explicit chat allowlist.
        allowed = self._telegram_observe_allowed_chats()
        if not allowed or chat_id_str not in allowed:
            return False
        # Only observe messages the require_mention gate would skip.
        if chat_id_str in self._telegram_free_response_chats() or self._telegram_is_free_response_topic(message):
            return False
        if not self._telegram_require_mention() or self._is_reply_to_bot(message) or self._message_mentions_bot(message):
            return False
        return not self._message_matches_mention_patterns(message)

    def _telegram_group_observe_shared_source(self, source):
        """Return a chat/topic-scoped source for observed Telegram group context."""
        from . import adapter as _adapter

        return _adapter.dataclasses.replace(source, user_id=None, user_name=None, user_id_alt=None)

    def _telegram_group_observe_attributed_text(self, event: MessageEvent) -> str:
        user_id = event.source.user_id or "unknown"
        return f"[{event.source.user_name or user_id}|{user_id}]\n{event.text or ''}"

    def _telegram_group_observe_channel_prompt(self) -> str:
        username = self._current_bot_username() or "unknown"
        bot_id = getattr(getattr(self, "_bot", None), "id", None) or "unknown"
        return (
            "You are handling a Telegram group chat message.\n"
            f"- Your identity: user_id={bot_id}, @-mention name in this group=@{username}\n"
            "- observed Telegram group context may be provided in a separate context-only block "
            "before the current message; it is not necessarily addressed to you.\n"
            "- Treat only the current new message as a request explicitly directed at you, "
            "and use observed context only when the current message asks for it.")

    def _apply_telegram_group_observe_attribution(self, event: MessageEvent) -> MessageEvent:
        """Align triggered group turns with observed-history attribution."""
        from . import adapter as _adapter

        if not self._telegram_observe_unmentioned_group_messages():
            return event
        raw_message = getattr(event, "raw_message", None)
        if not raw_message or not self._is_group_chat(raw_message):
            return event
        allowed = self._telegram_observe_allowed_chats()
        if not allowed or self._chat_id_str(raw_message) not in allowed:
            return event
        observe_prompt = self._telegram_group_observe_channel_prompt()
        channel_prompt = f"{event.channel_prompt}\n\n{observe_prompt}" if event.channel_prompt else observe_prompt
        if event.message_type == _adapter.MessageType.COMMAND:
            # Commands keep the original source (user_id) so _check_slash_access can identify the sender.
            return _adapter.dataclasses.replace(event, channel_prompt=channel_prompt)
        return _adapter.dataclasses.replace(
            event, text=self._telegram_group_observe_attributed_text(event),
            source=self._telegram_group_observe_shared_source(event.source), channel_prompt=channel_prompt)
