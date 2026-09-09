"""Slack lifecycle methods; SDK and mutable dependencies remain on the facade."""

import asyncio
from typing import Any, Callable, Dict, Optional, Tuple
try:
    from slack_bolt.async_app import AsyncApp
    from slack_sdk.web.async_client import AsyncWebClient
except ImportError:
    AsyncApp = AsyncWebClient = Any


class SlackLifecycleMixin:
    async def _close_workspace_clients(self) -> None:
        """Close any Slack SDK clients that may own aiohttp sessions."""
        from . import adapter as _adapter

        primary_client = getattr(self._app, "client", None) if self._app is not None else None
        clients = ([primary_client] if primary_client is not None else []) + list(
            self._team_clients.values())
        seen_ids: set[int] = set()
        for client in clients:
            if id(client) in seen_ids:
                continue
            seen_ids.add(id(client))
            for method_name in ("close", "aclose"):
                closer = getattr(client, method_name, None)
                if not callable(closer):
                    continue
                result = closer()
                if _adapter.inspect.isawaitable(result):
                    await result
                break

    @staticmethod
    def _slack_timestamp_sort_key(ts: Any) -> Tuple[int, int, str]:
        """Chronological, deterministic sort key for bare ts strings or ``(team_id, ts)`` markers."""
        from . import adapter as _adapter

        if isinstance(ts, tuple) and len(ts) == 2:
            ts = ts[1]
        seconds, _, fraction = str(ts).partition(".")
        return _adapter._int_or_zero(seconds), _adapter._int_or_zero((fraction + "000000")[:6] or "0"), str(ts)

    @classmethod
    def _discard_oldest_by_thread_ts(
        cls, entries: Any, count: int, ts_getter: Callable[[Any], Any] = lambda e: e) -> None:
        """Discard the *count* entries (set or dict keys) with the oldest embedded Slack ts.
        Sets iterate in arbitrary order, so ``list(entries)[:count]`` could evict the most ACTIVE
        entry; sort chronologically by the embedded ts instead.

        For bounded tracking sets whose members are keys CONTAINING a Slack timestamp (tuples or
        colon-joined strings) rather than bare ts values. See #51019.
        """
        if count <= 0:
            return
        oldest = sorted(entries, key=lambda e: cls._slack_timestamp_sort_key(ts_getter(e)))[:count]
        remove = entries.discard if isinstance(entries, set) else entries.pop
        for entry in oldest:
            remove(entry)

    def _evict_oldest_by_ts(
        self, entries: Any, cap: int, ts_getter: Callable[[Any], Any] = lambda e: e) -> None:
        """Once ``entries`` exceeds ``cap``, drop oldest-ts-first down to half the cap."""
        if len(entries) > cap:
            self._discard_oldest_by_thread_ts(entries, len(entries) - cap // 2, ts_getter)

    def _trim_bot_message_timestamps(self) -> None:
        self._evict_oldest_by_ts(self._bot_message_ts, self._BOT_TS_MAX)

    def _trim_mentioned_threads(self) -> None:
        if len(self._mentioned_threads) > self._MENTIONED_THREADS_MAX:
            # Keys are "team:channel:thread_ts[:user]" — evict the oldest threads first. Evicting an ACTIVE
            # thread's key would re-run its rehydration check and re-inject the missed delta (#51019-style
            # arbitrary eviction), so never pop in set order.
            self._discard_oldest_by_thread_ts(
                self._mentioned_threads, self._MENTIONED_THREADS_MAX // 2)

    @staticmethod
    def _trim_oldest_dict_entries(mapping: Dict[Any, Any], max_size: int) -> None:
        """Evict oldest-inserted entries down to half the cap once *mapping* exceeds *max_size*.
        Dict insertion order makes ``list(mapping)[:excess]`` truly oldest-first (sets would not
        be); halving amortizes eviction like the sibling caches.

        Evicts down to half the cap so eviction runs amortized-once per max_size//2 writes, matching the
        sibling tracking structures. See #51019.
        """
        if len(mapping) <= max_size:
            return
        excess = len(mapping) - max_size // 2
        for old_key in list(mapping)[:excess]:
            del mapping[old_key]

    def _lazy_attr(self, name: str, factory: Callable[[], Any]) -> Any:
        """``self.<name>``, created via ``factory`` when missing/None (object.__new__ test doubles
        never ran ``__init__``)."""
        value = getattr(self, name, None)
        if value is None:
            value = factory()
            setattr(self, name, value)
        return value

    def _remember_channel_team(self, channel_id: str, team_id: str) -> None:
        """Record which workspace owns *channel_id* (bounded oldest-first). Channel ids are
        workspace-local so one id CAN appear twice; the unqualified fallback is kept only while
        unambiguous. Explicit outbound team_id remains authoritative."""
        if not channel_id or not team_id:
            return
        channel_id = str(channel_id)
        team_id = str(team_id)
        channel_teams = self._lazy_attr("_channel_teams", dict)
        teams = channel_teams.setdefault(channel_id, set())
        teams.add(team_id)
        if len(teams) == 1:
            self._channel_team[channel_id] = team_id
        else:
            self._channel_team.pop(channel_id, None)
        self._trim_oldest_dict_entries(self._channel_team, self._CHANNEL_TEAM_MAX)
        self._trim_oldest_dict_entries(self._channel_teams, self._CHANNEL_TEAM_MAX)

    def _start_socket_mode_handler(self) -> None:
        """Start the Slack Socket Mode background task."""
        from . import adapter as _adapter

        if not self._app or not self._app_token:
            raise RuntimeError("Socket Mode requires an initialized app and app token")
        self._handler = _adapter.AsyncSocketModeHandler(self._app, self._app_token, proxy=self._proxy_url)
        _adapter._apply_slack_proxy(self._handler.client, self._proxy_url)
        task = _adapter.asyncio.create_task(self._handler.start_async())
        self._socket_mode_task = task
        self._socket_handler_started_monotonic = _adapter.time.monotonic()
        task.add_done_callback(self._on_socket_mode_task_done)

    async def _stop_socket_mode_handler(self) -> None:
        """Stop Socket Mode handler and task. Order matters: ``SocketModeClient.connect()`` is a
        ``while True`` retry loop that never checks ``closed``, so anything inside it when
        ``close_async()`` drops the session retries forever. Cancel every task that can reach
        ``connect()`` BEFORE closing (it rebinds task attrs on success, so a mid-close snapshot
        races a moving target).

        Everything that can reach ``connect()`` therefore has to be stopped first.
        ``monitor_current_session()`` and ``receive_messages()`` each get there on their own, and
        ``connect()`` rebinds the client's task attributes on success, so the set of live tasks changes
        across the awaits inside ``close()``. Cancelling from a snapshot taken partway through that would
        race a moving target. See slackapi/python-slack-sdk#1913.
        """
        from . import adapter as _adapter

        handler, task = self._handler, self._socket_mode_task
        self._handler = self._socket_mode_task = None
        client = getattr(handler, "client", None)
        await _adapter._cancel_socket_tasks(
            [task] + [getattr(client, attr, None) for attr in _adapter._SOCKET_CLIENT_TASK_ATTRS])
        if handler is not None:
            try:
                await handler.close_async()
            except Exception as e:  # pragma: no cover - defensive logging
                _adapter.logger.warning(
                    "[Slack] Error while closing Socket Mode handler: %s", e, exc_info=True)

    async def _socket_transport_connected(self) -> Optional[bool]:
        """Best-effort check of current Socket Mode transport state."""
        from . import adapter as _adapter

        state = getattr(getattr(self._handler, "client", None), "is_connected", None)
        if state is None:
            return None
        try:
            value = state() if callable(state) else state
            if _adapter.asyncio.iscoroutine(value):
                value = await value
            return bool(value)
        except Exception:  # pragma: no cover - optional client API
            _adapter.logger.debug("[Slack] Could not inspect Socket Mode transport state", exc_info=True)
            return None

    def _socket_ping_pong_stale(self) -> bool:
        """No recent ping/pong on the transport. Slack pings every ``ping_interval`` even when idle,
        and a client stuck on a closed session can still report ``is_connected()``, so staleness is
        the reliable "wedged" signal. Non-numeric attrs (mocked clients) never reconnect."""
        from . import adapter as _adapter

        client = getattr(self._handler, "client", None)
        if client is None:
            return False
        ping_interval = getattr(client, "ping_interval", None)
        if not isinstance(ping_interval, (int, float)) or ping_interval <= 0:
            return False
        last = getattr(client, "last_ping_pong_time", None)
        if last is None:
            # No ping yet: healthy right after (re)connect until the grace window elapses.
            started = self._socket_handler_started_monotonic
            grace = max(self._socket_first_ping_grace_s, ping_interval * 2)
            return started is not None and (_adapter.time.monotonic() - started) > grace
        if not isinstance(last, (int, float)):
            return False
        return (_adapter.time.time() - last) > (ping_interval * self._socket_ping_stale_factor)

    async def _restart_socket_mode(self, reason: str) -> None:
        """Reconnect Socket Mode without rebuilding adapter state."""
        from . import adapter as _adapter

        if not self._running:
            return
        async with self._socket_reconnect_lock:
            if not self._running or not self._app or not self._app_token:
                return
            _adapter.logger.warning("[Slack] Socket Mode unhealthy (%s); reconnecting", reason)
            await self._stop_socket_mode_handler()
            try:
                self._start_socket_mode_handler()
            except Exception as exc:  # pragma: no cover - defensive logging
                _adapter.logger.error("[Slack] Socket Mode reconnect failed: %s", exc, exc_info=True)

    async def _socket_watchdog_loop(self) -> None:
        """Monitor Socket Mode and reconnect if the task/transport dies.
        Broad except so a transient probe/restart bug can't kill self-healing."""
        from . import adapter as _adapter

        while self._running:
            try:
                await _adapter.asyncio.sleep(self._socket_watchdog_interval_s)
                if not self._running:
                    break
                task = self._socket_mode_task
                if task is None:
                    await self._restart_socket_mode("socket task missing")
                    continue
                if task.done():
                    await self._restart_socket_mode("socket task stopped")
                    continue
                connected = await self._socket_transport_connected()
                if connected is False:
                    await self._restart_socket_mode("transport disconnected")
                elif self._socket_ping_pong_stale():
                    # is_connected() can lie on a closed session; staleness catches the zombie.
                    await self._restart_socket_mode("ping/pong stale")
            except _adapter.asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive logging
                _adapter.logger.warning(
                    "[Slack] Socket Mode watchdog iteration failed; continuing", exc_info=True)

    def _on_socket_watchdog_done(self, task: asyncio.Task) -> None:
        from . import adapter as _adapter

        if task is not self._socket_watchdog_task:
            return
        if task.cancelled() or not self._running:
            return
        try:
            exc = task.exception()
        except (_adapter.asyncio.CancelledError, Exception):  # pragma: no cover
            exc = None
        if exc is not None:
            _adapter.logger.warning(
                "[Slack] Socket Mode watchdog exited with error; restarting: %s", exc, exc_info=True
            )
        else:
            _adapter.logger.warning("[Slack] Socket Mode watchdog exited; restarting")
        self._socket_watchdog_task = None
        self._ensure_socket_watchdog()

    async def _cancel_socket_watchdog(self, failure_msg: str) -> None:
        """Cancel and await the watchdog task (if any); exceptions are debug-logged."""
        from . import adapter as _adapter

        watchdog_task = self._socket_watchdog_task
        self._socket_watchdog_task = None
        if watchdog_task is None or watchdog_task.done():
            return
        watchdog_task.cancel()
        try:
            await watchdog_task
        except _adapter.asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover - defensive logging
            _adapter.logger.debug(failure_msg, exc_info=True)

    def _ensure_socket_watchdog(self) -> None:
        from . import adapter as _adapter

        if self._socket_watchdog_task is None or self._socket_watchdog_task.done():
            task = _adapter.asyncio.create_task(self._socket_watchdog_loop())
            self._socket_watchdog_task = task
            task.add_done_callback(self._on_socket_watchdog_done)

    def _on_socket_mode_task_done(self, task: asyncio.Task) -> None:
        # Ignore stale tasks from intentional reconnect/shutdown.
        from . import adapter as _adapter

        if task is not self._socket_mode_task or task.cancelled() or not self._running:
            return
        exc = None
        try:
            exc = task.exception()
        except _adapter.asyncio.CancelledError:
            return
        except Exception:  # pragma: no cover - defensive logging
            _adapter.logger.debug("[Slack] Could not inspect Socket Mode task exception", exc_info=True)
        if exc is not None:
            _adapter.logger.warning("[Slack] Socket Mode task exited with error: %s", exc, exc_info=True)
        else:
            _adapter.logger.warning("[Slack] Socket Mode task exited unexpectedly")
        try:
            loop = _adapter.asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._restart_socket_mode("socket task exited"))

    def _describe_slack_api_error(
        self, response: Any, *, file_obj: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Convert Slack API auth/permission failures into actionable user-facing text."""
        from . import adapter as _adapter

        if response is None or not hasattr(response, "get"):
            return None
        error = str(response.get("error", "") or "").strip()
        if not error:
            return None
        file_label = _adapter._attachment_label(file_obj)
        if error == "missing_scope":
            needed = str(response.get("needed", "") or "").strip()
            provided = str(response.get("provided", "") or "").strip()
            needed_hint = f"Missing scope: {needed}." if needed else "Missing required Slack scope."
            provided_hint = f" Current bot scopes: {provided}." if provided else ""
            return (
                f"Slack attachment access failed for {file_label}. {needed_hint}{provided_hint}"
                " Update the Slack app scopes/settings and reinstall the app to the workspace.")
        for codes, template in _adapter._SLACK_API_ERROR_TEMPLATES:
            if error in codes:
                return template.format(file_label=file_label, error=error)
        return None

    def _describe_slack_download_failure(
        self, exc: Exception, *, file_obj: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Translate Slack download exceptions into user-facing attachment diagnostics."""
        from . import adapter as _adapter

        file_label = _adapter._attachment_label(file_obj)
        response = getattr(exc, "response", None)
        api_detail = self._describe_slack_api_error(response, file_obj=file_obj)
        if api_detail:
            return api_detail
        try:
            import httpx
        except Exception:  # pragma: no cover
            httpx = None
        if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
            template = _adapter._SLACK_HTTP_STATUS_TEMPLATES.get(exc.response.status_code)
            if template:
                return template.format(file_label=file_label)
        message = str(exc)
        if "Slack returned HTML instead of media" in message or "non-image data" in message:
            return (
                f"Slack attachment access failed for {file_label}: Slack returned an HTML/login or non-media response. "
                "This usually means a scope, auth, or file-permission problem.")
        return None

    def _warn_if_missing_group_dm_scopes(self, auth_response, team_name: str) -> None:
        """Nudge a reinstall when group-DM scopes are absent: a missing ``message.mpim`` event
        delivers *nothing* (no runtime error), so ``auth.test``'s ``x-oauth-scopes`` header at
        connect time is the only detection point."""
        from . import adapter as _adapter

        try:
            # Warn once per team per process, not on every reconnect.
            warned = self._lazy_attr("_group_dm_scope_warned", set)
            headers = getattr(auth_response, "headers", None) or {}
            raw = headers.get("x-oauth-scopes") or headers.get("X-OAuth-Scopes") or ""
            if not raw:
                return  # Header absent (e.g. some proxies) — don't guess.
            granted = {s.strip() for s in raw.split(",") if s.strip()}
            team_key = team_name or ""
            # im:history without mpim:history == stale DM-capable manifest.
            if team_key not in warned and "im:history" in granted and "mpim:history" not in granted:
                warned.add(team_key)
                _adapter.logger.warning(
                    "[Slack] Group DMs (multi-person DMs) will not work in workspace %s: the app "
                    "is missing the 'mpim:history' scope and 'message.mpim' event. Add "
                    "'mpim:history' (and 'mpim:read') to bot scopes, add 'message.mpim' to event "
                    "subscriptions, then REINSTALL the app to the workspace. Regenerating the app "
                    "from `hermes slack` produces a manifest with these already included.",
                    team_key or "this workspace")
        except Exception:  # pragma: no cover - diagnostics must never break connect
            pass

    def _warn_if_not_bot_token(self, auth_response, team_name: str) -> None:
        """Warn once per workspace when the token authenticates as a human: ``auth.test`` on an
        ``xoxp-`` token returns the installer's ``user_id`` and no ``bot_id``, so mentions OF THAT
        PERSON become bot mentions. No runtime error exists; warn only (user tokens still work)."""
        from . import adapter as _adapter

        try:
            warned = self._lazy_attr("_user_token_warned", set)
            team_key = team_name or ""
            if team_key in warned:
                return
            # bot_id present only for bot tokens; absent + resolved user_id == user token.
            try:
                bot_id = auth_response.get("bot_id", "") or ""
                user_id = auth_response.get("user_id", "") or ""
            except Exception:
                # Attribute-only response shapes: fall back to .data.
                data = getattr(auth_response, "data", None) or {}
                bot_id = data.get("bot_id", "") or ""
                user_id = data.get("user_id", "") or ""
            if not user_id:
                return  # Nothing resolved — don't guess.
            if not bot_id:
                warned.add(team_key)
                _adapter.logger.warning(
                    "[Slack] The configured Slack token for workspace %s authenticated as a USER "
                    "(member %s), not a bot — the auth.test response has no 'bot_id'. This is "
                    "almost certainly a user token (xoxp-...) instead of a Bot User OAuth Token "
                    "(xoxb-...). The bot's identity is now bound to that member's ID, so mentions "
                    "OF THAT PERSON will be misrouted as mentions of the bot (the bot replies to "
                    "messages merely addressed to them). Use the 'Bot User OAuth Token' "
                    "(xoxb-...) from your Slack app's 'OAuth & Permissions' page in "
                    "SLACK_BOT_TOKEN.", team_key or "this workspace", user_id)
        except Exception:  # pragma: no cover - diagnostics must never break connect
            pass

    def _register_bolt_handlers(self) -> None:
        """Wire every Bolt listener onto ``self._app``; must run before Socket Mode starts."""
        # Bolt injects listener args by NAME (None for unknown), so every handler takes
        # (event, say, body). message + app_mention share an event ts, so the deduplicator drops
        # the second. file_created/file_change are acked (no-op) to avoid "unhandled request" noise.
        from . import adapter as _adapter

        async def _noop(event, body):
            return None

        def _reaction(removed: bool):
            async def _handler(event, body):
                await self._handle_slack_reaction(event, removed=removed)

            return _handler

        def _listener_for(handler):
            async def _listener(event, say, body):
                await handler(event, body)

            return _listener

        for event_type, handler in (
            ("message", self._handle_slack_message), ("app_mention", self._handle_slack_message),
            ("app_home_opened", self._handle_app_home_opened),
            ("app_context_changed", self._handle_app_context_changed),
            ("file_shared", self._handle_slack_file_shared), ("file_created", _noop),
            ("file_change", _noop), ("reaction_added", _reaction(False)),
            ("reaction_removed", _reaction(True)),
            ("assistant_thread_started", self._handle_assistant_thread_lifecycle_event),
            ("assistant_thread_context_changed", self._handle_assistant_thread_lifecycle_event)):
            self._app.event(event_type)(_listener_for(handler))
        # Catch-all ack: unacked envelopes count as failures and past 95%/60-min Slack disables
        # Event Subscriptions (ALL inbound). Registered AFTER all named handlers (first match wins).
        # Catch-all no-op ack for any other subscribed event type that Hermes has no listener for (e.g.
        # user_change, user_huddle_changed, member_joined_channel, channel_archive, pin_added, etc.). Two
        # reasons this must exist (issues #6572 and the Event Subscriptions auto-disable failure mode): 1.
        # Correctness at scale: without a matching listener, slack-bolt returns HTTP 404 for every unhandled
        # event envelope and never sends the Socket Mode ack. When the app is subscribed to high-volume
        # events (user_change fires on every presence/status change for the whole org), the flood of
        # un-acked 404s pushes Slack's failure rate past its 95%/60-min threshold and Slack auto-disables
        # the app's Event Subscriptions — silently killing ALL inbound delivery until manually re-enabled.
        # 2. Noise: each unhandled envelope also logs a slack_bolt "Unhandled request" WARNING, flooding
        # gateway logs in busy channels. Registered AFTER every named handler: bolt dispatches to the first
        # matching listener, so the named handlers above always win and this only fires for truly unhandled
        # types. The envelope is acked with 200, keeping the failure rate near 0% regardless of which events
        # the Slack app manifest subscribes to. A debug line preserves visibility into unknown event types
        # without per-message WARNING noise.
        @self._app.event(_adapter.re.compile(r".*"))
        async def handle_unhandled_event(event, body, logger):
            logger.debug(
                "[Slack] Ignoring unhandled event type=%s (no listener registered; subscribed "
                "events not handled by Hermes can be removed from the Slack app manifest via "
                "`hermes slack manifest`)",
                (event or {}).get("type", (body or {}).get("event", {}).get("type", "unknown")))

        # Every COMMAND_REGISTRY command is a native slash via one regex matcher. Commands must
        # ALSO be declared in the app manifest (`hermes slack manifest`): Socket Mode won't
        # deliver undeclared commands at all.
        from hermes_cli.commands_platforms import slack_native_slashes
        _slash_names = [name for name, _d, _h in slack_native_slashes()]
        if _slash_names:
            _slash_pattern = _adapter.re.compile(
                r"^/(?:" + "|".join(_adapter.re.escape(n) for n in _slash_names) + r")$")
        else:  # pragma: no cover - registry always non-empty
            _slash_pattern = _adapter.re.compile(r"^/hermes$")

        @self._app.command(_slash_pattern)
        async def handle_hermes_command(ack, command):
            slash = (command.get("command") or "").lstrip("/")
            await ack(response_type="ephemeral", text=f"Running `/{slash}`…")
            await self._handle_slash_command(command)

        # Approval buttons, slash-confirm buttons (tools/slash_confirm.py), feedback.
        for _action_id in self._APPROVAL_CHOICES:
            self._app.action(_action_id)(self._handle_approval_action)
        for _action_id in self._CONFIRM_CHOICES:
            self._app.action(_action_id)(self._handle_slash_confirm_action)
        self._app.action("hermes_feedback")(self._handle_feedback_action)
        # Clarify buttons (tools/clarify_gateway.py); indexed action IDs because
        # Block Kit requires unique IDs within an actions block.
        self._app.action(_adapter.re.compile(r"^hermes_clarify_choice_\d+$"))(self._handle_clarify_action)
        self._app.action("hermes_clarify_other")(self._handle_clarify_action)
        # Register Block Kit action handlers for the model picker
        # (provider/model static_select + Back/Cancel buttons).
        for _action_id in _adapter._MODEL_PICKER_ACTION_IDS:
            self._app.action(_action_id)(self._handle_model_picker_action)
        self._register_plugin_action_handlers()
        # ctx.register_platform_handler("slack", ...) factories get the full
        # AsyncApp surface (event/action/command), wired before Socket Mode starts.
        self._wire_plugin_handlers(self._app)

    def _register_plugin_action_handlers(self) -> None:
        """Wire ``ctx.register_slack_action_handler`` callbacks; each is wrapped so a plugin
        exception is logged and slack_bolt still sees a clean ack."""
        from . import adapter as _adapter

        try:
            from hermes_cli.plugins import get_plugin_manager
            _plugin_handlers = get_plugin_manager().get_slack_action_handlers()
        except Exception as e:  # pragma: no cover - defensive
            _adapter.logger.warning("[Slack] Could not load plugin action handlers: %s", e)
            _plugin_handlers = []
        # Closure factory: slack_bolt passes ``None`` for unrecognised listener params, so loop
        # vars captured as default args (``_cb=_cb``) would be silently clobbered at dispatch.
        def _make_wrapper(cb, plugin_name):
            async def _wrapped(ack, body, action):
                try:
                    await cb(ack, body, action)
                except Exception as exc:  # pragma: no cover - defensive
                    _adapter.logger.error(
                        "[Slack] Plugin '%s' action handler raised: %s", plugin_name, exc,
                        exc_info=True)
                    # Best-effort ack so Slack doesn't retry the click.
                    try:
                        await ack()
                    except Exception:
                        pass

            return _wrapped

        for _action_id, _cb, _plugin_name in _plugin_handlers:
            self._app.action(_action_id)(_make_wrapper(_cb, _plugin_name))
            _adapter.logger.debug(
                "[Slack] Registered plugin action handler %s (from %s)", _action_id, _plugin_name)
        if _plugin_handlers:
            _adapter.logger.info("[Slack] Wired %d plugin action handler(s)", len(_plugin_handlers))

    @staticmethod
    def _new_web_client(token: str, proxy_url: Optional[str]) -> Any:
        from . import adapter as _adapter

        client = _adapter.AsyncWebClient(token=token, user_agent_prefix=_adapter._HERMES_SLACK_USER_AGENT_PREFIX)
        _adapter._apply_slack_proxy(client, proxy_url)
        return client

    async def _authenticate_workspace(self, token: str, proxy_url: Optional[str]) -> None:
        """``auth.test`` one bot token and register its workspace client/identity.
        The first token wins as primary identity (cleared before reconnect)."""
        from . import adapter as _adapter

        client = self._new_web_client(token, proxy_url)
        auth_response = await client.auth_test()
        team_id = auth_response.get("team_id", "")
        bot_user_id = auth_response.get("user_id", "")
        bot_name = auth_response.get("user", "unknown")
        team_name = auth_response.get("team", "unknown")
        self._team_clients[team_id] = client
        self._team_bot_user_ids[team_id] = bot_user_id
        self._team_bot_names[team_id] = bot_name
        if self._bot_user_id is None:
            self._bot_user_id = bot_user_id
        if self._bot_display_name is None:
            self._bot_display_name = bot_name
        _adapter.logger.info(
            "[Slack] Authenticated as @%s in workspace %s (team: %s)", bot_name, team_name, team_id)
        self._warn_if_missing_group_dm_scopes(auth_response, team_name)
        self._warn_if_not_bot_token(auth_response, team_name)
        self._warn_if_inchannel_without_flat_reply(team_name)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to Slack via Socket Mode."""
        from . import adapter as _adapter

        if not _adapter.SLACK_AVAILABLE:
            _adapter.logger.error("[Slack] slack-bolt not installed. Run: pip install slack-bolt")
            self._set_fatal_error("missing_dependency", "slack-bolt not installed", retryable=False)
            return False
        raw_token = self.config.token
        # Scoped secret is authoritative; only an UNSCOPED read falls back to
        # process env, else a secondary profile inherits the default's app.
        try:
            # Multiplex: profile secrets live in the secret scope, not process os.environ. When a scope is
            # installed (secondary-profile connect), it is AUTHORITATIVE — do not fall through to os.getenv,
            # or a secondary profile missing SLACK_APP_TOKEN silently inherits the default profile's Socket
            # Mode app (#59739). Only an UNSCOPED read under multiplex (default-profile startup loop,
            # background reconnect rebuild) falls back to process env, which is that profile's own.
            app_token = _adapter.get_secret("SLACK_APP_TOKEN")
        except _adapter.UnscopedSecretError:
            app_token = _adapter.os.getenv("SLACK_APP_TOKEN")
        for env_name, value in (("SLACK_BOT_TOKEN", raw_token), ("SLACK_APP_TOKEN", app_token)):
            if not value:
                self._fatal_missing_env(env_name)
                return False
        proxy_url = _adapter._resolve_slack_proxy_url()
        if proxy_url:
            _adapter.logger.info("[Slack] Using proxy for Slack transport: %s", _adapter.safe_url_for_log(proxy_url))
        bot_tokens = _adapter._load_slack_bot_tokens(raw_token, quiet=False)
        lock_acquired = False
        try:
            if not self._acquire_platform_lock("slack-app-token", app_token, "Slack app token"):
                return False
            lock_acquired = True
            self._running = False
            # Cancel AND await the old watchdog so it can't see _running=False,
            # exit, and leave no monitor behind.
            await self._cancel_socket_watchdog("[Slack] Prior watchdog task failed while stopping")
            # A zombie Socket Mode handler would double-respond to every event.
            await self._stop_socket_mode_handler()
            await self._close_workspace_clients()
            # Close any previous handler before creating a new one so that calling connect() a second time
            # (e.g. during a gateway restart or in-process reconnect attempt) does not leave a zombie Socket
            # Mode connection alive. Both the old and new connections would otherwise receive every Slack
            # event and dispatch it twice, producing double responses — the same bug that affected
            # DiscordAdapter (#18187).
            self._app = None
            self._app_token = app_token
            self._proxy_url = proxy_url
            # Reset so a reconnect with dropped/rotated tokens carries no stale identities.
            self._bot_user_id = self._bot_display_name = None
            self._team_clients, self._team_bot_user_ids, self._team_bot_names = {}, {}, {}
            self._app = _adapter.AsyncApp(
                token=bot_tokens[0], client=self._new_web_client(bot_tokens[0], proxy_url),
                before_authorize=_adapter._slack_per_request_proxy_middleware(proxy_url))
            _adapter._apply_slack_proxy(self._app.client, proxy_url)
            for token in bot_tokens:
                await self._authenticate_workspace(token, proxy_url)
            self._register_bolt_handlers()
            # _running=True only once the handler is alive (watchdog needs the live
            # task); on failure keep it False so ``finally`` releases the lock.
            try:
                self._start_socket_mode_handler()
                self._running = True
                self._ensure_socket_watchdog()
            except Exception:
                self._running = False
                try:
                    await self._stop_socket_mode_handler()
                except Exception:  # pragma: no cover - defensive logging
                    _adapter.logger.debug("[Slack] Cleanup after failed start raised", exc_info=True)
                raise
            _adapter.logger.info("[Slack] Socket Mode connected (%d workspace(s))", len(self._team_clients))
            self._hint_allow_bots()
            return True
        except Exception as e:  # pragma: no cover - defensive logging
            _adapter.logger.error("[Slack] Connection failed: %s", e, exc_info=True)
            return False
        finally:
            if lock_acquired and not self._running:
                self._release_platform_lock()

    def _fatal_missing_env(self, env_name: str) -> None:
        """Log + record the permanent config error for a missing SLACK_* token."""
        from . import adapter as _adapter

        _adapter.logger.error(
            "[Slack] %s not set — this is a permanent config error; set %s via `hermes "
            "gateway setup` or in the active profile's ~/.hermes/.env file, then restart the "
            "gateway.", env_name, env_name)
        self._set_fatal_error(
            f"missing_{env_name.lower()}",
            f"{env_name} not configured. Use `hermes gateway setup` "
            "or add it to your active profile's ~/.hermes/.env file, then restart the gateway.",
            retryable=False)

    def _hint_allow_bots(self) -> None:
        """INFO hint: bot events can be swallowed upstream of allow_bots (manifest, allowlist)."""
        # Bot-event interop diagnostic. When the user has opted into bot messages via ``slack.allow_bots`` /
        # ``SLACK_ALLOW_BOTS``, surface the additional plumbing they almost certainly also need so
        # bot-to-bot interop doesn't silently fail. See #30091: a user reported that with ``allow_bots:
        # all`` configured, bot messages in shared threads were still dropped. Two things upstream of this
        # code can swallow them: 1. The Slack app's event subscriptions in the manifest — Socket Mode does
        # not deliver events the app hasn't subscribed to (``message.channels`` for public channels,
        # ``message.groups`` for private channels, ``message.im`` for DMs). 2. The SLACK_ALLOWED_USERS /
        # GATEWAY_ALLOWED_USERS per-user allowlists — the other bot's user id must be present (or
        # GATEWAY_ALLOW_ALL_USERS=true). Logging once at INFO keeps the startup line discoverable without
        # requiring DEBUG to enable.
        from . import adapter as _adapter

        _allow_bots_cfg = self._slack_allow_bots()
        if _allow_bots_cfg != "none":
            _adapter.logger.info(
                "[Slack] allow_bots=%s — for bot-to-bot interop also ensure: (a) the Slack "
                "app manifest subscribes to message.channels / message.groups / message.im as "
                "appropriate (run 'hermes slack manifest' if unsure), and (b) the other bot's "
                "Slack user id is in SLACK_ALLOWED_USERS or GATEWAY_ALLOW_ALL_USERS=true. "
                "Without these, bot events are silently dropped upstream of the allow_bots "
                "gate.", _allow_bots_cfg)

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Post a seed message and return its ``ts`` as the handoff ``thread_id``. Slack threads
        anchor to a parent message, not a channel-level object. Returns ``None`` on failure."""
        from . import adapter as _adapter

        if not self._app:
            return None
        try:
            client = self._get_client(parent_chat_id)
            if client is None:
                return None
            seed_text = f":thread: Hermes handoff — *{(name or 'session').strip()[:80]}*"
            result = await client.chat_postMessage(channel=parent_chat_id, text=seed_text)
            ts = _adapter._slack_response_payload(result).get("ts")
            return str(ts) if ts else None
        except Exception as exc:
            _adapter.logger.warning(
                "[%s] Handoff thread: seed-post failed for channel %s: %s", self.name,
                parent_chat_id, exc)
        return None

    async def disconnect(self) -> None:
        """Disconnect from Slack."""
        from . import adapter as _adapter

        self._running = False
        # Seal dangling native streams so no live-typing indicator survives a restart.
        for chat_id, stream in list(self._active_streams.items()):
            await self._seal_stream(chat_id, stream)
        self._active_streams.clear()
        # A watchdog that lost the cancel race must not block cleanup/lock release.
        await self._cancel_socket_watchdog("[Slack] Watchdog task raised during disconnect")
        # Finalize native streams while workspace clients are still live —
        # shutdown safety net for cancellation/reconnect races.
        for key, stream in list(self._native_task_card_streams.items()):
            await self._stop_native_task_card_stream(key, stream)
        await self._stop_socket_mode_handler()
        await self._close_workspace_clients()
        self._app = self._app_token = self._proxy_url = self._bot_user_id = None
        self._team_clients, self._team_bot_user_ids = {}, {}
        self._channel_team, self._dm_conversation_cache = {}, {}
        self._release_platform_lock()
        _adapter.logger.info("[Slack] Disconnected")
