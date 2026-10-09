"""Transactional integration for route-scoped token-budget policy.

The facade composes this mixin; provider/model helpers remain in their
0.21.5 sibling modules and are invoked lazily at transition boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging

logger = logging.getLogger(__name__)


class _TokenBudgetEffectSink:
    """Post-commit publication journal passed explicitly to route helpers."""

    def __init__(self) -> None:
        self._callbacks = []

    def defer(self, callback) -> bool:
        self._callbacks.append(callback)
        return True

    def checkpoint(self) -> int:
        return len(self._callbacks)

    def discard_since(self, checkpoint: int) -> None:
        del self._callbacks[checkpoint:]

    def discard(self) -> None:
        self._callbacks.clear()

    def publish(self) -> None:
        callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            try:
                callback()
            except Exception:
                # A committed route is valid even if a dashboard, notice or
                # billing sink is temporarily unavailable.
                logger.warning(
                    "failed to publish committed token-budget transition effect",
                    exc_info=True,
                )


class _TokenBudgetTransition:
    """Order component-owned tickets without inspecting their private state."""

    def __init__(self, owner) -> None:
        self.owner = owner
        self.effect_sink = _TokenBudgetEffectSink()
        self._guard_ticket = None
        self._compressor = None
        self._compressor_route = None
        self._compressor_ticket = None
        self._owner_tickets = []
        self._coordinator_commits = []
        self._client_candidate_journal = None

    @property
    def compressor_route(self):
        return self._compressor_route

    @property
    def has_final_compressor_ticket(self) -> bool:
        return self._compressor_ticket is not None

    def _require_ticket_factory(self, compressor):
        prepare = getattr(compressor, "prepare_route_update", None)
        if callable(prepare):
            return prepare
        from agent.token_budget_policy import TokenBudgetPolicyError

        raise TokenBudgetPolicyError(
            "enabled token-budget policy requires the context engine to expose "
            "prepare_route_update() for atomic route changes"
        )

    @staticmethod
    def _validate_compressor_ticket(ticket):
        """Return a usable owner ticket or fail closed before it can be stored.

        A callable ``prepare_route_update`` is only a factory capability.  The
        returned object is the actual atomicity capability and must implement
        both sides of the owner protocol.
        """
        if ticket is not None and all(
            callable(getattr(ticket, name, None)) for name in ("commit", "abort")
        ):
            return ticket
        abort = getattr(ticket, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                logger.debug("invalid compressor ticket abort failed", exc_info=True)
        from agent.token_budget_policy import TokenBudgetPolicyError

        raise TokenBudgetPolicyError(
            "prepare_route_update() must return a non-null ticket with callable "
            "commit() and abort()"
        )

    def prepare_compressor_guard(self) -> None:
        compressor = getattr(self.owner, "context_compressor", None)
        if compressor is None:
            return
        prepare = self._require_ticket_factory(compressor)
        route = self.owner._token_budget_compressor_route(compressor)
        self._compressor = compressor
        self._guard_ticket = self._validate_compressor_ticket(prepare(**route))

    def stage_compressor_route(self, compressor, **route) -> None:
        self._require_ticket_factory(compressor)
        self._compressor = compressor
        staged = dict(route)
        if staged.get("max_tokens") is None:
            current_max = getattr(compressor, "max_tokens", None)
            if current_max is not None:
                # ``update_model(max_tokens=None)`` preserves the live output
                # reservation. Materialize it so a later policy removal can
                # restore the exact pre-policy destination route.
                staged["max_tokens"] = current_max
        self._compressor_route = staged

    def capture_helper_route_if_needed(self) -> None:
        if self._compressor_route is not None:
            return
        compressor = getattr(self.owner, "context_compressor", None)
        if compressor is None:
            return
        self._require_ticket_factory(compressor)
        self._compressor = compressor
        self._compressor_route = self.owner._token_budget_compressor_route(compressor)

    def _release_guard(self) -> None:
        ticket, self._guard_ticket = self._guard_ticket, None
        if ticket is not None:
            ticket.abort()

    def stage_final_compressor(self, route) -> None:
        self._release_guard()
        if not isinstance(route, Mapping):
            return
        compressor = self._compressor or getattr(
            self.owner, "context_compressor", None
        )
        if compressor is None:
            return
        prepare = self._require_ticket_factory(compressor)
        self._compressor = compressor
        self._compressor_route = dict(route)
        self._compressor_ticket = self._validate_compressor_ticket(
            prepare(**self._compressor_route)
        )

    def add_owner_ticket(self, ticket) -> None:
        if ticket is not None:
            self._owner_tickets.append(ticket)

    def client_candidate_journal(self):
        if self._client_candidate_journal is None:
            from agent.client_lifecycle import ClientCandidateJournal

            self._client_candidate_journal = ClientCandidateJournal(self.owner)
            self.add_owner_ticket(self._client_candidate_journal)
        return self._client_candidate_journal

    def defer_coordinator(self, callback) -> None:
        self._coordinator_commits.append(callback)

    def has_owner_work(self) -> bool:
        return bool(self._owner_tickets or self._coordinator_commits)

    def _tickets_in_commit_order(self):
        tickets = []
        if self._compressor_ticket is not None:
            tickets.append(self._compressor_ticket)
        tickets.extend(self._owner_tickets)
        return tickets

    def commit_owner_tickets(self) -> None:
        self._release_guard()
        for ticket in self._tickets_in_commit_order():
            ticket.commit()
        for callback in self._coordinator_commits:
            callback()

    def abort(self) -> None:
        self.effect_sink.discard()
        for ticket in reversed(self._tickets_in_commit_order()):
            try:
                ticket.abort()
            except Exception:
                logger.debug("owner ticket abort failed", exc_info=True)
        try:
            self._release_guard()
        except Exception:
            logger.debug("compressor guard abort failed", exc_info=True)

    def publish(self) -> None:
        self.effect_sink.publish()


class TokenBudgetRuntimeMixin:
    _TOKEN_BUDGET_BOOKKEEPING_FIELDS = (
        "_fallback_index",
        "_unavailable_fallback_keys",
        "_restore_wait_logged",
        "_rate_limit_backoff_count",
        "_rate_limited_until",
    )

    def _load_preflight_token_budget_config(self):
        """Load and validate policy before a transition can mutate runtime."""
        import copy

        from agent.token_budget_policy import (
            TokenBudgetPolicyError,
            validate_token_budget_policy_config,
        )
        from hermes_cli.config import load_config_readonly
        from hermes_cli.config_read_errors import FailedConfigRead

        try:
            loaded = load_config_readonly()
            if isinstance(loaded, FailedConfigRead):
                raise loaded.read_error
            config = loaded or {}
        except Exception as exc:
            if not bool(
                getattr(self, "_token_budget_policy_config_validated", False)
            ):
                raise TokenBudgetPolicyError(
                    "token-budget config could not be read before initialization"
                ) from exc
            # A reload failure is distinct from an invalid reload. Retain only
            # a policy subtree that previously completed validation.
            config = copy.deepcopy(
                getattr(self, "_token_budget_policy_config", {}) or {}
            )
            logger.debug(
                "token-budget policy reload failed; using last known safe policy",
                exc_info=True,
            )
        validate_token_budget_policy_config(config)
        return config

    @staticmethod
    def _snapshot_token_budget_runtime_value(value, memo):
        """Snapshot only owned built-in containers; every foreign object is atomic.

        This explicit schema deliberately never reflects over ``vars`` or
        ``__slots__``. SDK clients, transports, locks, modules, classes and
        callables may contain unbounded or read-only graphs and are rollback
        resources by identity, not mutable policy state.
        """
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return {"kind": "atom", "value": value}
        existing = memo.get(id(value))
        if existing is not None:
            return existing
        if isinstance(value, dict):
            node = {"kind": "dict", "object": value, "items": []}
            memo[id(value)] = node
            node["items"] = [
                (
                    TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(key, memo),
                    TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo),
                )
                for key, item in value.items()
            ]
            return node
        if isinstance(value, list):
            node = {"kind": "list", "object": value, "items": []}
            memo[id(value)] = node
            node["items"] = [
                TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo)
                for item in value
            ]
            return node
        if isinstance(value, set):
            node = {"kind": "set", "object": value, "items": []}
            memo[id(value)] = node
            node["items"] = [
                TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo)
                for item in value
            ]
            return node
        if isinstance(value, tuple):
            node = {"kind": "tuple", "object": value, "items": []}
            memo[id(value)] = node
            node["items"] = [
                TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo)
                for item in value
            ]
            return node
        node = {"kind": "identity", "object": value}
        memo[id(value)] = node
        return node

    @staticmethod
    def _restore_token_budget_runtime_value(node, restored=None):
        """Restore one captured graph node in place, preserving aliases."""
        if not isinstance(node, dict) or node.get("kind") in {
            "atom",
            "identity",
            "missing",
        }:
            return
        if restored is None:
            restored = set()
        node_id = id(node)
        if node_id in restored:
            return
        restored.add(node_id)
        kind = node["kind"]
        for item in node.get("items", []):
            if kind == "dict":
                TokenBudgetRuntimeMixin._restore_token_budget_runtime_value(item[0], restored)
                TokenBudgetRuntimeMixin._restore_token_budget_runtime_value(item[1], restored)
            else:
                TokenBudgetRuntimeMixin._restore_token_budget_runtime_value(item, restored)
        value = node["object"]
        try:
            if kind == "dict":
                value.clear()
                value.update(
                    {
                        TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(key): TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(item)
                        for key, item in node["items"]
                    }
                )
            elif kind == "list":
                value[:] = [
                    TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(item) for item in node["items"]
                ]
            elif kind == "set":
                value.clear()
                value.update(
                    TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(item) for item in node["items"]
                )
        except Exception as exc:
            raise RuntimeError("cannot restore transactional runtime state") from exc

    @staticmethod
    def _token_budget_runtime_snapshot_value(node):
        """Return the original identity represented by a graph snapshot node."""
        if node["kind"] == "atom":
            return node["value"]
        return node.get("object")

    @staticmethod
    def _token_budget_runtime_value_matches(current, node, seen=None):
        """Check graph state without equality hooks on arbitrary SDK objects."""
        if seen is None:
            seen = set()
        kind = node["kind"]
        if kind == "atom":
            return current == node["value"]
        if kind == "missing":
            return False
        if current is not node.get("object"):
            return False
        node_id = id(node)
        if node_id in seen:
            return True
        seen.add(node_id)
        if kind == "dict":
            return len(current) == len(node["items"]) and all(
                TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(key) in current
                and TokenBudgetRuntimeMixin._token_budget_runtime_value_matches(
                    current[TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(key)], item, seen
                )
                for key, item in node["items"]
            )
        if kind in {"list", "tuple"}:
            return len(current) == len(node["items"]) and all(
                TokenBudgetRuntimeMixin._token_budget_runtime_value_matches(item, saved, seen)
                for item, saved in zip(current, node["items"])
            )
        if kind == "set":
            return current == {
                TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(item) for item in node["items"]
            }
        return kind == "identity"

    def _snapshot_token_budget_runtime(self):
        """Capture every rollback-critical field and mutable resource explicitly."""
        missing = object()
        fields = (
            "provider",
            "model",
            "requested_provider",
            "_base_url",
            "_base_url_lower",
            "_base_url_hostname",
            "api_mode",
            "api_key",
            "client",
            "_anthropic_client",
            "_anthropic_api_key",
            "_anthropic_base_url",
            "_is_anthropic_oauth",
            "_config_context_length",
            "_reasoning_echo_flag",
            "reasoning_config",
            "request_overrides",
            "runtime_capabilities",
            "_custom_providers",
            "_client_kwargs",
            "_credential_pool",
            "_credential_pool_entry_id",
            "_credential_pool_revert_id",
            "context_compressor",
            "max_tokens",
            "_configured_max_tokens",
            "_configured_max_tokens_captured",
            "_token_budget_status",
            "_token_budget_policy_config",
            "_token_budget_policy_config_validated",
            "_token_budget_route_baselines",
            "_token_budget_applied_identity",
            "_primary_runtime",
            "_use_prompt_caching",
            "_use_native_cache_layout",
            "_cached_system_prompt",
            "_transport_cache",
            "_fallback_activated",
            "_fallback_chain",
            "_fallback_model",
            "_provider_fallback_active",
            "_provider_fallback_route",
            "_fallback_index",
            "_pending_fallback_notice",
            "_unavailable_fallback_keys",
            "_restore_wait_logged",
            "_rate_limit_backoff_count",
            "_rate_limited_until",
            "_consecutive_stale_streams",
            "_compression_feasibility_checked",
            "_last_feasibility_notice",
            "_compression_warning",
        )
        values = vars(self)
        captured = {}
        memo = {}
        for name in fields:
            value = values.get(name, missing)
            captured[name] = {
                "present": value is not missing,
                "value": value,
                "state": (
                    self._snapshot_token_budget_runtime_value(value, memo)
                    if value is not missing
                    else None
                ),
            }
        return {
            "missing": missing,
            "fields": captured,
        }

    def _restore_token_budget_runtime(self, snapshot, candidate_journal=None):
        """Restore failed transitions before retiring only replacement clients."""
        if not isinstance(snapshot, dict):
            return
        missing = snapshot.get("missing")
        fields = snapshot.get("fields")
        if not isinstance(fields, dict):
            return

        live_clients = {
            name: getattr(self, name, missing) for name in ("client", "_anthropic_client")
        }
        for name, captured in fields.items():
            if not isinstance(captured, dict):
                continue
            try:
                if captured.get("present"):
                    setattr(self, name, captured.get("value"))
                elif name in vars(self):
                    delattr(self, name)
            except Exception:
                logger.debug("failed to restore rolled-back field %s", name, exc_info=True)

        # Restore agent-owned containers after references point back at them.
        # Foreign components (compressor and credential pool) are identities;
        # their tickets own compensation.
        restored_nodes = set()
        for captured in fields.values():
            if not isinstance(captured, dict) or not captured.get("present"):
                continue
            self._restore_token_budget_runtime_value(captured.get("state"), restored_nodes)
        original_clients = {
            id(captured.get("value"))
            for name, captured in fields.items()
            if name in {"client", "_anthropic_client"}
            and isinstance(captured, dict)
            and captured.get("present")
        }
        managed_resource_ids = set(
            getattr(candidate_journal, "managed_resource_ids", ()) or ()
        )
        borrowed_resource_ids = set(
            getattr(candidate_journal, "borrowed_resource_ids", ()) or ()
        )
        retired_client_ids = set()
        for name, replacement in live_clients.items():
            if (
                replacement is missing
                or replacement is None
                or id(replacement) in original_clients
                or id(replacement) in retired_client_ids
                or id(replacement) in managed_resource_ids
                or id(replacement) in borrowed_resource_ids
            ):
                continue
            retired_client_ids.add(id(replacement))
            close = getattr(replacement, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Rollback has already restored the agent and original
                    # resources. A best-effort close must not mask its result.
                    logger.debug("failed to close rolled-back %s", name, exc_info=True)

    def _failed_transition_mutated_runtime(self, snapshot):
        """Whether a False helper result dirtied rollback-critical runtime state.

        A normal False may still record retry/cooldown bookkeeping.  Detect
        only route, credentials, clients, policy baselines, and context-engine
        identity so that allowed bookkeeping remains visible to the caller.
        """
        if not isinstance(snapshot, dict):
            return False
        missing = snapshot.get("missing")
        fields = snapshot.get("fields")
        if not isinstance(fields, dict):
            return False
        for name, captured in fields.items():
            if name in self._TOKEN_BUDGET_BOOKKEEPING_FIELDS:
                continue
            if not isinstance(captured, dict):
                continue
            previous = captured.get("value", missing)
            present = captured.get("present", False)
            current = vars(self).get(name, missing)
            if present != (current is not missing):
                return True
            if not present:
                continue
            if current is not previous and current != previous:
                return True
            if not self._token_budget_runtime_value_matches(current, captured.get("state")):
                return True
        return False

    def _snapshot_token_budget_bookkeeping(self):
        """Capture legitimate helper progress that must survive a False result."""
        missing = object()
        memo = {}
        fields = {}
        for name in self._TOKEN_BUDGET_BOOKKEEPING_FIELDS:
            value = vars(self).get(name, missing)
            fields[name] = {
                "present": value is not missing,
                "value": value,
                "state": (
                    self._snapshot_token_budget_runtime_value(value, memo)
                    if value is not missing
                    else None
                ),
            }
        return {"fields": fields}

    def _restore_token_budget_bookkeeping(self, snapshot):
        fields = snapshot.get("fields") if isinstance(snapshot, dict) else None
        if not isinstance(fields, dict):
            return
        restored = set()
        for name, captured in fields.items():
            if captured.get("present"):
                setattr(self, name, captured.get("value"))
                self._restore_token_budget_runtime_value(
                    captured.get("state"), restored
                )
            else:
                vars(self).pop(name, None)

    @staticmethod
    def _token_budget_policy_enabled(config):
        policy = config.get("token_budget_policy") if isinstance(config, Mapping) else None
        return isinstance(policy, Mapping) and policy.get("enabled") is True

    def _run_token_budget_transition(self, config, helper):
        """Sequence owner tickets, then publish explicit post-commit effects."""
        if not self._token_budget_policy_enabled(config):
            return helper()

        snapshot = self._snapshot_token_budget_runtime()
        transition = _TokenBudgetTransition(self)
        try:
            # Capability check and owner snapshot happen before any route/client
            # helper mutation. Third-party engines therefore fail closed here.
            transition.prepare_compressor_guard()
            result = helper(transition, transition.effect_sink)
            if result is False:
                if transition.has_owner_work():
                    transition.commit_owner_tickets()
                    transition.publish()
                    return result
                transition.abort()
                if self._failed_transition_mutated_runtime(snapshot):
                    bookkeeping = self._snapshot_token_budget_bookkeeping()
                    self._restore_token_budget_runtime(
                        snapshot, transition._client_candidate_journal
                    )
                    self._restore_token_budget_bookkeeping(bookkeeping)
                return result
            transition.capture_helper_route_if_needed()
            self._apply_runtime_token_budget(config, transition)
            transition.commit_owner_tickets()
        except Exception:
            transition.abort()
            self._restore_token_budget_runtime(
                snapshot, transition._client_candidate_journal
            )
            raise
        transition.publish()
        return result

    def _cleanup_failed_token_budget_initialization(
        self, snapshot, *, injected_resource_ids=()
    ):
        """Retire each resource owned by an initialization that never committed."""
        injected = set(injected_resource_ids or ())
        retired = set()

        def close_once(resource, label):
            if (
                resource is None
                or id(resource) in injected
                or id(resource) in retired
            ):
                return
            retired.add(id(resource))
            close = getattr(resource, "close", None)
            if not callable(close):
                return
            try:
                close()
            except Exception:
                logger.debug(
                    "failed to close %s after token-budget init rollback",
                    label,
                    exc_info=True,
                )

        for name in ("client", "_anthropic_client", "_codex_session"):
            close_once(getattr(self, name, None), name)
        transports = getattr(self, "_transport_cache", None)
        if isinstance(transports, dict):
            for resource in transports.values():
                close_once(resource, "transport")

        engine = getattr(self, "context_compressor", None)
        if engine is not None and id(engine) not in injected:
            on_session_end = getattr(engine, "on_session_end", None)
            if callable(on_session_end):
                try:
                    on_session_end(getattr(self, "session_id", "") or "", [])
                except Exception:
                    logger.debug(
                        "failed to end context engine after token-budget init rollback",
                        exc_info=True,
                    )

        memory = getattr(self, "_memory_manager", None)
        if memory is not None and id(memory) not in injected:
            end_memory = getattr(memory, "on_session_end", None)
            if callable(end_memory):
                try:
                    end_memory([])
                except Exception:
                    logger.debug(
                        "failed to end memory manager after token-budget init rollback",
                        exc_info=True,
                    )
            shutdown = getattr(memory, "shutdown_all", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    logger.debug(
                        "failed to stop memory manager after token-budget init rollback",
                        exc_info=True,
                    )

        session_db = getattr(self, "_session_db", None)
        if (
            session_db is not None
            and bool(getattr(self, "_owns_session_db", False))
            and id(session_db) not in injected
        ):
            try:
                from hermes_state_registry import release_or_close

                release_or_close(session_db)
            except Exception:
                logger.debug(
                    "failed to release session DB after token-budget init rollback",
                    exc_info=True,
                )

    def _snapshot_token_budget_request_state(self):
        """Capture one-shot request state that builders consume before returning."""
        missing = object()
        memo = {}
        fields = {}
        for name in (
            "_ephemeral_max_output_tokens",
            "_ephemeral_reasoning_off",
            "_wire_reasoning_config",
        ):
            value = vars(self).get(name, missing)
            fields[name] = {
                "present": value is not missing,
                "value": value,
                "state": (
                    self._snapshot_token_budget_runtime_value(value, memo)
                    if value is not missing
                    else None
                ),
            }
        return {"fields": fields}

    def _restore_token_budget_request_state(self, snapshot):
        fields = snapshot.get("fields") if isinstance(snapshot, dict) else None
        if not isinstance(fields, dict):
            return
        restored = set()
        for name, captured in fields.items():
            if captured.get("present"):
                setattr(self, name, captured.get("value"))
                self._restore_token_budget_runtime_value(
                    captured.get("state"), restored
                )
            else:
                vars(self).pop(name, None)

    def _token_budget_compressor_route(self, engine=None):
        """Return public route inputs; owner-private state stays inside its ticket."""
        engine = engine or getattr(self, "context_compressor", None)
        if engine is None:
            return None
        context_length = getattr(engine, "context_length", None)
        if type(context_length) is not int or context_length <= 0:
            from agent.token_budget_policy import TokenBudgetPolicyError

            raise TokenBudgetPolicyError(
                "ticket-capable context engine must expose a positive context_length"
            )
        return {
            "model": getattr(engine, "model", getattr(self, "model", "")),
            "context_length": context_length,
            "base_url": getattr(engine, "base_url", getattr(self, "base_url", "")),
            "api_key": getattr(engine, "api_key", getattr(self, "api_key", "")),
            "provider": getattr(engine, "provider", getattr(self, "provider", "")),
            "api_mode": getattr(engine, "api_mode", getattr(self, "api_mode", "")),
            "max_tokens": getattr(engine, "max_tokens", None),
        }

    @staticmethod
    def _invoke_transition_helper(
        helper, args, kwargs, transition=None, effect_sink=None
    ):
        """Pass explicit sinks when supported; legacy test/plugin helpers stay callable."""
        if transition is None and effect_sink is None:
            return helper(*args, **kwargs)
        import inspect

        try:
            parameters = inspect.signature(helper).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            parameters, accepts_kwargs = {}, False
        call_kwargs = dict(kwargs)
        if accepts_kwargs or "transition" in parameters:
            call_kwargs["transition"] = transition
        if accepts_kwargs or "effect_sink" in parameters:
            call_kwargs["effect_sink"] = effect_sink
        return helper(*args, **call_kwargs)

    def _apply_runtime_token_budget(self, config=None, transition=None):
        """Synchronize the active route's fail-closed token budget.

        Policy owns agent scalar/container state. The context engine owns the
        staged route ticket and its durable compensation.
        """
        if config is None:
            config = self._load_preflight_token_budget_config()

        from agent.token_budget_policy import (
            _runtime_route_identity,
            apply_runtime_token_budget,
        )

        owns_transition = transition is None
        baselines = getattr(self, "_token_budget_route_baselines", None)
        if (
            owns_transition
            and not self._token_budget_policy_enabled(config)
            and not baselines
        ):
            resolution = apply_runtime_token_budget(self, config)
            self._token_budget_applied_identity = _runtime_route_identity(self)
            return resolution

        snapshot = self._snapshot_token_budget_runtime() if owns_transition else None
        active_transition = transition or _TokenBudgetTransition(self)
        try:
            if owns_transition:
                active_transition.prepare_compressor_guard()
            active_transition.capture_helper_route_if_needed()
            resolution = apply_runtime_token_budget(
                self,
                config,
                compressor_route=active_transition.compressor_route,
                compressor_target_sink=active_transition.stage_final_compressor,
            )
            if (
                resolution is None
                and not owns_transition
                and not active_transition.has_final_compressor_ticket
                and isinstance(active_transition.compressor_route, Mapping)
            ):
                # A policy-enabled config can legitimately have no rule for
                # the destination provider. The route helper still owns a
                # staged compressor change that must commit unchanged.
                active_transition.stage_final_compressor(
                    active_transition.compressor_route
                )
            if resolution is not None:
                status = self._token_budget_status
                status["runtime_context"] = resolution.effective_context
                status["runtime_soft_budget"] = resolution.soft_budget
                status["runtime_output_reserve"] = (
                    resolution.effective_context - resolution.soft_budget
                )
            self._token_budget_applied_identity = _runtime_route_identity(self)
            if owns_transition:
                active_transition.commit_owner_tickets()
        except Exception:
            active_transition.abort()
            if snapshot is not None:
                self._restore_token_budget_runtime(snapshot)
            raise
        if owns_transition:
            active_transition.publish()
        return resolution

    def switch_model(
        self,
        new_model,
        new_provider,
        api_key="",
        base_url="",
        api_mode="",
        capabilities=None,
    ):
        """Switch routes atomically, then reconcile the active token budget."""
        from agent.agent_runtime_helpers import switch_model

        config = self._load_preflight_token_budget_config()
        def run_transition(transition=None, effect_sink=None):
            args = (self, new_model, new_provider, api_key, base_url, api_mode)
            kwargs = {} if capabilities is None else {"capabilities": capabilities}
            return self._invoke_transition_helper(
                switch_model,
                args,
                kwargs,
                transition,
                effect_sink,
            )

        return self._run_token_budget_transition(config, run_transition)

    def _try_activate_fallback(self, reason=None, reset_at=None):
        """Activate fallback only after policy validation, rolling back failures."""
        from agent.chat_completion_helpers import try_activate_fallback

        config = self._load_preflight_token_budget_config()
        def run_transition(transition=None, effect_sink=None):
            kwargs = {} if reset_at is None else {"reset_at": reset_at}
            return self._invoke_transition_helper(
                try_activate_fallback,
                (self, reason),
                kwargs,
                transition,
                effect_sink,
            )

        return self._run_token_budget_transition(config, run_transition)

    def _restore_primary_runtime(self):
        """Restore primary only after policy validation, rolling back failures."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        config = self._load_preflight_token_budget_config()

        def run_transition(transition=None, effect_sink=None):
            return self._invoke_transition_helper(
                restore_primary_runtime,
                (self,),
                {},
                transition,
                effect_sink,
            )

        return self._run_token_budget_transition(config, run_transition)

    def _build_api_kwargs(self, api_messages, tools_for_api=None):
        """Build request kwargs while preserving explicit one-shot output caps."""
        from agent.chat_completion_helpers import build_api_kwargs
        from agent.token_budget_policy import _runtime_route_identity

        policy_config = getattr(self, "_token_budget_policy_config", {}) or {}
        if self._token_budget_policy_enabled(policy_config) and (
            getattr(self, "_token_budget_applied_identity", None)
            != _runtime_route_identity(self)
        ):
            config = self._load_preflight_token_budget_config()
            snapshot = self._snapshot_token_budget_runtime()
            try:
                self._apply_runtime_token_budget(config)
            except Exception:
                self._restore_token_budget_runtime(snapshot)
                raise

        ephemeral_output = getattr(self, "_ephemeral_max_output_tokens", None)
        request_snapshot = self._snapshot_token_budget_request_state()
        try:
            kwargs = build_api_kwargs(self, api_messages, tools_for_api=tools_for_api)
        except Exception:
            self._restore_token_budget_request_state(request_snapshot)
            raise
        policy_status = getattr(self, "_token_budget_status", None)
        if (
            ephemeral_output is None
            and isinstance(policy_status, dict)
            and policy_status.get("output_cap_enforcement") == "provider_default"
            and policy_status.get("request_output_cap") is None
        ):
            for name in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
                kwargs.pop(name, None)
        return kwargs
