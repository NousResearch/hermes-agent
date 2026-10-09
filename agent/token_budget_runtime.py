"""Transactional integration for route-scoped token-budget policy.

The facade composes this mixin; provider/model helpers remain in their
0.21.5 sibling modules and are invoked lazily at transition boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging

logger = logging.getLogger(__name__)


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
        compressor = values.get("context_compressor", missing)
        compressor_fields = {}
        if compressor is not missing and compressor is not None:
            for name in (
                "model",
                "context_length",
                "base_url",
                "api_key",
                "provider",
                "api_mode",
                "threshold_percent",
                "threshold_tokens",
                "max_tokens",
                "_tail_token_budget",
                "tail_token_budget",
                "summary_target_ratio",
                "model_thresholds",
            ):
                try:
                    value = getattr(compressor, name)
                except (AttributeError, TypeError):
                    continue
                compressor_fields[name] = {
                    "value": value,
                    "state": self._snapshot_token_budget_runtime_value(value, memo),
                }
        return {
            "missing": missing,
            "fields": captured,
            "compressor_fields": compressor_fields,
        }

    def _restore_token_budget_runtime(self, snapshot):
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

        # Restore original resources after references point back at them. This
        # includes in-place SDK-client mutation, mutable credential pools, and
        # policy baseline maps; no blind transport deepcopy is involved.
        restored_nodes = set()
        for captured in fields.values():
            if not isinstance(captured, dict) or not captured.get("present"):
                continue
            self._restore_token_budget_runtime_value(captured.get("state"), restored_nodes)
        compressor = getattr(self, "context_compressor", None)
        compressor_fields = snapshot.get("compressor_fields") or {}
        if compressor is not None and isinstance(compressor_fields, dict):
            for name, captured in compressor_fields.items():
                try:
                    setattr(compressor, name, captured.get("value"))
                    self._restore_token_budget_runtime_value(
                        captured.get("state"), restored_nodes
                    )
                except Exception:
                    logger.debug(
                        "failed to restore compressor field %s", name, exc_info=True
                    )

        original_clients = {
            id(captured.get("value"))
            for name, captured in fields.items()
            if name in {"client", "_anthropic_client"}
            and isinstance(captured, dict)
            and captured.get("present")
        }
        retired_client_ids = set()
        for name, replacement in live_clients.items():
            if (
                replacement is missing
                or replacement is None
                or id(replacement) in original_clients
                or id(replacement) in retired_client_ids
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
        only route, credentials, clients, policy baselines, and compressor
        state so that allowed bookkeeping remains visible to the caller.
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
        compressor = getattr(self, "context_compressor", None)
        for name, captured in (snapshot.get("compressor_fields") or {}).items():
            try:
                current = getattr(compressor, name)
            except (AttributeError, TypeError):
                return True
            if not self._token_budget_runtime_value_matches(
                current, captured.get("state")
            ):
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

    def _defer_token_budget_effect(self, callback):
        """Queue one external effect while an enabled-policy transition prepares."""
        effects = vars(self).get("_token_budget_deferred_effects")
        if not isinstance(effects, list):
            return False
        effects.append(callback)
        return True

    def _begin_token_budget_effects(self):
        if "_token_budget_deferred_effects" in vars(self):
            raise RuntimeError("nested token-budget transition is not supported")
        self._token_budget_deferred_effects = []

    def _discard_token_budget_effects(self):
        vars(self).pop("_token_budget_deferred_effects", None)

    def _commit_token_budget_effects(self):
        effects = vars(self).pop("_token_budget_deferred_effects", [])
        for callback in effects:
            try:
                callback()
            except Exception:
                # These surfaces were best-effort before staging. A committed
                # runtime remains valid even if its dashboard/status sink fails.
                logger.warning(
                    "failed to publish committed token-budget transition effect",
                    exc_info=True,
                )

    def _run_token_budget_transition(self, config, helper):
        """Prepare runtime+policy, then publish route side effects exactly once."""
        if not self._token_budget_policy_enabled(config):
            return helper()

        snapshot = self._snapshot_token_budget_runtime()
        self._begin_token_budget_effects()
        try:
            result = helper()
            if result is False:
                self._discard_token_budget_effects()
                if self._failed_transition_mutated_runtime(snapshot):
                    bookkeeping = self._snapshot_token_budget_bookkeeping()
                    self._restore_token_budget_runtime(snapshot)
                    self._restore_token_budget_bookkeeping(bookkeeping)
                return result
            self._apply_runtime_token_budget(config)
        except Exception:
            self._discard_token_budget_effects()
            self._restore_token_budget_runtime(snapshot)
            raise
        self._commit_token_budget_effects()
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

    def _synchronize_token_budget_compressor_derivatives(self, engine, resolution):
        """Recompute compressor-owned values invalidated by a soft-cap override."""
        # ContextCompressor exposes an invalidating private cache and a
        # mode-aware ``tail_token_budget`` property. Use it instead of
        # duplicating its lean-tail calculation. Legacy engines expose only a
        # summary ratio, for which the historical threshold-derived formula is
        # the compatible derivation.
        if hasattr(engine, "_tail_token_budget"):
            engine._tail_token_budget = None
            _ = engine.tail_token_budget
            return
        ratio = getattr(engine, "summary_target_ratio", None)
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool):
            engine.tail_token_budget = int(resolution.soft_budget * ratio)

    def _apply_runtime_token_budget(self, config=None):
        """Synchronize the active route's fail-closed token budget.

        Model switches, provider fallback and primary restoration all build or
        update the context engine before returning to ``AIAgent``.  Keeping
        this synchronization at the concrete runtime boundary ensures that
        ``max_tokens``, the compressor window, and status describe the same
        effective route without changing the cached system prompt.
        """
        if config is None:
            config = self._load_preflight_token_budget_config()

        from agent.token_budget_policy import (
            _runtime_route_identity,
            apply_runtime_token_budget,
        )

        resolution = apply_runtime_token_budget(self, config)
        if resolution is None:
            self._token_budget_applied_identity = _runtime_route_identity(self)
            return None

        engine = getattr(self, "context_compressor", None)
        if engine is not None:
            update_model = getattr(engine, "update_model", None)
            if callable(update_model):
                import inspect

                kwargs = {
                    "model": self.model,
                    "context_length": resolution.effective_context,
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "provider": self.provider,
                    "api_mode": self.api_mode,
                }
                try:
                    accepts_max_tokens = "max_tokens" in inspect.signature(
                        update_model
                    ).parameters
                except (TypeError, ValueError):
                    accepts_max_tokens = False
                if accepts_max_tokens:
                    # Built-in compressors use this reservation while deriving
                    # their internal budgets.  Older plugin ABI variants do
                    # not accept it, so preserve their historical call shape.
                    update_model(**kwargs, max_tokens=resolution.max_output)
                else:
                    update_model(**kwargs)

            # ``update_model`` may apply a model-specific compression ratio.
            # The policy's soft budget is route-authoritative, so calibrate
            # the effective engine state explicitly afterward.  This also
            # supports lean plugin engines that expose fields but no updater.
            for name, value in (
                ("context_length", resolution.effective_context),
                ("max_tokens", resolution.max_output),
                ("threshold_percent", resolution.compression_threshold),
                ("threshold_tokens", resolution.soft_budget),
            ):
                setattr(engine, name, value)
            self._synchronize_token_budget_compressor_derivatives(engine, resolution)

        status = self._token_budget_status
        status["runtime_context"] = resolution.effective_context
        status["runtime_soft_budget"] = resolution.soft_budget
        status["runtime_output_reserve"] = (
            resolution.effective_context - resolution.soft_budget
        )
        self._token_budget_applied_identity = _runtime_route_identity(self)

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
        def transition():
            args = (self, new_model, new_provider, api_key, base_url, api_mode)
            return (
                switch_model(*args)
                if capabilities is None
                else switch_model(*args, capabilities=capabilities)
            )

        return self._run_token_budget_transition(config, transition)

    def _try_activate_fallback(self, reason=None, reset_at=None):
        """Activate fallback only after policy validation, rolling back failures."""
        from agent.chat_completion_helpers import try_activate_fallback

        config = self._load_preflight_token_budget_config()
        def transition():
            if reset_at is None:
                return try_activate_fallback(self, reason)
            return try_activate_fallback(self, reason, reset_at=reset_at)

        return self._run_token_budget_transition(config, transition)

    def _restore_primary_runtime(self):
        """Restore primary only after policy validation, rolling back failures."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        config = self._load_preflight_token_budget_config()
        return self._run_token_budget_transition(
            config, lambda: restore_primary_runtime(self)
        )

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
