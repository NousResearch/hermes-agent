"""Transactional integration for route-scoped token-budget policy.

The facade composes this mixin; provider/model helpers remain in their
0.21.5 sibling modules and are invoked lazily at transition boundaries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TokenBudgetRuntimeMixin:
    def _load_preflight_token_budget_config(self):
        """Load and validate policy before a transition can mutate runtime."""
        from agent.token_budget_policy import validate_token_budget_policy_config
        from hermes_cli.config import load_config_readonly

        try:
            config = load_config_readonly() or {}
        except Exception:
            # A reload failure is distinct from an invalid reload. Retain the
            # last validated, non-secret policy snapshot for the transition.
            config = getattr(self, "_token_budget_policy_config", {}) or {}
            logger.debug(
                "token-budget policy reload failed; using last known safe policy",
                exc_info=True,
            )
        validate_token_budget_policy_config(config)
        return config

    @staticmethod
    def _token_budget_runtime_slot_names(value):
        """Return real slot attribute names, including inherited privates."""
        names = []
        for cls in type(value).__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in {"__dict__", "__weakref__"}:
                    continue
                if name.startswith("__") and not name.endswith("__"):
                    name = f"_{cls.__name__.lstrip('_')}{name}"
                if name not in names:
                    names.append(name)
        return names

    @staticmethod
    def _snapshot_token_budget_runtime_value(value, memo, *, allow_opaque=False):
        """Capture a restorable graph without cloning SDK transports or locks.

        The shared ``memo`` is deliberately used for every transactional field:
        independently copying each field loses aliases such as a reasoning and
        transport cache pointing to the same dictionary.
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
                    TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(key, memo, allow_opaque=allow_opaque),
                    TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo, allow_opaque=allow_opaque),
                )
                for key, item in value.items()
            ]
            return node
        if isinstance(value, list):
            node = {"kind": "list", "object": value, "items": []}
            memo[id(value)] = node
            node["items"] = [
                TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo, allow_opaque=allow_opaque)
                for item in value
            ]
            return node
        if isinstance(value, set):
            node = {"kind": "set", "object": value, "items": []}
            memo[id(value)] = node
            node["items"] = [
                TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo, allow_opaque=allow_opaque)
                for item in value
            ]
            return node
        if isinstance(value, tuple):
            node = {"kind": "tuple", "object": value, "items": []}
            memo[id(value)] = node
            node["items"] = [
                TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(item, memo, allow_opaque=allow_opaque)
                for item in value
            ]
            return node
        if allow_opaque:
            # Clients/compressors can own locks, sessions and callables. Keep
            # those identities, but capture their writable instance state.
            node = {"kind": "object", "object": value, "attrs": [], "slots": []}
            memo[id(value)] = node
            try:
                node["attrs"] = [
                    (
                        name,
                        TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(
                            item, memo, allow_opaque=True
                        ),
                    )
                    for name, item in vars(value).items()
                ]
            except TypeError:
                pass
            missing = object()
            for name in TokenBudgetRuntimeMixin._token_budget_runtime_slot_names(value):
                try:
                    item = getattr(value, name)
                except AttributeError:
                    node["slots"].append((name, {"kind": "missing", "value": missing}))
                else:
                    node["slots"].append(
                        (
                            name,
                            TokenBudgetRuntimeMixin._snapshot_token_budget_runtime_value(
                                item, memo, allow_opaque=True
                            ),
                        )
                    )
            return node
        # Configuration, prompts and cache state must be fully restorable. An
        # opaque mutable value here makes rollback unverifiable, so fail before
        # invoking a helper that may mutate runtime.
        raise RuntimeError(
            f"cannot snapshot transactional runtime state of {type(value).__name__}"
        )

    @staticmethod
    def _restore_token_budget_runtime_value(node, restored=None):
        """Restore one captured graph node in place, preserving aliases."""
        if not isinstance(node, dict) or node.get("kind") in {"atom", "missing"}:
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
        for _, item in node.get("attrs", []):
            TokenBudgetRuntimeMixin._restore_token_budget_runtime_value(item, restored)
        for _, item in node.get("slots", []):
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
            elif kind == "object":
                try:
                    attrs = vars(value)
                    attrs.clear()
                    attrs.update(
                        {
                            name: TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(item)
                            for name, item in node["attrs"]
                        }
                    )
                except TypeError:
                    pass
                for name, item in node["slots"]:
                    if item["kind"] == "missing":
                        try:
                            delattr(value, name)
                        except AttributeError:
                            pass
                    else:
                        setattr(value, name, TokenBudgetRuntimeMixin._token_budget_runtime_snapshot_value(item))
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
        try:
            attrs = vars(current)
        except TypeError:
            attrs = {}
        for name, saved in node["attrs"]:
            if name not in attrs or not TokenBudgetRuntimeMixin._token_budget_runtime_value_matches(
                attrs[name], saved, seen
            ):
                return False
        if len(attrs) != len(node["attrs"]):
            return False
        for name, saved in node["slots"]:
            try:
                current_slot = getattr(current, name)
            except AttributeError:
                if saved["kind"] != "missing":
                    return False
            else:
                if saved["kind"] == "missing" or not TokenBudgetRuntimeMixin._token_budget_runtime_value_matches(
                    current_slot, saved, seen
                ):
                    return False
        return True

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
            "_token_budget_route_baselines",
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
            critical_object = name in {
                "client",
                "_anthropic_client",
                "_credential_pool",
                "context_compressor",
                # Runtime helpers may retain these critical identities here.
                "_client_kwargs",
                "_primary_runtime",
                # Cache entries can be live transport adapters too.
                "_transport_cache",
            }
            captured[name] = {
                "present": value is not missing,
                "value": value,
                "state": (
                    self._snapshot_token_budget_runtime_value(
                        value, memo, allow_opaque=critical_object
                    )
                    if value is not missing
                    else None
                ),
            }
        return {"missing": missing, "fields": captured}

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

        from agent.token_budget_policy import apply_runtime_token_budget

        resolution = apply_runtime_token_budget(self, config)
        if resolution is None:
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
        snapshot = self._snapshot_token_budget_runtime()
        try:
            args = (self, new_model, new_provider, api_key, base_url, api_mode)
            result = (
                switch_model(*args)
                if capabilities is None
                else switch_model(*args, capabilities=capabilities)
            )
            if result is False:
                if self._failed_transition_mutated_runtime(snapshot):
                    self._restore_token_budget_runtime(snapshot)
                return result
            self._apply_runtime_token_budget(config)
            return result
        except Exception:
            self._restore_token_budget_runtime(snapshot)
            raise

    def _try_activate_fallback(self, reason=None, reset_at=None):
        """Activate fallback only after policy validation, rolling back failures."""
        from agent.chat_completion_helpers import try_activate_fallback

        config = self._load_preflight_token_budget_config()
        snapshot = self._snapshot_token_budget_runtime()
        try:
            if reset_at is None:
                activated = try_activate_fallback(self, reason)
            else:
                activated = try_activate_fallback(self, reason, reset_at=reset_at)
            if not activated:
                if self._failed_transition_mutated_runtime(snapshot):
                    self._restore_token_budget_runtime(snapshot)
                return activated
            self._apply_runtime_token_budget(config)
            return activated
        except Exception:
            self._restore_token_budget_runtime(snapshot)
            raise

    def _restore_primary_runtime(self):
        """Restore primary only after policy validation, rolling back failures."""
        from agent.agent_runtime_helpers import restore_primary_runtime

        config = self._load_preflight_token_budget_config()
        snapshot = self._snapshot_token_budget_runtime()
        try:
            restored = restore_primary_runtime(self)
            if not restored:
                if self._failed_transition_mutated_runtime(snapshot):
                    self._restore_token_budget_runtime(snapshot)
                return restored
            self._apply_runtime_token_budget(config)
            return restored
        except Exception:
            self._restore_token_budget_runtime(snapshot)
            raise

    def _build_api_kwargs(self, api_messages, tools_for_api=None):
        """Build request kwargs while preserving explicit one-shot output caps."""
        from agent.chat_completion_helpers import build_api_kwargs

        ephemeral_output = getattr(self, "_ephemeral_max_output_tokens", None)
        kwargs = build_api_kwargs(self, api_messages, tools_for_api=tools_for_api)
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
