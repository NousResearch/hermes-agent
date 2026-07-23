"""Public contract tests for opt-in auxiliary quota fallback."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import product
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.auxiliary_client as auxiliary_client


_MISSING = object()
_CHAIN = [
    {"provider": "nous", "model": "fallback-one"},
    {"provider": "openai", "model": "fallback-two"},
]


class _PrimaryError(Exception):
    pass


class _QuotaExhaustedError(_PrimaryError):
    pass


class _EntryDict(dict):
    pass


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _config(*, chain=_CHAIN, fallback_on=["quota_exhausted"]):
    return {"fallback_on": fallback_on, "fallback_chain": chain}


def _quota_error(
    field="code",
    value="insufficient_quota",
    *,
    envelope=False,
    status=_MISSING,
):
    error = _PrimaryError("primary request failed")
    marker = {field: value}
    error.body = {"error": marker} if envelope else marker
    if status is not _MISSING:
        error.status_code = status
    return error


class _PublicHarness:
    def __init__(self, mode, error, config, resolutions=(), outcomes=()):
        self.mode = mode
        self.error = error
        self.config = config
        self.resolutions = list(resolutions)
        self.resolve_order = []
        self.call_order = []
        self.primary = MagicMock()
        create = (
            AsyncMock(side_effect=error)
            if mode == "async"
            else MagicMock(side_effect=error)
        )
        self.primary.chat.completions.create = create
        self.candidates = [
            self._candidate(index, outcome) for index, outcome in enumerate(outcomes)
        ]
        self.implicit = {
            name: MagicMock(return_value=(None, None, ""))
            for name in (
                "_try_configured_fallback_chain",
                "_try_main_fallback_chain",
                "_try_payment_fallback",
                "_try_main_agent_model_fallback",
            )
        }

    def _candidate(self, index, outcome):
        client = MagicMock()

        def run(**_kwargs):
            self.call_order.append(index)
            if isinstance(outcome, BaseException):
                raise outcome
            return _response(outcome)

        client.chat.completions.create = (
            AsyncMock(side_effect=run)
            if self.mode == "async"
            else MagicMock(side_effect=run)
        )
        return client

    def _resolve(self, entry):
        self.resolve_order.append(entry["model"])
        selected = self.resolutions.pop(0)
        if selected is None:
            return None, None
        return self.candidates[selected], entry["model"]

    async def invoke(self, *, stream=False, call_overrides=None):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    auxiliary_client,
                    "_get_auxiliary_task_config",
                    return_value=self.config,
                )
            )
            stack.enter_context(
                patch.object(
                    auxiliary_client,
                    "_resolve_task_provider_model",
                    return_value=("primary", "primary-model", None, None, None),
                )
            )
            stack.enter_context(
                patch.object(
                    auxiliary_client,
                    "_get_cached_client",
                    return_value=(self.primary, "primary-model"),
                )
            )
            stack.enter_context(
                patch.object(
                    auxiliary_client,
                    "_resolve_fallback_entry",
                    side_effect=self._resolve,
                )
            )
            stack.enter_context(
                patch.object(
                    auxiliary_client,
                    "_to_async_client",
                    side_effect=lambda client, model, **_kwargs: (client, model),
                )
            )
            for name, mock in self.implicit.items():
                stack.enter_context(patch.object(auxiliary_client, name, mock))
            call = dict(
                task="title_generation",
                messages=[{"role": "user", "content": "title"}],
            )
            call.update(call_overrides or {})
            if self.mode == "async":
                return await auxiliary_client.async_call_llm(**call)
            return auxiliary_client.call_llm(**call, stream=stream)

    def assert_no_implicit_routes(self):
        for mock in self.implicit.values():
            mock.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "scenario,resolutions,outcomes,expected_calls,expected_content",
    [
        ("first-success", [0], ["first", "unused"], [0], "first"),
        (
            "second-success",
            [0, 1],
            [ValueError("candidate failed"), "second"],
            [0, 1],
            "second",
        ),
        ("resolve-unavailable", [None, 1], ["unused", "second"], [1], "second"),
    ],
)
async def test_configured_chain_walks_in_order(
    mode, scenario, resolutions, outcomes, expected_calls, expected_content
):
    chain = tuple(_CHAIN) if mode == "sync" and scenario == "first-success" else _CHAIN
    harness = _PublicHarness(
        mode, _quota_error(), _config(chain=chain), resolutions, outcomes
    )

    result = await harness.invoke()

    assert result.choices[0].message.content == expected_content
    assert harness.call_order == expected_calls
    assert harness.resolve_order == [
        entry["model"] for entry in chain[: len(resolutions)]
    ]
    harness.assert_no_implicit_routes()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_all_candidates_fail_reraises_same_primary_object(mode, caplog):
    primary = _quota_error(value=" quota_exhausted ", status=429)
    primary.body["secret"] = "do-not-log-this"
    harness = _PublicHarness(
        mode,
        primary,
        _config(),
        [0, 1],
        [ValueError("first failed"), RuntimeError("second failed")],
    )

    with caplog.at_level("INFO", logger="agent.auxiliary_client"):
        with pytest.raises(_PrimaryError) as caught:
            await harness.invoke()

    assert caught.value is primary
    assert caught.value.__cause__ is primary.__cause__
    assert caught.value.__context__ is primary.__context__
    assert harness.call_order == [0, 1]
    assert "do-not-log-this" not in caplog.text
    harness.assert_no_implicit_routes()


@pytest.mark.asyncio
async def test_duplicate_entries_are_each_attempted_once():
    duplicate = {"provider": "nous", "model": "same-model"}
    harness = _PublicHarness(
        "sync",
        _quota_error(),
        _config(chain=[duplicate, duplicate]),
        [0, 1],
        [ValueError("first index failed"), "second index"],
    )

    result = await harness.invoke()

    assert result.choices[0].message.content == "second index"
    assert harness.resolve_order == ["same-model", "same-model"]
    assert harness.call_order == [0, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,envelope,status",
    list(product(("code", "type", "reason"), (False, True), (_MISSING, 402, 429))),
)
async def test_exact_structured_quota_markers_trigger(field, envelope, status):
    harness = _PublicHarness(
        "sync",
        _quota_error(field, " Usage_Limit_Reached ", envelope=envelope, status=status),
        _config(),
        [0],
        ["fallback"],
    )

    result = await harness.invoke()

    assert result.choices[0].message.content == "fallback"
    assert harness.call_order == [0]
    harness.assert_no_implicit_routes()


def _nonquota_error(case):
    if case == "message-only":
        return _PrimaryError("quota_exhausted")
    if case == "bare-429":
        error = _PrimaryError("rate limited")
        error.status_code = 429
        return error
    if case in {"RESOURCE_EXHAUSTED", "quota_exceeded"}:
        return _quota_error(value=case)
    if case == "status-bool":
        return _quota_error(status=True)
    if case == "status-string":
        return _quota_error(status="429")
    if case.startswith("status-"):
        return _quota_error(status=int(case.removeprefix("status-")))
    if case == "body-message":
        error = _PrimaryError("primary failed")
        error.body = {"message": "quota_exhausted"}
        return error
    if case == "class-name":
        return _QuotaExhaustedError("primary failed")
    if case == "response-json":
        error = _PrimaryError("primary failed")
        setattr(error, "response", SimpleNamespace(json=MagicMock()))
        return error
    if case == "body-subclass":
        error = _PrimaryError("primary failed")
        setattr(error, "body", _EntryDict(code="quota_exhausted"))
        return error
    if case == "cause":
        error = _PrimaryError("outer")
        error.__cause__ = _quota_error()
        return error
    error = _PrimaryError("outer")
    error.__context__ = _quota_error()
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,case",
    [
        ("sync", "message-only"),
        ("async", "bare-429"),
        ("sync", "RESOURCE_EXHAUSTED"),
        ("async", "quota_exceeded"),
        ("sync", "status-401"),
        ("async", "status-403"),
        ("sync", "status-500"),
        ("async", "status-bool"),
        ("sync", "status-string"),
        ("async", "body-message"),
        ("sync", "class-name"),
        ("async", "response-json"),
        ("sync", "body-subclass"),
        ("sync", "cause"),
        ("async", "context"),
    ],
)
async def test_nonquota_signals_never_leave_primary_route(mode, case):
    primary = _nonquota_error(case)
    harness = _PublicHarness(mode, primary, _config(), [], [])

    with pytest.raises(_PrimaryError) as caught:
        await harness.invoke()

    assert caught.value is primary
    assert harness.resolve_order == []
    assert harness.call_order == []
    harness.assert_no_implicit_routes()
    if case == "response-json":
        getattr(primary, "response").json.assert_not_called()


_VALID_CHAIN = [{"provider": "nous", "model": "fallback-model"}]
_INVALID_CONFIGS = [
    ("sync", {"fallback_on": "quota_exhausted", "fallback_chain": _VALID_CHAIN}),
    ("async", {"fallback_on": ["other"], "fallback_chain": _VALID_CHAIN}),
    (
        "sync",
        {"fallback_on": ["quota_exhausted", "other"], "fallback_chain": _VALID_CHAIN},
    ),
    ("async", {"fallback_on": ["quota_exhausted"], "fallback_chain": []}),
    ("sync", {"fallback_on": ["quota_exhausted"], "fallback_chain": {}}),
    ("async", {"fallback_on": ["quota_exhausted"], "fallback_chain": [[]]}),
    (
        "sync",
        {
            "fallback_on": ["quota_exhausted"],
            "fallback_chain": [{"provider": "", "model": "m"}],
        },
    ),
    (
        "async",
        {
            "fallback_on": ["quota_exhausted"],
            "fallback_chain": [{"provider": "p", "model": ""}],
        },
    ),
    (
        "sync",
        {
            "fallback_on": ["quota_exhausted"],
            "fallback_chain": [{"provider": "auto", "model": "m"}],
        },
    ),
    (
        "async",
        {
            "fallback_on": ["quota_exhausted"],
            "fallback_chain": [{"provider": "main", "model": "m"}],
        },
    ),
    (
        "sync",
        {
            "fallback_on": ["quota_exhausted"],
            "fallback_chain": [_EntryDict(provider="p", model="m")],
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,config", _INVALID_CONFIGS)
async def test_invalid_opt_in_config_is_rejected_before_primary(mode, config):
    harness = _PublicHarness(mode, _quota_error(), config)

    with pytest.raises(ValueError, match="fallback"):
        await harness.invoke()

    harness.primary.chat.completions.create.assert_not_called()
    harness.assert_no_implicit_routes()


def test_sync_stream_is_rejected_before_primary():
    config = _config()
    with (
        patch.object(
            auxiliary_client, "_get_auxiliary_task_config", return_value=config
        ),
        patch.object(
            auxiliary_client, "_resolve_task_provider_model"
        ) as resolve_primary,
    ):
        with pytest.raises(ValueError, match="stream"):
            auxiliary_client.call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "title"}],
                stream=True,
            )
    resolve_primary.assert_not_called()


def test_missing_fallback_on_preserves_legacy_fallback():
    primary = MagicMock()
    payment_error = _PrimaryError("payment required")
    payment_error.status_code = 402
    primary.chat.completions.create.side_effect = payment_error
    fallback = MagicMock()
    fallback.chat.completions.create.return_value = _response("legacy fallback")
    with (
        patch.object(
            auxiliary_client,
            "_get_auxiliary_task_config",
            return_value={"fallback_chain": _CHAIN},
        ),
        patch.object(
            auxiliary_client,
            "_resolve_task_provider_model",
            return_value=("primary", "primary-model", None, None, None),
        ),
        patch.object(
            auxiliary_client,
            "_get_cached_client",
            return_value=(primary, "primary-model"),
        ),
        patch.object(
            auxiliary_client,
            "_try_configured_fallback_chain",
            return_value=(fallback, "legacy-model", "fallback_chain[0](nous)"),
        ) as legacy_chain,
    ):
        result = auxiliary_client.call_llm(
            task="title_generation",
            messages=[{"role": "user", "content": "title"}],
        )

    assert result.choices[0].message.content == "legacy fallback"
    legacy_chain.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("field", ["code", "type", "reason"])
async def test_direct_marker_is_not_shadowed_by_unrelated_error_mapping(mode, field):
    primary = _quota_error(field, "quota_exhausted")
    getattr(primary, "body")["error"] = {"message": "unrelated details"}
    harness = _PublicHarness(mode, primary, _config(), [0], ["fallback"])

    result = await harness.invoke()

    assert result.choices[0].message.content == "fallback"
    assert harness.call_order == [0]
    harness.assert_no_implicit_routes()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "body,expected_fallback",
    [
        ({"error": {"reason": "usage_limit_reached"}}, True),
        ({"error": {"error": {"reason": "usage_limit_reached"}}}, False),
    ],
)
async def test_error_envelope_is_exactly_one_level(mode, body, expected_fallback):
    primary = _PrimaryError("primary request failed")
    setattr(primary, "body", body)
    harness = _PublicHarness(
        mode,
        primary,
        _config(),
        [0] if expected_fallback else [],
        ["fallback"] if expected_fallback else [],
    )

    if expected_fallback:
        result = await harness.invoke()
        assert result.choices[0].message.content == "fallback"
        assert harness.call_order == [0]
    else:
        with pytest.raises(_PrimaryError) as caught:
            await harness.invoke()
        assert caught.value is primary
        assert harness.call_order == []
    harness.assert_no_implicit_routes()


def _auth_error(message="expired credential"):
    error = _PrimaryError(message)
    setattr(error, "status_code", 401)
    return error


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("final_kind", ["success", "all-fail", "nonquota"])
async def test_generic_credential_refresh_retry_preserves_quota_contract(
    mode, final_kind
):
    retry_error = (
        RuntimeError("ordinary retry failure")
        if final_kind == "nonquota"
        else _quota_error("reason", "usage_limit_reached")
    )
    resolutions = []
    outcomes = []
    if final_kind == "success":
        resolutions, outcomes = [0], ["fallback"]
    elif final_kind == "all-fail":
        resolutions = [0, 1]
        outcomes = [ValueError("candidate one"), RuntimeError("candidate two")]
    harness = _PublicHarness(mode, _auth_error(), _config(), resolutions, outcomes)
    retry_name = (
        "_retry_same_provider_async" if mode == "async" else "_retry_same_provider_sync"
    )
    retry_mock = (
        AsyncMock(side_effect=retry_error)
        if mode == "async"
        else MagicMock(side_effect=retry_error)
    )

    with (
        patch.object(
            auxiliary_client, "_refresh_provider_credentials", return_value=True
        ),
        patch.object(auxiliary_client, retry_name, retry_mock),
    ):
        if final_kind == "success":
            result = await harness.invoke()
            assert result.choices[0].message.content == "fallback"
            assert harness.call_order == [0]
        else:
            expected_calls = [0, 1] if final_kind == "all-fail" else []
            with pytest.raises(type(retry_error)) as caught:
                await harness.invoke()
            assert caught.value is retry_error
            assert harness.call_order == expected_calls
    harness.assert_no_implicit_routes()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("retry_class", ["unsupported-temperature", "max-tokens"])
async def test_parameter_repair_retry_quota_reaches_explicit_chain(mode, retry_class):
    retry_error = _quota_error("type", "insufficient_quota")
    parameter = (
        "temperature" if retry_class == "unsupported-temperature" else "max_tokens"
    )
    first_error = RuntimeError(f"Unsupported parameter: {parameter}")
    harness = _PublicHarness(mode, first_error, _config(), [0], ["fallback"])
    harness.primary.base_url = "https://example.test/anthropic"
    create = (
        AsyncMock(side_effect=[first_error, retry_error])
        if mode == "async"
        else MagicMock(side_effect=[first_error, retry_error])
    )
    harness.primary.chat.completions.create = create
    call_overrides = (
        {"temperature": 0.3}
        if retry_class == "unsupported-temperature"
        else {"max_tokens": 100}
    )

    result = await harness.invoke(call_overrides=call_overrides)

    assert result.choices[0].message.content == "fallback"
    assert harness.call_order == [0]
    harness.assert_no_implicit_routes()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_stale_candidate_warning_redacts_exception_body(mode, caplog):
    sentinel = "SENTINEL-API-KEY-super-secret"
    harness = _PublicHarness(
        mode,
        _quota_error(),
        _config(),
        [0, 1],
        [_auth_error(f"401 provider body API_KEY={sentinel}"), "second"],
    )

    with (
        patch.object(
            auxiliary_client, "_refresh_provider_credentials", return_value=False
        ),
        caplog.at_level("WARNING", logger="agent.auxiliary_client"),
    ):
        result = await harness.invoke()

    assert result.choices[0].message.content == "second"
    assert harness.call_order == [0, 1]
    assert sentinel not in caplog.text
    assert "_PrimaryError" in caplog.text
    harness.assert_no_implicit_routes()
