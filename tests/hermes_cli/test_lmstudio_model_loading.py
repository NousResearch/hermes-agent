"""Regression tests for LM Studio loaded-instance management.

``ensure_lmstudio_model_loaded`` has to satisfy two contracts at once:

* the runtime-context contract — an explicit ``context_length`` is only sent
  when the caller supplied one, and the returned context always comes from
  LM Studio itself (loaded instance, echoed ``load_config``, or a refreshed
  catalog read); and
* the resident-instance contract — a dirty LM Studio LLM state (competing
  LLMs, several instances, or the target loaded below an explicitly
  requested context) is unloaded before the target is loaded, so LM Studio
  does not keep several large local LLMs resident and fall back to RAM/CPU.
"""

from __future__ import annotations

import json

import pytest

from hermes_cli import models


TARGET = "publisher/target"
COMPETITOR = "publisher/competitor"
EMBEDDING = "publisher/embedder"
BASE_URL = "http://127.0.0.1:12345"
LOAD_URL = f"{BASE_URL}/api/v1/models/load"
UNLOAD_URL = f"{BASE_URL}/api/v1/models/unload"


class _JsonResponse:
    def __init__(self, payload: dict | None = None) -> None:
        self._body = json.dumps(payload if payload is not None else {}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


def _entry(
    model_id: str,
    *,
    loaded: tuple[tuple[str, int], ...] = (),
    model_type: str = "llm",
    maximum: int = 262_144,
    key: str | None = None,
) -> dict:
    entry: dict = {
        "id": model_id,
        "type": model_type,
        "max_context_length": maximum,
        "loaded_instances": [
            {"id": instance_id, "config": {"context_length": context}}
            for instance_id, context in loaded
        ],
    }
    if key is not None:
        entry["key"] = key
    return entry


def _catalogs(monkeypatch, *catalogs: list[dict]) -> None:
    """Serve ``catalogs`` in order; the last one repeats for later refreshes."""
    remaining = list(catalogs)

    def fake_fetch(**_kwargs):
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    monkeypatch.setattr(models, "_lmstudio_fetch_raw_models", fake_fetch)


def _record(monkeypatch, *, load_response: dict | None = None, fail_unload: bool = False):
    """Record every management request; return the recorded call list."""
    requests: list[dict] = []

    def fake_open(request, *, timeout):
        requests.append({
            "method": request.get_method(),
            "url": request.full_url,
            "body": json.loads(request.data.decode()) if request.data else None,
            "timeout": timeout,
            "headers": {name.lower(): value for name, value in request.header_items()},
        })
        if request.full_url == UNLOAD_URL:
            if fail_unload:
                raise OSError("unload refused")
            return _JsonResponse({})
        return _JsonResponse(load_response)

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fake_open)
    return requests


def _forbid_requests(monkeypatch) -> None:
    def unexpected(*_args, **_kwargs):
        raise AssertionError("clean LM Studio state must not unload or reload")

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", unexpected)


# --------------------------------------------------------------------------
# Cold loads
# --------------------------------------------------------------------------


def test_cold_load_without_explicit_context_omits_context_length(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET)])
    requests = _record(monkeypatch, load_response={"load_config": {"context_length": 131_072}})

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=None
    )

    assert loaded == 131_072
    assert [call["url"] for call in requests] == [LOAD_URL]
    assert requests[0]["body"] == {"model": TARGET, "echo_load_config": True}
    assert "context_length" not in requests[0]["body"]


def test_cold_load_with_explicit_context_sends_it(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET)])
    requests = _record(monkeypatch, load_response={"load_config": {"context_length": 64_000}})

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=64_000
    )

    assert loaded == 64_000
    assert requests[0]["body"] == {
        "model": TARGET,
        "echo_load_config": True,
        "context_length": 64_000,
    }


# --------------------------------------------------------------------------
# Clean no-op states
# --------------------------------------------------------------------------


def test_single_loaded_target_without_explicit_context_is_noop(monkeypatch):
    """With no override, the already-loaded context is authoritative."""
    _catalogs(monkeypatch, [_entry(TARGET, loaded=(("target-instance", 96_000),))])
    _forbid_requests(monkeypatch)

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=None
    )

    assert loaded == 96_000


def test_single_loaded_target_satisfying_explicit_context_is_noop(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET, loaded=(("target-instance", 131_072),))])
    _forbid_requests(monkeypatch)

    result = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=64_000,
        return_load_result=True,
    )

    assert result.context_length == 131_072
    assert result.load_attempted is False
    assert result.rejected is False


def test_embedding_instance_alongside_target_is_not_competing_state(monkeypatch):
    """An embedding model resident next to the target is still a clean state."""
    _catalogs(
        monkeypatch,
        [
            _entry(EMBEDDING, loaded=(("embed-instance", 512),), model_type="embedding"),
            _entry(TARGET, loaded=(("target-instance", 131_072),)),
        ],
    )
    _forbid_requests(monkeypatch)

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=64_000
    )

    assert loaded == 131_072


# --------------------------------------------------------------------------
# Dirty states — unload before (re)loading the target
# --------------------------------------------------------------------------


def test_insufficient_target_is_unloaded_and_reloaded(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET, loaded=(("target-instance", 8_192),))])
    requests = _record(monkeypatch, load_response={"load_config": {"context_length": 64_000}})

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=64_000,
        timeout=15,
    )

    assert loaded == 64_000
    assert [(call["url"], call["body"], call["timeout"]) for call in requests] == [
        (UNLOAD_URL, {"instance_id": "target-instance"}, 15),
        (
            LOAD_URL,
            {"model": TARGET, "echo_load_config": True, "context_length": 64_000},
            15,
        ),
    ]


def test_competing_llm_is_unloaded_before_the_target_loads(monkeypatch):
    """A competitor is evicted even though the target itself is not resident."""
    _catalogs(
        monkeypatch,
        [
            _entry(COMPETITOR, loaded=(("competing-instance", 131_072),)),
            _entry(TARGET),
        ],
    )
    requests = _record(monkeypatch, load_response={"load_config": {"context_length": 131_072}})

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=None
    )

    assert loaded == 131_072
    assert [(call["method"], call["url"], call["body"]) for call in requests] == [
        ("POST", UNLOAD_URL, {"instance_id": "competing-instance"}),
        ("POST", LOAD_URL, {"model": TARGET, "echo_load_config": True}),
    ]


def test_target_plus_competitor_cleans_both_then_reloads_target(monkeypatch):
    """The target is reloaded too — it was loaded under contended resources."""
    _catalogs(
        monkeypatch,
        [
            _entry(TARGET, loaded=(("target-instance", 131_072),), key=TARGET),
            _entry(COMPETITOR, loaded=(("competing-instance", 131_072),)),
        ],
    )
    requests = _record(monkeypatch, load_response={"load_config": {"context_length": 64_000}})

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=64_000
    )

    assert loaded == 64_000
    assert [(call["url"], call["body"]) for call in requests] == [
        (UNLOAD_URL, {"instance_id": "target-instance"}),
        (UNLOAD_URL, {"instance_id": "competing-instance"}),
        (
            LOAD_URL,
            {"model": TARGET, "echo_load_config": True, "context_length": 64_000},
        ),
    ]


def test_target_recognized_through_key_and_id_aliases(monkeypatch):
    """An entry whose ``id`` differs from its ``key`` is still the target.

    If either alias were dropped the sole resident instance would look like a
    competing LLM and get needlessly unloaded and reloaded.
    """
    _catalogs(
        monkeypatch,
        [_entry(f"{TARGET}@q4", loaded=(("target-instance", 131_072),), key=TARGET)],
    )
    _forbid_requests(monkeypatch)

    # The caller asks for the ``key`` alias; the ``id`` alias belongs to the
    # same entry and must not be mistaken for a competing LLM.
    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=64_000
    )

    assert loaded == 131_072


def test_embedding_instances_are_never_unloaded(monkeypatch):
    _catalogs(
        monkeypatch,
        [
            _entry(EMBEDDING, loaded=(("embed-instance", 512),), model_type="embedding"),
            _entry(COMPETITOR, loaded=(("competing-instance", 131_072),)),
            _entry(TARGET),
        ],
    )
    requests = _record(monkeypatch, load_response={"load_config": {"context_length": 131_072}})

    models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=None
    )

    unloaded = [call["body"]["instance_id"] for call in requests if call["url"] == UNLOAD_URL]
    assert unloaded == ["competing-instance"]


# --------------------------------------------------------------------------
# Result contract
# --------------------------------------------------------------------------


def test_unload_failure_reports_no_load_attempt(monkeypatch):
    _catalogs(
        monkeypatch,
        [
            _entry(COMPETITOR, loaded=(("competing-instance", 131_072),)),
            _entry(TARGET),
        ],
    )
    requests = _record(monkeypatch, fail_unload=True)

    result = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=64_000,
        return_load_result=True,
    )

    assert isinstance(result, models.LMStudioLoadResult)
    assert result.context_length is None
    assert result.load_attempted is False
    assert result.rejected is False
    assert [call["url"] for call in requests] == [UNLOAD_URL]


def test_load_failure_after_unload_reports_the_attempt(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET, loaded=(("target-instance", 8_192),))])

    def fake_open(request, *, timeout):
        if request.full_url == UNLOAD_URL:
            return _JsonResponse({})
        raise OSError("load refused")

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fake_open)

    result = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=64_000,
        return_load_result=True,
    )

    assert result.context_length is None
    assert result.load_attempted is True
    assert result.rejected is False


def test_return_load_result_false_yields_the_bare_context(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET)])
    _record(monkeypatch, load_response={"load_config": {"context_length": 64_000}})

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=64_000,
        return_load_result=False,
    )

    assert loaded == 64_000
    assert not isinstance(loaded, models.LMStudioLoadResult)


def test_explicit_override_above_maximum_is_rejected_without_touching_state(monkeypatch):
    _catalogs(
        monkeypatch,
        [
            _entry(TARGET, loaded=(("target-instance", 64_000),), maximum=128_000),
            _entry(COMPETITOR, loaded=(("competing-instance", 64_000),)),
        ],
    )
    _forbid_requests(monkeypatch)

    result = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=256_000,
        return_load_result=True,
    )

    assert result.context_length is None
    assert result.load_attempted is False
    assert result.rejected is True


# --------------------------------------------------------------------------
# The verified context always comes from LM Studio
# --------------------------------------------------------------------------


def test_echoed_load_config_overrides_the_requested_context(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET, loaded=(("target-instance", 8_192),))])
    _record(monkeypatch, load_response={"load_config": {"context_length": 32_000}})

    loaded = models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=64_000
    )

    assert loaded == 32_000


def test_refreshed_state_is_authoritative_when_echo_is_missing(monkeypatch):
    _catalogs(
        monkeypatch,
        [
            _entry(COMPETITOR, loaded=(("competing-instance", 131_072),)),
            _entry(TARGET),
        ],
        [_entry(TARGET, loaded=(("target-instance", 88_000),))],
    )
    _record(monkeypatch, load_response={"status": "loaded"})

    result = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=100_000,
        return_load_result=True,
    )

    assert result.context_length == 88_000
    assert result.load_attempted is True


def test_unverifiable_load_returns_no_context(monkeypatch):
    """Neither an echo nor a refreshed instance means no context is verified."""
    _catalogs(monkeypatch, [_entry(TARGET)])
    _record(monkeypatch, load_response={"status": "loaded"})

    result = models.ensure_lmstudio_model_loaded(
        TARGET,
        BASE_URL,
        api_key="lm-secret",
        target_context_length=64_000,
        return_load_result=True,
    )

    assert result.context_length is None
    assert result.load_attempted is True


def test_management_requests_carry_the_bearer_and_json_content_type(monkeypatch):
    _catalogs(monkeypatch, [_entry(TARGET, loaded=(("target-instance", 8_192),))])
    requests = _record(monkeypatch, load_response={"load_config": {"context_length": 64_000}})

    models.ensure_lmstudio_model_loaded(
        TARGET, BASE_URL, api_key="lm-secret", target_context_length=64_000
    )

    assert requests, "expected unload + load requests"
    for call in requests:
        assert call["headers"]["authorization"] == "Bearer lm-secret"
        assert call["headers"]["content-type"] == "application/json"


@pytest.mark.parametrize("bad_context", [0, -1, True])
def test_invalid_explicit_context_is_refused(monkeypatch, bad_context):
    _catalogs(monkeypatch, [_entry(TARGET, loaded=(("target-instance", 8_192),))])
    _forbid_requests(monkeypatch)

    assert (
        models.ensure_lmstudio_model_loaded(
            TARGET, BASE_URL, api_key="lm-secret", target_context_length=bad_context
        )
        is None
    )
