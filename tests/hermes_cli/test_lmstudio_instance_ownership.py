"""LM Studio switches preserve instances that Hermes did not load."""

import json
import urllib.error
from hermes_cli import models, models_local

BASE_URL = "http://127.0.0.1:1234/v1"
SERVER_ROOT = "http://127.0.0.1:1234"
MODEL = "publisher/next"
PREVIOUS = "publisher/previous"


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _instance(instance_id, context=8192):
    return {"id": instance_id, "config": {"context_length": context}}


def _server(monkeypatch, previous_instances, *, target_instances=(), load_response=None):
    calls = []
    catalog = [
        {"key": PREVIOUS, "loaded_instances": previous_instances},
        {"key": MODEL, "max_context_length": 32768, "loaded_instances": list(target_instances)},
    ]

    def open_request(request, *, timeout):
        payload = json.loads(request.data) if request.data else None
        calls.append((request.full_url, payload))
        if request.full_url.endswith("/unload"):
            return _Response({})
        if request.full_url.endswith("/load"):
            return _Response(load_response if load_response is not None else {"instance_id": "owned-next", "load_config": {"context_length": 8192}})
        return _Response({"models": catalog})

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", open_request)
    return calls


def _load(**kwargs):
    return models_local.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, "", None, previous_model=PREVIOUS, return_load_result=True, **kwargs,
    )


def test_unowned_previous_instances_are_preserved(monkeypatch):
    calls = _server(monkeypatch, [_instance("user-one"), _instance("user-two")])
    result = _load()
    assert result.context_length == 8192
    assert result.load_attempted is True
    assert not any(url.endswith("/unload") for url, _ in calls)
    assert [payload for url, payload in calls if url.endswith("/load")] == [
        {"model": MODEL, "echo_load_config": True}
    ]


def test_owned_previous_instance_unloads_without_its_sibling(monkeypatch):
    from hermes_cli.models_lmstudio_instances import forget_instance, remember_instance
    calls = _server(monkeypatch, [_instance("owned-previous"), _instance("user-copy")])
    remember_instance(SERVER_ROOT, PREVIOUS, "owned-previous")
    try:
        result = _load()
    finally:
        forget_instance(SERVER_ROOT, PREVIOUS)
    assert result.context_length == 8192
    assert [payload for url, payload in calls if url.endswith("/unload")] == [
        {"instance_id": "owned-previous"}
    ]


def test_never_keeps_owned_previous_and_target_context(monkeypatch):
    from hermes_cli.models_lmstudio_instances import forget_instance, remember_instance
    calls = _server(monkeypatch, [_instance("owned-previous")], target_instances=[_instance("user-next", 4096)])
    remember_instance(SERVER_ROOT, PREVIOUS, "owned-previous")
    try:
        result = _load(unload_policy="never")
    finally:
        forget_instance(SERVER_ROOT, PREVIOUS)
    assert result.context_length == 4096
    assert result.load_attempted is False
    assert all(payload is None for _, payload in calls)


def test_unverified_resident_target_keeps_previous_owned_instance(monkeypatch):
    from hermes_cli.models_lmstudio_instances import forget_instance, remember_instance
    calls = _server(monkeypatch, [_instance("owned-previous")], target_instances=[{"id": "unknown-context"}])
    remember_instance(SERVER_ROOT, PREVIOUS, "owned-previous")
    try:
        result = _load()
    finally:
        forget_instance(SERVER_ROOT, PREVIOUS)
    assert result.context_length is None
    assert result.load_attempted is False
    assert all(payload is None for _, payload in calls)


def test_stale_owned_id_does_not_unload_user_copy(monkeypatch):
    from hermes_cli.models_lmstudio_instances import forget_instance, owned_instance, remember_instance
    calls = _server(monkeypatch, [_instance("user-copy")])
    remember_instance(SERVER_ROOT, PREVIOUS, "vanished")
    try:
        _load()
        assert owned_instance(SERVER_ROOT, PREVIOUS) is None
    finally:
        forget_instance(SERVER_ROOT, PREVIOUS)
    assert not any(url.endswith("/unload") for url, _ in calls)


def test_owned_claim_is_limited_to_profile(monkeypatch, tmp_path):
    from hermes_cli.models_lmstudio_instances import forget_instance, owned_instance, remember_instance
    remember_instance(SERVER_ROOT, PREVIOUS, "owned-previous")
    try:
        with monkeypatch.context() as other_profile:
            other_profile.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
            assert owned_instance(SERVER_ROOT, PREVIOUS) is None
    finally:
        forget_instance(SERVER_ROOT, PREVIOUS)


def test_failed_destination_load_restores_previous_owned_context(monkeypatch):
    from hermes_cli.models_lmstudio_instances import forget_instance, remember_instance
    calls = _server(monkeypatch, [_instance("owned-previous", 4096)], load_response={})
    remember_instance(SERVER_ROOT, PREVIOUS, "owned-previous")
    try:
        result = _load()
    finally:
        forget_instance(SERVER_ROOT, PREVIOUS)
    assert result.context_length is None
    assert result.load_attempted is True
    assert [payload for url, payload in calls if url.endswith("/load")] == [
        {"model": MODEL, "echo_load_config": True},
        {"model": PREVIOUS, "echo_load_config": True, "context_length": 4096},
    ]


def test_unload_404_does_not_restore_a_missing_instance(monkeypatch):
    from hermes_cli.models_lmstudio_instances import forget_instance, owned_instance, remember_instance
    calls = _server(monkeypatch, [_instance("owned-previous")], load_response={})
    open_request = models._urlopen_model_catalog_request

    def vanished(request, *, timeout):
        if request.full_url.endswith("/unload"):
            raise urllib.error.HTTPError(request.full_url, 404, "missing instance", None, None)
        return open_request(request, timeout=timeout)

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", vanished)
    remember_instance(SERVER_ROOT, PREVIOUS, "owned-previous")
    try:
        result = _load()
        assert owned_instance(SERVER_ROOT, PREVIOUS) is None
    finally:
        forget_instance(SERVER_ROOT, PREVIOUS)
    assert result.context_length is None
    assert [payload["model"] for url, payload in calls if url.endswith("/load")] == [MODEL]


def test_web_policy_partial_update_preserves_route_and_context():
    from hermes_cli.config import load_config, save_config
    from hermes_cli.web_server_config import _denormalize_config_from_web, _normalize_config_for_web
    save_config({"model": {"provider": "lmstudio", "default": MODEL, "base_url": BASE_URL, "context_length": 4096}})
    update = _denormalize_config_from_web({"model_lmstudio_unload_policy": "never"})
    save_config(update)
    model = load_config()["model"]
    assert model["lmstudio_unload_policy"] == "never"
    assert model["context_length"] == 4096
    assert model["provider"] == "lmstudio"
    assert model["base_url"] == BASE_URL
    assert _normalize_config_for_web(load_config())["model_lmstudio_unload_policy"] == "never"
    assert "model_lmstudio_unload_policy" not in load_config()


def test_web_policy_can_upgrade_bare_model_and_reset_default():
    from hermes_cli.config import load_config, save_config
    from hermes_cli.web_server_config import _denormalize_config_from_web
    save_config({"model": MODEL})
    save_config(_denormalize_config_from_web({"model_lmstudio_unload_policy": "never"}))
    save_config(_denormalize_config_from_web({"model_lmstudio_unload_policy": "always"}))
    assert load_config()["model"] == {"default": MODEL, "lmstudio_unload_policy": "always"}
