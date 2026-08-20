"""The reasoning-capability disk mirror.

Every consumer of these capabilities sits on a per-request hot path that must
never block on HTTP, so a process whose in-memory cache is cold answers
"unknown" — and on that answer the Nous profile drops a "thinking off" disable
rather than risk a 400. A short-lived process (``hermes -p``, a cron job, a
freshly booted gateway) is ALWAYS cold, so without a disk copy that fallback is
the only behavior those runs ever get and the user keeps paying for reasoning
they turned off.

These tests pin the mirror that makes every run after the first correct from
its first turn.
"""

import json

import pytest

import hermes_cli.models as models_mod


_CATALOG = json.dumps({
    "data": [
        {
            "id": "deepseek/deepseek-v4-pro",
            "supported_parameters": ["reasoning", "tools"],
            "reasoning": {"mandatory": False},
        },
        {
            "id": "arcee-ai/trinity-large-thinking",
            "supported_parameters": ["reasoning"],
            "reasoning": {"mandatory": True},
        },
    ]
}).encode()


def _response(body: bytes):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return body

    return _Resp()


@pytest.fixture
def cold_process(monkeypatch):
    """Put the module in the state a freshly started process would be in."""

    def _reset():
        for name in (
            "_nous_reasoning_caps_cache",
            "_nous_reasoning_caps_failed_at",
            "_openrouter_reasoning_caps_cache",
            "_openrouter_reasoning_caps_failed_at",
        ):
            monkeypatch.setattr(models_mod, name, None)
        for name in (
            "_nous_caps_disk_checked",
            "_nous_caps_warm_started",
            "_openrouter_caps_disk_checked",
            "_openrouter_caps_warm_started",
        ):
            monkeypatch.setattr(models_mod, name, False)

    _reset()
    return _reset


@pytest.fixture
def offline():
    """Fail the test if anything reaches for the network."""

    def _boom(req, *, timeout):
        raise AssertionError(f"must not fetch: {req.full_url}")

    return _boom


def test_fetched_catalog_answers_a_later_process_offline(
    cold_process, offline, monkeypatch
):
    """The whole point: run once online, and the next run starts out correct.

    Without the mirror this second lookup is the reported bug — an unknown
    verdict, and a silently-ignored "thinking off" for the whole turn.
    """
    monkeypatch.setattr(
        models_mod, "_urlopen_model_catalog_request",
        lambda req, *, timeout: _response(_CATALOG),
    )
    assert models_mod.nous_model_reasoning_capabilities(
        "deepseek/deepseek-v4-pro", allow_fetch=True
    ) is not None

    cold_process()
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    caps = models_mod.nous_model_reasoning_capabilities("deepseek/deepseek-v4-pro")
    assert caps["mandatory"] is False
    mandatory = models_mod.nous_model_reasoning_capabilities(
        "arcee-ai/trinity-large-thinking"
    )
    assert mandatory["mandatory"] is True


def test_mirror_is_keyed_by_catalog_url(cold_process, offline, monkeypatch):
    """One catalog's verdicts must never answer for another's.

    The Portal and OpenRouter list different models, and a staging Portal
    answering for production would decide the reasoning-mandatory question for
    the wrong deployment.
    """
    monkeypatch.setattr(
        models_mod, "_urlopen_model_catalog_request",
        lambda req, *, timeout: _response(_CATALOG),
    )
    models_mod.nous_model_reasoning_capabilities(
        "deepseek/deepseek-v4-pro", allow_fetch=True
    )

    cold_process()
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    assert models_mod.openrouter_model_reasoning_capabilities(
        "deepseek/deepseek-v4-pro"
    ) is None


def test_staging_portal_does_not_read_productions_mirror(
    cold_process, offline, monkeypatch
):
    monkeypatch.setattr(
        models_mod, "_urlopen_model_catalog_request",
        lambda req, *, timeout: _response(_CATALOG),
    )
    models_mod.nous_model_reasoning_capabilities(
        "deepseek/deepseek-v4-pro", allow_fetch=True
    )

    cold_process()
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://staging.nousresearch.com")
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    assert models_mod.nous_model_reasoning_capabilities(
        "deepseek/deepseek-v4-pro"
    ) is None


def test_stale_copy_is_still_served(cold_process, offline, monkeypatch):
    """A stale verdict beats no verdict — capabilities change rarely.

    Refusing to read an aged mirror would put every long-idle install back on
    the cold-start fallback it exists to prevent.
    """
    url = models_mod.nous_catalog_url()
    models_mod._save_reasoning_caps_disk(
        url, {"deepseek/deepseek-v4-pro": {"supports_reasoning": True, "mandatory": False}}
    )
    raw = json.loads(models_mod._reasoning_caps_disk_path().read_text())
    raw[url]["ts"] = 0  # epoch — far past any TTL
    models_mod._reasoning_caps_disk_path().write_text(json.dumps(raw))

    cold_process()
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    caps = models_mod.nous_model_reasoning_capabilities("deepseek/deepseek-v4-pro")
    assert caps["mandatory"] is False


def test_unreadable_mirror_degrades_to_unknown(cold_process, offline, monkeypatch):
    """A corrupt file answers "unknown", never raises into the request path."""
    path = models_mod._reasoning_caps_disk_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)
    assert models_mod.nous_model_reasoning_capabilities("deepseek/deepseek-v4-pro") is None


def test_pricing_fetch_seeds_the_mirror(cold_process, offline, monkeypatch):
    """The picker's pricing call already holds the catalog — mirror it for free.

    Every surface that renders prices goes through here, so the common case
    never pays a second round-trip to learn the same thing.
    """
    monkeypatch.setattr(models_mod, "_pricing_cache", {})
    monkeypatch.setattr(models_mod, "_pricing_cache_retry_after", {})
    monkeypatch.setattr(
        models_mod, "_urlopen_model_catalog_request",
        lambda req, *, timeout: _response(_CATALOG),
    )
    models_mod.fetch_models_with_pricing(
        base_url="https://inference-api.nousresearch.com"
    )

    cold_process()
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    caps = models_mod.nous_model_reasoning_capabilities("deepseek/deepseek-v4-pro")
    assert caps is not None


def test_a_missing_mirror_is_looked_for_once_per_process(cold_process, monkeypatch):
    """Coming up empty must not re-cost the lookup on every later turn.

    This path runs once per request, and naming the Portal catalog resolves
    Portal credentials — which can itself reach the network to refresh a
    token. Both halves have to be paid at most once.
    """
    counts = {"url": 0, "read": 0}
    real_url = models_mod.nous_catalog_url
    real_read = models_mod._read_reasoning_caps_disk

    def _counting_url():
        counts["url"] += 1
        return real_url()

    def _counting_read():
        counts["read"] += 1
        return real_read()

    monkeypatch.setattr(models_mod, "nous_catalog_url", _counting_url)
    monkeypatch.setattr(models_mod, "_read_reasoning_caps_disk", _counting_read)
    for _ in range(5):
        models_mod.nous_model_reasoning_capabilities("deepseek/deepseek-v4-pro")

    assert counts == {"url": 1, "read": 1}
