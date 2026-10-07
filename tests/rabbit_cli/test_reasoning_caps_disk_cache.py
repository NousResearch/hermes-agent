"""The reasoning-capability disk mirror.

Every consumer of these capabilities sits on a per-request hot path that must
never block on HTTP, so a process whose in-memory cache is cold answers
"unknown" — and on that answer the profile drops a "thinking off" disable
rather than risk a 400. A short-lived process (``rabbit -p``, a cron job, a
freshly booted gateway) is ALWAYS cold, so without a disk copy that fallback is
the only behavior those runs ever get and the user keeps paying for reasoning
they turned off.

These tests pin the mirror that makes every run after the first correct from
its first turn.
"""

import json

import pytest

import rabbit_cli.models as models_mod
from rabbit_cli import models_pricing
from rabbit_cli import models_reasoning_caps


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
            "_openrouter_reasoning_caps_cache",
            "_openrouter_reasoning_caps_failed_at",
        ):
            monkeypatch.setattr(models_mod, name, None)
        for name in (
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
    assert models_reasoning_caps.openrouter_model_reasoning_capabilities(
        "deepseek/deepseek-v4-pro", allow_fetch=True
    ) is not None

    cold_process()
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    caps = models_reasoning_caps.openrouter_model_reasoning_capabilities("deepseek/deepseek-v4-pro")
    assert caps["mandatory"] is False
    mandatory = models_reasoning_caps.openrouter_model_reasoning_capabilities(
        "arcee-ai/trinity-large-thinking"
    )
    assert mandatory["mandatory"] is True


def test_stale_copy_is_still_served(cold_process, offline, monkeypatch):
    """A stale verdict beats no verdict — capabilities change rarely.

    Refusing to read an aged mirror would put every long-idle install back on
    the cold-start fallback it exists to prevent.
    """
    url = models_reasoning_caps._OPENROUTER_CATALOG_URL
    models_reasoning_caps._save_reasoning_caps_disk(
        url, {"deepseek/deepseek-v4-pro": {"supports_reasoning": True, "mandatory": False}}
    )
    raw = json.loads(models_reasoning_caps._reasoning_caps_disk_path().read_text())
    raw[url]["ts"] = 0  # epoch — far past any TTL
    models_reasoning_caps._reasoning_caps_disk_path().write_text(json.dumps(raw))

    cold_process()
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    caps = models_reasoning_caps.openrouter_model_reasoning_capabilities("deepseek/deepseek-v4-pro")
    assert caps["mandatory"] is False


def test_unreadable_mirror_degrades_to_unknown(cold_process, offline, monkeypatch):
    """A corrupt file answers "unknown", never raises into the request path."""
    path = models_reasoning_caps._reasoning_caps_disk_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not json")

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)
    assert models_reasoning_caps.openrouter_model_reasoning_capabilities("deepseek/deepseek-v4-pro") is None


def test_pricing_fetch_seeds_the_mirror(cold_process, offline, monkeypatch):
    """The picker's pricing call already holds the catalog — mirror it for free.

    Every surface that renders prices goes through here, so the common case
    never pays a second round-trip to learn the same thing.
    """
    monkeypatch.setattr(models_pricing, "_pricing_cache", {})
    monkeypatch.setattr(models_pricing, "_pricing_cache_retry_after", {})
    monkeypatch.setattr(
        models_mod, "_urlopen_model_catalog_request",
        lambda req, *, timeout: _response(_CATALOG),
    )
    models_pricing.fetch_models_with_pricing(
        base_url="https://openrouter.ai/api"
    )

    cold_process()
    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", offline)

    caps = models_reasoning_caps.openrouter_model_reasoning_capabilities("deepseek/deepseek-v4-pro")
    assert caps is not None


def test_a_missing_mirror_is_looked_for_once_per_process(cold_process, monkeypatch):
    """Coming up empty must not re-cost the lookup on every later turn.

    This path runs once per request, so both halves of the disk check have to
    be paid at most once per process.
    """
    counts = {"read": 0}
    real_read = models_reasoning_caps._read_reasoning_caps_disk

    def _counting_read():
        counts["read"] += 1
        return real_read()

    monkeypatch.setattr(models_reasoning_caps, "_read_reasoning_caps_disk", _counting_read)
    for _ in range(5):
        models_reasoning_caps.openrouter_model_reasoning_capabilities("deepseek/deepseek-v4-pro")

    assert counts == {"read": 1}
