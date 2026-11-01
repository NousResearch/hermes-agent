"""A job that pins its own model can still name the providers that may rescue it.

Before this, a pinned cron job had no fallback chain at all (``scoped_fallback_chain(pinned=True)``
with ``declared=None``), so one throttled provider meant a failed run even when a chain was
configured. ``_job_fallback_chain`` now reads the job's own ``fallback_providers`` first, then
``cron.fallback_providers`` as a fleet-wide chain for pinned jobs, and keeps the global chain for
unpinned jobs only.
"""


from cron.scheduler import _job_fallback_chain

FLEET = [
    {"provider": "openrouter", "model": "deepseek/deepseek-v4.1-flash"},
    {"provider": "openai-codex", "model": "gpt-5.6-terra"},
]
PIN = {"id": "abc123", "model": "claude-haiku-5-5", "provider": "anthropic"}


def _models(chain):
    return [entry["model"] for entry in (chain or [])]


def test_pinned_job_with_its_own_chain_uses_it():
    job = {**PIN, "fallback_providers": FLEET}
    assert _models(_job_fallback_chain(job, {"fallback_providers": [{"provider": "x", "model": "y"}]})) == [
        "deepseek/deepseek-v4.1-flash", "gpt-5.6-terra"]


def test_pinned_job_falls_back_to_the_cron_fleet_chain():
    cfg = {"fallback_providers": [{"provider": "other", "model": "global-only"}], "cron": {"fallback_providers": FLEET}}
    assert _models(_job_fallback_chain(PIN, cfg)) == ["deepseek/deepseek-v4.1-flash", "gpt-5.6-terra"]


def test_pinned_job_without_any_chain_stays_pinned():
    assert _job_fallback_chain(PIN, {"fallback_providers": FLEET}) is None


def test_explicit_empty_job_chain_disables_fallback():
    cfg = {"fallback_providers": FLEET, "cron": {"fallback_providers": FLEET}}
    assert _job_fallback_chain({**PIN, "fallback_providers": []}, cfg) is None


def test_unpinned_job_inherits_the_global_chain():
    assert _models(_job_fallback_chain({"id": "abc123"}, {"fallback_providers": FLEET})) == [
        "deepseek/deepseek-v4.1-flash", "gpt-5.6-terra"]
