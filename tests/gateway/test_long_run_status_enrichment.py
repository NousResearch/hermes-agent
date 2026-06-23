import asyncio

from gateway import long_run_status as lrs


def test_status_enrichment_config_defaults_disabled_and_small_local_model():
    cfg = lrs.config_from_gateway_config({})

    assert cfg.enabled is False
    assert cfg.local_model == "qwen2.5:0.5b"
    assert cfg.local_keep_alive == "0s"
    assert cfg.cloud_model == "deepseek/deepseek-v4-flash"


def test_status_enrichment_config_reads_nested_gateway_values():
    cfg = lrs.config_from_gateway_config(
        {
            "agent": {
                "gateway_status_enrichment": {
                    "enabled": "true",
                    "local": {
                        "enabled": "true",
                        "model": "qwen3:0.6b",
                        "timeout_seconds": "0.7",
                        "keep_alive": "0s",
                    },
                    "cloud": {
                        "enabled": "false",
                        "max_calls_per_day": "3",
                    },
                }
            }
        }
    )

    assert cfg.enabled is True
    assert cfg.local_model == "qwen3:0.6b"
    assert cfg.local_timeout_seconds == 0.7
    assert cfg.cloud_enabled is False
    assert cfg.cloud_max_calls_per_day == 3


def test_sanitize_activity_snapshot_keeps_only_safe_telemetry():
    snap = lrs.sanitize_activity_snapshot(
        {
            "elapsed_seconds": 42,
            "api_call_count": 3,
            "current_tool": "terminal",
            "last_activity_desc": "OPENROUTER_API_KEY=secret should not pass",
            "user_message": "never include transcript",
        }
    )

    assert snap == {
        "elapsed_seconds": 42,
        "api_call_count": 3,
        "current_tool": "terminal",
    }


def test_normalize_status_line_rejects_secret_or_multiline():
    assert lrs.normalize_status_line("running focused tests") == "running focused tests"
    assert lrs.normalize_status_line("line one\nline two") is None
    assert lrs.normalize_status_line("using bearer token") is None


def test_normalize_status_label_accepts_only_known_labels():
    assert lrs.normalize_status_label("Running tests.") == "running tests"
    assert (
        lrs.normalize_status_label(
            "Running focused tests, optimizing code quality, enhancing system performance."
        )
        == "running tests"
    )
    assert lrs.normalize_status_label("inventing a detailed story") is None


def test_enrich_prefers_local_and_skips_cloud(monkeypatch):
    calls = {"cloud": 0}

    monkeypatch.setattr(lrs, "_ollama_chat_sync", lambda _cfg, _snap: "running tests")

    def fake_cloud(_cfg, _snap):
        calls["cloud"] += 1
        return "cloud should not run"

    monkeypatch.setattr(lrs, "_openrouter_chat_sync", fake_cloud)
    cfg = lrs.LongRunStatusEnrichmentConfig(enabled=True)

    result = asyncio.run(lrs.enrich_long_run_status({"elapsed_seconds": 40}, cfg))

    assert result == "running tests"
    assert calls["cloud"] == 0


def test_enrich_falls_back_to_cloud_when_local_unavailable(monkeypatch):
    monkeypatch.setattr(lrs, "_ollama_chat_sync", lambda _cfg, _snap: None)
    monkeypatch.setattr(lrs, "_openrouter_chat_sync", lambda _cfg, _snap: "waiting on network")
    cfg = lrs.LongRunStatusEnrichmentConfig(enabled=True)

    result = asyncio.run(lrs.enrich_long_run_status({"elapsed_seconds": 40}, cfg))

    assert result == "waiting on network"


def test_enrich_disabled_returns_none_without_provider_calls(monkeypatch):
    monkeypatch.setattr(lrs, "_ollama_chat_sync", lambda _cfg, _snap: (_ for _ in ()).throw(AssertionError("local called")))
    monkeypatch.setattr(lrs, "_openrouter_chat_sync", lambda _cfg, _snap: (_ for _ in ()).throw(AssertionError("cloud called")))

    result = asyncio.run(
        lrs.enrich_long_run_status(
            {"elapsed_seconds": 40},
            lrs.LongRunStatusEnrichmentConfig(enabled=False),
        )
    )

    assert result is None
