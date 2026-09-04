import json
import logging
from pathlib import Path
from types import SimpleNamespace

from agent.model_router import ModelProfile, RouterPipeline, RoutingRequest
from agent.model_router.pipeline import default_db_path
from agent.model_router.telemetry import RouterTelemetry
from gateway.run import GatewayRunner
from hermes_cli.router_cmd import cmd_router


_HYDRA_PROMPT = "run a shell command, read the file, and search the output. " * 50
_SESSION_KEY = "gateway:test-platform:chat-42"


def _runtime(provider="test-provider"):
    return {
        "api_key": None,
        "base_url": None,
        "provider": provider,
        "requested_provider": provider,
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "capabilities": {},
    }


def test_gateway_hydra_decision_is_visible_to_router_cli(tmp_path, monkeypatch, capsys):
    """The real gateway seam and CLI must share one durable telemetry DB."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text(
        """\
model_router:
  mode: suggest
  candidates:
    - model: current
      provider: test-provider
      tier: economical
      reasoning: true
      quality: 0.9
      cost: 0.2
  session_pin:
    enabled: false
  telemetry:
    enabled: true
""",
        encoding="utf-8",
    )

    runner = SimpleNamespace(_service_tier=None)
    route = GatewayRunner._resolve_turn_agent_config(
        runner,
        _HYDRA_PROMPT,
        "current",
        _runtime(),
        session_id=_SESSION_KEY,
    )

    assert route["model"] == "current"  # suggest mode never changes runtime behavior
    assert route["model_router"]["stage"] == "hydra_match"
    assert route["model_router"]["reason"] == "multi_objective"
    assert default_db_path() == hermes_home / "state" / "model_router" / "router.db"

    cmd_router(SimpleNamespace(router_action="stats"))
    stats = json.loads(capsys.readouterr().out)
    assert stats["total"] == 1
    assert stats["by_stage"] == {"hydra_match": 1}

    cmd_router(
        SimpleNamespace(router_action="history", limit=20, session=_SESSION_KEY)
    )
    history = json.loads(capsys.readouterr().out)
    assert len(history) == 1
    assert history[0]["session_id"] == _SESSION_KEY
    assert history[0]["mode"] == "suggest"
    assert history[0]["stage"] == "hydra_match"
    assert history[0]["selected_model"] == "current"

    # Candidate telemetry is metadata only; prompt content is never persisted.
    import sqlite3

    with sqlite3.connect(default_db_path()) as conn:
        candidates_json, prompt_chars = conn.execute(
            "SELECT candidates_json, prompt_chars FROM routing_history"
        ).fetchone()
    candidates = json.loads(candidates_json)
    assert len(candidates) == 1
    assert candidates[0]["model"] == "current"
    assert isinstance(candidates[0]["score"], float)
    assert candidates[0]["rejected"] is None
    assert prompt_chars == len(_HYDRA_PROMPT)
    assert _HYDRA_PROMPT not in Path(default_db_path()).read_bytes().decode(
        "utf-8", errors="ignore"
    )


def test_sqlite_write_failure_is_observable_but_does_not_break_routing(
    tmp_path, monkeypatch, caplog
):
    telemetry = RouterTelemetry(tmp_path / "router.db")

    def fail_connect():
        raise OSError("simulated telemetry failure")

    monkeypatch.setattr(telemetry, "_connect", fail_connect)
    pipeline = RouterPipeline(
        [ModelProfile("current", tier="economical", reasoning=True)],
        telemetry=telemetry,
    )

    with caplog.at_level(logging.WARNING, logger="agent.model_router.telemetry"):
        decision = pipeline.route(
            RoutingRequest(_HYDRA_PROMPT, session_id=_SESSION_KEY),
            current_model="current",
            mode="suggest",
        )

    assert decision.selected_model == "current"
    assert decision.stage == "hydra_match"
    assert "Model router telemetry write failed" in caplog.text
    assert _HYDRA_PROMPT not in caplog.text


def test_unexpected_recorder_failure_is_fail_closed(caplog):
    class BrokenTelemetry:
        def record(self, request, decision, *, mode, session_id):
            raise OSError("simulated telemetry failure")

    pipeline = RouterPipeline(
        [ModelProfile("current", tier="economical", reasoning=True)],
        telemetry=BrokenTelemetry(),
    )

    with caplog.at_level(logging.WARNING, logger="agent.model_router.pipeline"):
        decision = pipeline.route(
            RoutingRequest(_HYDRA_PROMPT, session_id=_SESSION_KEY),
            current_model="current",
            mode="suggest",
        )

    assert decision.selected_model == "current"
    assert decision.stage == "hydra_match"
    assert "Model router telemetry write failed" in caplog.text
    assert _HYDRA_PROMPT not in caplog.text
