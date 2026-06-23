import json
from types import SimpleNamespace


def test_codex_service_tier_log_records_safe_request_and_response(tmp_path, monkeypatch):
    from agent import codex_service_tier_log as tier_log

    monkeypatch.setattr(tier_log, "get_hermes_home", lambda: tmp_path)

    tier_log.record_codex_service_tier_request(
        model="gpt-5.5",
        issuer_kind="chatgpt_codex",
        requested_service_tier="priority",
        session_id="raw-session-id-should-not-appear",
        request_id="raw-request-id-should-not-appear",
        source="transport.build_kwargs",
    )
    tier_log.record_codex_service_tier_response(
        model="gpt-5.5",
        issuer_kind="chatgpt_codex",
        requested_service_tier="priority",
        effective_service_tier="default",
        session_id="raw-session-id-should-not-appear",
        response_id="resp-secret-should-not-appear",
        status="completed",
        source="transport.normalize_response",
    )

    log_path = tmp_path / "logs" / "codex-service-tier.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    request = json.loads(lines[0])
    response = json.loads(lines[1])

    assert request["event"] == "request"
    assert request["model"] == "gpt-5.5"
    assert request["requested_service_tier"] == "priority"
    assert request["session_hash"].startswith("sha256:")
    assert "session_id" not in request
    assert "request_id" not in request

    assert response["event"] == "response"
    assert response["requested_service_tier"] == "priority"
    assert response["effective_service_tier"] == "default"
    assert response["response_hash"].startswith("sha256:")
    assert "response_id" not in response

    raw_log = log_path.read_text(encoding="utf-8")
    assert "raw-session-id-should-not-appear" not in raw_log
    assert "raw-request-id-should-not-appear" not in raw_log
    assert "resp-secret-should-not-appear" not in raw_log


def test_codex_transport_logs_requested_and_effective_tiers(tmp_path, monkeypatch):
    from agent import codex_service_tier_log as tier_log
    from agent.transports.codex import ResponsesApiTransport

    monkeypatch.setattr(tier_log, "get_hermes_home", lambda: tmp_path)

    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        session_id="secret-session",
        request_overrides={"service_tier": "priority"},
        is_codex_backend=True,
    )
    assert kwargs["service_tier"] == "priority"

    response = SimpleNamespace(
        id="resp-secret",
        model="gpt-5.5",
        status="completed",
        service_tier="default",
        output=[],
        output_text="OK",
    )
    transport.normalize_response(response)

    log_path = tmp_path / "logs" / "codex-service-tier.jsonl"
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["request", "response"]
    assert records[0]["requested_service_tier"] == "priority"
    assert records[1]["requested_service_tier"] == "priority"
    assert records[1]["effective_service_tier"] == "default"
    assert "secret-session" not in log_path.read_text(encoding="utf-8")
    assert "resp-secret" not in log_path.read_text(encoding="utf-8")


def test_logs_command_can_tail_codex_tier_jsonl(tmp_path, monkeypatch, capsys):
    import hermes_cli.logs as logs_mod

    monkeypatch.setattr(logs_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(logs_mod, "display_hermes_home", lambda: "~/.hermes")
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "codex-service-tier.jsonl"
    log_file.write_text('{"event":"response","effective_service_tier":"default"}\n', encoding="utf-8")

    logs_mod.tail_log("codex-tier", num_lines=5)

    out = capsys.readouterr().out
    assert "codex-service-tier.jsonl" in out
    assert "effective_service_tier" in out
    assert "default" in out
