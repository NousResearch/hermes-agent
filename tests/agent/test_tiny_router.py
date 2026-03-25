from agent.tiny_router import (
    HeadPrediction,
    RouterOutput,
    build_interaction_from_history,
    classify_turn,
    validate_tiny_router_config,
)


def test_classify_turn_disabled_returns_disabled_source():
    out = classify_turn({"enabled": False}, "hello", [])
    assert out.source == "disabled"
    assert out.actionability.label == "none"


def test_classify_turn_heuristic_fallback_when_subprocess_missing():
    out = classify_turn(
        {
            "enabled": True,
            "backend": "subprocess",
            "repo_root": "/no/such/repo",
            "model_dir": "/no/such/model",
            "fallback_mode": "heuristic",
        },
        "please implement this fix now",
        [],
    )
    assert out.source == "heuristic"
    assert out.actionability.label in {"act", "review", "none"}


def test_classify_turn_parses_subprocess_json_heads(tmp_path, monkeypatch):
    repo_root = tmp_path / "tiny-router"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "predict.py").write_text("print('ok')", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class _Proc:
        returncode = 0
        stdout = """
{
  "relation_to_previous": {"label": "new", "confidence": 0.95},
  "actionability": {"label": "review", "confidence": 0.44},
  "retention": {"label": "useful", "confidence": 0.64},
  "urgency": {"label": "low", "confidence": 0.53},
  "overall_confidence": 0.64
}
""".strip()
        stderr = ""

    monkeypatch.setattr("agent.tiny_router.subprocess.run", lambda *args, **kwargs: _Proc())

    out = classify_turn(
        {
            "enabled": True,
            "backend": "subprocess",
            "repo_root": str(repo_root),
            "model_dir": str(model_dir),
            "enforce_pinned_commit": False,
            "fallback_mode": "heuristic",
        },
        "what time is it in tokyo?",
        [],
    )
    assert out.source == "subprocess"
    assert out.actionability.label == "review"
    assert out.actionability.confidence > 0.0
    assert out.overall_confidence > 0.0


def test_classify_turn_accepts_multimodal_list_input():
    out = classify_turn(
        {
            "enabled": True,
            "backend": "heuristic",
            "fallback_mode": "heuristic",
        },
        [
            {"type": "text", "text": "please implement this now"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        ],
        [],
    )
    assert out.source == "heuristic"
    assert out.actionability.label == "act"


def test_build_interaction_from_history_picks_previous_turn():
    interaction = build_interaction_from_history(
        [
            {"role": "user", "content": "set a reminder for friday"},
            {"role": "assistant", "content": "done"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "create_reminder", "arguments": "{}"}}],
            },
        ]
    )
    assert interaction["previous_text"] == "set a reminder for friday"
    assert interaction["previous_outcome"] == "success"


def test_build_interaction_from_history_handles_multimodal_content():
    interaction = build_interaction_from_history(
        [
            {"role": "user", "content": [{"type": "text", "text": "send this to alex"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
    )
    assert interaction["previous_text"] == "send this to alex"
    assert interaction["previous_action"] == "none"


def test_router_output_policy_helpers():
    out = RouterOutput(
        relation_to_previous=HeadPrediction("follow_up", 0.6),
        actionability=HeadPrediction("review", 0.9),
        retention=HeadPrediction("remember", 0.8),
        urgency=HeadPrediction("high", 0.8),
        overall_confidence=0.9,
        source="heuristic",
    )
    cfg = {
        "apply_approval_posture": True,
        "confidence_thresholds": {
            "overall": 0.4,
            "actionability": 0.6,
            "retention": 0.6,
            "urgency": 0.6,
        },
    }
    assert out.should_boost_memory_nudge(cfg) is True
    assert out.needs_terminal_review_escalation(cfg) is True
    assert out.should_use_cheap_model_route(cfg) is False


def test_validate_tiny_router_config_checks_paths(tmp_path):
    repo_root = tmp_path / "tiny-router"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "predict.py").write_text("print('ok')", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    ok, err = validate_tiny_router_config(
        {
            "enabled": True,
            "backend": "subprocess",
            "repo_root": str(repo_root),
            "model_dir": str(model_dir),
            "enforce_pinned_commit": False,
        }
    )
    assert ok is True
    assert err == ""


def test_validate_tiny_router_config_enforces_pinned_commit(tmp_path, monkeypatch):
    repo_root = tmp_path / "tiny-router"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "predict.py").write_text("print('ok')", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class _Proc:
        returncode = 0
        stdout = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        stderr = ""

    monkeypatch.setattr("agent.tiny_router.subprocess.run", lambda *args, **kwargs: _Proc())

    ok, err = validate_tiny_router_config(
        {
            "enabled": True,
            "backend": "subprocess",
            "repo_root": str(repo_root),
            "model_dir": str(model_dir),
            "pinned_commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "enforce_pinned_commit": True,
        }
    )
    assert ok is False
    assert "not pinned" in err


def test_validate_tiny_router_config_uses_revision_file_when_git_unavailable(tmp_path, monkeypatch):
    repo_root = tmp_path / "tiny-router"
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "predict.py").write_text("print('ok')", encoding="utf-8")
    (repo_root / "REVISION").write_text(
        "9d6b2a718a205d90ebe85e9a28f9b8a1f20801e4\n", encoding="utf-8"
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "not a git repository"

    monkeypatch.setattr("agent.tiny_router.subprocess.run", lambda *args, **kwargs: _Proc())

    ok, err = validate_tiny_router_config(
        {
            "enabled": True,
            "backend": "subprocess",
            "repo_root": str(repo_root),
            "model_dir": str(model_dir),
            "pinned_commit": "9d6b2a718a205d90ebe85e9a28f9b8a1f20801e4",
            "enforce_pinned_commit": True,
            "source_revision_file": "REVISION",
        }
    )
    assert ok is True
    assert err == ""
