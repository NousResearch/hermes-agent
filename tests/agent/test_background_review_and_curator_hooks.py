"""Behavior-contract tests for ``pre_background_review``, ``post_background_review``,
and ``pre_curator_run`` plugin lifecycle hooks."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _plugins():
    return importlib.import_module("hermes_cli.plugins")


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    yield {"home": home, "curator": curator}


@pytest.fixture(autouse=True)
def _fresh_plugin_manager():
    plugins = _plugins()
    original = plugins._plugin_manager
    plugins._plugin_manager = plugins.PluginManager()
    yield
    plugins._plugin_manager = original


def _make_parent_agent() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="sess-42",
        provider="anthropic",
        model="claude-sonnet-4-6",
        memory_notifications="on",
        background_review_callback=None,
        _safe_print=lambda *_a, **_kw: None,
        _emit_auxiliary_failure=lambda *_a, **_kw: None,
    )


def _snapshot_with_skill(skill_name: str = "deploy-helper") -> list[dict]:
    return [
        {"role": "user", "content": "Help deploy."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-prior-1",
                    "function": {"name": "skill_view", "arguments": json.dumps({"name": skill_name})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-prior-1", "content": json.dumps({"success": True})},
        {"role": "assistant", "content": "Done."},
    ]


def test_pre_background_review_augment_and_post_background_review_observe(monkeypatch):
    from agent import background_review as br
    from hermes_cli.hooks import _DEFAULT_PAYLOADS

    pre_seen: list[dict] = []
    post_seen: list[dict] = []
    fork_calls: list[dict] = []

    pm = _plugins().get_plugin_manager()
    pm._hooks.setdefault("pre_background_review", []).append(
        lambda **kw: (pre_seen.append(kw), {"append_prompt": "SIM_HINT: check P1", "extra_tools": ["custom_probe"]})[1]
    )
    pm._hooks.setdefault("post_background_review", []).append(lambda **kw: post_seen.append(kw))

    def _fake_fork(agent, messages_snapshot, prompt, task_cfg, review_run, st, review_memory, explicit):
        fork_calls.append({"prompt": prompt, "task_cfg": task_cfg, "review_memory": review_memory, "explicit": explicit})
        st.review_usage.update({"model": "claude-sonnet-4-6", "provider": "anthropic", "input_tokens": 500, "api_calls": 1})
        st.review_messages = list(messages_snapshot) + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-new-1",
                        "function": {
                            "name": "skill_manage",
                            "arguments": json.dumps({"action": "patch", "name": "deploy-helper", "old_string": "a", "new_string": "b"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-new-1",
                "content": json.dumps({"success": True, "message": "Skill 'deploy-helper' patched"}),
            },
        ]

    monkeypatch.setattr(br, "_run_review_fork", _fake_fork)
    br._run_review_in_thread(
        _make_parent_agent(),
        _snapshot_with_skill("deploy-helper"),
        "BASE_PROMPT",
        task_cfg={"enabled": True},
        review_memory=False,
    )

    assert len(pre_seen) == 1
    assert set(pre_seen[0]) - {"telemetry_schema_version"} == set(_DEFAULT_PAYLOADS["pre_background_review"])
    assert pre_seen[0]["skills_loaded"] == ["deploy-helper"]
    assert len(fork_calls) == 1
    assert fork_calls[0]["prompt"] == "BASE_PROMPT\n\nSIM_HINT: check P1"
    assert fork_calls[0]["task_cfg"]["extra_tools"] == ["custom_probe"]

    assert len(post_seen) == 1
    assert set(post_seen[0]) - {"telemetry_schema_version"} == set(_DEFAULT_PAYLOADS["post_background_review"])
    assert post_seen[0]["outcome"] == "ok"
    assert post_seen[0]["result_kind"] == "skill"
    assert post_seen[0]["skills_touched"] == ["deploy-helper"]
    assert post_seen[0]["tool_calls"][0]["action"] == "patch"


def test_pre_background_review_skip_overrides_fork_and_notifies_post(monkeypatch):
    from agent import background_review as br

    fork_called = False
    post_seen: list[dict] = []

    def _should_not_run(*_a, **_kw):
        nonlocal fork_called
        fork_called = True

    pm = _plugins().get_plugin_manager()
    pm._hooks.setdefault("pre_background_review", []).append(
        lambda **_kw: {"action": "skip", "reason": "handled_by_plugin"}
    )
    pm._hooks.setdefault("post_background_review", []).append(lambda **kw: post_seen.append(kw))
    monkeypatch.setattr(br, "_run_review_fork", _should_not_run)

    run = br._BackgroundReviewRun()
    br._run_review_in_thread(
        _make_parent_agent(),
        _snapshot_with_skill(),
        "BASE_PROMPT",
        review_run=run,
    )

    assert not fork_called
    assert run.request_done.is_set()
    assert len(post_seen) == 1
    assert post_seen[0]["outcome"] == "skipped"
    assert post_seen[0]["skip_reason"] == "handled_by_plugin"


def test_pre_background_review_fail_open_on_callback_error(monkeypatch):
    from agent import background_review as br

    fork_prompts: list[str] = []

    def _boom(**_kw):
        raise RuntimeError("plugin bug")

    pm = _plugins().get_plugin_manager()
    pm._hooks.setdefault("pre_background_review", []).append(_boom)
    monkeypatch.setattr(
        br,
        "_run_review_fork",
        lambda _a, _m, prompt, *_rest: fork_prompts.append(prompt),
    )

    br._run_review_in_thread(_make_parent_agent(), _snapshot_with_skill(), "BASE_PROMPT")
    assert fork_prompts == ["BASE_PROMPT"]


def test_pre_curator_run_skip_prevents_mutations_and_matches_payload_shape(isolated_home):
    from hermes_cli.hooks import _DEFAULT_PAYLOADS

    curator = isolated_home["curator"]
    seen: list[dict] = []
    pm = _plugins().get_plugin_manager()
    pm._hooks.setdefault("pre_curator_run", []).append(
        lambda **kw: (seen.append(kw), {"action": "skip", "reason": "plugin_owns_curation"})[1]
    )

    result = curator.run_curator_review(synchronous=True, consolidate=True, trigger="scheduled")
    assert result["skipped"] is True
    assert result["skip_reason"] == "plugin_owns_curation"
    assert curator.load_state().get("run_count", 0) == 0
    assert len(seen) == 1
    assert set(seen[0]) - {"telemetry_schema_version"} == set(_DEFAULT_PAYLOADS["pre_curator_run"])


def test_pre_curator_run_augment_and_skip_llm(isolated_home, monkeypatch):
    curator = isolated_home["curator"]
    seen_prompts: list[str] = []

    # Seed one agent-created skill so the LLM pass has a candidate.
    skill_dir = isolated_home["home"] / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo-skill\ndescription: Demo.\n---\n# Demo\n", encoding="utf-8")
    from tools import skill_usage
    skill_usage.record_created("demo-skill", agent_created=True)

    monkeypatch.setattr(
        curator,
        "_run_llm_review",
        lambda prompt: (seen_prompts.append(prompt), curator._llm_meta("ok"))[1],
    )

    pm = _plugins().get_plugin_manager()
    pm._hooks.setdefault("pre_curator_run", []).append(
        lambda **_kw: {"append_prompt": "GUARD_NOTE: pin mastered rules"}
    )
    curator.run_curator_review(synchronous=True, consolidate=True, trigger="manual")
    assert len(seen_prompts) == 1
    assert "GUARD_NOTE: pin mastered rules" in seen_prompts[0]

    seen_prompts.clear()
    pm._hooks["pre_curator_run"] = [lambda **_kw: {"action": "skip_llm", "reason": "budget_cap"}]
    curator.run_curator_review(synchronous=True, consolidate=True, trigger="manual")
    assert seen_prompts == []
    assert "skipped by hook (budget_cap)" in curator.load_state()["last_run_summary"]
