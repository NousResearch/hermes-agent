"""Regression guard: a cron job's own skill list must not be filtered by ``skills.platform_disabled``.

The per-platform disabled list prunes the *interactive* skill surface: `/skills`, the agent's skill
toolset, and the slash-command gate all resolve the caller's platform and drop those skills. Cron
jobs declare their skills in the job definition instead, and the same job therefore behaves
differently depending on how it was launched:

* scheduled fire -> the run has no chat platform, ``platform_disabled`` resolves to nothing, the
  job's skills load;
* manual run triggered from a chat session -> the run inherits ``HERMES_SESSION_PLATFORM`` (e.g.
  ``feishu``), a skill that job depends on happens to be in ``platform_disabled['feishu']``, and
  ``_load_cron_skill_parts`` gets a "disabled" error back from ``skill_view`` — the job's playbook is
  silently replaced by "could not be found and were skipped" and the job degrades to running without
  its instructions.

Fix: ``skill_view(..., allow_platform_disabled=True)``, passed explicitly by
``cron/scheduler_prompt.py``. The default stays ``False``, so every interactive path is unchanged,
and the global ``skills.disabled`` list (a skill that was retired or removed) still blocks the
explicit load.

Covered here:
  T1 the interactive path (no flag) is still blocked by ``platform_disabled`` — the gate is not
     being widened for anyone else;
  T2 the explicit load (flag set) succeeds and really returns the playbook body;
  T3 global ``skills.disabled`` still wins over the flag;
  T4 end-to-end: ``_load_cron_skill_parts`` injects the playbook into the cron prompt and no longer
     emits the "skipped" notice;
  T5 the readiness preflight (``cron/scheduler_preflight._preflight_check_skills``) can still see
     the job's skill — it fails open on an unreadable skill, so the gate silently turned its
     missing-prerequisite check into a no-op.
  T6 both cron call sites pass the flag *themselves* — a widened fake signature would otherwise
     absorb a refactor that drops it and silently restore the bug.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tools.skills_tool as skills_tool_module
from tools.skills_tool import skill_view

PLAYBOOK_BODY = "PLAYBOOK-BODY: step 1, step 2"
NEEDY_ENV_VAR = "CRON_SKILL_PLATFORM_TEST_NEEDY_KEY"


@pytest.fixture
def cron_skill_env(tmp_path, monkeypatch):
    """Isolated skills tree holding the job's skills, disabled for the chat platform we fake."""
    skills_dir = tmp_path / ".hermes" / "skills"
    playbook_dir = skills_dir / "cron-playbook"
    playbook_dir.mkdir(parents=True)
    (playbook_dir / "SKILL.md").write_text(
        "---\nname: cron-playbook\ndescription: job playbook\n---\n\n"
        f"# cron-playbook\n\n{PLAYBOOK_BODY}\n",
        encoding="utf-8",
    )
    needy_dir = skills_dir / "cron-needy"
    needy_dir.mkdir(parents=True)
    (needy_dir / "SKILL.md").write_text(
        "---\nname: cron-needy\ndescription: job playbook with an unset prerequisite\n"
        "required_environment_variables:\n"
        f"  - name: {NEEDY_ENV_VAR}\n"
        "    prompt: test key\n---\n\n# cron-needy\n\nbody\n",
        encoding="utf-8",
    )
    # `tools.skills_tool` snapshots SKILLS_DIR at import time; patch the module constant so
    # skill_view() resolves against the tree we just planted.
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", skills_dir)
    # A manual run launched from a chat session carries that platform into the cron turn.
    monkeypatch.setenv("HERMES_PLATFORM", "feishu")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "feishu")
    monkeypatch.delenv(NEEDY_ENV_VAR, raising=False)
    return skills_dir


def _config(monkeypatch, *, platform_disabled=("cron-playbook", "cron-needy"), disabled=()):
    import hermes_cli.config as config_module
    cfg = {
        "skills": {
            "platform_disabled": {"feishu": list(platform_disabled)},
            "disabled": list(disabled),
        }
    }
    monkeypatch.setattr(config_module, "load_config", lambda *a, **k: cfg)


def test_interactive_view_still_blocked_by_platform_disabled(cron_skill_env, monkeypatch):
    """T1: the interactive path keeps its gate — nothing is widened by default."""
    _config(monkeypatch)
    result = json.loads(skill_view("cron-playbook"))
    assert result.get("success") is False
    assert "disabled" in (result.get("error") or "")


def test_explicit_load_bypasses_platform_disabled(cron_skill_env, monkeypatch):
    """T2: a caller that named the skill itself can load it."""
    _config(monkeypatch)
    result = json.loads(skill_view("cron-playbook", allow_platform_disabled=True))
    assert result.get("success") is True, result.get("error")
    assert PLAYBOOK_BODY in result.get("content", "")


def test_global_disabled_still_blocks_explicit_load(cron_skill_env, monkeypatch):
    """T3: globally disabled (retired skill) is not resurrected by the flag."""
    _config(monkeypatch, platform_disabled=(), disabled=("cron-playbook",))
    result = json.loads(skill_view("cron-playbook", allow_platform_disabled=True))
    assert result.get("success") is False
    assert "disabled" in (result.get("error") or "")


def test_cron_prompt_includes_platform_disabled_skill(cron_skill_env, monkeypatch):
    """T4: the cron prompt really carries the playbook, with no 'skipped' notice."""
    _config(monkeypatch)
    import tools.skill_usage as skill_usage_module
    monkeypatch.setattr(skill_usage_module, "bump_use", lambda *a, **k: None)

    from cron.scheduler_prompt import _load_cron_skill_parts
    parts = _load_cron_skill_parts({"id": "job-1", "name": "playbook job"}, ["cron-playbook"])
    blob = "\n".join(parts)
    assert PLAYBOOK_BODY in blob
    assert "could not be found" not in blob


def test_preflight_still_checks_platform_disabled_job_skill(cron_skill_env, monkeypatch):
    """T5: the readiness preflight must not fall open on a job skill the platform gate hides.

    ``_preflight_check_skills`` skips (fail-open) any skill whose payload is not a success, so
    under the gate a job that needs an unset credential ran anyway with no warning.
    """
    _config(monkeypatch)
    from cron.scheduler_preflight import _preflight_check_skills
    reason = _preflight_check_skills({"id": "job-2", "name": "needy job", "skills": ["cron-needy"]})
    assert reason is not None and NEEDY_ENV_VAR in reason


def test_cron_call_sites_pass_the_bypass_flag_explicitly(cron_skill_env, monkeypatch):
    """T6: both cron call sites pass ``allow_platform_disabled=True`` themselves.

    ``_load_cron_skill_parts`` and ``_preflight_check_skills`` import ``skill_view`` locally, so
    the flag is the only thing tying them to the fix. Recording the kwarg (instead of widening a
    fake to ``**_kwargs`` and ignoring it) makes a future refactor that drops the argument fail
    here rather than silently restoring the skipped-playbook behaviour.
    """
    _config(monkeypatch)
    seen: list[tuple[str, object]] = []

    def _record(name, **kwargs):
        seen.append((name, kwargs.get("allow_platform_disabled")))
        return json.dumps({"success": True, "content": f"# {name}\n{PLAYBOOK_BODY}"})

    import tools.skill_usage as skill_usage_module
    monkeypatch.setattr(skill_usage_module, "bump_use", lambda *a, **k: None)
    monkeypatch.setattr(skills_tool_module, "skill_view", _record)

    from cron.scheduler_prompt import _load_cron_skill_parts
    from cron.scheduler_preflight import _preflight_check_skills

    _load_cron_skill_parts({"id": "job-3", "name": "prompt path"}, ["cron-playbook"])
    _preflight_check_skills({"id": "job-4", "name": "preflight path", "skills": ["cron-playbook"]})

    assert seen, "neither cron call site reached skill_view"
    assert all(flag is True for _, flag in seen), (
        f"a cron call site dropped allow_platform_disabled: {seen}"
    )
