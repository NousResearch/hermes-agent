from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache


def _make_skill(tmp_path, rel_path, body):
    p = tmp_path / rel_path
    p.mkdir(parents=True, exist_ok=True)
    (p / 'SKILL.md').write_text(body)


def test_skills_prompt_default_mode_keeps_full_guidance(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    clear_skills_system_prompt_cache(clear_snapshot=True)
    _make_skill(
        tmp_path,
        'skills/coding/python-debug',
        '---\nname: python-debug\ndescription: Debug Python scripts\n---\n',
    )

    import agent.prompt_builder as pb
    monkeypatch.setattr(pb, '_compact_skills_prompt_enabled', lambda: False)

    result = build_skills_system_prompt()
    assert 'Before replying, scan the skills below.' in result
    assert 'python-debug' in result
    assert 'Debug Python scripts' in result


def test_skills_prompt_compact_mode_shortens_preamble(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    clear_skills_system_prompt_cache(clear_snapshot=True)
    _make_skill(
        tmp_path,
        'skills/coding/python-debug',
        '---\nname: python-debug\ndescription: Debug Python scripts\n---\n',
    )

    import agent.prompt_builder as pb
    monkeypatch.setattr(pb, '_compact_skills_prompt_enabled', lambda: True)

    compact = build_skills_system_prompt()
    monkeypatch.setattr(pb, '_compact_skills_prompt_enabled', lambda: False)
    clear_skills_system_prompt_cache(clear_snapshot=True)
    full = build_skills_system_prompt()

    assert 'Scan the skill index below before replying.' in compact
    assert 'Before replying, scan the skills below.' not in compact
    assert 'For Hermes Agent setup/config/tools/providers/gateway/skills/plugins/troubleshooting, load `hermes-agent` first.' in compact
    assert 'python-debug' in compact
    assert 'Debug Python scripts' in compact
    assert len(compact) < len(full)


def test_skills_prompt_cache_key_reflects_compact_toggle(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    clear_skills_system_prompt_cache(clear_snapshot=True)
    _make_skill(
        tmp_path,
        'skills/coding/python-debug',
        '---\nname: python-debug\ndescription: Debug Python scripts\n---\n',
    )

    import agent.prompt_builder as pb
    monkeypatch.setattr(pb, '_compact_skills_prompt_enabled', lambda: False)
    full = build_skills_system_prompt()

    monkeypatch.setattr(pb, '_compact_skills_prompt_enabled', lambda: True)
    compact = build_skills_system_prompt()

    assert full != compact
    assert len(compact) < len(full)
