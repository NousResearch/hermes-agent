from agent.prompt_builder import build_context_files_prompt


def test_context_files_prompt_compaction_disabled(monkeypatch, tmp_path):
    import agent.prompt_builder as pb

    monkeypatch.setattr(pb, '_compact_context_files_enabled', lambda: False)
    long_text = (
        'Please carefully make sure to use pytest for tests. '
        'It is important to note that this project simply uses Ruff. '
        'Do not forget to add type hints. '
    ) * 30
    (tmp_path / 'AGENTS.md').write_text(long_text)
    result = build_context_files_prompt(cwd=str(tmp_path))
    assert 'Please carefully make sure to use pytest for tests.' in result


def test_context_files_prompt_compaction_enabled(monkeypatch, tmp_path):
    import agent.prompt_builder as pb

    monkeypatch.setattr(pb, '_compact_context_files_enabled', lambda: True)
    long_text = (
        'Please carefully make sure to use pytest for tests. '
        'It is important to note that this project simply uses Ruff. '
        'Do not forget to add type hints. '
    ) * 30
    (tmp_path / 'AGENTS.md').write_text(long_text)
    result = build_context_files_prompt(cwd=str(tmp_path))
    assert 'pytest' in result
    assert 'Ruff' in result
    assert 'type hints' in result
    assert 'context files have been loaded' in result
    assert len(result) < len('# Project Context\n\nLoaded context files:\n\n## AGENTS.md\n\n' + long_text)


def test_context_files_prompt_preserves_url_code_and_path(monkeypatch, tmp_path):
    import agent.prompt_builder as pb

    monkeypatch.setattr(pb, '_compact_context_files_enabled', lambda: True)
    text = (
        'Please review `pytest -q` and see https://example.com/docs carefully before editing /tmp/demo/file.py. '
        'It is important to note that this project simply uses Ruff. '
    ) * 20
    (tmp_path / 'AGENTS.md').write_text(text)
    result = build_context_files_prompt(cwd=str(tmp_path))
    assert '`pytest -q`' in result
    assert 'https://example.com/docs' in result
    assert '/tmp/demo/file.py' in result
