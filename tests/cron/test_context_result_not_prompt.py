from cron.scheduler_prompt import _inject_context_from


def test_continuity_keeps_latest_response_not_recursive_prompt(tmp_path,monkeypatch):
    import cron.jobs
    monkeypatch.setattr(cron.jobs,'get_cron_output_dir',lambda:tmp_path)
    folder=tmp_path/'abcdef';folder.mkdir()
    (folder/'one.md').write_text('# Cron Job: test\n\n## Prompt\nold prompt\n## Response\nold nested response\n```\nmore prompt\n\n## Response\nNEW REAL RESULT\n')
    prompt,injected=_inject_context_from({'id':'abcdef','context_from':['self']},'TASK')
    assert injected and 'NEW REAL RESULT' in prompt
    assert 'old prompt' not in prompt and 'old nested response' not in prompt


def test_continuity_retains_actual_error(tmp_path,monkeypatch):
    import cron.jobs
    monkeypatch.setattr(cron.jobs,'get_cron_output_dir',lambda:tmp_path)
    folder=tmp_path/'abcdef';folder.mkdir()
    (folder/'one.md').write_text('# Cron Job: test\n\n## Prompt\nSECRET OLD CONTEXT\n\n## Error\nFailed actual probe\n')
    prompt,injected=_inject_context_from({'id':'abcdef','context_from':['self']},'TASK')
    assert injected and 'Failed actual probe' in prompt
    assert 'SECRET OLD CONTEXT' not in prompt
