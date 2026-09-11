from agent.prompt_builder import _render_skills_index


def test_skills_large_index_keeps_every_name_and_discovery_route():
    entries = {f'category-{c}': [(f'procedure-{c}-{n}', 'Use when investigating a specific operational issue. ' * 4) for n in range(100)] for c in range(6)}
    result = _render_skills_index(entries, {}, None, {'skills_list', 'skill_view'})
    assert len(result) <= 25_000
    for category in entries.values():
        for name, _ in category:
            assert name in result
    assert 'skills_list' in result
    assert result == _render_skills_index(dict(reversed(list(entries.items()))), {}, None, {'skills_list', 'skill_view'})


def test_skills_compaction_preserves_collision_warnings_and_small_catalog():
    entries = {'general': [('conflicted', '[name collision — load via category path] Use when fixing deployment.')], 'ops': [('deploy', 'Use when deploying.') ]}
    small = _render_skills_index(entries, {}, None, None)
    assert 'deploy: Use when deploying.' in small
    entries['large'] = [(f'procedure-{n}', 'A' * 200) for n in range(200)]
    result = _render_skills_index(entries, {}, None, None)
    assert '[name collision — load via category path]' in result
    assert 'conflicted' in result
