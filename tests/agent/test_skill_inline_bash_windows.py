"""Native Windows execution contract for opt-in skill snippets."""
import pytest

from agent.skill_preprocessing import preprocess_skill_content, run_inline_shell


@pytest.mark.platforms("windows")
@pytest.mark.parametrize(('command', 'expected'), [
    ('printf "hello"', 'hello'),
    (':', ''),
    ('printf "diagnostic" >&2; exit 3', 'diagnostic'),
    ('printf "preferred"; printf "ignored" >&2', 'preferred'),
])
def test_inline_shell_keeps_output_contract(tmp_path, command, expected):
    assert run_inline_shell(command, tmp_path, 5) == expected


@pytest.mark.platforms("windows")
def test_inline_shell_resolver_failure_is_a_marker(tmp_path, monkeypatch):
    from tools.environments import local

    def unavailable():
        raise RuntimeError('Git Bash unavailable')

    monkeypatch.setattr(local, '_find_bash', unavailable)
    assert run_inline_shell(':', tmp_path, 5) == '[inline-shell error: Git Bash unavailable]'


@pytest.mark.platforms("windows")
def test_disabled_inline_shell_does_not_resolve_bash(tmp_path, monkeypatch):
    from tools.environments import local

    def unexpected():
        pytest.fail('disabled inline shell must not resolve an interpreter')

    monkeypatch.setattr(local, '_find_bash', unexpected)
    assert preprocess_skill_content('!`printf hello`', tmp_path, skills_cfg={}) == '!`printf hello`'


@pytest.mark.platforms("windows")
def test_inline_shell_executes_in_native_skill_directory(tmp_path):
    (tmp_path / 'marker.txt').write_text('native-skill', encoding='utf-8')
    result = preprocess_skill_content(
        'value=!`read -r value < marker.txt; printf "%s" "$value"`',
        tmp_path,
        skills_cfg={'inline_shell': True, 'inline_shell_timeout': 5},
    )
    assert result == 'value=native-skill'
