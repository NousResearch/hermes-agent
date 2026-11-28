"""``kanban_create`` refuses a card whose assignee profile cannot load a requested skill.

The board is shared, but skills are profile islands (``~/.hermes/profiles/<name>/skills`` vs the
default tree). Before this guard, ``skills: [home-assistant-operations]`` on a fabro card was
accepted and queued, then every dispatch died at boot — four wasted attempts on 2026-11-02.
A refusal must name the skill and the profile, and must not create the card.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    kb.init_db()
    return home


def _profile(home, name, skills=()):
    """A live named profile (identity marker) with optional loadable skills."""
    profile = home / 'profiles' / name
    profile.mkdir(parents=True)
    (profile / 'config.yaml').write_text('{}\n', encoding='utf-8')
    for skill in skills:
        directory = profile / 'skills' / skill
        directory.mkdir(parents=True)
        (directory / 'SKILL.md').write_text(
            f'---\nname: {skill}\ndescription: Fixture skill.\n---\nBODY\n', encoding='utf-8')
    return profile


def _cards():
    with kbc.connect_closing() as conn:
        return kb.list_tasks(conn)


def _tool_create(**args):
    from model_tools import handle_function_call
    return json.loads(handle_function_call('kanban_create', args))


def test_tool_refuses_a_skill_the_assignee_cannot_load(home):
    _profile(home, 'adapter')

    payload = _tool_create(title='probe', assignee='adapter', skills=['nowhere-skill'])

    assert payload.get('error'), payload
    text = json.dumps(payload)
    assert 'nowhere-skill' in text and 'adapter' in text, text
    assert _cards() == [], 'an unrunnable card must never be queued'


def test_tool_creates_the_card_when_the_skill_resolves(home):
    _profile(home, 'adapter', skills=['reachable-skill'])

    payload = _tool_create(title='probe', assignee='adapter', skills=['reachable-skill'])

    assert payload.get('success') is not False, payload
    assert 'task_id' in payload, payload
    assert [task.id for task in _cards()] == [payload['task_id']]


def test_tool_still_creates_a_card_without_skills_for_an_unknown_profile(home):
    """No live profile here → its skills cannot be judged; the dispatcher's own refusal owns it."""
    payload = _tool_create(title='probe', assignee='not-a-profile', skills=['whatever'])
    assert 'task_id' in payload, payload


def _cli_create(argv):
    root = argparse.ArgumentParser(prog='hermes')
    kc.build_parser(root.add_subparsers())
    return kc._cmd_create(root.parse_args(['kanban', 'create', *argv]))


def test_cli_refuses_a_skill_the_assignee_cannot_load(home, capsys):
    _profile(home, 'adapter')

    assert _cli_create(['PROBE', '--assignee', 'adapter', '--skill', 'nowhere-skill']) != 0

    stderr = capsys.readouterr().err
    assert 'nowhere-skill' in stderr and 'adapter' in stderr, stderr
    assert _cards() == [], 'an unrunnable card must never be queued'


def test_cli_still_creates_the_card_when_the_skill_resolves(home, capsys):
    _profile(home, 'adapter', skills=['reachable-skill'])

    assert _cli_create(['PROBE', '--assignee', 'adapter', '--skill', 'reachable-skill']) == 0
    assert len(_cards()) == 1
