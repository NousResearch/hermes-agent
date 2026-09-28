"""Resolve knowledge paths at profile/turn scope without per-profile guide copies."""
from pathlib import Path


def guides_root():
    return Path(__file__).resolve().parent.parent / 'guides'


def is_guide(path):
    return Path(path).resolve().is_relative_to(guides_root())


def render(text):
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config_readonly
    from agent.runtime_cwd import resolve_agent_cwd
    config = load_config_readonly().get('employee', {})
    values = {'profile_home': str(get_hermes_home()), 'guides_root': str(guides_root()),
              'workdir': str(resolve_agent_cwd()), 'employee_name': str(config.get('name') or 'Hermes')}
    for key, value in values.items():
        text = text.replace('{' + key + '}', value)
    return text
