"""Resolve knowledge paths at profile/turn scope without per-profile guide copies."""
from pathlib import Path


def guides_root():
    return Path(__file__).resolve().parent.parent / 'guides'


def render(text):
    from hermes_constants import get_hermes_home
    from agent.runtime_cwd import resolve_agent_cwd
    values = {'profile_home': str(get_hermes_home()), 'guides_root': str(guides_root()),
              'workdir': str(resolve_agent_cwd())}
    for key, value in values.items():
        text = text.replace('{' + key + '}', value)
    return text
