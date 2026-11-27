"""Behavior contract: `sessions list` hides automation sources by default.

Every cron fire and every delegate_task subagent run creates an untitled session
row in the store. Listing them alongside real user conversations floods
`hermes sessions list` (and the desktop session panel that consumes it). The
default exclusion covers ``tool``, ``cron`` and ``subagent``; an explicit
``--source`` opts back into any of them.
"""

from argparse import Namespace

from hermes_cli.sessions_cmd import _default_exclude


def _args(**kw):
    base = dict(sessions_action="list", source=None)
    base.update(kw)
    return Namespace(**base)


def test_default_excludes_automation_sources():
    excluded = _default_exclude(_args())
    assert excluded is not None
    assert set(excluded) == {"tool", "cron", "subagent"}


def test_explicit_source_honored_no_exclusion():
    assert _default_exclude(_args(source="cron")) is None
    assert _default_exclude(_args(source="subagent")) is None
