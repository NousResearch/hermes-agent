"""The /model switch summary names the pooled credential the next request will use.

A provider can hold several pooled keys (`hermes auth pool`), and the switch summary named only the
model and provider — so "which key am I on?" was unanswerable from the CLI without `hermes auth
list`. The summary now prints the label of the credential the pool will actually select
(``credential_pool.load_pool(provider).peek()``, the same source ``hermes auth list`` marks with
``←``), plus the reorder hint when the pool has more than one entry.

Contract under test: the lines follow the pool, not the model name — and
``display.show_switch_credentials: false`` suppresses them without touching the rest of the block.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.cli_model_switch_mixin import _print_switch_summary


class _FakeEntry:
    def __init__(self, label="work-key", priority=0):
        self.label = label
        self.priority = priority


class _FakePool:
    def __init__(self, entries):
        self._entries = entries

    def peek(self):
        return self._entries[0] if self._entries else None

    def entries(self):
        return list(self._entries)


def _result():
    """Minimal switch result: `model_info=None` keeps Context/Max output out of the block."""
    return SimpleNamespace(
        new_model="deepseek-flash", target_provider="deepseek", provider_label="DeepSeek",
        model_info=None, base_url="", api_key="", api_mode=None, warning_message=None)


def _cli(**overrides):
    cli = SimpleNamespace(agent=None, base_url="", api_key="", conversation_history=[])
    cli.__dict__.update(overrides)
    return cli


def _run(cli, pool):
    printed = []
    with patch("cli._cprint", side_effect=printed.append), \
            patch("agent.credential_pool.load_pool", return_value=pool), \
            patch("hermes_cli.model_switch.resolve_display_context_length", return_value=None):
        _print_switch_summary(cli, _result(), "glm-5.3-flash", one_turn=False, strict_context=False)
    return printed


def test_summary_names_the_pooled_credential_and_reorder_hint():
    """Two pooled keys: the selected label shows, and the reorder hint names the provider."""
    printed = _run(_cli(), _FakePool([_FakeEntry("work-key", 0), _FakeEntry("backup-key", 1)]))

    assert any("work-key" in line for line in printed), printed
    assert any("work-key (priority 0)" in line for line in printed), printed
    assert any("hermes auth priority deepseek" in line for line in printed), printed
    assert not any("backup-key" in line for line in printed), (
        "only the credential the pool will use is named, not the whole pool")


def test_single_entry_pool_has_no_reorder_hint():
    """One key cannot be reordered before another, so the hint would be noise."""
    printed = _run(_cli(), _FakePool([_FakeEntry("only-key", 0)]))

    assert any("only-key" in line for line in printed), printed
    assert not any("hermes auth priority" in line for line in printed), printed


def test_disabled_flag_suppresses_credential_lines_only():
    """`display.show_switch_credentials: false` drops these lines, not the switch summary."""
    printed = _run(_cli(show_switch_credentials=False), _FakePool([_FakeEntry("work-key")]))

    assert any("Model switched: deepseek-flash" in line for line in printed), printed
    assert any("Provider: DeepSeek" in line for line in printed), printed
    assert not any("work-key" in line for line in printed), printed
