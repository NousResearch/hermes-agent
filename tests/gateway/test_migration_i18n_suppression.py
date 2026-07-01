"""Fork migration: gateway.system_messages custom-text override.

Locks the fork's ``gateway.system_messages.<key>`` override layer grafted onto
upstream's i18n engine (agent/i18n.py) after the upstream/main rebase. Upstream's
own i18n behaviour is covered by tests/agent/test_i18n.py (unchanged, still
passing). Category suppression is a separate feature (see its own test module).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import agent.i18n as i18n


def _cfg(**system_messages):
    return {"gateway": {"system_messages": system_messages}, "display": {"language": "ru"}}


# ---- custom-text override (gateway.system_messages.<key>) -------------------

def test_override_replaces_catalog_value():
    with patch.object(i18n, "_load_config_dict", return_value=_cfg(draining="СВОЙ {count}")):
        i18n.reset_language_cache()
        assert i18n.t("gateway.draining", count=3) == "СВОЙ 3"
    i18n.reset_language_cache()


def test_override_missing_placeholder_stays_literal():
    with patch.object(i18n, "_load_config_dict", return_value=_cfg(draining="X {count} {bogus}")):
        i18n.reset_language_cache()
        out = i18n.t("gateway.draining", count=1)
        assert "{bogus}" in out and "1" in out  # safe formatter degrades
    i18n.reset_language_cache()


def test_name_autofill():
    with patch.object(i18n, "_load_config_dict",
                      return_value={"ui": {"theme": {"branding": {"agent_name": "Гермес"}}},
                                    "gateway": {"system_messages": {"draining": "{name} drains {count}"}},
                                    "display": {"language": "ru"}}):
        i18n.reset_language_cache()
        assert i18n.t("gateway.draining", count=2) == "Гермес drains 2"
    i18n.reset_language_cache()


# ---- marker-coupling preserved ---------------------------------------------

def test_provider_auth_envelope_stays_english_literal():
    """The raw 'Provider authentication failed: {exc}' envelope must remain a
    literal in run.py so _GATEWAY_PROVIDER_ERROR_SHAPE_RE still rewrites it."""
    src = Path("gateway/run.py").read_text(encoding="utf-8")
    assert 'f"⚠️ Provider authentication failed: {exc}"' in src


# ---- sample net-new keys resolve in Russian --------------------------------

@pytest.mark.parametrize("key,kwargs", [
    ("gateway.long_running", {"minutes": 5, "status_detail": ""}),
    ("gateway.kanban_done", {"tag": "", "task_id": "t1", "title": "X", "handoff": ""}),
    (
        "gateway.codex_gpt55_autoraise_notice",
        {"model": "gpt-5.6-terra", "cap": "272K", "from_pct": 80, "to_pct": 85},
    ),
    ("gateway.subgoal_cleared_many", {"count": 3}),
    ("gateway.tool_guardrail_halted", {"tool": "exec", "code": "E1"}),
])
def test_sample_keys_resolve_russian(key, kwargs):
    i18n.reset_language_cache()
    ru = i18n.t(key, lang="ru", **kwargs)
    en = i18n.t(key, lang="en", **kwargs)
    assert ru and ru != en
    for name in kwargs:
        assert "{" + name + "}" not in ru
