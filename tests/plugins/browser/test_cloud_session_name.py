"""Cloud browser session names keep agent-browser's socket path inside the AF_UNIX budget.

Regression for #141390: ``CloudBrowserProvider._session_name`` embedded the task id, so a
default UUID task id produced a 129-byte socket path on macOS (``/tmp`` root) and every cloud
browser call failed with "Socket path would be 129 bytes (max 103)". Same overflow on longer
temp roots: #131231.
"""
from __future__ import annotations

import os
import uuid

import pytest

from hermes_constants import SOCKET_TMPDIR_MAX_LEN, socket_safe_tmpdir

# agent-browser refuses a daemon socket path longer than this ("... (max 103)").
AGENT_BROWSER_SOCKET_PATH_MAX = 103

CLOUD_PROVIDERS = ("browserbase", "browser-use", "firecrawl")

TASK_IDS = (
    str(uuid.uuid4()),  # AIAgent.run_conversation default
    "20261019_172030_ab12cd34",  # CLI session id
    "t" * 200,
)


def _provider(name: str):
    from hermes_cli.plugins import _ensure_plugins_discovered
    from agent.browser_registry import get_provider

    _ensure_plugins_discovered()
    provider = get_provider(name)
    assert provider is not None
    return provider


def _socket_path(root: str, session_name: str) -> str:
    # Hermes points AGENT_BROWSER_SOCKET_DIR at <root>/agent-browser-<name>
    # (tools/browser_tool_session.py); agent-browser binds <name>.sock inside it.
    return os.path.join(root, f"agent-browser-{session_name}", f"{session_name}.sock")


@pytest.mark.parametrize("provider_name", CLOUD_PROVIDERS)
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_socket_path_fits_on_every_socket_root(provider_name: str, task_id: str) -> None:
    name = _provider(provider_name)._session_name(task_id)
    # This host's root, and the longest root socket_safe_tmpdir() accepts on any host.
    for root in (socket_safe_tmpdir(), "/" + "r" * (SOCKET_TMPDIR_MAX_LEN - 1)):
        path = _socket_path(root, name)
        assert len(path.encode()) <= AGENT_BROWSER_SOCKET_PATH_MAX, (root, name, len(path.encode()))


@pytest.mark.parametrize("provider_name", CLOUD_PROVIDERS)
def test_sessions_for_one_task_get_distinct_reapable_names(provider_name: str) -> None:
    provider = _provider(provider_name)
    names = {provider._session_name("task-1") for _ in range(50)}
    assert len(names) == 50  # a recreated session must not share the old socket dir
    # The orphan reaper only scans agent-browser-hermes_* (plus the local prefixes).
    assert all(name.startswith("hermes_") for name in names)
