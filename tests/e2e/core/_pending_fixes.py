"""Merge-order-safe expected failures for live gaps whose fix is an open PR.

A plain ``xfail(strict=True)`` turns main red the moment its fix merges (XPASS), and a
non-strict one guards nothing. Instead, each gap here has a PROBE: a few lines that reproduce
the defect's mechanism on the tree under test, in a throwaway interpreter with its own
``HOME``/``HERMES_HOME`` (no state leaks into the suite process). ``expect_gaps`` applies an
xfail only while a probe still reproduces its defect, and only for the bug-specific exception
that cell raises when it observes THAT leak (``raises=``): a timeout, a boot failure or any
other assertion still fails the cell. Once the fix is in the tree the cell runs as a plain
test, so it must pass. Whichever lands first, suite or fix, main stays green.

``strict=False`` is only for a gap the end-to-end cell reproduces probabilistically (a race);
the probe itself is deterministic, so the mark still disappears when the fix lands.

Same pattern as ``tests/e2e/core/delivery/_pending_fixes.py`` (PR #120344). When a fix has
landed, delete its probe and every ``Gap`` naming it.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Type

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# ``verdict`` writes to a file, not stdout: importing tui_gateway.server repoints stdout (its JSON-RPC
# channel) at stderr.
_PRELUDE = ("import json, os, sys\nfrom pathlib import Path\nsys.path.insert(0, os.getcwd())\n"
            "HOME = Path(os.environ['HERMES_HOME'])\n"
            "def verdict(v):\n    Path(os.environ['GAP_VERDICT']).write_text(v)\n")

# PR -> (extra env, script). A script records ``open`` while the defect reproduces, ``fixed`` once
# it no longer does; anything else (including a crash) fails the cell that asked.
PROBES: Dict[int, Tuple[Dict[str, str], str]] = {
    # Desktop backend: a secondary profile's session.create / lazy session info reported (and
    # stored) the LAUNCH profile's configured model.
    120319: ({}, r'''
(HOME / "config.yaml").write_text("model:\n  default: launch-model\n")
alpha = HOME / "profiles" / "alpha"
alpha.mkdir(parents=True)
(alpha / "config.yaml").write_text("model:\n  default: alpha-model\n")
from tui_gateway import server
model = server._lazy_resume_info(str(HOME), profile="alpha")["model"]
verdict({"launch-model": "open", "alpha-model": "fixed"}.get(model, f"unexpected model {model!r}"))
'''),
    # Multiplexed host: (a) the first import of gateway.run under a routed profile's home override
    # loaded THAT profile's .env into the process env; (b) session-less work (cron) for a routed
    # profile reused the shared "default" terminal environment.
    120307: ({}, r'''
alpha = HOME / "profiles" / "alpha"
alpha.mkdir(parents=True)
(alpha / ".env").write_text("C7_PROBE_ROUTED_CANARY=alpha\n")
from hermes_constants import set_hermes_home_override
set_hermes_home_override(str(alpha))
import gateway.run  # noqa: F401
bridged = os.environ.get("C7_PROBE_ROUTED_CANARY") == "alpha"
from tools.terminal_tool import _resolve_container_task_id
shared = _resolve_container_task_id(None) == "default"
verdict("open" if bridged or shared else "fixed")
'''),
    # Credential routing: (a) with no OpenRouter key, the OpenRouter resolver fell back to an
    # OPENAI_API_KEY that OPENAI_BASE_URL binds to another host; (b) the pre-request Anthropic
    # credential refresh ran for provider 'anthropic' on a foreign (alias) host.
    120299: ({"OPENAI_API_KEY": "sk-probe-bound-elsewhere", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}, r'''
(HOME / "config.yaml").write_text("model:\n  provider: custom\n  default: m\n")
from hermes_cli.runtime_provider_backends import _resolve_openrouter_runtime
openrouter_leak = _resolve_openrouter_runtime(requested_provider="openrouter").get("api_key") == "sk-probe-bound-elsewhere"
from types import SimpleNamespace
from unittest.mock import MagicMock
import agent.anthropic_credentials as creds
from agent.client_lifecycle import ClientLifecycleMixin
creds.resolve_anthropic_token = lambda **_kw: "sk-ant-probe-refreshed"
agent = SimpleNamespace(api_mode="anthropic_messages", provider="anthropic", model="claude-probe",
                        _anthropic_api_key="", _anthropic_base_url="http://127.0.0.1:9",
                        _anthropic_client=MagicMock(), _build_direct_anthropic_client=lambda *a: MagicMock(),
                        _anthropic_oauth_flag=lambda _t: False)
refresh_leak = ClientLifecycleMixin._try_refresh_anthropic_client_credentials(agent) is True
verdict("open" if openrouter_leak or refresh_leak else "fixed")
'''),
    # /model <id> --provider X adopted another provider's alias (its base_url and key) that
    # exposes the same model id.
    120295: ({"PROVIDER_B_KEY": "sk-provider-b"}, r'''
(HOME / "config.yaml").write_text(
    "model:\n  provider: provider-a\n  default: old-model\n"
    "providers:\n"
    "  provider-a:\n    base_url: https://api-a.example.com/v1\n"
    "  provider-b:\n    base_url: https://api-b.example.com/v1\n    key_env: PROVIDER_B_KEY\n")
import hermes_cli.model_switch as ms
import hermes_cli.models_validate as mv
from hermes_cli.config import load_config
ms.DIRECT_ALIASES = {"a-alias": ms.DirectAlias("shared-model", "custom", "https://alias-host.example.com/v1",
                                               api_key="sk-alias-host")}
mv.validate_requested_model = lambda *a, **kw: {"accepted": True, "persist": True, "recognized": True, "message": None}
r = ms.switch_model("shared-model", "provider-a", "old-model", current_base_url="https://api-a.example.com/v1",
                    current_api_key="sk-provider-a", explicit_provider="provider-b",
                    user_providers=load_config()["providers"])
assert r.success, r.error_message
verdict("fixed" if (r.base_url, r.api_key) == ("https://api-b.example.com/v1", "sk-provider-b") else "open")
'''),
    # tui_gateway: when the approval wait had already ended by the time the settle hook was
    # attached (answered by RPC, resolved on another surface), the sent request was never
    # withdrawn and stayed in open_requests.
    120374: ({}, r'''
from tui_gateway import server, server_requests
settled = []
server_requests.send_async = lambda method, sid, params, on_result: settled.append
server._sessions["probe"] = {"session_key": "probe-key"}
server._emit_approval_request("probe", {"request_id": "already-resolved", "description": "probe"})
verdict("fixed" if settled else "open")
'''),
}


@functools.lru_cache(maxsize=None)
def gap_open(pr: int) -> bool:
    """True while the defect PR ``pr`` fixes still reproduces on this tree."""
    extra_env, script = PROBES[pr]
    with tempfile.TemporaryDirectory(prefix=f"gap-{pr}-") as tmp:
        home = Path(tmp) / "home"
        (home / ".hermes").mkdir(parents=True)
        out = Path(tmp) / "verdict"
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PYTEST_", "HERMES_")) and not k.endswith(("_API_KEY", "_BASE_URL"))}
        env.update({"HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "GAP_VERDICT": str(out),
                    "PYTHONPATH": str(REPO_ROOT), **extra_env})
        proc = subprocess.run([sys.executable, "-c", _PRELUDE + script], cwd=str(REPO_ROOT), env=env,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        verdict = out.read_text() if proc.returncode == 0 and out.exists() else ""
    if verdict not in ("open", "fixed"):
        raise AssertionError(f"probe for #{pr} is broken (rc={proc.returncode}, verdict={verdict!r}): "
                             f"{json.dumps(proc.stdout[-800:])} {proc.stderr[-2000:]}")
    return verdict == "open"


@dataclass(frozen=True)
class Gap:
    pr: int
    reason: str
    raises: Type[BaseException] | Tuple[Type[BaseException], ...]
    strict: bool = True

    def __post_init__(self) -> None:
        assert f"#{self.pr}" in self.reason, f"reason for a #{self.pr} gap must name the PR: {self.reason!r}"
        assert self.pr in PROBES, f"#{self.pr} has no probe"


def expect_gaps(request: pytest.FixtureRequest, *gaps: Gap) -> None:
    """One xfail for this cell covering every gap whose probe still reproduces; a plain test once none does.

    pytest honours only the first matching xfail mark, so open gaps are merged: ``raises`` is the
    union of their bug-specific exceptions, and the mark is strict when any open gap is deterministic.
    """
    open_gaps = [g for g in gaps if gap_open(g.pr)]
    if not open_gaps:
        return
    raises: tuple = ()
    for g in open_gaps:
        raises += g.raises if isinstance(g.raises, tuple) else (g.raises,)
    request.applymarker(pytest.mark.xfail(
        strict=any(g.strict for g in open_gaps), raises=tuple(dict.fromkeys(raises)),
        reason=" | ".join(g.reason for g in open_gaps)))
