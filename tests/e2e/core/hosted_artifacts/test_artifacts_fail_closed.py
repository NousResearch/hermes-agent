"""The hosted artifacts server stays out unless the profile is entitled, in a real ``hermes chat -q`` turn.

Each case runs one turn against a live fake artifacts server and a fake Portal, then reads what the
model was offered (every provider request's ``tools[]``, including any Tool Search catalog) and what
the artifacts server received. A refused case must offer no ``hermes-artifacts`` tool and send the
server nothing; the ``entitled`` control proves the same harness does attach, so the refusals are
not vacuous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.hosted_artifacts._helpers import (
    SERVER,
    FakeArtifactsServer,
    build_artifacts_home,
    closed_port_url,
    configure,
    fake_portal,
    nous_token,
    offered,
)
from tests.e2e.core.mcp_plugins._helpers import FINAL, provider, run_chat_q
from tests.e2e.core.mcp_plugins._plugin_helpers import reap_tagged
from tests.fakes.fake_llm_provider import Text
from tests.fakes.nous_portal_account import account_body

pytestmark = [
    pytest.mark.platforms("linux"),  # the orphan sweep reads /proc
    pytest.mark.live_system_guard_bypass,  # reap_tagged signals only this run's tagged tree
]

TOOL_PREFIX = "mcp__hermes_artifacts__"

# case -> (account body, extra ``artifacts`` config, portal reachable, attaches)
CASES: dict[str, tuple[dict[str, Any], dict[str, Any], bool, bool]] = {
    "entitled": (account_body(), {}, True, True),
    "artifacts-access-false": (account_body(artifacts_access=False), {}, True, False),
    "artifacts-access-absent": (account_body(artifacts_access=None), {}, True, False),
    "paid-access-false": (account_body(paid_access=False), {}, True, False),
    "artifacts-disabled": (account_body(), {"enabled": False}, True, False),
    "portal-down": (account_body(), {}, False, False),
}


def _turn(root: Path, case: str) -> dict[str, Any]:
    body, extra, portal_up, _ = CASES[case]
    token = nous_token()
    with provider(lambda _record: Text(FINAL)) as llm, fake_portal(body) as portal:
        eh = build_artifacts_home(root, llm.base_url)
        server = FakeArtifactsServer(root, token, "http://portal.localhost", eh.tag).start()
        try:
            configure(eh, base_url=server.base_url, portal_url=portal.origin if portal_up else closed_port_url(),
                      token=token, **extra)
            proc = run_chat_q(eh, "Say hello.")
        finally:
            server.stop()
            reap_tagged(eh)
        mains = llm.main_requests()
        reads = portal.account_reads()
    assert proc.returncode == 0 and FINAL in proc.stdout, (
        f"turn did not complete: exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
    assert mains, "the model was never called"
    return {"mains": mains, "entries": server.entries(), "reads": reads, "eh": eh}


@pytest.mark.parametrize("case", list(CASES))
def test_hosted_artifacts_attach_only_for_an_entitled_profile(tmp_path: Path, case: str) -> None:
    attaches = CASES[case][3]
    got = _turn(tmp_path, case)
    offered_tools = [offered(body) for body in got["mains"]]
    methods = [e["msg"].get("method") for e in got["entries"] if isinstance(e["msg"], dict)]
    if attaches:
        assert all(TOOL_PREFIX in tools for tools in offered_tools), (
            f"control: the entitled profile was not offered the {SERVER} tools (harness broken?)")
        assert "tools/list" in methods, f"control: the server was never listed: {methods}"
        return
    leaked = [i for i, tools in enumerate(offered_tools) if TOOL_PREFIX in tools]
    assert not leaked, f"{case}: {SERVER} tools offered to the model in requests {leaked}"
    assert not got["entries"], f"{case}: the artifacts server received requests: {methods}"
    if case == "artifacts-disabled":
        assert not got["reads"], f"{case}: the account was read although artifacts are disabled: {got['reads']}"
    elif CASES[case][2]:
        assert got["reads"], f"{case}: refused without reading the account (vacuous)"
