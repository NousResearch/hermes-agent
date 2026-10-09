"""Binds the shared contract (``_contract.py``) to one adapter driver for a test module.

A test module declares ``KNOWN`` (scenario -> (pattern, "#<issue> <symptom>")) and gets:

* two module-scoped rigs: ``rig`` (the adapter's default delivery config, ``agent.disabled_toolsets:
  [file]``, supervisor-owned so ``/restart`` exits 75) and ``rig_stream`` (streaming on with the
  platform's default transport, ``platform_toolsets.<platform>: [file]``);
* one parametrized ``test_contract`` over every scenario. A KNOWN scenario runs its final assertions
  under ``known_failure(pattern, reason)`` (``Rig.gate``): it xfails only while it fails with that
  bug's own message, fails on anything else, and passes once the fix lands, in either merge order.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from tests.e2e.core.platforms import _contract as C
from tests.e2e.core.platforms._helpers import Director, GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer

# scenario -> (rig fixture, runner(rig, tag, tmp_dir))
SCENARIOS: Dict[str, Tuple[str, Callable[..., None]]] = {
    "dm_one_reply": ("rig", lambda r, t, d: C.dm_gets_exactly_one_reply(r, t)),
    "group_require_mention": ("rig", lambda r, t, d: C.group_obeys_require_mention(r, t)),
    "long_reply_split": ("rig", lambda r, t, d: C.long_reply_is_split_in_order(r, t)),
    "failed_continuation": ("rig", lambda r, t, d: C.failed_continuation_is_retried_or_reported(r, t)),
    "redelivery_one_reply": ("rig", lambda r, t, d: C.redelivered_inbound_gets_one_reply(r, t)),
    "approval_click": ("rig", lambda r, t, d: C.approval_click_by_allowlisted_user_runs_command(
        r, t, d / "victim")),
    "disabled_toolsets": ("rig", lambda r, t, d: C.disabled_toolsets_are_honored(r, t)),
    "heic_as_image": ("rig", lambda r, t, d: C.heic_document_reaches_agent_as_image(r, t)),
    "stream_reply_once": ("rig_stream", lambda r, t, d: C.streamed_reply_shown_once(r, t)),
    "stream_finalize_rejected": ("rig_stream", lambda r, t, d: C.rejected_finalize_leaves_one_copy(r, t)),
    "stream_finalize_rejected_group": ("rig_stream",
                                       lambda r, t, d: C.rejected_finalize_leaves_one_copy(r, t, group=True)),
    "stream_trailing_whitespace": ("rig_stream",
                                   lambda r, t, d: C.streamed_reply_ending_in_whitespace_shown_once(r, t)),
    "platform_toolsets": ("rig_stream", lambda r, t, d: C.platform_toolsets_are_honored(r, t)),
    # last: it restarts the default rig's gateway
    "planned_restart_notice": ("rig", lambda r, t, d: C.planned_restart_notice_once(r, t, r.restart)),
}


# platform -> the SDK distribution its adapter imports (all from the `messaging` extra)
_SDK = {"telegram": "python-telegram-bot", "discord": "discord.py", "slack": "slack-bolt"}


def scenario_params(skip: Optional[Dict[str, str]] = None) -> List[Any]:
    return [pytest.param(name, id=name, marks=[pytest.mark.skip(reason=skip[name])] if skip and name in skip else [])
            for name in SCENARIOS]


def run_scenario(name: str, request: pytest.FixtureRequest, tmp_path: Path,
                 known: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
    fixture, runner = SCENARIOS[name]
    rig = request.getfixturevalue(fixture)
    assert rig.gw.alive(), f"gateway died before {name}\n{rig.gw.tail()}"
    rig.gw.wait_idle()  # no agent run of the previous scenario still in flight
    rig.known = (known or {}).get(name)
    try:
        runner(rig, name.replace("_", ""), tmp_path)
    finally:
        rig.known = None


class _RestartableRig(C.Rig):
    def restart(self) -> None:
        self.gw.stop()
        self.gw.start()


def _short_root(factory: pytest.TempPathFactory, name: str) -> Path:
    # The gateway binds an AF_UNIX tick socket under its home: keep the path well under 108 bytes.
    return factory.mktemp(name)


def rig_fixtures(driver_cls: type) -> Tuple[Any, Any]:
    """``(rig, rig_stream)`` module-scoped fixtures for ``driver_cls``."""

    def _make(factory: pytest.TempPathFactory, label: str, extra_cfg: Dict[str, Any], extra_env: Dict[str, str]):
        # The gateway child runs this interpreter: without the adapter's SDK it only times out later.
        # (A distribution lookup: the unit-test conftest may leave an SDK stub in sys.modules.)
        sdk = _SDK[driver_cls.name]
        try:
            importlib.metadata.version(sdk)
        except importlib.metadata.PackageNotFoundError:
            pytest.fail(f"{sdk} is not installed for {sys.executable}: the test environment lacks the "
                        "`messaging` extra (`source ./activate --test-extras all,messaging`; CI passes it "
                        "through setup-pm `extras`)", pytrace=False)
        drv = driver_cls()
        drv.start()
        director = Director()
        llm = FakeLLMServer(director)
        llm.start()
        cfg = C.merge(drv.gateway_config(), extra_cfg)
        gw = GatewayUnderTest(_short_root(factory, f"{drv.name[:2]}{label}"), llm_base_url=llm.base_url,
                              config=cfg, env={**drv.gateway_env(), **extra_env}, ready=drv.connected)
        rig = _RestartableRig(gw=gw, drv=drv, director=director, llm=llm)
        try:
            gw.start()
        except BaseException:
            gw.stop()
            llm.stop()
            drv.stop()
            raise
        return rig

    def _teardown(rig: C.Rig) -> None:
        rig.gw.stop()
        for pid in rig.gw.pids:
            try:
                os.kill(pid, 9)
            except (ProcessLookupError, PermissionError):
                pass
        rig.llm.stop()
        rig.drv.stop()

    @pytest.fixture(scope="module")
    def rig(tmp_path_factory: pytest.TempPathFactory):
        # model.supports_vision: the fake model has no vision metadata, so photos would be routed
        # through vision pre-analysis and no image part could ever reach the model request
        r = _make(tmp_path_factory, "a", {"agent": {"disabled_toolsets": ["file"]}, "model": {"supports_vision": True}},
                  {"HERMES_GATEWAY_EXTERNAL_SUPERVISOR": "1"})
        yield r
        _teardown(r)

    @pytest.fixture(scope="module")
    def rig_stream(tmp_path_factory: pytest.TempPathFactory):
        name = driver_cls.name
        r = _make(tmp_path_factory, "b", {
            "streaming": {"enabled": True, "edit_interval": 0.3},
            "display": {"platforms": {name: {"streaming": True}}},
            "platform_toolsets": {name: ["file"]},
        }, {})
        yield r
        _teardown(r)

    return rig, rig_stream
