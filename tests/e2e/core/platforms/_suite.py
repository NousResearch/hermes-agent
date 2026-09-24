"""Binds the shared contract (``_contract.py``) to one adapter driver for a test module.

A test module declares ``KNOWN`` (scenario -> "#<issue> <symptom>") and gets:

* two module-scoped rigs: ``rig`` (the adapter's default delivery config, ``agent.disabled_toolsets:
  [file]``, supervisor-owned so ``/restart`` exits 75) and ``rig_stream`` (edit-streaming on,
  ``platform_toolsets.<platform>: [file]``);
* one parametrized ``test_contract`` over every scenario, KNOWN ones as ``xfail(strict=True)`` so
  a fix turns them red until the entry is removed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

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
    "stream_finalize_rejected": ("rig_stream", lambda r, t, d: C.rejected_finalize_leaves_one_copy(r, t)),
    "stream_trailing_whitespace": ("rig_stream",
                                   lambda r, t, d: C.streamed_reply_ending_in_whitespace_shown_once(r, t)),
    "platform_toolsets": ("rig_stream", lambda r, t, d: C.platform_toolsets_are_honored(r, t)),
    # last: it restarts the default rig's gateway
    "planned_restart_notice": ("rig", lambda r, t, d: C.planned_restart_notice_once(r, t, r.restart)),
}


def scenario_params(known: Dict[str, str], skip: Dict[str, str] | None = None) -> List[Any]:
    out = []
    for name in SCENARIOS:
        marks = []
        if name in known:
            marks.append(pytest.mark.xfail(strict=True, reason=known[name]))
        if skip and name in skip:
            marks.append(pytest.mark.skip(reason=skip[name]))
        out.append(pytest.param(name, id=name, marks=marks))
    return out


def run_scenario(name: str, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    fixture, runner = SCENARIOS[name]
    rig = request.getfixturevalue(fixture)
    assert rig.gw.alive(), f"gateway died before {name}\n{rig.gw.tail()}"
    runner(rig, name.replace("_", ""), tmp_path)


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
        r = _make(tmp_path_factory, "a", {"agent": {"disabled_toolsets": ["file"]}},
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
