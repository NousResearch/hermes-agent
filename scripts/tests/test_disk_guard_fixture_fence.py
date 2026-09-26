"""The fixture fence: a FAKE free-space figure must never reach a real channel.

CLASS under test — measured on this host, 2026-09-25:

  sqlite3 ~/.hermes/kanban.db \
    "select id,datetime(created_at,'unixepoch','localtime') from tasks
     where created_by='disk-guard';"
      t_849189ed | 2026-09-25 19:04:47
      t_0fbf5b8d | 2026-09-25 19:04:49

Both cards say "8.0Gi free (floor 10.0Gi)". At 19:04 the host really had 23Gi
free and the pinned production floor is 0.5Gi, so NEITHER number is a
measurement of this host — 8/10 are the literals
tests/test_disk_guard_incident_key.py feeds to dga.main(). The bytecode
mtime of that test file is the same second (19:04:43). One test there
(test_suppressed_tick_does_not_extend_the_ttl) patches deliver_wake and
deliver_telegram but NOT deliver_kanban_cto — the route that has been live
since 2026-09-24 — so the unpatched channel ran for real and manufactured two
CTO incidents out of a unit test.

Cost of the class: a synthetic page consumes a human's attention and a worker
slot, and it spends the credibility of the one channel that must be believed
during a real ENOSPC. A guard that cries wolf is the same defect as a guard
that stays silent.

The fence is derived, not enumerated: any call that would page while
DISK_GUARD_FAKE_FREE_GI is set is refused, so a channel added tomorrow is
covered without anyone remembering to list it here, and a test that forgets to
patch a channel fails loudly instead of paging production.

Run: python3 -m pytest ~/.hermes/scripts/tests/test_disk_guard_fixture_fence.py -q
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(os.path.expanduser("~/.hermes/scripts/disk_guard_alert.py"))
spec = importlib.util.spec_from_file_location("disk_guard_alert_fence", SCRIPT)
dga = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dga)

# Every function in the module that performs a real, externally visible side
# effect. DERIVED as "module-level callable named deliver_*", so a channel
# added later is fenced by this test without being named here.
CHANNELS = sorted(
    n for n in dir(dga)
    if n.startswith("deliver_") and callable(getattr(dga, n))
)


def test_there_is_at_least_one_channel_to_fence():
    """Anti-vacuity: if the naming convention changes, this test must fail
    rather than silently assert over an empty set."""
    assert CHANNELS, "no deliver_* channels discovered — the fence tests nothing"


@pytest.mark.parametrize("name", CHANNELS)
def test_no_channel_pages_while_a_fixture_free_figure_is_set(monkeypatch, name):
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    fn = getattr(dga, name)
    with pytest.raises(RuntimeError) as exc:
        fn("DISK GUARD RED\nsynthetic body")
    assert "FIXTURE" in str(exc.value), (
        f"{name} refused for the wrong reason: {exc.value}")


@pytest.mark.parametrize("name", CHANNELS)
def test_the_fence_is_the_FIRST_thing_each_channel_does(monkeypatch, name):
    """The refusal must precede every external call, not merely happen
    somewhere inside. Proven by making the environment hostile: no kanban
    binary, no bot token, no wake secret. Without the fence these raise their
    OWN error; with it they raise the FIXTURE error, which is only possible if
    the check ran first."""
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    monkeypatch.setattr(dga, "KANBAN_BIN", "/nonexistent/hermes")

    def _no_network(*a, **kw):
        raise AssertionError("channel reached the network before the fence")

    monkeypatch.setattr(dga.urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(dga.subprocess, "run", _no_network)
    with pytest.raises(RuntimeError, match="FIXTURE"):
        getattr(dga, name)("DISK GUARD RED\nbody")


@pytest.mark.parametrize("name", CHANNELS)
def test_the_fence_does_not_disarm_a_real_page(monkeypatch, name):
    """The fence must not become the new silence. With NO fixture set, each
    channel gets past fixture_active() and fails for its own honest reason.

    NOTE on how this test is made safe, learned the hard way while writing it:
    an earlier version set DISK_GUARD_BOT_TOKEN='' expecting deliver_telegram
    to fail. It does not — _bot_token() falls back to the REAL token in
    ~/.hermes/.env, and the call really sent Den a synthetic page. That is the
    same class this whole file exists to close, reproduced inside its own
    regression test. The network is therefore severed at urlopen/subprocess,
    not at a credential, because severing a credential is a guess about the
    code path while severing the transport is a fact about it."""
    monkeypatch.delenv("DISK_GUARD_FAKE_FREE_GI", raising=False)
    monkeypatch.setattr(dga, "KANBAN_BIN", "/nonexistent/hermes")

    def _no_network(*a, **kw):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(dga.urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(dga.subprocess, "run", _no_network)
    monkeypatch.setattr(dga, "wake_config",
                        lambda: {"url": "http://127.0.0.1:1/x",
                                 "secret_file": "/nonexistent/secret"})
    with pytest.raises(Exception) as exc:
        getattr(dga, name)("DISK GUARD RED\nbody")
    assert "FIXTURE" not in str(exc.value), (
        f"{name} refused a REAL page as if it were a fixture")


def test_main_with_a_fixture_lands_no_page_and_still_reports_red(
        monkeypatch, tmp_path):
    """End to end: the exact call shape the leaking unit test used
    (--floor 10 with a fake 8Gi, no channel patched) must create NO card and
    send NO message, while still returning rc=1 so a harness asserting on the
    RED branch keeps working."""
    monkeypatch.setenv("DISK_GUARD_STATE", str(tmp_path / "s.json"))
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    ran = []
    real = dga.subprocess.run
    monkeypatch.setattr(dga.subprocess, "run",
                        lambda argv, **kw: ran.append(argv) or real(argv, **kw))
    assert dga.main(["--floor", "10", "--detail", "synthetic"]) == 1
    assert not any("kanban" in str(a) for a in ran), (
        f"a fixture run shelled out to the board: {ran}")


def test_dry_run_remains_the_supported_way_to_render(monkeypatch, capsys):
    """The fence removes a footgun, not a capability."""
    monkeypatch.setenv("DISK_GUARD_FAKE_FREE_GI", "8")
    assert dga.main(["--floor", "10", "--detail", "d", "--dry-run"]) == 1
    assert "DISK GUARD RED" in capsys.readouterr().out
