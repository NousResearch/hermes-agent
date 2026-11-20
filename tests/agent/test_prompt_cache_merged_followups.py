"""Follow-ups to the prompt-cache lifecycle and the sandbox cwd fallback, found after they merged.

- A catalog that lists a cache write at zero (free / automatic caching) does not buy cache markers.
- A learned two-session rule survives one session that keeps missing (the sample trim keeps every session).
- The keep-alive decision reads the learned-route store once.
- A sandbox cwd whose name ends in a space survives the fallback marker round trip.
"""

import itertools
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agent.prompt_cache_routes as routes
from agent.prompt_cache_lifecycle import learned_window
from agent.prompt_caching import catalog_cache_markers
from tools.environments.base import BaseEnvironment

_NOUS = "https://inference-api.nousresearch.com/v1"
_T0 = 1_000_000.0
_SESSIONS = itertools.count()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(routes.time, "time", lambda: _T0)
    return tmp_path


def _miss(model, session, *, explicit=True):
    routes.observe(provider="nous", base_url=_NOUS, model=model, explicit=explicit, read=0, write=6013,
                   gap_s=10.0, tier_s=3600.0, now=_T0, session_id=session)


@pytest.mark.parametrize(("cache_write", "markers"), [
    ("0.0000000000", False), (0, False), (0.0, False), ("0.0", False), ("0", False), (None, False), ("", False),
    ("free", False), ("nan", False), ("-1", False), ("0.000001", True), ("3.75", True), (3.75, True)])
def test_only_a_positive_catalog_cache_write_price_buys_markers(cache_write, markers):
    profile = SimpleNamespace(prompt_cache_markers_from_catalog=True, base_url=_NOUS)
    rows = {"m": {"pricing": {"cache_write": cache_write}}}
    with patch("providers.get_provider_profile", return_value=profile), \
            patch("agent.model_metadata.fetch_endpoint_model_metadata", return_value=rows):
        assert catalog_cache_markers("nous", _NOUS, "m") is markers


def test_a_two_session_rule_survives_one_session_that_keeps_missing(home):
    model = "vendor/model-a"
    _miss(model, "s1")
    _miss(model, "s2")
    assert routes.rule_for("nous", _NOUS, model, True) == "no-cache"
    for _ in range(routes._GAPS_KEPT + 4):
        _miss(model, "s2")
    assert routes.rule_for("nous", _NOUS, model, True) == "no-cache"
    entry = routes._entry(routes.route_key("nous", _NOUS, model, True), _T0)
    assert len(entry["miss_gaps"]) == len(entry["miss_gaps_sessions"]) == routes._GAPS_KEPT
    assert entry["miss_gaps_sessions"].count("s1") == 1


def test_the_sample_trim_still_drops_the_oldest_when_every_session_is_unique(home):
    model = "vendor/model-b"
    for i in range(routes._GAPS_KEPT + 3):
        _miss(model, f"chat-{i}")
    entry = routes._entry(routes.route_key("nous", _NOUS, model, True), _T0)
    assert entry["miss_gaps_sessions"] == [f"chat-{i}" for i in range(3, routes._GAPS_KEPT + 3)]


def test_unmarked_keep_alive_decision_reads_the_store_once(home, monkeypatch):
    model = "vendor/model-a"
    _miss(model, "s1")
    _miss(model, "s2")  # the explicit cache never read back: markers dropped
    loads = []
    real = routes._load
    monkeypatch.setattr(routes, "_load", lambda path, now: loads.append(1) or real(path, now))
    agent = SimpleNamespace(provider="nous", base_url=_NOUS, model=model)

    assert learned_window(agent, (_T0 + 1800, 1800.0), False) == (None, "no-cache")
    assert len(loads) == 1
    loads.clear()
    assert routes.marker_preference("nous", _NOUS, model) is False
    assert len(loads) == 1


def test_an_unmarked_route_whose_automatic_cache_read_is_warmed(home):
    model = "vendor/model-a"
    _miss(model, "s1")
    _miss(model, "s2")
    routes.observe(provider="nous", base_url=_NOUS, model=model, explicit=False, read=6000, write=0,
                   gap_s=10.0, tier_s=1800.0, now=_T0, session_id="s3")
    agent = SimpleNamespace(provider="nous", base_url=_NOUS, model=model)
    window = (_T0 + 1800, 1800.0)
    assert learned_window(agent, window, False) == (window, None)


class _BashSandbox(BaseEnvironment):
    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        return subprocess.Popen(["bash", "-c", cmd_string], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, cwd="/")

    def cleanup(self):
        pass


@pytest.mark.platforms("linux", "macos")
def test_a_fallback_into_a_directory_whose_name_ends_in_a_space_stays_there(tmp_path):
    parent = tmp_path / "parent "
    (parent / "gone").mkdir(parents=True)
    env = _BashSandbox(cwd=str(parent / "gone"), timeout=30)
    shutil.rmtree(parent / "gone")

    first = env.execute("pwd", timeout=30, allow_cwd_fallback=True)
    assert first["cwd_fallback"]["used"] == str(parent.resolve())
    assert env.cwd == str(parent.resolve())
    again = env.execute("pwd", timeout=30, allow_cwd_fallback=True)
    assert again.get("cwd_fallback") is None and again["output"].rstrip("\n") == str(parent.resolve())
