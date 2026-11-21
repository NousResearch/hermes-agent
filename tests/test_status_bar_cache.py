"""T5.3: cache-hit segment exists upstream (◎ pct) — gate it with display.show_cache.

Premise adaptation: the plan asked to ADD the segment; it already ships (``cache_hit`` in
_status_bar_segments). What was missing is the knob: display.show_cache (new default True)
turns it off.
"""
import hermes_cli.cli_status_bar_mixin as sbm
from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin


class _Bar(CLIStatusBarMixin):
    pass


def _flat(segs):
    return "".join(text for frags in segs for _, text in frags)


def _snap():
    return {"model_short": "sonnet", "duration": "0s", "context_percent": 50,
            "context_length": 0, "context_tokens": 0, "cache_hit_pct": 85.0}


def test_cache_segment_shows_by_default(monkeypatch):
    import cli
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"show_cache": True}})
    segs = _Bar()._status_bar_segments(_snap(), 90, None, False, styled=False)
    assert "85" in _flat(segs)


def test_show_cache_false_hides_cache_segment(monkeypatch):
    import cli
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"show_cache": False}})
    segs = _Bar()._status_bar_segments(_snap(), 90, None, False, styled=False)
    assert "85" not in _flat(segs)
