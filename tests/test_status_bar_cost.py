"""T2.5: display.show_cost must actually show a $ segment (reader count was 0 upstream).

Data source: ``agent.session_estimated_cost_usd`` (maintained upstream); no data -> no segment.
"""
import hermes_cli.cli_status_bar_mixin as sbm
from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin


class _Bar(CLIStatusBarMixin):
    pass


def _flat(segs):
    return "".join(text for frags in segs for _, text in frags)


def _snap(**extra):
    base = {"model_short": "sonnet", "duration": "0s", "context_percent": 50,
            "context_length": 0, "context_tokens": 0}
    base.update(extra)
    return base


def test_show_cost_true_adds_dollar_segment(monkeypatch):
    import cli
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"show_cost": True}})
    segs = _Bar()._status_bar_segments(_snap(cost_usd=0.1234), 90, None, False, styled=False)
    assert "$0.1234" in _flat(segs)


def test_show_cost_false_omits_cost(monkeypatch):
    import cli
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"show_cost": False}})
    segs = _Bar()._status_bar_segments(_snap(cost_usd=0.1234), 90, None, False, styled=False)
    assert "$" not in _flat(segs)


def test_no_cost_data_omits_segment(monkeypatch):
    import cli
    monkeypatch.setattr(cli, "CLI_CONFIG", {"display": {"show_cost": True}})
    segs = _Bar()._status_bar_segments(_snap(), 90, None, False, styled=False)
    assert "$" not in _flat(segs)
