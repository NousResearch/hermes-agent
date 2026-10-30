"""Run controls are fenced to the exact canonical admission."""
from tests.gateway.test_api_authority_execution import _probe


def test_run_stop_cancels_only_its_queued_admission(tmp_path):
    _probe(tmp_path, runs="controls")
