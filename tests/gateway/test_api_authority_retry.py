"""Busy API retries observe one committed execution."""
from tests.gateway.test_api_authority_execution import _probe


def test_api_busy_fifo_retries_observe_one_durable_result(tmp_path):
    _probe(tmp_path, advanced=True)
