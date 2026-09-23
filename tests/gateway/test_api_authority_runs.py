"""Run acceptance follows the same durable FIFO as HTTP/WS turns."""
from tests.gateway.test_api_authority_execution import _probe


def test_runs_commit_before_accepted_response(tmp_path):
    _probe(tmp_path, runs=True)
