"""``gateway.agent_executor_workers`` sizes the gateway-owned agent-turn pool.

Every synchronous messaging turn holds one worker for its whole duration, so
the pool size (not the CPU) bounds turn concurrency. The knob was promised
with the dedicated pool (#38909) but the size stayed hardcoded at 10.
"""
import threading

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def _bare_runner(config=None):
    runner = object.__new__(GatewayRunner)
    runner._executor = None
    runner._executor_lock = threading.Lock()
    runner._executor_closing = False
    if config is not None:
        runner.config = config
    return runner


def test_executor_pool_sized_from_config():
    runner = _bare_runner(GatewayConfig.from_dict({"gateway": {"agent_executor_workers": 3}}))
    try:
        pool = runner._get_executor()
        assert pool._max_workers == 3
        assert pool._thread_name_prefix == "hermes-gateway"
    finally:
        runner._shutdown_executor()


def test_executor_pool_default_without_config_matches_default_config():
    """A bare runner (no ``config``) and a default GatewayConfig agree on the pool size."""
    bare, configured = _bare_runner(), _bare_runner(GatewayConfig.from_dict({}))
    try:
        assert bare._get_executor()._max_workers == configured._get_executor()._max_workers
    finally:
        bare._shutdown_executor()
        configured._shutdown_executor()
