"""Finite storage observation budgets, separate from model/tool execution time."""
from dataclasses import dataclass
import math
import threading
from contextvars import copy_context
from pathlib import Path
import time


class BoundedIO:
    """Retain one physical transport step after its observer times out.

    Threads cannot cancel an OS/filesystem call. New mutation admission must
    remain fenced until this exact thread settles; deadline != cancellation.
    """
    def __init__(self, operation, phase):
        self.done = threading.Event()
        self.result = None
        self.error = None
        self.phase = phase
        context = copy_context()
        def run():
            try:
                self.result = context.run(operation)
            except BaseException as exc:
                self.error = exc
            finally:
                self.done.set()
        self.thread = threading.Thread(target=run, name='worker-rpc-' + phase, daemon=True)
        self.thread.start()

    def wait(self, deadline):
        if not self.done.wait(remaining(deadline)):
            raise TimeoutError('worker_receipt_timeout')
        self.thread.join()
        if self.error is not None:
            raise self.error
        return self.result


@dataclass(frozen=True)
class WorkerRPCBudget:
    # 60s DB patience + 20s discovery/transport + 20s queue/maintenance,
    # followed by a reserved 20s immutable receipt observation window.
    rpc_timeout_seconds: float = 120.0
    recovery_timeout_seconds: float = 20.0
    progress_interval_seconds: float = 5.0
    close_timeout_seconds: float = 1.0

    @classmethod
    def load(cls, home):
        from agent.safe_worker_policy import worker_config_snapshot
        config = worker_config_snapshot()
        if config is None:
            from hermes_cli.config import fast_safe_load
            path = Path(home) / 'config.yaml'
            try:
                with path.open(encoding='utf-8-sig') as stream:
                    config = fast_safe_load(stream)
            except FileNotFoundError:
                return cls()
        if config is None:
            return cls()
        if not isinstance(config, dict):
            raise ValueError('invalid_worker_persistence_budget')
        settings = config.get('worker_persistence', {})
        if not isinstance(settings, dict):
            raise ValueError('invalid_worker_persistence_budget')
        defaults = cls()
        values = {}
        for name, lower, upper in (
            ('rpc_timeout_seconds', 90, 3600),
            ('recovery_timeout_seconds', 5, 300),
            ('progress_interval_seconds', .1, 30),
        ):
            value = settings.get(name, getattr(defaults, name))
            if type(value) not in (float, int) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError('invalid_worker_persistence_budget')
            values[name] = float(value)
        # Never configure the receipt wait shorter than the permitted DB retry
        # plus a finite transport margin. This does not promise DB cancellation.
        if values['rpc_timeout_seconds'] - values['recovery_timeout_seconds'] < 80:
            raise ValueError('invalid_worker_persistence_budget')
        return cls(**values)


def remaining(deadline, cap=None):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError('worker_receipt_timeout')
    return seconds if cap is None else min(seconds, cap)
