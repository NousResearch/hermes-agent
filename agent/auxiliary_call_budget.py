"""Opt-in synchronous auxiliary dispatch limits, shared across a caller's retries.

The scope never abandons a running request or starts cancellation threads. Network
SDK timeouts bound in-flight work; gates stop subsequent requests. SDK clients in
auxiliary_client disable internal retries, so each create is one provider attempt.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import threading
import time
from typing import Callable
from types import SimpleNamespace


class AuxiliaryBudgetExceeded(TimeoutError):
    """A caller's deadline or physical request allowance was exhausted."""


@dataclass
class AuxiliaryCallBudget:
    deadline: float
    max_attempts: int
    check_current: Callable | None = None
    clock: Callable = time.monotonic
    attempts: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def remaining(self) -> float:
        if self.check_current is not None:
            self.check_current()
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise AuxiliaryBudgetExceeded("deadline_exceeded")
        if self.attempts >= self.max_attempts:
            raise AuxiliaryBudgetExceeded("request_budget_exceeded")
        return remaining

    def dispatch(self, kwargs: dict) -> dict:
        with self._lock:
            remaining = self.remaining()
            request = dict(kwargs)
            request["timeout"] = min(request.get("timeout") or remaining, remaining)
            self.attempts += 1
            return request


_budget: ContextVar[AuxiliaryCallBudget | None] = ContextVar("auxiliary_call_budget", default=None)


@contextmanager
def auxiliary_call_budget(budget: AuxiliaryCallBudget):
    """Bind a caller-owned budget; tokens restore nested and concurrent scopes."""
    token = _budget.set(budget)
    try:
        yield budget
    finally:
        _budget.reset(token)


def check_auxiliary_budget() -> None:
    budget = _budget.get()
    if budget is not None:
        budget.remaining()


def auxiliary_budget_active() -> bool:
    return _budget.get() is not None


def call_on_budgeted_worker(callback, kwargs, cancel_check):
    check_auxiliary_budget()
    if callable(cancel_check) and cancel_check():
        from agent.auxiliary_client import AuxiliaryExplicitCancellation
        raise AuxiliaryExplicitCancellation()
    # No cancel-return daemon: the caller retains its worker until SDK timeout/return.
    return callback(kwargs)


def budgeted_create(client, kwargs):
    budget = _budget.get()
    if budget is None:
        return client.chat.completions.create(**kwargs)
    adapter = client.chat.completions
    if getattr(adapter, "AUXILIARY_BUDGET_UNSUPPORTED", False) is True:
        raise ValueError("provider_budget_unsupported")
    if getattr(adapter, "BUDGET_AT_NATIVE_DISPATCH", False) is True:
        budget.remaining()
        return adapter.create(**kwargs)
    return client.chat.completions.create(**budget.dispatch(kwargs))


class _BudgetedMessages:
    def __init__(self, messages, budget):
        self.messages = messages
        self.budget = budget

    def create(self, **kwargs):
        return self.messages.create(**self.budget.dispatch(kwargs))

    @contextmanager
    def stream(self, **kwargs):
        # SDK stream managers dispatch on enter, not when constructed.
        with self.messages.stream(**self.budget.dispatch(kwargs)) as stream:
            yield stream


def budgeted_messages_client(client):
    budget = _budget.get()
    return client if budget is None else SimpleNamespace(messages=_BudgetedMessages(client.messages, budget))


def acquire_auxiliary_slot(semaphore) -> None:
    budget = _budget.get()
    if budget is None:
        semaphore.acquire()
        return
    while True:
        # Poll policy while queued, without launching a request or another worker.
        if semaphore.acquire(timeout=min(0.1, budget.remaining())):
            return


def auxiliary_retry_wait(seconds: float) -> None:
    budget = _budget.get()
    if budget is None:
        time.sleep(seconds)
        return
    until = budget.clock() + seconds
    while True:
        remaining = budget.remaining()
        delay = until - budget.clock()
        if delay <= 0:
            return
        if delay >= remaining:
            raise AuxiliaryBudgetExceeded("deadline_exceeded")
        time.sleep(min(0.1, delay))
