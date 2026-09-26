"""Balance observed turns when an early return or exception skips the finalizer."""

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _Observation:
    agent: Any
    pending: dict | None = None
    result: dict | None = None


_current: ContextVar[_Observation | None] = ContextVar("turn_observation", default=None)


def mark_turn_started(agent, **identity) -> None:
    observation = _current.get()
    if observation is not None and observation.agent is agent:
        observation.pending = identity


def mark_turn_finished(agent, turn_id) -> None:
    observation = _current.get()
    if (observation is not None and observation.agent is agent
            and observation.pending is not None
            and observation.pending["turn_id"] == turn_id):
        observation.pending = None


@contextmanager
def observe_turn_completion(agent):
    observation = _Observation(agent)
    token = _current.set(observation)
    error = None
    try:
        yield observation
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            if observation.pending is not None:
                from hermes_cli.lifecycle import invoke_hook

                result = observation.result or {}
                interrupted = isinstance(error, (KeyboardInterrupt, InterruptedError, asyncio.CancelledError))
                interrupted = interrupted or bool(result.get("interrupted"))
                invoke_hook(
                    "on_session_end", **observation.pending,
                    completed=bool(result.get("completed", False)) and error is None and not interrupted,
                    failed=(error is not None and not interrupted) or bool(result.get("failed", False)),
                    interrupted=interrupted,
                    turn_exit_reason=result.get("turn_exit_reason") or (
                        "interrupted" if interrupted else "exception" if error is not None else "early_return"
                    ),
                )
        except Exception:
            logger.warning("Turn-end observer notification failed", exc_info=True)
        finally:
            _current.reset(token)
