"""Opt-in, bounded re-arm of a durably observed terminal 429; not a turn scheduler."""
from __future__ import annotations

import logging
import math
import time

from hermes_cli.codex_goals import native_goal_resource_limited, native_objective
from hermes_cli.goals import load_goal

logger = logging.getLogger(__name__)


def recovery_delays(value):
    if not isinstance(value, (list, tuple)) or len(value) > 6 or any(
        isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or not 0 < n <= 3600 for n in value
    ):
        raise ValueError('agent.codex_goal_rate_limit_delays must contain at most 6 positive seconds, each <= 3600')
    return tuple(value)


def terminal_rate_limit(turn):
    """Only the completed turn's own structured HTTP error can authorize recovery."""
    if turn.interrupted or turn.should_retire or turn.terminal_status != 'failed':
        return False
    error = turn.terminal_error or {}
    message = str(error.get('message', '')).lower()
    if any(n in message for n in ('insufficient_quota', 'quota exceeded', 'credit', 'billing', 'payment', 'usage limit')):
        return False
    info = error.get('codexErrorInfo') or {}
    detail = info.get('responseTooManyFailedAttempts') if isinstance(info, dict) else None
    return isinstance(detail, dict) and detail.get('httpStatusCode') == 429


class NativeRateLimitRecovery:
    def __init__(self, session, session_id, goal_id, objective, delays, control, on_recovery):
        self.session, self.session_id, self.goal_id, self.objective = session, session_id, goal_id, objective
        self.delays, self.attempts = recovery_delays(delays), 0
        self.control, self.on_recovery = control, on_recovery

    def eligible(self, turn, native):
        return (self.attempts < len(self.delays) and terminal_rate_limit(turn)
                and native.get('status') == 'blocked' and not native_goal_resource_limited(native))

    def _authorized(self, native):
        state = load_goal(self.session_id)
        return (state is not None and state.runtime == 'codex' and state.goal_id == self.goal_id
                and state.status == 'active' and native_objective(state) == self.objective
                and state.token_budget == native.get('tokenBudget') and not self.session._interrupt_event.is_set())

    def recover(self, turn, native):
        from agent.transports.codex_app_server_goals import _native_request
        if not self.eligible(turn, native) or not self._authorized(native):
            return None
        delay = self.delays[self.attempts]
        self.attempts += 1
        notice = (f'Codex provider rate-limited (HTTP 429). Waiting {delay:g}s before native Goal recovery '
                  f'{self.attempts}/{len(self.delays)}; progress and budget preserved, no tool replay.')
        logger.warning(notice)
        if self.on_recovery is not None:
            try:
                self.on_recovery(notice)
            except Exception:
                logger.exception('Native Goal recovery notice failed')
        deadline = time.monotonic() + delay
        while True:
            self.control()
            if not self._authorized(native) or self.session._subprocess_died(turn, self.session._client):
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self.session._interrupt_event.wait(min(.25, remaining))
        current = _native_request(self.session, 'thread/goal/get')
        fields = ('threadId', 'objective', 'createdAt', 'tokenBudget')
        if (any(current.get(k) != native.get(k) for k in fields) or current.get('status') != 'blocked'
                or native_goal_resource_limited(current) or not self._authorized(current)):
            return None
        # Codex admits its OWN next turn from the persisted history. No turn/start, user input,
        # clear, budget override, new thread or hand-built continuation is sent by Hermes.
        resumed = _native_request(self.session, 'thread/goal/set', status='active')
        if any(resumed.get(k) != current.get(k) for k in fields) or resumed.get('tokensUsed', 0) < current.get('tokensUsed', 0):
            raise RuntimeError('Native Goal ledger changed during rate-limit recovery')
        self.control()  # A concurrent /stop, pause, clear or replacement still wins.
        return resumed if self._authorized(resumed) else None
