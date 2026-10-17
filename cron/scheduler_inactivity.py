"""Activity-generation fencing for cron inactivity verdicts."""

from contextlib import suppress
import logging


class InactivityObservation:
    """Bind an idle sample to the same activity generation consumed by interrupt."""

    def __init__(self, agent):
        self.agent = agent
        self.token = None

    def read_idle(self):
        agent = self.agent
        lock = getattr(agent, "_liveness_activity_lock", None)
        if callable(lock):
            with lock():
                self.token = (
                    getattr(agent, "_turn_liveness_activity_generation", 0),
                    getattr(agent, "_last_activity_ts", None),
                )
        try:
            return float(agent.get_activity_summary().get("seconds_since_activity", 0.0) or 0.0)
        except Exception:
            logging.getLogger(__name__).debug("Could not sample cron inactivity", exc_info=True)
            return 0.0

    def commit(self):
        agent = self.agent
        if self.token is None:
            from agent.interrupt_compat import request_hard_interrupt
            request_hard_interrupt(
                agent, "Cron job timed out (inactivity)", tool_reason="cron inactivity watchdog")
            return True
        with agent._liveness_activity_lock():
            if self.token != (
                getattr(agent, "_turn_liveness_activity_generation", 0),
                getattr(agent, "_last_activity_ts", None),
            ):
                return False
        from agent.interrupt_compat import _accepts_keyword, request_hard_interrupt
        from agent.interrupt_control import InterruptControlMixin

        if isinstance(agent, InterruptControlMixin):
            # Like hard_interrupt, bypass legacy overrides while retaining the generation fence.
            interrupt = InterruptControlMixin.interrupt.__get__(agent, type(agent))
        else:
            interrupt = agent.interrupt
            if not all(_accepts_keyword(interrupt, key) for key in (
                "hard_cancel", "tool_reason", "require_generation",
            )):
                # Compatibility only: this legacy ABI cannot atomically fence publication.
                request_hard_interrupt(
                    agent, "Cron job timed out (inactivity)", tool_reason="cron inactivity watchdog")
                return True
        # The core primitive consumes this generation at first publication under the activity lock.
        return interrupt(
            "Cron job timed out (inactivity)", hard_cancel=True,
            tool_reason="cron inactivity watchdog", require_generation=self.token[0],
        ) is not False


def raise_inactivity_timeout(agent, job_name, limit_s, observation=None):
    """Decline stale observations; only a committed stop may become a timeout."""
    from cron import scheduler

    activity = {}
    if hasattr(agent, "get_activity_summary"):
        with suppress(Exception):
            activity = agent.get_activity_summary()
    if observation is None:
        scheduler.request_hard_interrupt(
            agent, "Cron job timed out (inactivity)", tool_reason="cron inactivity watchdog")
    elif not observation.commit():
        return False
    description = activity.get("last_activity_desc", "unknown")
    seconds = activity.get("seconds_since_activity", 0)
    scheduler.logger.error(
        "Job '%s' idle for %.0fs (inactivity limit %.0fs) "
        "| last_activity=%s | iteration=%s/%s | tool=%s",
        job_name, seconds, limit_s, description, activity.get("api_call_count", 0),
        activity.get("max_iterations", 0), activity.get("current_tool") or "none")
    raise TimeoutError(
        f"Cron job '{job_name}' idle for {int(seconds)}s (limit {int(limit_s)}s) "
        f"— last activity: {description}")
