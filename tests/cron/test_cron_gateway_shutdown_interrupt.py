from cron import scheduler


class _Agent:
    def __init__(self):
        self.messages = []

    def interrupt(self, message):
        self.messages.append(message)


def test_interrupt_active_cron_jobs_interrupts_registered_agents():
    first = _Agent()
    second = _Agent()
    with scheduler._active_cron_agents_lock:
        scheduler._active_cron_agents.clear()
        scheduler._active_cron_agents["job-a"] = first
        scheduler._active_cron_agents["job-b"] = second

    try:
        count = scheduler.interrupt_active_cron_jobs("restart")

        assert count == 2
        assert first.messages == ["restart"]
        assert second.messages == ["restart"]
    finally:
        with scheduler._active_cron_agents_lock:
            scheduler._active_cron_agents.clear()
