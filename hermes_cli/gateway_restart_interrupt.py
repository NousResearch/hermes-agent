"""``hermes gateway restart`` once the operator agreed to interrupt live work: restart now, not after it.

``refuse_while_sessions_live`` names the live work and asks before a restart interrupts it. Every
supervised restart (systemd, launchd, an external supervisor) then sends SIGUSR1, which carries no
arguments, so the gateway still waited up to ``agent.restart_after_turn_timeout`` (30 min by default)
for that work to finish, and the CLI waited with it. The ``restart-interrupt`` control verb carries the
consent to the gateway; ``stop()``'s drain still parks the interrupted sessions for resume.
"""

from __future__ import annotations


def restart_exit_wait(pid: int, interrupt: bool) -> tuple[float, bool]:
    """``(seconds, interrupting)``: how long the CLI waits for gateway ``pid`` to exit after SIGUSR1.

    With ``interrupt`` the gateway is first asked to restart at once; once it acknowledges, the budget
    drops the after-turn wait. Without consent, or when the gateway cannot take the request (it predates
    the verb, its control socket is down, another PID answered), the full drain-aware budget applies."""
    from hermes_cli import gateway as gw

    if interrupt:
        from gateway.control_socket import query_gateway_control
        from gateway.restart import resolve_restart_exit_wait_budget
        from gateway.run_restart_interrupt_verb import RESTART_INTERRUPT_VERB

        answer = query_gateway_control(gw.get_hermes_home(), RESTART_INTERRUPT_VERB, params={"pid": pid})
        if answer and answer.get("interrupting") and answer.get("pid") == pid:
            drain, cron = gw._get_restart_drain_timeout(), gw._get_cron_drain_timeout()
            return resolve_restart_exit_wait_budget(drain, 0.0, cron), True
        print(f"⚠ Gateway PID {pid} did not accept an immediate restart (it may predate it); "
              "it finishes its live work first")
    return gw._get_restart_exit_wait_budget(), False
