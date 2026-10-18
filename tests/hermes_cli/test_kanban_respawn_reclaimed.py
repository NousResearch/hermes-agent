"""Regressão: um last_failure_error de quota antigo não prende o card se o último run é reclaimed.

Medido em t_482df3f3 (17/10): erro de quota de 12/10 em last_failure_error e último
run reclaimed (#4026) → respawn_guarded blocker_auth para sempre. ``reclaimed`` é
recuperação do dispatcher, não diagnóstico de quota/auth.
"""

from hermes_cli import kanban_db_dispatch as kbd


def test_reclaimed_supersedes_stale_quota_blocker():
    assert "reclaimed" in kbd._RESPAWN_BLOCKER_SUPERSEDING_OUTCOMES


def test_stale_quota_text_regex_still_matches():
    # Controle positivo: o texto de quota continua reconhecido como bloqueio.
    text = "pid 85547 exited rate-limited (quota wall) — requeued without counting a failure"
    assert kbd._RESPAWN_BLOCKER_RE.search(text)
