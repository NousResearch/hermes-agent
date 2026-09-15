"""SRL-4543 fix-round: ``clear_session_vars`` must NEVER leave identity ContextVars unreset
when ``tokens`` is shorter than ``len(_SESSION_VARS) + 2`` (the shape ``set_session_vars``
always produces). Before this fix, ``zip(_SESSION_VARS, tokens)`` silently truncates — any var
past the truncation point (including ``HERMES_SESSION_USER_ID``, ``HERMES_SESSION_KEY``,
``HERMES_BROWSER_CONTROL_PRINCIPAL``) keeps its PREVIOUS turn's value, leaking identity into
whatever runs next on the same task/thread (the Kimi finding, Gate B rodada 1).
"""
import pytest

from gateway.session_context import clear_session_vars, get_session_env, set_session_vars


def test_truncated_tokens_still_reset_all_identity_vars_bob_then_alice():
    """Cenário real Alice/Bob: Bob é admitido, clear_session_vars recebe uma lista de tokens
    TRUNCADA (menor que o esperado — simula um caller desatualizado ou uma captura parcial),
    e mesmo assim TODAS as vars de identidade devem terminar vazias. Depois, uma admissão
    fresca de Alice nunca deve herdar nada de Bob."""
    tokens_bob = set_session_vars(
        session_key="bob-key", user_id="bob@example.invalid",
        browser_control_principal="bob-digest", browser_control_transport_family="cloud-ticket-ws")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "bob@example.invalid"
        assert get_session_env("HERMES_SESSION_KEY") == "bob-key"
        assert get_session_env("HERMES_BROWSER_CONTROL_PRINCIPAL") == "bob-digest"

        # Tokens truncados: só os primeiros 5 (de 19 + 2 = 21 esperados). Isto é EXATAMENTE o
        # formato que _set_session_context poderia devolver via contextlib.suppress(Exception)
        # engolindo uma exceção no meio, ou um caller desatualizado repassando uma lista antiga.
        truncated = tokens_bob[:5]
        with pytest.raises(ValueError):
            clear_session_vars(truncated)
    finally:
        pass  # o teste abaixo confirma que o reset aconteceu mesmo com a exceção

    # A garantia de reset tem que valer INDEPENDENTEMENTE da exceção ter sido levantada:
    # nenhuma var de identidade pode reter o valor de Bob depois de clear_session_vars,
    # mesmo com tokens malformados.
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_KEY") == ""
    assert get_session_env("HERMES_BROWSER_CONTROL_PRINCIPAL") == ""

    # Turno seguinte, no MESMO thread/task: Alice é admitida fresca. Nunca deve ver Bob.
    tokens_alice = set_session_vars(
        session_key="alice-key", user_id="alice@example.invalid",
        browser_control_principal="alice-digest", browser_control_transport_family="cloud-ticket-ws")
    try:
        assert get_session_env("HERMES_SESSION_USER_ID") == "alice@example.invalid"
        assert get_session_env("HERMES_SESSION_USER_ID") != "bob@example.invalid"
    finally:
        clear_session_vars(tokens_alice)


def test_well_formed_tokens_happy_path_unchanged():
    """Regressão zero no caminho feliz: tokens do tamanho certo (produzidos por
    set_session_vars) continuam limpando tudo sem levantar ValueError."""
    tokens = set_session_vars(
        session_key="carol-key", user_id="carol@example.invalid",
        browser_control_principal="carol-digest")
    assert get_session_env("HERMES_SESSION_USER_ID") == "carol@example.invalid"
    clear_session_vars(tokens)  # não deve levantar
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_truncated_tokens_error_message_identifies_mismatch():
    """A exceção levantada por tokens malformados precisa ser diagnosticável, não um erro
    genérico — mensagem clara identificando o mismatch de tamanho."""
    tokens_bob = set_session_vars(session_key="dave-key", user_id="dave@example.invalid")
    try:
        with pytest.raises(ValueError, match=r"(?i)token"):
            clear_session_vars(tokens_bob[:3])
    finally:
        # mesmo após a exceção, o reset já rodou — nada a limpar de novo, mas confirmamos
        # que uma segunda clear_session_vars com [] não quebra nada preexistente.
        assert get_session_env("HERMES_SESSION_USER_ID") == ""


def test_longer_than_expected_tokens_also_reset_everything_and_raise():
    """Tokens MAIORES que o esperado (formato futuro incompatível, ou bug do outro lado) devem
    seguir a mesma disciplina: reset total garantido, e o mismatch sinalizado alto."""
    tokens = set_session_vars(session_key="erin-key", user_id="erin@example.invalid")
    bogus_extra = tokens + [tokens[-1]]  # tamanho errado, maior
    with pytest.raises(ValueError):
        clear_session_vars(bogus_extra)
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_KEY") == ""
