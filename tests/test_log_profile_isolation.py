"""Regresja: rekordy jednej sesji NIE moga trafiac do logow innego profilu.

Bramka z ``gateway.multiplex_profiles`` to JEDEN proces obslugujacy wiele
profili. ``hermes_logging`` trzyma handlery plikowe na root loggerze i dokłada
je addytywnie, wiec po zainicjowaniu drugiego profilu w tym samym procesie
kazdy rekord leci do WSZYSTKICH plikow - takze prywatna tresc wlasciciela do
``profiles/<gosc>/logs/``.

Uruchomienie:
    ~/.hermes/hermes-agent/venv/bin/python -m pytest tests/test_log_profile_isolation.py -q
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def czysty_stan_logow():
    """Zdejmij handlery i zresetuj stan modulu miedzy testami."""
    import hermes_logging as hl

    hl.flush_log_queue()
    hl._stop_queue_listener()
    root = logging.getLogger()
    zapisane = list(root.handlers)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in list(hl._queued_file_handlers):
        try:
            h.close()
        except Exception:
            pass
    hl._queued_file_handlers.clear()
    hl._log_queue = None
    hl._logging_initialized = False
    hl._primary_home_key = None
    yield hl
    hl.flush_log_queue()
    hl._stop_queue_listener()
    for h in list(hl._queued_file_handlers):
        try:
            h.close()
        except Exception:
            pass
    hl._queued_file_handlers.clear()
    hl._log_queue = None
    hl._logging_initialized = False
    hl._primary_home_key = None
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in zapisane:
        root.addHandler(h)


def _tresc(root: Path, nazwa: str) -> str:
    p = root / "logs" / nazwa
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def test_rekord_wlasciciela_nie_trafia_do_logu_profilu(tmp_path, czysty_stan_logow):
    """Rekord wyemitowany BEZ scope profilu nie moze wyladowac w jego logach."""
    hl = czysty_stan_logow
    glowny = tmp_path / "main"
    gosc = tmp_path / "profiles" / "guest-x"

    hl.setup_logging(hermes_home=glowny, mode="gateway")
    hl.setup_logging(hermes_home=gosc)  # tak robi agent_init dla profilu goscia

    logging.getLogger("gateway.run").warning("PRYWATNA TRESC WLASCICIELA")
    hl.flush_log_queue()

    assert "PRYWATNA TRESC" in _tresc(glowny, "agent.log"), \
        "log glowny musi zawierac rekord wlasciciela"
    assert "PRYWATNA TRESC" not in _tresc(gosc, "agent.log"), \
        "WYCIEK: rekord wlasciciela trafil do agent.log profilu goscia"
    assert "PRYWATNA TRESC" not in _tresc(gosc, "errors.log"), \
        "WYCIEK: rekord wlasciciela trafil do errors.log profilu goscia"


def test_rekord_profilu_trafia_do_jego_logu(tmp_path, czysty_stan_logow):
    """Filtrowanie nie moze osiroca logow profilu - jego rekordy MUSZA tam byc."""
    hl = czysty_stan_logow
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    glowny = tmp_path / "main"
    gosc = tmp_path / "profiles" / "guest-x"
    hl.setup_logging(hermes_home=glowny, mode="gateway")
    hl.setup_logging(hermes_home=gosc)

    token = set_hermes_home_override(str(gosc))
    try:
        logging.getLogger("gateway.run").warning("TRESC SESJI GOSCIA")
    finally:
        reset_hermes_home_override(token)
    hl.flush_log_queue()

    assert "TRESC SESJI GOSCIA" in _tresc(gosc, "agent.log"), \
        "rekord z sesji goscia musi trafic do JEGO logu"
    assert "TRESC SESJI GOSCIA" not in _tresc(glowny, "agent.log"), \
        "rekord sesji goscia nie powinien trafiac do logu glownego"


def test_bez_multipleksowania_nic_sie_nie_zmienia(tmp_path, czysty_stan_logow):
    """Instalacja jednoprofilowa: zachowanie MUSI byc identyczne jak dotad."""
    hl = czysty_stan_logow
    glowny = tmp_path / "main"
    hl.setup_logging(hermes_home=glowny, mode="gateway")

    logging.getLogger("gateway.run").warning("ZWYKLY REKORD")
    logging.getLogger("agent.conversation_loop").info("REKORD INFO")
    hl.flush_log_queue()

    tresc = _tresc(glowny, "agent.log")
    assert "ZWYKLY REKORD" in tresc
    assert "REKORD INFO" in tresc
    assert "ZWYKLY REKORD" in _tresc(glowny, "errors.log")
    assert "REKORD INFO" not in _tresc(glowny, "errors.log"), \
        "errors.log ma zostac przy WARNING+"


def test_rekordy_bez_kontekstu_ida_do_logu_procesu(tmp_path, czysty_stan_logow):
    """Rekordy tla (cron, housekeeping) nie moga zginac."""
    hl = czysty_stan_logow
    glowny = tmp_path / "main"
    gosc = tmp_path / "profiles" / "guest-x"
    hl.setup_logging(hermes_home=glowny, mode="gateway")
    hl.setup_logging(hermes_home=gosc)

    logging.getLogger("hermes_cli.mem_trim").warning("HOUSEKEEPING")
    hl.flush_log_queue()

    assert "HOUSEKEEPING" in _tresc(glowny, "agent.log"), \
        "rekord bez scope profilu musi wyladowac w logu procesu"


def test_ostrzezenie_goscia_trafia_tylko_do_jego_errors_log(
    tmp_path, czysty_stan_logow
):
    """errors.log podlega tej samej izolacji co agent.log — to byl realny wyciek."""
    hl = czysty_stan_logow
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    glowny = tmp_path / "main"
    gosc = tmp_path / "profiles" / "guest-x"
    hl.setup_logging(hermes_home=glowny, mode="gateway")
    hl.setup_logging(hermes_home=gosc)

    token = set_hermes_home_override(str(gosc))
    try:
        logging.getLogger("agent.conversation_loop").warning("BLAD TYLKO GOSCIA")
    finally:
        reset_hermes_home_override(token)
    hl.flush_log_queue()

    assert "BLAD TYLKO GOSCIA" in _tresc(gosc, "errors.log")
    assert "BLAD TYLKO GOSCIA" not in _tresc(glowny, "errors.log")


def test_filtr_profilu_laczy_sie_z_filtrem_komponentu_gateway(
    tmp_path, czysty_stan_logow
):
    """Nowy filtr nie moze usunac istniejacego _ComponentFilter gateway.log."""
    hl = czysty_stan_logow
    glowny = tmp_path / "main"
    hl.setup_logging(hermes_home=glowny, mode="gateway")

    logging.getLogger("gateway.run").info("REKORD GATEWAY")
    logging.getLogger("agent.conversation_loop").info("REKORD AGENTA")
    hl.flush_log_queue()

    gateway_log = _tresc(glowny, "gateway.log")
    assert "REKORD GATEWAY" in gateway_log
    assert "REKORD AGENTA" not in gateway_log, \
        "gateway.log nadal musi odrzucac rekordy spoza komponentu gateway"


def test_gateway_log_tez_jest_izolowany_miedzy_profilami(
    tmp_path, czysty_stan_logow
):
    """Tam gdzie oba filtry dzialaja naraz, oba musza dzialac poprawnie."""
    hl = czysty_stan_logow
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    glowny = tmp_path / "main"
    gosc = tmp_path / "profiles" / "guest-x"
    hl.setup_logging(hermes_home=glowny, mode="gateway")
    hl.setup_logging(hermes_home=gosc, mode="gateway")

    token = set_hermes_home_override(str(gosc))
    try:
        logging.getLogger("gateway.run").info("GATEWAY GOSCIA")
        logging.getLogger("agent.conversation_loop").info("AGENT GOSCIA")
    finally:
        reset_hermes_home_override(token)
    hl.flush_log_queue()

    assert "GATEWAY GOSCIA" in _tresc(gosc, "gateway.log")
    assert "GATEWAY GOSCIA" not in _tresc(glowny, "gateway.log")
    assert "AGENT GOSCIA" not in _tresc(gosc, "gateway.log"), \
        "filtr komponentu musi dzialac takze w profilu goscia"


def test_powtorzony_setup_nie_dubluje_wpisow(tmp_path, czysty_stan_logow):
    """Deduplikacja po sciezce pliku musi przetrwac nowa logike primary-home."""
    hl = czysty_stan_logow
    glowny = tmp_path / "main"
    for _ in range(3):
        hl.setup_logging(hermes_home=glowny, mode="gateway")

    logging.getLogger("gateway.run").warning("JEDEN RAZ")
    hl.flush_log_queue()

    assert _tresc(glowny, "agent.log").count("JEDEN RAZ") == 1, \
        "wielokrotny setup_logging() nie moze dublowac wpisow"


def test_pierwszy_zainicjowany_home_bierze_rekordy_bez_scope(
    tmp_path, czysty_stan_logow
):
    """Semantyka 'first home wins' jest celowa - zakotwicz ja testem."""
    hl = czysty_stan_logow
    glowny = tmp_path / "main"
    gosc = tmp_path / "profiles" / "guest-x"
    # KOLEJNOSC ODWROTNA: profil goscia inicjalizowany PIERWSZY.
    hl.setup_logging(hermes_home=gosc, mode="gateway")
    hl.setup_logging(hermes_home=glowny)

    logging.getLogger("hermes_cli.mem_trim").warning("BEZ SCOPE")
    hl.flush_log_queue()

    assert "BEZ SCOPE" in _tresc(gosc, "agent.log"), \
        "rekord bez scope idzie do PIERWSZEGO zainicjowanego home"
    assert "BEZ SCOPE" not in _tresc(glowny, "agent.log")
