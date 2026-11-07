"""`hermes vault add` can save a payment card for any site, but only when the user types ``any`` and confirms it.

A blank answer never means any site: the wizard asks again. Exact cards are unchanged. ``hermes vault list``
shows the scope in the Origin column. The store is the real profile-scoped vault under a temp HERMES_HOME.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.vault_store import get_vault_store
from hermes_cli import vault as vault_cli

_CARD_ANSWERS = ["4111111111111111", "", "07", "2029", "123", ""]  # hidden prompts, in PAYMENT_FIELDS order


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("agent.vault_backends.base.is_installed", lambda name: False)
    return h


def _scripted(monkeypatch, answers):
    """Answer the wizard's visible prompts in order and record each prompt; card fields are read hidden."""
    answers, prompts, hidden = list(answers), [], list(_CARD_ANSWERS)
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or answers.pop(0))
    monkeypatch.setattr(vault_cli.getpass, "getpass", lambda prompt="": hidden.pop(0))
    return prompts


def _add(monkeypatch, answers):
    prompts = _scripted(monkeypatch, ["Visa", *answers])
    vault_cli._cmd_add(SimpleNamespace(kind="payment"))
    return prompts


def test_typing_any_and_confirming_saves_an_any_site_card(home, monkeypatch, capsys):
    _add(monkeypatch, ["any", "y"])
    [meta] = get_vault_store().list_items()
    assert meta.origin is None and meta.origin_match == "any_site"
    assert "origin=any site" in capsys.readouterr().out


def test_declining_the_any_site_confirmation_asks_for_a_site(home, monkeypatch):
    _add(monkeypatch, ["any", "n", "https://shop.example"])
    [meta] = get_vault_store().list_items()
    assert meta.origin == "https://shop.example" and meta.origin_match == "exact"


def test_blank_origin_never_means_any_site(home, monkeypatch):
    prompts = _add(monkeypatch, ["", "", "https://shop.example"])
    [meta] = get_vault_store().list_items()
    assert meta.origin == "https://shop.example" and meta.origin_match == "exact"
    assert len([p for p in prompts if "origin" in p.lower()]) == 3


def test_vault_list_shows_any_site(home, monkeypatch, capsys):
    store = get_vault_store()
    store.add_item("payment", "Anywhere", dict(zip(("card_number", "exp_month", "exp_year", "cvc"),
                                                   ("4111111111111111", "07", "2029", "123"))),
                   origin_match="any_site")
    store.add_item("address", "Home", {"address_line1": "1 Test Way", "city": "Testville", "postal_code": "94107",
                                       "country": "US"})
    vault_cli._cmd_list(SimpleNamespace())
    rows = {line.split("│")[4].strip(): line.split("│")[6].strip()
            for line in capsys.readouterr().out.splitlines() if line.count("│") >= 7}
    assert rows["Anywhere"] == "any site" and rows["Home"] == "-"
