"""``hermes vault add`` and the site question: an address may be saved for any site (blank
origin, the agent confirms each page at fill time), a card still needs one, and ``hermes vault
list`` shows the difference."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.vault_store import VaultStore
from hermes_cli.vault import _cmd_add, _cmd_list


def _scripted(answers):
    """An ``input``/``getpass`` stand-in that fails loudly once the script runs out, so a wizard that
    re-asks a question errors instead of hanging the runner."""
    queue = list(answers)

    def read(prompt=""):
        assert queue, f"wizard asked for more input than scripted: {prompt!r}"
        return queue.pop(0)

    return read


def test_address_saved_with_blank_origin_binds_to_no_site(tmp_path):
    store = VaultStore(tmp_path / "vault")
    # label, origin (blank = any site), line 1, line 2, city, state, postal code, country
    answers = _scripted(["Home", "", "1 Main St", "", "Springfield", "", "12345", "US"])
    with patch("agent.vault_store.get_vault_store", return_value=store), patch("builtins.input", answers):
        _cmd_add(SimpleNamespace(kind="address"))
    (meta,) = store.list_items()
    assert meta.kind == "address" and meta.origin is None


def test_payment_keeps_asking_until_an_origin_is_typed(tmp_path):
    store = VaultStore(tmp_path / "vault")
    prompts = _scripted(["Visa", "", "https://shop.test"])
    hidden = _scripted(["4111111111111111", "A User", "7", "2029", "123", ""])
    with patch("agent.vault_store.get_vault_store", return_value=store), patch("builtins.input", prompts), \
         patch("getpass.getpass", hidden):
        _cmd_add(SimpleNamespace(kind="payment"))
    (meta,) = store.list_items()
    assert meta.kind == "payment" and meta.origin == "https://shop.test"


def test_list_shows_an_unbound_address_as_any_site_and_an_unbound_card_as_none(tmp_path, monkeypatch, capsys):
    """Blank means two different things in the Origin column, and the list must not make a deliberately
    unbound address look like a card that was saved without a site and cannot be filled."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = VaultStore(home / "vault")
    store.add_item(kind="address", label="Home", secret={"address_line1": "1 Main St", "city": "Springfield",
                                                           "postal_code": "12345", "country": "US"})
    store.add_item(kind="payment", label="Visa", secret={"card_number": "4111111111111111", "cardholder_name": "A",
                                                          "exp_month": "7", "exp_year": "2029", "cvc": "123"})
    _cmd_list(SimpleNamespace())
    rows = {line.split("│")[4].strip(): line.split("│")[-2].strip()
            for line in capsys.readouterr().out.splitlines() if line.count("│") >= 6}
    assert rows["Home"] == "any site" and rows["Visa"] == "-"
