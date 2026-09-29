import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "optional-skills/finance/fakturownia/scripts/fakturownia.py"
spec = importlib.util.spec_from_file_location("fakturownia", SCRIPT)
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)


def test_url_targets_the_accounts_own_host_and_carries_the_token_only_as_query():
    url = fk.build_url("acme", "/invoices.json", "TKN", {"period": "this_month", "client_name": ""})

    assert url.startswith("https://acme.fakturownia.pl/invoices.json?")
    assert "period=this_month" in url and "client_name" not in url and "api_token=TKN" in url


@pytest.mark.parametrize("bad", ["", "evil.com/x", "a b", "x@y"])
def test_subdomain_cannot_redirect_the_request_elsewhere(bad):
    with pytest.raises(ValueError):
        fk.base_url(bad)


def test_missing_credentials_stop_before_any_request(monkeypatch):
    monkeypatch.delenv("FAKTUROWNIA_SUBDOMAIN", raising=False)
    monkeypatch.delenv("FAKTUROWNIA_API_TOKEN", raising=False)
    monkeypatch.setattr(fk, "fetch", lambda url: pytest.fail("no request expected"))

    with pytest.raises(SystemExit):
        fk.main(["invoices"])


def test_output_keeps_only_whitelisted_fields(monkeypatch, capsys):
    monkeypatch.setenv("FAKTUROWNIA_SUBDOMAIN", "acme")
    monkeypatch.setenv("FAKTUROWNIA_API_TOKEN", "TKN")
    monkeypatch.setattr(fk, "fetch", lambda url: [{"id": 1, "number": "FV 1/2026", "secret_note": "x"}])

    fk.main(["invoices"])

    out = capsys.readouterr().out
    assert "FV 1/2026" in out and "secret_note" not in out and "TKN" not in out
