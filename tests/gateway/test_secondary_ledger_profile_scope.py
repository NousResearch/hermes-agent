"""A secondary-completion ledger walk must run under the swept profile's REAL secret scope.

``_each_secondary_ledger`` passed ``{}`` as ``_profile_runtime_scope``'s ``prepared_secret_scope``.
That argument *is* the mapping to install — not an "already hydrated" flag — so the walk ran with an
EMPTY scope stamped with the profile's home. Under multiplexing a scope miss is fail-closed: every
``${env:...}`` ref of the swept profile's config stayed an unresolved placeholder (one journal line
per ref, every sweep interval) and any credential read inside the span returned nothing, while the
sweep itself believes it is serving that profile correctly.
"""

from types import SimpleNamespace

import pytest


def _harness(tmp_path, monkeypatch):
    """A launch home plus one secondary profile that owns a secret the launch home does not."""
    launch = tmp_path / ".hermes"
    sec = launch / "profiles" / "ledgersec"
    sec.mkdir(parents=True)
    (launch / ".env").write_text("OPENROUTER_API_KEY=LAUNCH-or\n")
    (sec / ".env").write_text("OPENROUTER_API_KEY=SEC-or\nLEDGER_ONLY_TOKEN=SEC-ledger\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("LEDGER_ONLY_TOKEN", raising=False)
    return launch, sec


@pytest.mark.parametrize("multiplex", [True, False])
def test_secondary_ledger_walk_binds_the_profiles_own_secret_scope(tmp_path, monkeypatch, multiplex):
    _launch, sec = _harness(tmp_path, monkeypatch)
    from agent import secret_scope
    from gateway.run_notifications import GatewayNotificationsMixin

    seen = {}

    def fn() -> int:
        seen["token"] = secret_scope.get_secret("LEDGER_ONLY_TOKEN")
        return 0

    owner = SimpleNamespace(_primary_profile_name="default")
    secret_scope.set_multiplex_active(multiplex)
    try:
        GatewayNotificationsMixin._each_secondary_ledger(owner, [("ledgersec", sec)], fn, "replayed")
    finally:
        secret_scope.set_multiplex_active(False)

    assert seen["token"] == "SEC-ledger", (
        "the ledger walk ran without the swept profile's secret scope"
    )
