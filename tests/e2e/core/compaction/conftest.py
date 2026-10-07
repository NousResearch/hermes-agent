"""Lane-private fixtures for the compaction suites (one fake provider per module, one session per test)."""

from __future__ import annotations

import pytest

from tests.e2e.core.compaction._helpers import (
    SLOW_SUMMARY_TIMEOUT_SECONDS,
    THRESHOLD_TOKENS,
    Scenario,
    compaction_config,
    start_provider,
    write_home,
)


@pytest.fixture(scope="module")
def provider():
    server, dispatch = start_provider()
    try:
        yield server, dispatch
    finally:
        server.stop()


@pytest.fixture
def make_scenario(provider, tmp_path, monkeypatch):
    """Build a Scenario with its own RABBIT_HOME/state.db; closes every agent + DB at teardown."""
    server, dispatch = provider
    made: list[Scenario] = []

    def make(mode: str = "good", *, extra: str = "", platform: str = "cli", window: int | None = None,
             threshold_tokens: int = THRESHOLD_TOKENS, name: str = "rabbit_home") -> Scenario:
        # The hermetic root conftest already isolates RABBIT_HOME per test; HOME stays put because the
        # state.db live-system guard treats $HOME/.rabbit as production.
        rabbit_home = tmp_path / name
        monkeypatch.setenv("RABBIT_HOME", str(rabbit_home))
        import rabbit_state

        monkeypatch.setattr(rabbit_state, "DEFAULT_DB_PATH", rabbit_home / "state.db")
        slow = mode == "slow"
        if slow:
            # The compression request timeout has a 300 s production floor; pin it (and the host idle
            # watchdog) to seconds so a hung summarizer is judged inside a unit-test budget.
            import agent.auxiliary_client as aux

            monkeypatch.setattr(aux, "_COMPRESSION_TIMEOUT_FLOOR_SECONDS", SLOW_SUMMARY_TIMEOUT_SECONDS)
            extra += f"  context_timeout_seconds: {SLOW_SUMMARY_TIMEOUT_SECONDS}\n"
        write_home(rabbit_home, server.base_url,
                   compaction_config(extra=extra, slow=slow, threshold_tokens=threshold_tokens))
        sc = Scenario(server=server, dispatch=dispatch, home=rabbit_home, mode=mode, platform=platform, window=window)
        made.append(sc)
        return sc

    yield make
    for sc in made:
        sc.close()
