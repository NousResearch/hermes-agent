"""Tests for the secret-source tracking in ``hermes_cli.env_loader``.

These cover the small public surface that lets `hermes model` / `hermes setup`
label detected credentials with their origin ("from Bitwarden") so users
don't see an unexplained "credentials ✓" line when their .env is empty.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_cli import env_loader  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_sources():
    """Each test starts with a clean source map and applied-home guard."""
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader.reset_secret_source_cache()
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader._SECRET_SOURCE_VALUES_BY_HOME.clear()
    env_loader.reset_secret_source_cache()


def test_get_secret_source_returns_none_for_untracked_var():
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None


def test_get_secret_source_returns_label_for_tracked_var():
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "bitwarden"
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"


def test_get_secret_source_values_returns_home_snapshot_copy(tmp_path):
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()

    env_loader._SECRET_SOURCE_VALUES_BY_HOME[str(home_a.resolve())] = {
        "ANTHROPIC_API_KEY": "sk-profile-a"
    }

    snapshot = env_loader.get_secret_source_values(home_a)
    assert snapshot == {
        "ANTHROPIC_API_KEY": "sk-profile-a"
    }
    assert env_loader.get_secret_source_values(home_b) == {}
    snapshot["ANTHROPIC_API_KEY"] = "mutated"
    assert env_loader.get_secret_source_values(home_a) == {
        "ANTHROPIC_API_KEY": "sk-profile-a"
    }


def test_format_secret_source_suffix_empty_for_untracked():
    # Credentials from .env or the shell shouldn't add noise — the
    # implicit case stays unlabeled.
    assert env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY") == ""


def test_format_secret_source_suffix_bitwarden_uses_proper_name():
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "bitwarden"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Bitwarden)"
    )


def test_format_secret_source_suffix_generic_label_for_future_sources():
    # Future-proofing: a new secret source (e.g. "vault") should still
    # produce a sensible label without needing to edit every call site.
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "vault"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from vault)"
    )


def test_format_secret_source_suffix_onepassword_uses_proper_name():
    env_loader._SECRET_SOURCES["OPENAI_API_KEY"] = "onepassword"
    assert (
        env_loader.format_secret_source_suffix("OPENAI_API_KEY")
        == " (from 1Password)"
    )


def test_apply_external_secret_sources_records_bitwarden_origin(tmp_path, monkeypatch):
    """End-to-end: when the Bitwarden source fetches keys, applied vars
    end up in ``_SECRET_SOURCES`` so the UI can label them."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    # Stub the fetch layer under the SecretSource adapter.
    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(
        bw_module,
        "fetch_bitwarden_secrets",
        lambda **_kw: ({"ANTHROPIC_API_KEY": "sk-ant-test"}, []),
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from Bitwarden)"
    )


def test_cold_profile_bitwarden_uses_profile_bootstrap_without_global_env(
    tmp_path, monkeypatch
):
    """Real Bitwarden adapter reads its token from the profile-local view."""
    monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "BWS_ACCESS_TOKEN=profile-bootstrap\n", encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    import agent.secret_sources.bitwarden as bw_module
    from agent.secret_sources import registry as reg_module

    captured = {}
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))

    def _fake_fetch(**kwargs):
        captured.update(kwargs)
        return {"ANTHROPIC_API_KEY": "profile-provider-key"}, []

    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets", _fake_fetch)
    reg_module._reset_registry_for_tests()

    assert env_loader.hydrate_profile_secret_sources(tmp_path) == {
        "ANTHROPIC_API_KEY": "profile-provider-key"
    }
    assert captured["access_token"] == "profile-bootstrap"
    assert os.environ.get("BWS_ACCESS_TOKEN") is None
    assert os.environ.get("ANTHROPIC_API_KEY") is None


def test_single_profile_scoped_load_keeps_override_behavior(tmp_path, monkeypatch):
    """Without multiplex, a scoped load keeps its historical override behaviour.

    Ported from #77970 (@DonShelly): the guard must key on the multiplex flag,
    not on the home override alone -- single-profile ``-p`` runs still load.
    """
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.delenv("HERMES_TEST_SHARED_ADAPTER_CONFIG", raising=False)
    other_home = tmp_path / "other"
    other_home.mkdir()
    (other_home / ".env").write_text("HERMES_TEST_SHARED_ADAPTER_CONFIG=second\n")

    was_active = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    home_token = set_hermes_home_override(other_home)
    try:
        loaded = env_loader.load_hermes_dotenv(hermes_home=other_home)
    finally:
        secret_scope.set_multiplex_active(was_active)
        reset_hermes_home_override(home_token)

    try:
        assert os.environ.get("HERMES_TEST_SHARED_ADAPTER_CONFIG") == "second"
        assert (other_home / ".env") in loaded
    finally:
        os.environ.pop("HERMES_TEST_SHARED_ADAPTER_CONFIG", None)


def test_multiplex_dotenv_load_hydrates_sources_without_global_env(
    tmp_path, monkeypatch
):
    """The safe multiplex path must still refresh profile secret sources."""
    from agent import secret_scope
    import agent.secret_sources.bitwarden as bw_module
    from agent.secret_sources import registry as reg_module
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    monkeypatch.delenv("BWS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "BWS_ACCESS_TOKEN=profile-bootstrap\n", encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(
        bw_module,
        "fetch_bitwarden_secrets",
        lambda **_kw: ({"ANTHROPIC_API_KEY": "profile-provider-key"}, []),
    )
    reg_module._reset_registry_for_tests()

    was_active = secret_scope.is_multiplex_active()
    home_token = set_hermes_home_override(tmp_path)
    secret_scope.set_multiplex_active(True)
    try:
        assert env_loader.load_hermes_dotenv(hermes_home=tmp_path) == []
    finally:
        secret_scope.set_multiplex_active(was_active)
        reset_hermes_home_override(home_token)

    assert env_loader.get_secret_source_values(tmp_path) == {
        "ANTHROPIC_API_KEY": "profile-provider-key"
    }
    assert os.environ.get("BWS_ACCESS_TOKEN") is None
    assert os.environ.get("ANTHROPIC_API_KEY") is None


def test_cold_profile_hydration_seeds_op_env_bootstrap(tmp_path, monkeypatch):
    """The .op.env bootstrap file must feed cold-profile hydration.

    load_hermes_dotenv() reads <home>/.op.env for OP_SERVICE_ACCOUNT_TOKEN
    (the documented gitignored 1Password bootstrap); hydration must mirror
    that or a cold profile using the supported .op.env flow fails 1Password
    resolution (sweeper review on #74549). .env wins on conflict.
    """
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    (tmp_path / ".env").write_text("UNRELATED=x\n", encoding="utf-8")
    (tmp_path / ".op.env").write_text(
        "OP_SERVICE_ACCOUNT_TOKEN=ops_from-op-env\n", encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    from agent.secret_sources import registry as reg_module

    seen_env = {}

    def _capture_apply_all(_cfg, home_path, environ=None):
        from agent.secret_sources.registry import ApplyReport
        seen_env.update(environ or {})
        return ApplyReport(sources=[], provenance={})

    monkeypatch.setattr(reg_module, "apply_all", _capture_apply_all)
    reg_module._reset_registry_for_tests()

    env_loader.hydrate_profile_secret_sources(tmp_path)

    assert seen_env.get("OP_SERVICE_ACCOUNT_TOKEN") == "ops_from-op-env"
    # Never leaked into the process env.
    assert os.environ.get("OP_SERVICE_ACCOUNT_TOKEN") is None


def test_cold_profile_hydration_dotenv_wins_over_op_env(tmp_path, monkeypatch):
    """.env takes precedence over .op.env for the same key (setdefault)."""
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    (tmp_path / ".env").write_text(
        "OP_SERVICE_ACCOUNT_TOKEN=ops_from-dotenv\n", encoding="utf-8"
    )
    (tmp_path / ".op.env").write_text(
        "OP_SERVICE_ACCOUNT_TOKEN=ops_from-op-env\n", encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    from agent.secret_sources import registry as reg_module

    seen_env = {}

    def _capture_apply_all(_cfg, home_path, environ=None):
        from agent.secret_sources.registry import ApplyReport
        seen_env.update(environ or {})
        return ApplyReport(sources=[], provenance={})

    monkeypatch.setattr(reg_module, "apply_all", _capture_apply_all)
    reg_module._reset_registry_for_tests()

    env_loader.hydrate_profile_secret_sources(tmp_path)

    assert seen_env.get("OP_SERVICE_ACCOUNT_TOKEN") == "ops_from-dotenv"


def test_apply_external_secret_sources_noop_when_disabled(tmp_path, monkeypatch):
    """Disabled Bitwarden config must not touch the source map."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") is None


def test_apply_external_secret_sources_dedupes_within_process(tmp_path, monkeypatch):
    """``load_hermes_dotenv()`` is called at module-import time from several
    hot modules (cli.py, hermes_cli/main.py, run_agent.py, ...).  The
    Bitwarden status line previously printed once per call — 3-5x per
    startup.  The applied-home guard must short-circuit subsequent calls
    so the heavy work (config re-parse, Bitwarden lookup, status print)
    runs exactly once per HERMES_HOME per process.
    """

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    call_count = {"n": 0}
    def _fake_fetch(**_kwargs):
        call_count["n"] += 1
        return {"ANTHROPIC_API_KEY": "sk-ant-test"}, []

    import agent.secret_sources.bitwarden as bw_module
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets", _fake_fetch)

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    # Five calls in a row, simulating module-import-time invocations from
    # cli.py, hermes_cli/main.py, run_agent.py, trajectory_compressor.py,
    # gateway/run.py.  Only the first should actually call the backend.
    for _ in range(5):
        env_loader._apply_external_secret_sources(tmp_path)

    assert call_count["n"] == 1, (
        "Bitwarden backend was called {} time(s); expected exactly 1 — "
        "the applied-home guard is broken.".format(call_count["n"])
    )

    # Source tracking still works after dedup.
    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "bitwarden"
    assert env_loader.get_secret_source_values(tmp_path) == {
        "ANTHROPIC_API_KEY": "sk-ant-test"
    }

    # reset_secret_source_cache() forces a fresh pull on the next call.
    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)
    assert call_count["n"] == 2


def test_apply_external_secret_sources_status_line_suppresses_secret_names(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")
    monkeypatch.delenv("LEAK_THIS_API_KEY", raising=False)
    monkeypatch.delenv("LEAK_THIS_TOKEN", raising=False)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: test-project\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n",
        encoding="utf-8",
    )

    import agent.secret_sources.bitwarden as bw_module

    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(
        bw_module,
        "fetch_bitwarden_secrets",
        lambda **_kw: (
            {"LEAK_THIS_API_KEY": "sk-test", "LEAK_THIS_TOKEN": "tok-test"},
            [],
        ),
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(tmp_path)

    err = capsys.readouterr().err
    assert "Bitwarden Secrets Manager: applied 2 secrets" in err
    assert "LEAK_THIS_API_KEY" not in err
    assert "LEAK_THIS_TOKEN" not in err


def test_external_secret_values_are_isolated_between_homes(tmp_path, monkeypatch):
    """A later apply for the same key must not mutate an earlier home snapshot."""
    from agent.secret_scope import build_profile_secret_scope
    from agent.secret_sources.base import FetchResult
    from agent.secret_sources.registry import (
        AppliedVar,
        ApplyReport,
        SourceReport,
    )
    from agent.secret_sources import registry as reg_module

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    for home in (home_a, home_b):
        home.mkdir()
        (home / "config.yaml").write_text(
            "secrets:\n  test-source:\n    enabled: true\n",
            encoding="utf-8",
        )

    values = {
        str(home_a.resolve()): "value-a",
        str(home_b.resolve()): "value-b",
    }

    def _fake_apply_all(_cfg, home_path):
        value = values[str(Path(home_path).resolve())]
        monkeypatch.setenv("SHARED_API_KEY", value)
        return ApplyReport(
            # Real apply_all always appends a SourceReport per enabled
            # source; the env_loader guard (#40597) early-returns on an
            # empty sources list, so the fake must match the real shape.
            sources=[
                SourceReport(
                    name="test-source",
                    label="Test Source",
                    result=FetchResult(),
                    applied=["SHARED_API_KEY"],
                )
            ],
            provenance={
                "SHARED_API_KEY": AppliedVar(
                    name="SHARED_API_KEY",
                    source="test-source",
                    shape="mapped",
                    overrode_env=True,
                )
            }
        )

    monkeypatch.setattr(reg_module, "apply_all", _fake_apply_all)

    env_loader._apply_external_secret_sources(home_a)
    env_loader._apply_external_secret_sources(home_b)

    assert os.environ["SHARED_API_KEY"] == "value-b"
    assert env_loader.get_secret_source_values(home_a) == {
        "SHARED_API_KEY": "value-a"
    }
    assert env_loader.get_secret_source_values(home_b) == {
        "SHARED_API_KEY": "value-b"
    }
    assert build_profile_secret_scope(home_a) == {
        "SHARED_API_KEY": "value-a"
    }
    assert build_profile_secret_scope(home_b) == {
        "SHARED_API_KEY": "value-b"
    }


def test_apply_external_secret_sources_records_onepassword_origin(tmp_path, monkeypatch):
    """When the 1Password source resolves refs, applied vars end up in
    ``_SECRET_SOURCES`` labeled ``onepassword``."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    env:\n"
        "      ANTHROPIC_API_KEY: 'op://Private/Anthropic/credential'\n",
        encoding="utf-8",
    )

    import agent.secret_sources.onepassword as op_module

    monkeypatch.setattr(op_module, "find_op", lambda *_a, **_kw: Path("/fake/op"))
    monkeypatch.setattr(
        op_module,
        "fetch_onepassword_secrets",
        lambda **_kw: ({"ANTHROPIC_API_KEY": "sk-ant-test"}, []),
    )

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(tmp_path)

    assert env_loader.get_secret_source("ANTHROPIC_API_KEY") == "onepassword"
    assert (
        env_loader.format_secret_source_suffix("ANTHROPIC_API_KEY")
        == " (from 1Password)"
    )


def test_apply_external_secret_sources_survives_non_dict_section(tmp_path, monkeypatch):
    """A malformed `secrets:` section must not abort startup (fail-open).

    Both `onepassword: true` (non-dict) and a bad bitwarden section must be
    coerced to empty config instead of raising AttributeError up through
    load_hermes_dotenv().
    """

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden: true\n"
        "  onepassword: true\n",
        encoding="utf-8",
    )

    # Must not raise and must not record anything.
    env_loader._apply_external_secret_sources(tmp_path)
    assert env_loader.get_secret_source("ANYTHING") is None


def test_apply_external_secret_sources_bad_ttl_does_not_crash(tmp_path, monkeypatch):
    """A non-numeric cache_ttl_seconds must be coerced, not crash startup."""

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  onepassword:\n"
        "    enabled: true\n"
        "    cache_ttl_seconds: not-a-number\n"
        "    env:\n"
        "      K: 'op://V/I/F'\n",
        encoding="utf-8",
    )

    captured = {}

    def _fake_fetch(**kwargs):
        captured.update(kwargs)
        return {}, []

    import agent.secret_sources.onepassword as op_module
    monkeypatch.setattr(op_module, "find_op", lambda *_a, **_kw: Path("/fake/op"))
    monkeypatch.setattr(op_module, "fetch_onepassword_secrets", _fake_fetch)

    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(tmp_path)

    # Coerced to the 300s default rather than raising ValueError.
    assert captured["cache_ttl_seconds"] == 300


@pytest.fixture
def _fresh_registry():
    from agent.secret_sources import registry as reg_module

    reg_module._reset_registry_for_tests()
    yield
    reg_module._reset_registry_for_tests()


def _register_fake_bulk_source(value_for_home):
    """One bulk source supplying GLM_API_KEY, resolved per home."""
    from agent.secret_sources import registry as reg_module
    from agent.secret_sources.base import FetchResult, SecretSource

    class _Fake(SecretSource):
        name = "fakebulk"
        label = "Fake"
        shape = "bulk"

        def fetch(self, cfg, home_path):
            result = FetchResult()
            result.secrets = {"GLM_API_KEY": value_for_home(Path(home_path))}
            return result

    reg_module.register_source(_Fake(), replace=True)


def test_env_shadowed_reapply_keeps_home_snapshot(tmp_path, monkeypatch, _fresh_registry):
    """#102041: a re-apply whose every key is ``skipped_existing`` (the previous apply's own write-back,
    or a systemd ``EnvironmentFile=`` value) must still snapshot the home's effective values. Latching
    an empty snapshot made ``build_profile_secret_scope`` drop every vault credential for the process
    lifetime under multiplex."""
    from agent.secret_scope import build_profile_secret_scope

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("secrets:\n  fakebulk:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    _register_fake_bulk_source(lambda _home: "vault-value")

    env_loader.load_hermes_dotenv(hermes_home=home)
    assert env_loader.get_secret_source_values(home) == {"GLM_API_KEY": "vault-value"}

    # cron per-fire / plugin-discovery re-pull: reset + reload with the key now shadowing itself.
    env_loader.reset_secret_source_cache()
    env_loader.load_hermes_dotenv(hermes_home=home)

    assert str(home.resolve()) in env_loader._APPLIED_HOMES
    assert env_loader.hydrate_profile_secret_sources(home) == {"GLM_API_KEY": "vault-value"}
    assert build_profile_secret_scope(home)["GLM_API_KEY"] == "vault-value"


def test_home_scoped_reset_preserves_sibling_snapshot(tmp_path, monkeypatch, _fresh_registry):
    """A cron fire / discovery refresh for one profile resets only THAT home: a multiplex sibling's
    hydrated snapshot stays intact instead of running empty until it re-hydrates."""
    home = tmp_path / ".hermes"
    sibling = home / "profiles" / "b"
    sibling.mkdir(parents=True)
    for h in (home, sibling):
        (h / "config.yaml").write_text("secrets:\n  fakebulk:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    _register_fake_bulk_source(lambda h: f"vault-{h.name}")

    env_loader.load_hermes_dotenv(hermes_home=home)
    assert env_loader.hydrate_profile_secret_sources(sibling) == {"GLM_API_KEY": "vault-b"}

    env_loader.reset_secret_source_cache(home)

    assert env_loader.get_secret_source_values(home) == {}
    assert env_loader.get_secret_source_values(sibling) == {"GLM_API_KEY": "vault-b"}
    assert str(sibling.resolve()) in env_loader._APPLIED_HOMES


def test_profile_scope_carries_vault_secrets_only_when_hydrated_first(tmp_path, monkeypatch):
    """``build_profile_secret_scope`` only READS the per-home source map, so hydration must run
    BEFORE the scope is frozen — the order ``gateway/run.py`` and the external cron worker use.

    This is load-bearing once the process-global dotenv write is suppressed for a scoped home:
    a vault-backed secret then has no other route into the run.
    """
    from pathlib import Path

    from agent.secret_scope import build_profile_secret_scope
    from agent.secret_sources import registry as reg_module

    home = tmp_path / "routed"
    home.mkdir()
    (home / ".env").write_text("BWS_ACCESS_TOKEN=0.test-token\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "secrets:\n  bitwarden:\n    enabled: true\n    project_id: p\n"
        "    access_token_env: BWS_ACCESS_TOKEN\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.test-token")

    import agent.secret_sources.bitwarden as bw_module
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets",
                        lambda **_kw: ({"VAULT_ONLY_KEY": "from-vault"}, []))
    reg_module._reset_registry_for_tests()
    env_loader._APPLIED_HOMES.discard(str(home.resolve()))

    # Frozen before hydration: the vault value cannot be in the scope.
    assert "VAULT_ONLY_KEY" not in build_profile_secret_scope(home)

    # Hydrated first — as run_one_job now does — the scope carries it.
    env_loader.hydrate_profile_secret_sources(home)
    assert build_profile_secret_scope(home).get("VAULT_ONLY_KEY") == "from-vault"


def test_a_plugin_source_discovered_mid_fire_reaches_the_installed_scope(tmp_path, monkeypatch):
    """A routed cron fire freezes its scope before its first agent build discovers plugin secret
    sources; under multiplex semantics the post-discovery reload is hydrate-only, so without an
    in-place refresh THIS fire never saw the plugin credential (#107692 review)."""
    import os

    from agent import secret_scope
    from agent.secret_sources import registry as reg_module
    from agent.secret_sources.base import SECRET_SOURCE_API_VERSION, FetchResult, SecretSource
    from hermes_cli.plugins import PluginManager
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    class _LateVault(SecretSource):
        api_version = SECRET_SOURCE_API_VERSION
        shape = "bulk"
        name = "latevault"

        def is_enabled(self, cfg: dict) -> bool:
            return True

        def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
            return FetchResult(secrets={"PLUGIN_ONLY_KEY": "from-plugin"})

    launch, home = tmp_path / "launch", tmp_path / "launch" / "profiles" / "ops"
    home.mkdir(parents=True)
    (home / ".env").write_text("XAI_API_KEY=routed\n", encoding="utf-8")
    (home / "config.yaml").write_text("secrets:\n  latevault:\n    enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("PLUGIN_ONLY_KEY", raising=False)
    reg_module._reset_registry_for_tests()
    env_loader.reset_secret_source_cache()

    context_token = secret_scope.set_multiplex_context(True)
    home_token = set_hermes_home_override(str(home))
    scope_token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(home))
    try:
        assert secret_scope.get_secret("PLUGIN_ONLY_KEY") is None  # frozen before the plugin existed
        assert reg_module.register_source(_LateVault())  # discovery registers the source...
        PluginManager(scope_key=str(home))._refresh_secret_sources_after_discovery()  # ...and re-pulls
        assert secret_scope.get_secret("PLUGIN_ONLY_KEY") == "from-plugin"
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.reset_multiplex_context(context_token)
        reg_module._reset_registry_for_tests()
    assert "PLUGIN_ONLY_KEY" not in os.environ
