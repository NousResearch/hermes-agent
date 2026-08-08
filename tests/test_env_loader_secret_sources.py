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


# ---------------------------------------------------------------------------
# #74283 — multiplex regression: a profile's dotenv value must not be replaced
# by another profile's external secret across load_hermes_dotenv() calls.
# ---------------------------------------------------------------------------


def test_multiplex_dotenv_value_not_clobbered_by_other_profile_external_secret(
    tmp_path, monkeypatch
):
    """Regression for #74283: per-home isolation must hold across profiles.

    Home A resolves ``ANTHROPIC_API_KEY`` via an external secret source
    (``"aaa"``).  Home B's ``.env`` sets the same key to a different
    value (``"bbb"``).  Loading home B's dotenv must NOT replace B's
    value with A's external secret snapshot.

    The pre-repair implementation snapshotted the process-global
    ``_SECRET_SOURCES`` (which conflates profiles because the latest
    writer wins on the ``name -> source`` map).  Restoring from that
    global snapshot would clobber B's dotenv value with A's external
    value.  The fix snapshots ``get_secret_source_values(home_path)``
    so each home's protection is independent.
    """
    from agent.secret_sources.base import FetchResult
    from agent.secret_sources.registry import (
        AppliedVar,
        ApplyReport,
        SourceReport,
    )
    from agent.secret_sources import registry as reg_module

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()

    # Home A: external source resolves ANTHROPIC_API_KEY="aaa".
    (home_a / "config.yaml").write_text(
        "secrets:\n  test-source:\n    enabled: true\n",
        encoding="utf-8",
    )
    # Home B: dotenv sets ANTHROPIC_API_KEY="bbb" (no external source).
    (home_b / ".env").write_text(
        "ANTHROPIC_API_KEY=bbb\n", encoding="utf-8"
    )

    external_values = {
        str(home_a.resolve()): "aaa",
    }

    def _fake_apply_all(_cfg, home_path):
        home_key = str(Path(home_path).resolve())
        if home_key not in external_values:
            # Home B has no external source — return an empty report.
            return ApplyReport(sources=[], provenance={})
        value = external_values[home_key]
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
        return ApplyReport(
            sources=[
                SourceReport(
                    name="test-source",
                    label="Test Source",
                    result=FetchResult(),
                    applied=["ANTHROPIC_API_KEY"],
                )
            ],
            provenance={
                "ANTHROPIC_API_KEY": AppliedVar(
                    name="ANTHROPIC_API_KEY",
                    source="test-source",
                    shape="mapped",
                    overrode_env=True,
                )
            },
        )

    monkeypatch.setattr(reg_module, "apply_all", _fake_apply_all)

    # Seed the home-A snapshot directly so the test is independent of the
    # apply path's own side effects (the bug is purely about which dict
    # load_hermes_dotenv() snapshots from).
    home_a_key = str(home_a.resolve())
    env_loader._SECRET_SOURCES["ANTHROPIC_API_KEY"] = "test-source"
    env_loader._SECRET_SOURCE_VALUES_BY_HOME[home_a_key] = {
        "ANTHROPIC_API_KEY": "aaa",
    }
    env_loader._APPLIED_HOMES.add(home_a_key)

    # Simulate the first call having resolved A's external secret into the
    # shared os.environ (this is what an earlier load_hermes_dotenv(home_a)
    # would have done).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "aaa")

    # Now load home B.  Its .env must take precedence over the stale
    # process-global state.
    env_loader.load_hermes_dotenv(hermes_home=home_b)

    assert os.environ["ANTHROPIC_API_KEY"] == "bbb", (
        "Home B's dotenv value was replaced by home A's external secret "
        "(#74283). load_hermes_dotenv() must snapshot per-home, not "
        "process-global, when protecting external-source values."
    )

    # Home A's snapshot is untouched.
    assert env_loader.get_secret_source_values(home_a) == {
        "ANTHROPIC_API_KEY": "aaa",
    }


def test_multiplex_external_secret_survives_reload_for_own_home(
    tmp_path, monkeypatch
):
    """#74283 — sibling path: a home's external-secret value must still
    survive a *second* load_hermes_dotenv() for the SAME home.  This
    preserves the original #74265 protection while confirming the
    per-home snapshot is in effect.

    Home A: external source resolves ANTHROPIC_API_KEY="aaa" and stores
    a placeholder in .env.  Loading A twice must keep the resolved
    "aaa" (not the placeholder).
    Home B: independent .env with ANTHROPIC_API_KEY="bbb" and no
    external source.  Loading B once must keep "bbb" even though
    A's external source is still tracked in the process-global state.
    """
    from agent.secret_sources.base import FetchResult
    from agent.secret_sources.registry import (
        AppliedVar,
        ApplyReport,
        SourceReport,
    )
    from agent.secret_sources import registry as reg_module

    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()

    (home_a / "config.yaml").write_text(
        "secrets:\n  test-source:\n    enabled: true\n",
        encoding="utf-8",
    )
    (home_a / ".env").write_text(
        "ANTHROPIC_API_KEY=__PLACEHOLDER__\n", encoding="utf-8"
    )
    (home_b / ".env").write_text(
        "ANTHROPIC_API_KEY=bbb\n", encoding="utf-8"
    )

    home_a_key = str(home_a.resolve())

    def _fake_apply_all(_cfg, home_path):
        home_key = str(Path(home_path).resolve())
        if home_key != home_a_key:
            return ApplyReport(sources=[], provenance={})
        value = "aaa"
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
        return ApplyReport(
            sources=[
                SourceReport(
                    name="test-source",
                    label="Test Source",
                    result=FetchResult(),
                    applied=["ANTHROPIC_API_KEY"],
                )
            ],
            provenance={
                "ANTHROPIC_API_KEY": AppliedVar(
                    name="ANTHROPIC_API_KEY",
                    source="test-source",
                    shape="mapped",
                    overrode_env=True,
                )
            },
        )

    monkeypatch.setattr(reg_module, "apply_all", _fake_apply_all)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # First load: external source resolves A's placeholder to "aaa".
    env_loader.load_hermes_dotenv(hermes_home=home_a)
    assert os.environ["ANTHROPIC_API_KEY"] == "aaa"

    # Second load of A: placeholder must NOT clobber the resolved "aaa"
    # (this is the original #74265 protection, preserved).
    env_loader.load_hermes_dotenv(hermes_home=home_a)
    assert os.environ["ANTHROPIC_API_KEY"] == "aaa", (
        "Home A's external-secret value was clobbered on second "
        "load_hermes_dotenv() — #74265 protection regressed."
    )

    # Load B: B's .env "bbb" must be preserved, not replaced by A's "aaa".
    env_loader.load_hermes_dotenv(hermes_home=home_b)
    assert os.environ["ANTHROPIC_API_KEY"] == "bbb", (
        "Home B's dotenv value was replaced by home A's external secret "
        "snapshot — #74283 multiplex isolation regressed."
    )
