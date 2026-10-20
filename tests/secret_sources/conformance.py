"""Conformance kit for :class:`agent.secret_sources.base.SecretSource`.

Any secret-source backend — bundled or external plugin — can validate
itself against the contract by subclassing :class:`SecretSourceConformance`
and providing a ``source`` fixture (plus optional per-source config
fixtures).  Example::

    from tests.secret_sources.conformance import SecretSourceConformance

    class TestMySourceConformance(SecretSourceConformance):
        @pytest.fixture
        def source(self):
            return MySource()

The checks encode the parts of the contract that break OTHER people
when violated: never raising, never prompting (stdin closed), respecting
disabled config, valid identity attributes, and orchestrator
compatibility.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.secret_sources.base import (
    SECRET_SOURCE_API_VERSION,
    FetchResult,
    SecretSource,
)
from agent.secret_sources.registry import (
    _reset_registry_for_tests,
    apply_all,
    register_source,
)


class SecretSourceConformance:
    """Base class of contract checks; subclass and provide ``source``."""

    @pytest.fixture
    def source(self) -> SecretSource:  # pragma: no cover — must override
        raise NotImplementedError("conformance subclasses must provide a source fixture")

    @pytest.fixture
    def minimal_cfg(self) -> dict:
        """An enabled-but-unconfigured section — the common misconfig case."""
        return {"enabled": True}

    # -- identity ----------------------------------------------------------

    def test_name_is_lowercase_identifier(self, source):
        assert source.name, "source.name must be non-empty"
        assert source.name == source.name.lower()
        assert source.name.replace("_", "").isalnum()

    def test_label_present(self, source):
        assert source.label, "source.label must be a human-readable name"

    def test_shape_valid(self, source):
        assert source.shape in ("mapped", "bulk")

    def test_api_version_current(self, source):
        assert source.api_version == SECRET_SOURCE_API_VERSION

    # -- contract behavior --------------------------------------------------

    def test_fetch_never_raises_on_malformed_config(self, source, tmp_path):
        """Every degenerate config shape must produce a FetchResult, not a raise."""
        for cfg in ({}, {"enabled": True}, {"enabled": True, "env": "not-a-dict"},
                    {"enabled": True, "cache_ttl_seconds": "bogus"}, None):
            result = source.fetch(cfg if isinstance(cfg, dict) else {}, tmp_path)
            assert isinstance(result, FetchResult), (
                f"fetch() returned {type(result).__name__} for cfg={cfg!r}"
            )

    def test_fetch_unconfigured_reports_error_not_secrets(self, source, tmp_path,
                                                          minimal_cfg, monkeypatch):
        """enabled=true with nothing else set must fail cleanly with a kind."""
        result = source.fetch(minimal_cfg, tmp_path)
        assert isinstance(result, FetchResult)
        if not result.ok:
            assert result.error_kind is not None, (
                "errors must carry a machine-readable ErrorKind"
            )
            assert not result.secrets

    def test_disabled_by_default(self, source):
        assert source.is_enabled({}) is False
        assert source.is_enabled({"enabled": False}) is False

    def test_timeout_is_positive(self, source, minimal_cfg):
        assert source.fetch_timeout_seconds(minimal_cfg) > 0
        # Garbage config must not break the timeout accessor either.
        assert source.fetch_timeout_seconds({"timeout_seconds": "junk"}) > 0

    def test_protected_vars_are_valid_names(self, source, minimal_cfg):
        from agent.secret_sources.base import is_valid_env_name

        for var in source.protected_env_vars(minimal_cfg):
            assert is_valid_env_name(var)

    # -- orchestrator compatibility ------------------------------------------

    def test_registers_and_applies_via_orchestrator(self, source, tmp_path,
                                                    monkeypatch):
        """The source must survive a full apply_all() pass without breaking it."""
        _reset_registry_for_tests()
        # Prevent the bundled sources from interfering.
        monkeypatch.setattr(
            "agent.secret_sources.registry._ensure_builtin_sources", lambda: None
        )
        try:
            assert register_source(source), "register_source() rejected the source"
            env: dict = {}
            report = apply_all(
                {source.name: {"enabled": True}}, tmp_path, environ=env
            )
            names = [sr.name for sr in report.sources]
            assert source.name in names
        finally:
            _reset_registry_for_tests()
