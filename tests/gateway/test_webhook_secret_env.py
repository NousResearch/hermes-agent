"""Environment expansion for webhook route secrets (``secret: "${VAR}"``).

Route config is routinely a tracked file, so a literal ``secret:`` means
committing a credential.  These tests pin the three behaviours the deployment
depends on: a literal secret is untouched, ``${VAR}`` resolves at load time
(preferring the active profile secret scope), and an UNSET variable resolves to
empty so the adapter's existing "route without a secret is refused" guards fire
instead of silently HMAC-ing against the literal string ``"${VAR}"``.

That last property is the security one, and it has to hold for EVERY
reference-shaped value, not just the spelling this module resolves.  The
placeholder lives in the tracked config.yaml the feature exists to let you
commit, so a ``${...}`` that survives verbatim becomes a publicly-readable HMAC
key — anyone with the repo can then sign a delivery.  ``TestRefsNeverBecomeKeys``
pins the fail-closed rule across every shape: the canonical ``${env:VAR}``
SecretRef spelling, non-env sources, and malformed names.
"""

import asyncio
import json

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import (
    WebhookAdapter,
    _DYNAMIC_ROUTES_FILENAME,
    _normalize_routes,
    _resolve_secret,
)


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _adapter(routes, extra=None):
    _extra = dict(extra or {})
    _extra["routes"] = routes
    return WebhookAdapter(PlatformConfig(enabled=True, extra=_extra))


class TestResolveSecret:
    def test_literal_passes_through(self):
        assert _resolve_secret("s3cr3t") == "s3cr3t"

    def test_env_reference_resolves(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        assert _resolve_secret("${HERMES_TEST_WEBHOOK_SECRET}") == "from-env"

    def test_surrounding_whitespace_tolerated(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        assert _resolve_secret("  ${HERMES_TEST_WEBHOOK_SECRET}  ") == "from-env"

    def test_unset_variable_resolves_empty(self, monkeypatch):
        monkeypatch.delenv("HERMES_TEST_MISSING_SECRET", raising=False)
        assert _resolve_secret("${HERMES_TEST_MISSING_SECRET}") == ""

    def test_partial_reference_is_a_literal(self, monkeypatch):
        # Only a whole-string reference expands. A real secret that happens to
        # contain braces must survive verbatim or we'd silently corrupt it.
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        assert (
            _resolve_secret("prefix-${HERMES_TEST_WEBHOOK_SECRET}")
            == "prefix-${HERMES_TEST_WEBHOOK_SECRET}"
        )

    def test_non_string_untouched(self):
        assert _resolve_secret(None) is None
        assert _resolve_secret(42) == 42

    def test_profile_secret_scope_wins_over_process_env(self, monkeypatch):
        # _getenv prefers the active profile secret scope; the whole point of
        # routing through it rather than os.environ is multiplexed startup.
        from gateway import config as gw_config

        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "process-env")
        monkeypatch.setattr(gw_config, "current_secret_scope", lambda: object())
        monkeypatch.setattr(
            gw_config,
            "_get_secret",
            lambda name, default=None: "scoped" if name == "HERMES_TEST_WEBHOOK_SECRET" else default,
        )
        assert _resolve_secret("${HERMES_TEST_WEBHOOK_SECRET}") == "scoped"

    def test_env_prefixed_reference_resolves(self, monkeypatch):
        # ${env:VAR} is the canonical SecretRef spelling in this tree —
        # tools/mcp_tool._env_ref_name, and config.yaml's own expander since
        # #69267. A config author must not have to know that this one field
        # takes a different spelling from every other one.
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        assert _resolve_secret("${env:HERMES_TEST_WEBHOOK_SECRET}") == "from-env"

    def test_env_prefixed_reference_honours_secret_scope(self, monkeypatch):
        from gateway import config as gw_config

        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "process-env")
        monkeypatch.setattr(gw_config, "current_secret_scope", lambda: object())
        monkeypatch.setattr(
            gw_config,
            "_get_secret",
            lambda name, default=None: "scoped" if name == "HERMES_TEST_WEBHOOK_SECRET" else default,
        )
        assert _resolve_secret("${env:HERMES_TEST_WEBHOOK_SECRET}") == "scoped"

    def test_partial_env_prefixed_reference_is_a_literal(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        assert (
            _resolve_secret("prefix-${env:HERMES_TEST_WEBHOOK_SECRET}")
            == "prefix-${env:HERMES_TEST_WEBHOOK_SECRET}"
        )


class TestRefsNeverBecomeKeys:
    """No whole-string ``${...}`` may survive as an HMAC key.

    The placeholder is written in the tracked config.yaml this feature exists
    to enable, so passing one through verbatim publishes the signing key: it is
    non-empty, the "route has no secret" guards stay quiet, and the gateway
    happily validates deliveries signed with a string anyone can read out of
    the repo. Every unresolvable shape must fail CLOSED to ``""`` instead.
    """

    @pytest.mark.parametrize(
        "value",
        [
            "${env:HERMES_TEST_MISSING_SECRET}",  # canonical spelling, unset
            "${HERMES_TEST_MISSING_SECRET}",  # bare spelling, unset
            "${bitwarden:GH_WEBHOOK_SECRET}",  # backend injects via secrets:
            "${vault:gh/webhook}",
            "${file:/run/secrets/webhook}",
            "${HERMES_TEST_MISSING_SECRET:-fallback}",  # shell default syntax
            "${9INVALID}",  # not an identifier
            "${env:}",  # prefix with no name
            "${}",
            "${ }",
        ],
    )
    def test_unresolvable_reference_resolves_empty(self, value, monkeypatch):
        monkeypatch.delenv("HERMES_TEST_MISSING_SECRET", raising=False)
        assert _resolve_secret(value) == ""

    @pytest.mark.parametrize(
        "value",
        [
            "${env:HERMES_TEST_MISSING_SECRET}",
            "${bitwarden:GH_WEBHOOK_SECRET}",
            "${HERMES_TEST_MISSING_SECRET:-fallback}",
        ],
    )
    def test_unresolvable_reference_is_refused_at_startup(self, value, monkeypatch):
        # End-to-end: the empty resolution has to reach the existing guard, so
        # the gateway refuses to start rather than serving a route whose key is
        # a string published in config.yaml.
        monkeypatch.delenv("HERMES_TEST_MISSING_SECRET", raising=False)
        adapter = _adapter({"pr-review": {"secret": value}})
        with pytest.raises(ValueError, match="no HMAC secret"):
            asyncio.run(adapter.connect())

    def test_refusal_does_not_echo_the_reference(self, monkeypatch, caplog):
        # A value that merely LOOKS like a ref could be a real secret; the
        # warning must not become the leak this feature exists to prevent.
        with caplog.at_level("WARNING"):
            assert _resolve_secret("${bitwarden:s3cr3t-looking-value}") == ""
            assert _resolve_secret("${actually-the-secret:-oops}") == ""
        assert "s3cr3t-looking-value" not in caplog.text
        assert "actually-the-secret" not in caplog.text


class TestNormalizeRoutes:
    def test_copies_rather_than_mutates(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        source = {"r": {"secret": "${HERMES_TEST_WEBHOOK_SECRET}", "events": ["push"]}}
        out = _normalize_routes(source)
        assert out["r"]["secret"] == "from-env"
        assert out["r"]["events"] == ["push"]
        # The caller's dict (config.extra["routes"]) is shared state.
        assert source["r"]["secret"] == "${HERMES_TEST_WEBHOOK_SECRET}"

    def test_route_without_secret_untouched(self):
        source = {"r": {"events": ["push"]}}
        assert _normalize_routes(source)["r"] == {"events": ["push"]}

    def test_non_dict_input(self):
        assert _normalize_routes(None) == {}
        assert _normalize_routes([1, 2]) == {}


class TestAdapterWiring:
    def test_static_route_secret_resolved_at_construction(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        adapter = _adapter({"pr-review": {"secret": "${HERMES_TEST_WEBHOOK_SECRET}"}})
        assert adapter._routes["pr-review"]["secret"] == "from-env"

    def test_global_secret_resolved(self, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        adapter = _adapter({}, extra={"secret": "${HERMES_TEST_WEBHOOK_SECRET}"})
        assert adapter._global_secret == "from-env"

    def test_unset_variable_is_refused_at_startup(self, monkeypatch):
        # The empty resolution must reach the existing guard, not bypass it.
        # Driven with asyncio.run so the suite needs no async pytest plugin.
        monkeypatch.delenv("HERMES_TEST_MISSING_SECRET", raising=False)
        adapter = _adapter({"pr-review": {"secret": "${HERMES_TEST_MISSING_SECRET}"}})
        with pytest.raises(ValueError, match="no HMAC secret"):
            asyncio.run(adapter.connect())

    def test_dynamic_route_env_secret_resolved(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_TEST_WEBHOOK_SECRET", "from-env")
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"dyn": {"secret": "${HERMES_TEST_WEBHOOK_SECRET}"}})
        )
        adapter = _adapter({})
        adapter._reload_dynamic_routes()
        assert adapter._routes["dyn"]["secret"] == "from-env"

    def test_dynamic_route_unset_env_secret_is_skipped(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_TEST_MISSING_SECRET", raising=False)
        (tmp_path / _DYNAMIC_ROUTES_FILENAME).write_text(
            json.dumps({"dyn": {"secret": "${HERMES_TEST_MISSING_SECRET}"}})
        )
        adapter = _adapter({})
        adapter._reload_dynamic_routes()
        assert "dyn" not in adapter._routes
