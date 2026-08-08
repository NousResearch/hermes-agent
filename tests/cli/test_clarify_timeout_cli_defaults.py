"""Regression: the CLI defaults must not inject a legacy ``clarify.timeout``
that silently shadows ``agent.clarify_timeout``.

``resolve_clarify_timeout`` (tools/clarify_gateway.py) treats any top-level
``clarify.timeout`` as an explicit user override that wins over the canonical
``agent.clarify_timeout`` (default 3600, documented in
``hermes_cli/config.py`` DEFAULT_CONFIG).  For a long time ``load_cli_config``
seeded its built-in defaults with ``clarify.timeout: 120``.  Because the
resolver honours that key first, the stale hardcoded 120 permanently shadowed
``agent.clarify_timeout``, so the CLI surfaced a 2-minute clarify wait no
matter what the user set under ``agent.clarify_timeout`` — even though every
other surface (gateway, TUI/desktop) read 3600.

This test pins the contract: with no user config present, the CLI defaults must
NOT carry a ``clarify.timeout`` key at all, so the resolver falls through to
``agent.clarify_timeout`` — the single source of truth.
"""

from __future__ import annotations

import textwrap


def _load_cli_with_empty_user_config(tmp_path, monkeypatch):
    """Point cli._hermes_home at an empty tmp_path and reload, so
    load_cli_config sees no user config.yaml and returns only defaults."""
    import cli

    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    # No config.yaml written → load_cli_config returns built-in defaults only.
    return cli.load_cli_config()


class TestCliDefaultsDoNotShadowClarifyTimeout:
    """The built-in CLI defaults must let agent.clarify_timeout speak."""

    def test_no_legacy_clarify_key_in_defaults(self, tmp_path, monkeypatch):
        """Built-in defaults must not carry a top-level clarify.timeout.

        If they did, resolve_clarify_timeout would treat it as an explicit
        override and shadow agent.clarify_timeout (the documented key).
        """
        cfg = _load_cli_with_empty_user_config(tmp_path, monkeypatch)

        # The legacy section must be ABSENT from the defaults. Its presence is
        # the bug — any value here shadows agent.clarify_timeout regardless of
        # what the user sets there.
        assert "clarify" not in cfg or cfg["clarify"].get("timeout") is None, (
            "CLI defaults inject a top-level clarify.timeout, which "
            "resolve_clarify_timeout treats as an explicit override that "
            "shadows agent.clarify_timeout. See tools/clarify_gateway.py."
        )

    def test_resolve_falls_through_to_agent_key(self, tmp_path, monkeypatch):
        """resolve_clarify_timeout(load_cli_config()) == 3600 when the user has
        not set a legacy clarify.timeout.

        This is the end-to-end invariant the bug violated: with only built-in
        defaults in play, the resolver must reach the canonical 1-hour default,
        not the old hardcoded 120. (The CLI's local defaults dict doesn't seed
        agent.clarify_timeout itself; the resolver supplies 3600 as its own
        fallback, which is the documented default in DEFAULT_CONFIG.)
        """
        from tools.clarify_gateway import resolve_clarify_timeout

        cfg = _load_cli_with_empty_user_config(tmp_path, monkeypatch)
        resolved = resolve_clarify_timeout(cfg)

        # 3600 is the canonical default (DEFAULT_CONFIG / resolver fallback).
        # The old bug returned 120 here because a legacy clarify.timeout
        # shadowed everything.
        assert resolved == 3600

    def test_user_agent_clarify_timeout_is_honored(self, tmp_path, monkeypatch):
        """A user who sets agent.clarify_timeout in config.yaml gets that value,
        not a shadow of the old hardcoded default."""
        from tools.clarify_gateway import resolve_clarify_timeout

        config_yaml = textwrap.dedent(
            """
            agent:
              clarify_timeout: 900
            """
        ).lstrip()
        (tmp_path / "config.yaml").write_text(config_yaml)

        import cli

        monkeypatch.setattr(cli, "_hermes_home", tmp_path)
        cfg = cli.load_cli_config()
        assert resolve_clarify_timeout(cfg) == 900

    def test_explicit_legacy_clarify_still_overrides(self, tmp_path, monkeypatch):
        """Back-compat: a user who explicitly set the legacy clarify.timeout
        (before agent.clarify_timeout existed) still wins. This is the resolver
        contract that test_legacy_clarify_key_overrides already pins; we assert
        it still holds through the full load_cli_config path so removing the
        hardcoded default did not regress intentional legacy precedence."""
        from tools.clarify_gateway import resolve_clarify_timeout

        config_yaml = textwrap.dedent(
            """
            agent:
              clarify_timeout: 900
            clarify:
              timeout: 42
            """
        ).lstrip()
        (tmp_path / "config.yaml").write_text(config_yaml)

        import cli

        monkeypatch.setattr(cli, "_hermes_home", tmp_path)
        cfg = cli.load_cli_config()
        # Explicit legacy key wins over agent key — back-comat preserved.
        assert resolve_clarify_timeout(cfg) == 42
