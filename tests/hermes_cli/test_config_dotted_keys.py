"""Tests for double-quote-aware dotted config-key addressing (`_split_dotted_key`).

A key segment that legitimately contains a dot — every real model id does
(``gpt-5.6-sol``) — must be addressable so ``model_routes.<model-id>`` can be
written, read and removed from the CLI. Unquoted keys must keep tokenizing
exactly as ``str.split(".")`` did.
"""

import os
from unittest.mock import patch

import pytest

from hermes_cli.dotted_path import join_dotted_key, normalize_dotted_key
from hermes_cli.config import (
    _split_dotted_key,
    _get_nested,
    _unset_nested,
    _default_value_for_key,
    set_config_value,
    get_config_value,
    unset_config_value,
)


@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    (tmp_path / ".env").touch()
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        yield tmp_path


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestSplitDottedKey:
    def test_unquoted_matches_str_split(self):
        assert _split_dotted_key("a.b.c") == ["a", "b", "c"]

    def test_quoted_span_kept_literal(self):
        assert _split_dotted_key('a."b.c".d') == ["a", "b.c", "d"]

    def test_model_id_route_path(self):
        assert _split_dotted_key(
            'platforms.api_server.extra.model_routes."gpt-5.6-sol".model'
        ) == [
            "platforms", "api_server", "extra", "model_routes",
            "gpt-5.6-sol", "model",
        ]

    def test_numeric_list_segment_preserved(self):
        # The #17876 list-navigation guard must be unaffected.
        assert _split_dotted_key("custom_providers.0.api_key") == [
            "custom_providers", "0", "api_key",
        ]

    def test_unterminated_quote_raises(self):
        with pytest.raises(ValueError):
            _split_dotted_key('a."b')


# ---------------------------------------------------------------------------
# Round trip through set/get/unset on a dotted model-id leaf
# ---------------------------------------------------------------------------

class TestDottedRouteRoundTrip:
    KEY = 'platforms.api_server.extra.model_routes."gpt-5.6-sol".model'

    def test_set_get_unset_addresses_single_key(self, capsys):
        set_config_value(self.KEY, "openai/gpt-5.6")
        capsys.readouterr()

        from hermes_cli.config import load_config
        routes = (
            load_config()
            .get("platforms", {})
            .get("api_server", {})
            .get("extra", {})
            .get("model_routes", {})
        )
        # The route lands under the SINGLE key "gpt-5.6-sol", not a nested
        # "gpt-5" -> "6-sol" mangling.
        assert routes == {"gpt-5.6-sol": {"model": "openai/gpt-5.6"}}

        get_config_value(self.KEY)
        assert "openai/gpt-5.6" in capsys.readouterr().out

        unset_config_value(self.KEY)
        capsys.readouterr()
        routes_after = (
            load_config()
            .get("platforms", {})
            .get("api_server", {})
            .get("extra", {})
            .get("model_routes", {})
        )
        # Unset removes the addressed leaf (`.model`), and hermes' existing
        # empty-container cleanup then collapses the route entry and its now
        # empty ancestors — the same behaviour `config unset` already has for
        # any other key. Quoting changes only how the path is addressed.
        assert routes_after == {}

    def test_get_nested_directly(self):
        cfg = {"model_routes": {"gpt-5.6-sol": {"model": "x"}}}
        assert _get_nested(cfg, 'model_routes."gpt-5.6-sol".model') == "x"

    def test_unset_nested_directly(self):
        cfg = {"model_routes": {"gpt-5.6-sol": {"model": "x"}}}
        assert _unset_nested(cfg, 'model_routes."gpt-5.6-sol"') is True
        # The quoted segment is removed, then the existing empty-container
        # cleanup drops the emptied `model_routes` parent.
        assert cfg == {}


# ---------------------------------------------------------------------------
# A quoted model-id leaf must not be coerced away from a string
# ---------------------------------------------------------------------------

def test_quoted_key_does_not_mis_coerce_default_lookup():
    # A quoted model-id path misses DEFAULT_CONFIG, so the default lookup
    # returns None and the value keeps its historical best-effort coercion —
    # a model-id string stays a string.
    key = 'platforms.api_server.extra.model_routes."gpt-5.6-sol".model'
    assert _default_value_for_key(key) is None


# ---------------------------------------------------------------------------
# Malformed key is rejected cleanly (no traceback, non-zero exit)
# ---------------------------------------------------------------------------

class TestMalformedKeyGuard:
    def test_set_exits_on_unterminated_quote(self):
        with pytest.raises(SystemExit) as exc:
            set_config_value('model_routes."gpt-5.6-sol', "x")
        assert exc.value.code == 1

    def test_get_exits_on_unterminated_quote(self):
        with pytest.raises(SystemExit) as exc:
            get_config_value('model_routes."gpt-5.6-sol')
        assert exc.value.code == 1

    def test_unset_exits_on_unterminated_quote(self):
        with pytest.raises(SystemExit) as exc:
            unset_config_value('model_routes."gpt-5.6-sol')
        assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Serializer + round-trip invariants
#
# The tokenizer and the serializer are one representation, shared with
# ``managed_scope`` via ``hermes_cli.dotted_path``. These pin the contract that
# lets a managed key and a CLI key compare equal.
# ---------------------------------------------------------------------------

class TestJoinDottedKey:
    def test_dot_free_segments_join_exactly_as_str_join(self):
        """Backward compat, stated as an equality against the old behaviour.

        If ``join_dotted_key`` ever quoted unconditionally, every one of these
        would gain quotes and fail — this is what makes the invariant proven
        rather than assumed.
        """
        for parts in (
            ["a"],
            ["a", "b", "c"],
            ["model", "default"],
            ["platforms", "api_server", "extra", "model_routes", "alias", "model"],
            ["providers", "0", "name"],
            ["a", "", "b"],
            ["a-b", "a_b", "a b"],
        ):
            assert join_dotted_key(parts) == ".".join(parts)

    def test_dot_bearing_segment_is_quoted(self):
        assert join_dotted_key(["model_routes", "gpt-5.6-sol", "model"]) == (
            'model_routes."gpt-5.6-sol".model'
        )

    def test_non_string_segments_are_str_coerced(self):
        # Matches what managed_scope's flattening did with non-string YAML keys.
        assert join_dotted_key(["a", 0, True]) == "a.0.True"
        # A float key contains a dot, so it is quoted — deliberate: bare 4.5
        # addresses 4 -> 5, a different path.
        assert join_dotted_key([4.5, "a"]) == '"4.5".a'

    def test_quote_bearing_segment_is_emitted_bare_and_never_raises(self):
        """A ``"`` cannot be represented — the tokenizer treats quotes as
        structure and strips them, so there is no escape. Emitting bare is
        byte-identical to the pre-fix behaviour; raising would add a new crash
        surface to ``hermes doctor``, which calls managed_config_keys()
        outside its try/except.
        """
        assert join_dotted_key(['a"b', "c"]) == 'a"b.c'


class TestRoundTrip:
    def test_split_of_join_recovers_the_segments(self):
        """split(join(parts)) == parts for any segments containing no quote."""
        for parts in (
            ["a", "b"],
            ["model", "default"],
            ["platforms", "api_server", "extra", "model_routes", "gpt-5.6-sol", "model"],
            ["claude-sonnet-4.5"],
            ["a.b", "c.d"],
            ["a", "", "b"],
            ["4.5", "a"],
            ["only-one-segment"],
        ):
            assert _split_dotted_key(join_dotted_key(parts)) == parts

    def test_join_of_split_is_the_identity_on_dot_free_keys(self):
        """join(split(k)) == k byte-for-byte for any key with no dot-bearing
        segment. This is the backward-compatibility guarantee: every existing
        managed key in the wild is of this shape and must not change spelling.
        """
        for key in (
            "a",
            "a.b.c",
            "model.default",
            "platforms.api_server.extra.model_routes.alias.model",
            "toolsets.enabled",
            "providers.0.name",
            "",
            "a..b",
        ):
            assert join_dotted_key(_split_dotted_key(key)) == key

    def test_normalize_is_join_of_split(self):
        assert normalize_dotted_key("model.default") == "model.default"
        assert normalize_dotted_key('model."default"') == "model.default"
        assert normalize_dotted_key('a."b.c".d') == 'a."b.c".d'
        # Two spellings of the same path normalize to one string — this is what
        # lets is_key_managed compare a CLI key against a flattened managed key.
        assert normalize_dotted_key('"model".default') == normalize_dotted_key(
            "model.default"
        )

    def test_normalize_raises_on_unterminated_quote(self):
        with pytest.raises(ValueError):
            normalize_dotted_key('model."default')
