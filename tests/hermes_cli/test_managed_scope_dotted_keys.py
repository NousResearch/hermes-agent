"""Managed scope must speak the same dotted-key representation as the CLI.

Regression: a managed model-route alias contains a dot (every real model id
does — ``gpt-5.6-sol``). ``_flatten_keys`` used to join segments with a plain
``f"{prefix}.{k}"``, producing a string that re-tokenizes to a *different* path
(``gpt-5`` → ``6-sol``). So ``is_key_managed`` never matched the key the CLI
builds, and ``_strip_dotted_keys`` never found the leaf: a user ``config set``
of that managed key was neither refused nor stripped. It persisted into
config.yaml and was then silently overridden by the managed overlay on load.

That is a "the user is misled" bug, not a privilege escalation — the managed
layer still wins at load time. What broke was the user's ability to know it.
"""
import textwrap

import pytest

QUOTED_KEY = 'platforms.api_server.extra.model_routes."gpt-5.6-sol".model'
# The same characters without quotes are NOT the same path: they address
# model_routes → gpt-5 → 6-sol → model. Asserting False on this is the point;
# a prefix or fuzzy match in is_key_managed would make it True.
UNQUOTED_KEY = "platforms.api_server.extra.model_routes.gpt-5.6-sol.model"

MANAGED_YAML = """
    platforms:
      api_server:
        extra:
          model_routes:
            "gpt-5.6-sol":
              model: managed/route-target
            plain-alias:
              model: managed/plain-target
    model:
      default: managed/model
"""


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    (managed / "config.yaml").write_text(
        textwrap.dedent(MANAGED_YAML), encoding="utf-8"
    )
    managed_scope.invalidate_managed_cache()
    return home, managed


# ---------------------------------------------------------------------------
# Flattening emits the canonical (quoted) form
# ---------------------------------------------------------------------------

def test_managed_config_keys_quotes_the_dot_bearing_alias(homes):
    from hermes_cli import managed_scope

    keys = managed_scope.managed_config_keys()
    assert QUOTED_KEY in keys
    # The old unquoted join must be gone, not merely accompanied.
    assert UNQUOTED_KEY not in keys


def test_dot_free_keys_flatten_byte_for_byte_as_before(homes):
    """Backward compat: the quoting rule must fire ONLY on a dot-bearing segment.

    Reference implementation is the pre-fix join, inlined. If ``join_dotted_key``
    ever quoted unconditionally (or dropped quoting entirely), this diverges.
    """
    from hermes_cli import managed_scope

    def reference_flatten(d, prefix=""):
        keys = set()
        for k, v in d.items():
            dotted = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict) and v:
                keys |= reference_flatten(v, dotted)
            else:
                keys.add(dotted)
        return keys

    dot_free = {
        "model": {"default": "x", "planner": "y"},
        "toolsets": {"enabled": ["a", "b"]},
        "platforms": {
            "api_server": {"extra": {"model_routes": {"plain-alias": {"model": "z"}}}}
        },
        "empty": {},
        1: {"numeric_parent": True},
    }
    assert managed_scope._flatten_keys(dot_free) == reference_flatten(dot_free)

    # And the dot-free siblings of the real fixture are untouched by the fix.
    keys = managed_scope.managed_config_keys()
    assert "model.default" in keys
    assert (
        "platforms.api_server.extra.model_routes.plain-alias.model" in keys
    )


def test_yaml_float_key_is_quoted(homes):
    """A YAML float key (``4.5:``) is the same defect class and is now quoted.

    Pinned deliberately: bare ``4.5`` addresses ``4`` → ``5``. This is an
    intentional behaviour change, not a regression to revert.
    """
    from hermes_cli import managed_scope

    assert managed_scope._flatten_keys({4.5: {"a": 1}}) == {'"4.5".a'}


# ---------------------------------------------------------------------------
# Comparison is exact, on the canonical form
# ---------------------------------------------------------------------------

def test_is_key_managed_matches_the_quoted_key(homes):
    from hermes_cli import managed_scope

    assert managed_scope.is_key_managed(QUOTED_KEY) is True


def test_is_key_managed_rejects_the_unquoted_key(homes):
    """The correct negative. ``...model_routes.gpt-5.6-sol.model`` genuinely
    addresses a different path (``gpt-5`` → ``6-sol``), which the managed layer
    does NOT pin. If this ever returns True, ``is_key_managed`` has been
    weakened into a prefix/fuzzy match — it is a config-authority check and
    must stay exact.
    """
    from hermes_cli import managed_scope

    assert managed_scope.is_key_managed(UNQUOTED_KEY) is False


def test_is_key_managed_normalizes_redundant_quoting(homes):
    """Quoting a segment that needs no quotes addresses the same path."""
    from hermes_cli import managed_scope

    assert managed_scope.is_key_managed('model."default"') is True


def test_is_key_managed_still_false_for_unmanaged_key(homes):
    from hermes_cli import managed_scope

    assert managed_scope.is_key_managed("model.planner") is False
    assert managed_scope.is_key_managed(
        'platforms.api_server.extra.model_routes."claude-sonnet-4.5".model'
    ) is False


def test_is_key_managed_does_not_raise_on_a_malformed_key(homes):
    """``doctor``/authority paths must not gain a new crash surface."""
    from hermes_cli import managed_scope

    assert managed_scope.is_key_managed('model."unterminated') is False


# ---------------------------------------------------------------------------
# The write guard now actually fires
# ---------------------------------------------------------------------------

def test_save_config_strips_the_dot_bearing_managed_leaf(homes, capsys):
    """A bulk write must not persist a value the overlay would override.

    Positive control in the same test: ``model.planner`` sits under a managed
    parent but is NOT itself managed, and must survive. Without it, this test
    would also pass if ``save_config`` had bailed out entirely or written an
    empty document.
    """
    from hermes_cli.config import save_config, get_config_path
    from utils import fast_safe_load

    home, _managed = homes
    save_config(
        {
            "model": {"default": "user/override", "planner": "user/planner"},
            "platforms": {
                "api_server": {
                    "extra": {
                        "model_routes": {
                            "gpt-5.6-sol": {"model": "user/route-override"},
                        }
                    }
                }
            },
        },
        strip_defaults=False,
    )

    written = fast_safe_load(get_config_path().read_text(encoding="utf-8")) or {}

    # Positive control — the unmanaged leaf landed, so the save really ran.
    assert written["model"]["planner"] == "user/planner"

    # Managed leaves are gone.
    assert "default" not in written.get("model", {})
    routes = (
        written.get("platforms", {})
        .get("api_server", {})
        .get("extra", {})
        .get("model_routes", {})
    )
    assert "model" not in routes.get("gpt-5.6-sol", {})

    # And the user was told, using the canonical spelling they must type.
    err = capsys.readouterr().err
    assert "managed setting(s) were not saved" in err
    assert QUOTED_KEY in err


def test_strip_dotted_keys_reports_the_quoted_key_as_stripped(homes):
    """Assert the returned set, not just the absence — absence alone passes
    vacuously if the nesting were wrong and the leaf never existed."""
    from hermes_cli.config import _strip_dotted_keys

    cfg = {
        "platforms": {
            "api_server": {
                "extra": {"model_routes": {"gpt-5.6-sol": {"model": "user/x"}}}
            }
        }
    }
    pruned, stripped = _strip_dotted_keys(cfg, {QUOTED_KEY})
    assert stripped == {QUOTED_KEY}
    assert pruned["platforms"]["api_server"]["extra"]["model_routes"][
        "gpt-5.6-sol"
    ] == {}


def test_strip_dotted_keys_ignores_a_malformed_key(homes):
    """A hand-built malformed key must not abort the whole save."""
    from hermes_cli.config import _strip_dotted_keys

    cfg = {"model": {"default": "x"}}
    pruned, stripped = _strip_dotted_keys(cfg, {'model."unterminated', "model.default"})
    assert stripped == {"model.default"}
    assert pruned == {"model": {}}


def test_config_set_of_the_dot_bearing_managed_key_is_refused(homes, capsys):
    """The user-facing half of the bug: this used to be accepted silently."""
    from hermes_cli.config import set_config_value

    with pytest.raises(SystemExit) as exc:
        set_config_value(QUOTED_KEY, "user/override")
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "managed" in (captured.out + captured.err).lower()


def test_config_show_renders_the_typeable_spelling(homes, capsys):
    """`hermes config show` lists managed keys — it must print the exact string
    the user has to type to address the leaf.

    Without this, the "Managed config keys:" banner would advertise
    ``...model_routes.gpt-5.6-sol.model``, and a user copying it straight back
    into `config set` would address a different path and get no warning at all.
    The existing surfacing tests only assert the banner is ABSENT when no
    managed scope exists, so nothing else covers the rendered strings.
    """
    from hermes_cli.config import show_config

    show_config()
    out = capsys.readouterr().out
    assert "managed by your administrator" in out.lower()
    assert QUOTED_KEY in out
    assert UNQUOTED_KEY not in out
