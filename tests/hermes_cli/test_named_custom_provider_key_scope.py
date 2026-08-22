"""The named-custom-provider setup flow must resolve its credential through
the per-profile secret scope.

``_model_flow_named_custom`` probes the provider's ``/models`` endpoint with
whatever credential it resolves, so a scope-blind read does not merely pick
the wrong value — it transmits another profile's API key to *this* provider's
base URL, which may be an entirely different third party.

``0569c001d08`` closed this class on the ``/model`` picker path
(``hermes_cli/model_switch.py``), and ``_scoped_key_env``'s docstring records
that "the picker's ``key_env`` reads were not covered". These tests pin the
``hermes model`` setup-flow twin, which resolves the same two credential
shapes: a ``${VAR}`` config ref and a ``key_env`` name.
"""

import pytest

from agent import secret_scope
from hermes_cli.config import invalidate_env_cache


def _provider_info(**overrides):
    """A named custom_providers entry as ``_named_custom_provider_map`` builds it."""
    info = {
        "name": "MyCorp",
        "base_url": "https://mycorp.example/v1",
        "api_mode": "chat_completions",
        "api_key": "",
        "key_env": "",
        "model": "mycorp-1",
        "models": {},
        "discover_models": True,
        "provider_key": "mycorp",
        "api_key_ref": "",
        "base_url_ref": "",
    }
    info.update(overrides)
    return info


@pytest.fixture
def probed_keys(monkeypatch, tmp_path):
    """Capture the api_key ``_model_flow_named_custom`` probes with.

    ``HERMES_HOME`` is redirected at a tmp dir so the ``~/.hermes/.env``
    fallback inside ``get_env_value`` cannot see the developer's real file.
    The menu is stubbed to "Cancel" so the flow returns right after the probe
    without writing config or reading stdin.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    invalidate_env_cache()

    seen: list[str] = []

    def _fetch(api_key, base_url, **kwargs):
        seen.append(api_key)
        return ["mycorp-1"]

    monkeypatch.setattr("hermes_cli.models.fetch_api_models", _fetch)
    monkeypatch.setattr("hermes_cli.curses_ui.curses_radiolist", lambda *a, **k: -1)
    monkeypatch.setattr("hermes_cli.main._save_custom_provider", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.auth._save_model_choice", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.auth.deactivate_provider", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {}, "providers": {}, "custom_providers": []},
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)
    try:
        yield seen
    finally:
        invalidate_env_cache()


@pytest.fixture
def scope():
    """Install/remove a profile secret scope and restore the multiplex flag."""
    tokens = []
    previous_multiplex = secret_scope.is_multiplex_active()

    def _install(secrets, *, multiplex=True):
        tokens.append(secret_scope.set_secret_scope(secrets))
        secret_scope.set_multiplex_active(multiplex)

    try:
        yield _install
    finally:
        for token in reversed(tokens):
            secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(previous_multiplex)


def test_key_env_resolves_from_the_profile_scope_not_the_process_env(
    monkeypatch, probed_keys, scope
):
    """The scope is authoritative for a ``key_env`` credential.

    Under the multiplexed gateway ``os.environ`` may hold a *different*
    profile's key for the same variable name. Resolving the probe credential
    from it sends that profile's key to this provider's endpoint.
    """
    monkeypatch.setenv("MYCORP_API_KEY", "sk-other-profile")
    scope({"MYCORP_API_KEY": "sk-this-profile"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom({}, _provider_info(key_env="MYCORP_API_KEY"))

    assert probed_keys == ["sk-this-profile"], (
        "the named-custom-provider probe must use this profile's scoped "
        "credential, not the process environment's"
    )


def test_unresolved_env_ref_api_key_is_resolved_not_sent_as_the_bearer_token(
    monkeypatch, probed_keys, scope
):
    """``load_config`` keeps an unresolvable ``${VAR}`` ref verbatim.

    ``_expand_env_vars`` deliberately leaves the literal in place "so callers
    can detect them", so ``provider_info["api_key"]`` can be the placeholder
    string itself. It is truthy, so the flow accepts it as a credential and
    sends ``${MYCORP_API_KEY}`` as the bearer token — a guaranteed 401 with a
    misleading cause. Resolve the ref instead, through the profile scope.
    """
    monkeypatch.delenv("MYCORP_API_KEY", raising=False)
    scope({"MYCORP_API_KEY": "sk-this-profile"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom({}, _provider_info(api_key="${MYCORP_API_KEY}"))

    assert probed_keys == ["sk-this-profile"], (
        "an unresolved ${VAR} api_key ref must be resolved through the "
        "scope-aware reader, never sent to the endpoint verbatim"
    )


def test_unresolvable_env_ref_api_key_falls_through_to_key_env(
    monkeypatch, probed_keys, scope
):
    """A ref that resolves to nothing must not shadow ``key_env``.

    Both fields may be present on one entry (``api_key: ${A}`` plus
    ``key_env: B``). While the placeholder stays truthy the ``key_env``
    branch is unreachable, so the configured fallback silently never runs.
    """
    monkeypatch.delenv("MYCORP_MISSING_KEY", raising=False)
    scope({"MYCORP_FALLBACK_KEY": "sk-fallback"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom(
        {},
        _provider_info(
            api_key="${MYCORP_MISSING_KEY}",
            key_env="MYCORP_FALLBACK_KEY",
        ),
    )

    assert probed_keys == ["sk-fallback"], (
        "an unresolvable ${VAR} ref must fall through to key_env instead of "
        "being sent to the endpoint as a literal"
    )


def test_already_expanded_env_ref_api_key_is_re_resolved_through_the_scope(
    monkeypatch, probed_keys, scope
):
    """The ``${VAR}`` case that is *invisible* in ``provider_info["api_key"]``.

    ``_expand_env_vars`` only keeps the ``${VAR}`` literal when the variable is
    unset. When it *is* set, the expansion substitutes it — out of the
    process-global ``os.environ``, via ``config.py::_env_expand_match``, with no
    scope check — and ``api_key`` arrives as a plain string indistinguishable
    from a directly-configured inline key.

    So the placeholder shape detects only the harmless branch. This is the
    harmful one: the value is another profile's key and it is about to be sent
    to this provider's ``base_url``. The unexpanded template is still on
    ``api_key_ref``, so the resolution must key off that.
    """
    monkeypatch.setenv("MYCORP_API_KEY", "sk-other-profile")
    scope({"MYCORP_API_KEY": "sk-this-profile"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom(
        {},
        _provider_info(
            # What ``_expand_env_vars`` produced from the process environment.
            api_key="sk-other-profile",
            # What config.yaml actually says.
            api_key_ref="${MYCORP_API_KEY}",
        ),
    )

    assert probed_keys == ["sk-this-profile"], (
        "an api_key whose config ref was already expanded from the process "
        "environment must be re-resolved through the profile scope"
    )


def test_already_expanded_env_ref_api_key_is_dropped_when_out_of_scope(
    monkeypatch, probed_keys, scope
):
    """No credential for this profile means no credential — not the other one.

    With the referenced variable absent from the scope, ``get_env_value``
    returns ``None`` (scope is authoritative under multiplexing). The expanded
    process-env value must be discarded rather than used as the fallback,
    otherwise the scope check accomplishes nothing on this path.
    """
    monkeypatch.setenv("MYCORP_API_KEY", "sk-other-profile")
    scope({"UNRELATED_KEY": "sk-unrelated"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom(
        {},
        _provider_info(
            api_key="sk-other-profile",
            api_key_ref="${MYCORP_API_KEY}",
        ),
    )

    assert probed_keys == [""], (
        "a config ref the profile scope cannot resolve must not fall back to "
        "the process environment's expanded value"
    )


def test_composite_api_key_ref_is_rebuilt_from_the_profile_scope(
    monkeypatch, probed_keys, scope
):
    """A ``${...}`` inside a larger string is the same cross-profile read.

    ``_expand_env_vars`` substitutes *every* ``${...}`` it finds, not only a
    template that is exactly one whole ref, and every one of those
    substitutions comes from the process-global ``os.environ``. So
    ``sk-${MYCORP_SUFFIX}`` expands into a perfectly well-formed key built out
    of another profile's secret, and it keeps no ``${`` afterwards to give
    itself away. The surrounding literal text does not make it a different
    question — the whole value has to be rebuilt from the template through the
    scope-aware reader.
    """
    monkeypatch.setenv("MYCORP_SUFFIX", "other-profile")
    scope({"MYCORP_SUFFIX": "this-profile"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom(
        {},
        _provider_info(
            # What ``_expand_env_vars`` produced from the process environment.
            api_key="sk-other-profile",
            # What config.yaml actually says.
            api_key_ref="sk-${MYCORP_SUFFIX}",
        ),
    )

    assert probed_keys == ["sk-this-profile"], (
        "every ${...} in a composite api_key ref must resolve through the "
        "profile scope, not survive from the process-environment expansion"
    )


def test_multi_ref_api_key_ref_is_rebuilt_from_the_profile_scope(
    monkeypatch, probed_keys, scope
):
    """Two refs in one template, neither of which is the whole template.

    ``${A}-${B}`` both starts with ``${`` and ends with ``}`` while being no
    kind of bare ref at all, so a shape test that inspects only the template's
    two ends misreads it. ``_expand_env_vars`` treats it as two independent
    substitutions; the re-resolution has to do the same.
    """
    monkeypatch.setenv("MYCORP_PREFIX", "other")
    monkeypatch.setenv("MYCORP_SUFFIX", "profile")
    scope({"MYCORP_PREFIX": "this", "MYCORP_SUFFIX": "profile-key"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom(
        {},
        _provider_info(
            api_key="other-profile",
            api_key_ref="${MYCORP_PREFIX}-${MYCORP_SUFFIX}",
        ),
    )

    assert probed_keys == ["this-profile-key"], (
        "each ${...} in a multi-ref api_key ref must resolve through the "
        "profile scope independently"
    )


def test_composite_api_key_ref_is_dropped_when_a_ref_is_out_of_scope(
    monkeypatch, probed_keys, scope
):
    """A hole in a rebuilt credential must fail closed, not fall back.

    The scope is authoritative under multiplexing, so a ref it cannot resolve
    yields nothing *for this profile*. Splicing the process environment's value
    back in to keep the string well-formed would defeat the check entirely, and
    a half-built key is worth nothing to the endpoint in any case.
    """
    monkeypatch.setenv("MYCORP_SUFFIX", "other-profile")
    scope({"UNRELATED_KEY": "sk-unrelated"})

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    _model_flow_named_custom(
        {},
        _provider_info(
            api_key="sk-other-profile",
            api_key_ref="sk-${MYCORP_SUFFIX}",
        ),
    )

    assert probed_keys == [""], (
        "a composite ref the profile scope cannot resolve must not keep the "
        "process environment's expanded value"
    )


def test_key_env_is_fail_closed_when_multiplexing_runs_without_a_scope(
    monkeypatch, probed_keys, scope
):
    """Multiplexing on with no scope installed must not read ``os.environ``.

    This is the other half of ``get_secret``'s policy and a distinct branch
    from the scope-installed case above: with no scope there is nothing to
    identify the caller's profile, so ``os.environ`` is unattributable and the
    read raises rather than guessing. Silently probing with whatever the
    process env holds is exactly the cross-profile disclosure the scope exists
    to prevent.
    """
    monkeypatch.setenv("MYCORP_API_KEY", "sk-other-profile")
    scope(None, multiplex=True)

    from hermes_cli.model_setup_flows import _model_flow_named_custom

    with pytest.raises(secret_scope.UnscopedSecretError):
        _model_flow_named_custom({}, _provider_info(key_env="MYCORP_API_KEY"))

    assert probed_keys == [], (
        "no probe may be issued with an unattributable process-env credential"
    )
