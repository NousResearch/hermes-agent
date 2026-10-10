"""Every provider-declared ``post_setup`` key must resolve to a registered hook.

A provider row (bundled or plugin) advertises ``post_setup: <key>`` so that picking
it in ``hermes tools`` installs its optional dependency. ``valid_post_setup_keys()``
derives its allowlist from exactly those declarations, so a declared key with no
entry in ``_POST_SETUP_HOOKS`` passes validation and then silently no-ops:
``_run_post_setup`` resolves through ``.get(key, lambda: None)``.

That is a real failure mode, not a hypothetical: when a provider's installer moves
between the if/elif ``_run_post_setup`` ladder and the ``_POST_SETUP_HOOKS`` table,
the key keeps validating while the installation branch can be dropped — the user
picks the provider, reads "Saved", and the dependency is never installed.

This is a contract between two pieces of data, not a snapshot: it holds for any
provider set, so new no-key providers get the same protection.
"""

from __future__ import annotations


def test_every_declared_post_setup_key_has_a_hook():
    from hermes_cli.tools_config_post_setup import _POST_SETUP_HOOKS, valid_post_setup_keys

    missing = sorted(valid_post_setup_keys() - set(_POST_SETUP_HOOKS))
    assert not missing, (
        f"provider(s) declare post_setup key(s) with no registered hook: {missing} — "
        "picking that provider would silently skip its dependency install. "
        "Add a _PYTHON_POST_SETUP_HOOKS entry (or a bespoke _POST_SETUP_HOOKS hook) for each key."
    )


def test_python_post_setup_hooks_are_reachable_from_dispatch():
    """``_POST_SETUP_HOOKS`` is built by spreading ``_PYTHON_POST_SETUP_HOOKS`` into
    itself, so a python hook present in the inner table but not wired to dispatch
    would still validate (its provider declares the key) and no-op at pick time."""
    from hermes_cli.tools_config_post_setup import _PYTHON_POST_SETUP_HOOKS, _POST_SETUP_HOOKS

    unwired = sorted(set(_PYTHON_POST_SETUP_HOOKS) - set(_POST_SETUP_HOOKS))
    assert not unwired, f"python post-setup hook(s) not reachable from dispatch: {unwired}"


def test_python_post_setup_specs_are_complete():
    """Every python hook needs the fields ``_post_setup_python`` reads.

    That reader indexes ``spec["label"]``, ``spec["installing"]``, ``spec["extra"]``
    and passes ``spec["on_install"]``/``spec["always"]`` to ``_info_lines`` —
    a missing field raises at picker time, which is the worst moment to find out.
    ``extra`` must also name a real pyproject extra, or ``pm.sync_venv`` fails at
    install time; keeping it non-empty here catches the common copy/paste slip.
    """
    from hermes_cli.tools_config_post_setup import _PYTHON_POST_SETUP_HOOKS

    required = {"module", "extra", "label", "installing", "on_install", "always"}
    for key, spec in _PYTHON_POST_SETUP_HOOKS.items():
        missing = sorted(required - set(spec))
        assert not missing, f"python post-setup hook {key!r} is missing field(s): {missing}"
        assert spec["extra"], f"python post-setup hook {key!r} names no pyproject extra"
        assert spec["module"], f"python post-setup hook {key!r} names no import module"
