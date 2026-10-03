"""The ``hermes model`` provider menu must honor ``providers.<name>.enabled: false``.

``is_provider_enabled`` owns one contract: an explicit ``enabled: false`` hides a
provider from the picker, ``/models``, the runtime resolver and doctor. Every
surface that builds rows through ``list_authenticated_providers`` (gateway, TUI,
desktop, dashboard) inherits that, but ``hermes model`` builds its provider menu
from ``CANONICAL_PROVIDERS`` directly, so a provider the user disabled in
config.yaml still showed up there.

Invariant (both directions, no catalog snapshot): a provider block that disables
a provider removes it from the menu; a block that leaves it enabled keeps it.
"""

from __future__ import annotations


def _menu_slugs(config: dict) -> set[str]:
    """Provider keys offered by the ``hermes model`` menu, group members expanded."""
    from hermes_cli.main_provider_setup import _build_provider_picker_rows

    ordered, _default_idx = _build_provider_picker_rows(config, "", {}, {})
    slugs: set[str] = set()
    for key, _label, members in ordered:
        if members:
            slugs.update(members)
        elif not key.startswith(("group:", "custom", "remove-custom", "reasoning", "aux-config", "cancel")):
            slugs.add(key)
    return slugs


def test_disabled_provider_block_hides_it_from_the_model_menu():
    # Baseline: the provider is offered when config says nothing about it.
    assert "openrouter" in _menu_slugs({})

    # The same provider, explicitly disabled under `providers:` -> gone.
    hidden = _menu_slugs({"providers": {"openrouter": {"enabled": False}}})
    assert "openrouter" not in hidden
    # Sanity: only that provider was removed, the menu is otherwise intact.
    assert hidden and hidden != {"openrouter"}


def test_enabled_provider_block_keeps_the_provider_offered():
    """The relation is two-sided: the filter keys on the flag, not on the mere
    presence of a ``providers:`` entry."""
    assert "openrouter" in _menu_slugs({"providers": {"openrouter": {"enabled": True}}})
    assert "openrouter" in _menu_slugs({"providers": {"openrouter": {}}})
