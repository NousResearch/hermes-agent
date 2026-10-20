"""Layered i18n: user overlay > bundled, plugin pack > overlay, partial packs fall through, pack-only
languages become supported, and unload drops the layer. Real temp homes, a real plugin directory with a
manifest loaded through the real discovery path — no loader mocks."""

from __future__ import annotations

import hermes_yaml as yaml
import pytest

from agent import i18n, i18n_layers
from hermes_cli.plugins import PluginManager

# A bundled key every test can lean on (approval prompts ship in every locale).
_KEY = "approval.denied"


def _en(key: str = _KEY) -> str:
    return i18n.t(key, lang="en")


@pytest.fixture
def clean_layers():
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()
    yield
    i18n_layers._reset_registry_for_tests()
    i18n.reset_language_cache()


@pytest.fixture
def home(tmp_path, monkeypatch, clean_layers):
    """A temp HERMES_HOME with no plugins and an empty bundled plugin dir."""
    from hermes_cli import plugins as plugins_mod

    home = tmp_path / "home"
    (home / "locales").mkdir(parents=True)
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    i18n.reset_language_cache()
    return home


def _write_pack_plugin(home, name="hermes-lang-pl", *, lang="pl", core=None, tui=None, desktop=None,
                       manifest_extra=None, with_init=False):
    plugin = home / "plugins" / name
    (plugin / "locales").mkdir(parents=True)
    manifest = {"name": name, "version": "1.0.0", "description": f"{lang} language pack",
                "provides_locales": [lang], **(manifest_extra or {})}
    (plugin / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    for surface, data in (("", core), (".tui", tui), (".desktop", desktop)):
        if data is not None:
            (plugin / "locales" / f"{lang}{surface}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True),
                                                                      encoding="utf-8")
    if with_init:
        (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [name]}}), encoding="utf-8")
    return plugin


def _load(home) -> PluginManager:
    manager = PluginManager()
    manager.discover_and_load()
    return manager


# ── overlay ───────────────────────────────────────────────────────────────────────────────────


def test_user_overlay_overrides_bundled_and_is_partial(home):
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Überschrieben"}}),
                                              encoding="utf-8")
    i18n.reset_language_cache()
    assert i18n.t(_KEY, lang="de") == "Überschrieben"
    # A key the overlay does not carry still comes from the bundled German catalog, not English.
    bundled_de = i18n_layers.parse_locale_file(i18n._locales_dir() / "de.yaml")
    other = next(k for k in bundled_de if k != _KEY and bundled_de[k] != _en(k))
    assert i18n.t(other, lang="de") == bundled_de[other]


def test_overlay_only_language_is_supported_and_falls_back_to_english(home):
    (home / "locales" / "eo.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Esperanto titolo"}}),
                                              encoding="utf-8")
    i18n.reset_language_cache()
    assert "eo" in i18n.supported_languages()
    assert i18n.t(_KEY, lang="eo") == "Esperanto titolo"
    other = next(k for k in i18n._load_bundled("en") if k != _KEY)
    assert i18n.t(other, lang="eo") == _en(other)
    assert i18n.supported_languages()[0] == "en"


def test_overlay_is_profile_scoped_across_two_homes(tmp_path, monkeypatch, clean_layers):
    """Home A overlays de; home B does not. A → B → A must never serve A's overlay to B or B's miss to A."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    (home_a / "locales").mkdir(parents=True)
    home_b.mkdir()
    (home_a / "locales" / "de.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Nur A"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    i18n.reset_language_cache()
    bundled_de = i18n_layers.parse_locale_file(i18n._locales_dir() / "de.yaml")[_KEY]

    assert i18n.t(_KEY, lang="de") == "Nur A"
    token = set_hermes_home_override(home_b)
    try:
        assert i18n.t(_KEY, lang="de") == bundled_de
        assert "eo" not in i18n.supported_languages()
    finally:
        reset_hermes_home_override(token)
    assert i18n.t(_KEY, lang="de") == "Nur A"


# ── plugin packs through the real loader ──────────────────────────────────────────────────────


def test_manifest_only_pack_registers_language_and_display_language_resolves(home):
    _write_pack_plugin(home, core={"approval": {"denied": "Zatwierdzenie"}},
                       tui={"status": {"ready": "Gotowy"}}, desktop={"settings": {"title": "Ustawienia"}})
    assert "pl" not in i18n.supported_languages()

    manager = _load(home)
    loaded = manager._plugins["hermes-lang-pl"]
    assert loaded.enabled, loaded.error

    assert "pl" in i18n.supported_languages()
    assert i18n.t(_KEY, lang="pl") == "Zatwierdzenie"
    # Partial pack: a missing key falls through to English (there is no bundled pl).
    other = next(k for k in i18n._load_bundled("en") if k != _KEY)
    assert i18n.t(other, lang="pl") == _en(other)
    # Surfaces are kept apart and served separately.
    assert i18n.surface_catalog("pl", "tui") == {"status.ready": "Gotowy"}
    assert i18n.surface_catalog("pl", "desktop") == {"settings.title": "Ustawienia"}
    assert i18n.surface_catalog("pl", "core") == {"approval.denied": "Zatwierdzenie"}
    # Registration never touches display.language; setting it makes the pack the active language.
    assert i18n.get_language() == "en"
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["hermes-lang-pl"]}, "display": {"language": "pl"}}), encoding="utf-8")
    i18n.reset_language_cache()
    assert i18n.get_language() == "pl"
    assert i18n.t(_KEY) == "Zatwierdzenie"
    option = next(o for o in i18n.language_options() if o["id"] == "pl")
    assert option == {"id": "pl", "endonym": "pl", "rtl": False, "source": "plugin:hermes-lang-pl"}


def test_pack_with_register_function_and_metadata(home):
    _write_pack_plugin(home, core={"approval": {"denied": "Zatwierdzenie"}}, with_init=True,
                       manifest_extra={"provides_locales": [{"id": "pl", "endonym": "Polski", "rtl": False}]})
    manager = _load(home)
    assert manager._plugins["hermes-lang-pl"].enabled
    option = next(o for o in i18n.language_options() if o["id"] == "pl")
    assert option["endonym"] == "Polski" and option["source"] == "plugin:hermes-lang-pl"


def test_pack_overrides_overlay_and_unload_drops_it(home):
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Overlay"}}), encoding="utf-8")
    _write_pack_plugin(home, "hermes-lang-de", lang="de", core={"approval": {"denied": "Pack"}})
    i18n.reset_language_cache()
    assert i18n.t(_KEY, lang="de") == "Overlay"

    manager = _load(home)
    assert manager._plugins["hermes-lang-de"].enabled
    assert i18n.t(_KEY, lang="de") == "Pack"

    manager.unload()
    assert i18n.t(_KEY, lang="de") == "Overlay"
    assert i18n_layers.registered_packs() == ()


def test_pack_only_language_disappears_on_unload(home):
    _write_pack_plugin(home, core={"approval": {"denied": "Zatwierdzenie"}})
    manager = _load(home)
    assert "pl" in i18n.supported_languages()
    manager.unload()
    assert "pl" not in i18n.supported_languages()
    assert i18n.t(_KEY, lang="pl") == _en()  # unknown id → English, never the bare key


def test_later_pack_wins_over_earlier_pack(home):
    _write_pack_plugin(home, "aaa-lang-pl", core={"approval": {"denied": "First"}, "approval.cancelled": "Only first"})
    _write_pack_plugin(home, "zzz-lang-pl", core={"approval": {"denied": "Second"}})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["aaa-lang-pl", "zzz-lang-pl"]}}),
                                      encoding="utf-8")
    _load(home)
    assert i18n.t(_KEY, lang="pl") == "Second"
    assert i18n.t("approval.cancelled", lang="pl") == "Only first"


def test_register_locale_accepts_dicts_and_rejects_bad_ids(home):
    from hermes_cli.plugins import PluginContext
    from hermes_cli.plugins_manifest import PluginManifest

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="inline-pack", source="user", path=str(home)), manager)
    handle = ctx.register_locale("PT_BR", {"approval": {"denied": "Título"}}, endonym="Português (Brasil)")
    assert handle.key == "pt-br.core"
    assert i18n.t(_KEY, lang="pt-BR") == "Título"  # the supplied id beats the pt-br → pt alias
    with pytest.raises(ValueError):
        ctx.register_locale("not a lang", {"a": "b"})
    with pytest.raises(ValueError):
        ctx.register_locale("pl", {"a": "b"}, surface="web")
    with pytest.raises(FileNotFoundError):
        ctx.register_locale("pl", home / "missing.yaml")


# ── cyclic YAML anchors (issue #132006) ───────────────────────────────────────────────────────

# 14 bytes: a legal YAML anchor+alias that parses into a self-referential dict.
_CYCLE_DOC = "a: &x\n  b: *x\n"


def test_flatten_and_non_text_leaves_reject_a_cycle_with_valueerror():
    document = yaml.safe_load(_CYCLE_DOC)
    assert document["a"]["b"] is document["a"]  # ruamel keeps the shared object
    with pytest.raises(ValueError, match="cyclic"):
        i18n_layers.flatten(document)
    with pytest.raises(ValueError, match="cyclic"):
        i18n_layers.non_text_leaves(document)


def test_flatten_rejects_a_two_node_cycle():
    """``first: &a {second: &b {back: *a}}`` — the cycle passes through two distinct mappings."""
    document = yaml.safe_load("first: &a\n  second: &b\n    back: *a\n")
    assert document["first"]["second"]["back"] is document["first"]
    with pytest.raises(ValueError, match="cyclic"):
        i18n_layers.flatten(document)
    with pytest.raises(ValueError, match="cyclic"):
        i18n_layers.non_text_leaves(document)


def test_deep_acyclic_mapping_still_flattens():
    """Deep but acyclic nesting (a hostile locale file shape) flattens normally well past real catalogs."""
    document: dict = {"leaf": "v"}
    for i in reversed(range(300)):
        document = {f"l{i}": document}
    expected = ".".join(f"l{i}" for i in range(300)) + ".leaf"
    assert i18n_layers.flatten(document) == {expected: "v"}
    assert i18n_layers.non_text_leaves(document) == []


def test_flatten_keeps_a_shared_anchor_reused_by_sibling_keys():
    shared = {"k": "v"}
    document = {"a": shared, "b": shared}  # what `a: &x {k: v}` / `b: *x` parses into
    assert i18n_layers.flatten(document) == {"a.k": "v", "b.k": "v"}
    assert i18n_layers.non_text_leaves(document) == []


def test_parse_locale_file_rejects_a_cycle(tmp_path):
    path = tmp_path / "pl.yaml"
    path.write_text(_CYCLE_DOC, encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic"):
        i18n_layers.parse_locale_file(path)


def test_cyclic_overlay_degrades_to_empty_without_raising(home):
    (home / "locales" / "pl.yaml").write_text(_CYCLE_DOC, encoding="utf-8")
    assert i18n_layers.overlay_layer(home, "pl", i18n_layers.CORE_SURFACE) == {}
    assert i18n.t(_KEY, lang="pl") == _en()  # falls back to the bundled catalog
