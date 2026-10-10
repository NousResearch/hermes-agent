"""Layered i18n: user overlay > bundled, plugin pack > overlay, partial packs fall through, pack-only
languages become supported, and unload drops the layer. Real temp homes, a real plugin directory with a
manifest loaded through the real discovery path — no loader mocks."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


# ── resets racing a cache fill ────────────────────────────────────────────────────────────────


def _pause(monkeypatch, name, thread_name, *, before=False, only=lambda *args: True):
    """Patch ``i18n_layers.<name>`` so that its first matching call on the thread called *thread_name*
    signals ``reached`` and waits for ``resume``, before or after the real call. Returns both events."""
    real = getattr(i18n_layers, name)
    reached, resume = threading.Event(), threading.Event()

    def wait_here(args):
        if threading.current_thread().name == thread_name and only(*args) and not reached.is_set():
            reached.set()
            assert resume.wait(timeout=10)

    def paused(*args, **kwargs):
        if before:
            wait_here(args)
        result = real(*args, **kwargs)
        if not before:
            wait_here(args)
        return result

    monkeypatch.setattr(i18n_layers, name, paused)
    return reached, resume


@dataclass(frozen=True)
class _FillRace:
    """A cache fill that pauses after reading the layer *pause*, while *change* resets the caches."""

    pause: str
    fill: Callable[[], object]
    change: Callable[[Path], None]
    observe: Callable[[], object]
    expected: object
    only: Callable[..., bool] = lambda home, *args: True


def _register_pack(lang: str, messages: dict[str, str]) -> None:
    i18n_layers.register_pack(lang, "core", messages, source="test")


def _rewrite_de_overlay(home: Path) -> None:
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump({"approval": {"denied": "New"}}), encoding="utf-8")
    i18n.reset_language_cache()


def _add_pl_overlay(home: Path) -> None:
    (home / "locales" / "pl.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Odmowa"}}), encoding="utf-8")
    i18n.reset_language_cache()


def _t_de() -> str:
    return i18n.t(_KEY, lang="de")


@pytest.mark.parametrize("race", [
    pytest.param(_FillRace("pack_layer", _t_de, lambda home: _register_pack("de", {_KEY: "Pack"}), _t_de, "Pack"),
                 id="merged-catalog"),
    pytest.param(_FillRace("layered_languages", i18n.supported_languages, lambda home: _register_pack("pl", {}),
                           lambda: "pl" in i18n.supported_languages(), True),
                 id="supported-languages"),
    pytest.param(_FillRace("parse_locale_file", _t_de, _rewrite_de_overlay, _t_de, "New",
                           only=lambda home, path: path.parent == i18n_layers.overlay_dir(home)),
                 id="overlay-layer"),
    pytest.param(_FillRace("scan_locale_dir", i18n.supported_languages, _add_pl_overlay,
                           lambda: "pl" in i18n.supported_languages(), True),
                 id="overlay-languages"),
])
def test_fill_that_overlaps_a_reset_does_not_cache_the_old_view(home, monkeypatch, race):
    """A worker reads a layer, a reset lands, then the worker stores what it built. The next lookup must
    see the state after the reset, not the worker's result."""
    (home / "locales" / "de.yaml").write_text(yaml.safe_dump({"approval": {"denied": "Old"}}), encoding="utf-8")
    i18n.reset_language_cache()
    reached, resume = _pause(monkeypatch, race.pause, "i18n-fill", only=lambda *args: race.only(home, *args))
    worker = threading.Thread(target=race.fill, name="i18n-fill")
    worker.start()
    assert reached.wait(timeout=10)

    race.change(home)
    resume.set()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert race.observe() == race.expected


def test_fill_during_a_reset_does_not_cache_the_old_pack_layer(home, monkeypatch):
    """A reset clears the merged catalogs and the layer views in two steps. A catalog built between
    those steps must not keep the pack layer from before the reset."""
    assert i18n_layers.pack_layer("de") == {}
    reached, resume = _pause(monkeypatch, "clear_cache", "i18n-reset", before=True)
    registrar = threading.Thread(target=_register_pack, args=("de", {_KEY: "Pack"}), name="i18n-reset")
    registrar.start()
    assert reached.wait(timeout=10)

    _t_de()
    resume.set()
    registrar.join(timeout=10)

    assert not registrar.is_alive()
    assert _t_de() == "Pack"
