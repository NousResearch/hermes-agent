"""``hermes plugins enable/disable`` must not silently no-op on passive kinds.

Model providers (``kind: model-provider``) register through
``providers/__init__.py``'s own discovery and are selected via ``hermes model``
/ ``model.provider``; exclusive plugins (``kind: exclusive``, e.g. memory
providers) activate via ``<category>.provider``. The general plugin loader
skips both kinds, so a ``plugins.enabled``/``disabled`` entry for them is dead
config. Before this fix, ``hermes plugins enable gemini`` printed a green
success message while changing nothing that any loader would ever read.
"""
import pytest


@pytest.fixture
def fake_plugins(tmp_path, monkeypatch):
    """A discovery view with one plugin per kind, backed by real manifests."""
    import hermes_cli.plugins_cmd as pc

    entries = []
    for name, kind in [
        ("gemini", "model-provider"),
        ("honcho", "exclusive"),
        ("nemo_relay", "standalone"),
    ]:
        d = tmp_path / name
        d.mkdir()
        (d / "plugin.yaml").write_text(
            f"name: {name}\nkind: {kind}\nversion: 1.0.0\n", encoding="utf-8"
        )
        key = f"model-providers/{name}" if kind == "model-provider" else name
        entries.append((name, "1.0.0", "desc", "bundled", str(d), key))

    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: entries)
    # config writes must land in a scratch HERMES_HOME
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    return pc


def _enabled_disabled(pc):
    return pc._get_enabled_set(), pc._get_disabled_set()


def test_enable_model_provider_is_refused_with_hint(fake_plugins, capsys):
    pc = fake_plugins
    pc.cmd_enable("gemini")
    out = capsys.readouterr().out
    assert "model provider" in out
    assert "hermes model" in out
    assert "Nothing was changed" in out
    enabled, disabled = _enabled_disabled(pc)
    assert "model-providers/gemini" not in enabled
    assert "model-providers/gemini" not in disabled


def test_disable_model_provider_is_refused_with_hint(fake_plugins, capsys):
    pc = fake_plugins
    pc.cmd_disable("gemini")
    out = capsys.readouterr().out
    assert "model provider" in out
    enabled, disabled = _enabled_disabled(pc)
    assert "model-providers/gemini" not in disabled


def test_enable_exclusive_is_refused_with_hint(fake_plugins, capsys):
    pc = fake_plugins
    pc.cmd_enable("honcho")
    out = capsys.readouterr().out
    assert "exclusive" in out
    assert "provider" in out
    enabled, _ = _enabled_disabled(pc)
    assert "honcho" not in enabled


def test_enable_standalone_still_works(fake_plugins, capsys):
    pc = fake_plugins
    pc.cmd_enable("nemo_relay", allow_tool_override=False)
    out = capsys.readouterr().out
    assert "enabled" in out
    enabled, _ = _enabled_disabled(pc)
    assert "nemo_relay" in enabled


def test_plugin_kind_defaults_to_standalone(fake_plugins):
    pc = fake_plugins
    assert pc._plugin_kind("no-such-key") == "standalone"


# ---------------------------------------------------------------------------
# Production-path coverage (not just synthetic explicit ``kind:`` manifests).
#
# The fixture above declares ``kind`` explicitly in each manifest. Real bundled
# memory providers (e.g. plugins/memory/honcho/plugin.yaml) do NOT, and are
# excluded from _discover_all_plugins() entirely; user providers rely on the
# loader's register_memory_provider / register_provider heuristic. These tests
# exercise both of those real paths.
# ---------------------------------------------------------------------------


def _scratch_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as cfg

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()


def test_bundled_memory_provider_recognized_via_memory_discovery(
    tmp_path, monkeypatch, capsys
):
    """Bundled memory providers ship without ``kind: exclusive`` and are absent
    from _discover_all_plugins(); they must still be classified exclusive via
    plugins.memory discovery and refused (not "not installed")."""
    import hermes_cli.plugins_cmd as pc
    import plugins.memory as mem

    # The general plugin scan does NOT surface memory providers...
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: [])
    # ...but the memory category's own discovery does: (name, desc, available).
    monkeypatch.setattr(
        mem, "discover_memory_providers", lambda: [("honcho", "Honcho memory", True)]
    )

    assert pc._plugin_kind("honcho") == "exclusive"
    assert pc._plugin_kind("memory/honcho") == "exclusive"

    _scratch_home(tmp_path, monkeypatch)

    pc.cmd_enable("honcho")
    out = capsys.readouterr().out
    assert "exclusive" in out
    assert "Nothing was changed" in out
    assert "honcho" not in pc._get_enabled_set()

    pc.cmd_disable("honcho")
    out = capsys.readouterr().out
    assert "exclusive" in out
    assert "honcho" not in pc._get_disabled_set()


def test_heuristic_user_memory_provider_classified_exclusive(tmp_path, monkeypatch):
    """A user provider without explicit ``kind`` whose __init__.py registers a
    memory provider is classified ``exclusive`` by the shared loader heuristic —
    _plugin_kind must not fall back to ``standalone``."""
    import hermes_cli.plugins_cmd as pc
    import plugins.memory as mem

    d = tmp_path / "mymem"
    d.mkdir()
    (d / "plugin.yaml").write_text("name: mymem\nversion: 1.0.0\n", encoding="utf-8")
    (d / "__init__.py").write_text(
        "from plugins.memory import MemoryProvider\n\n"
        "def setup(ctx):\n"
        "    ctx.register_memory_provider(MemoryProvider())\n",
        encoding="utf-8",
    )
    entry = ("mymem", "1.0.0", "desc", "user", str(d), "providers/mymem")
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: [entry])
    # Force the heuristic (manifest) path, not memory discovery.
    monkeypatch.setattr(mem, "discover_memory_providers", lambda: [])

    assert pc._plugin_kind("providers/mymem") == "exclusive"


def test_heuristic_user_model_provider_classified_model_provider(tmp_path, monkeypatch):
    """Same shared-classification guarantee for the model-provider heuristic
    (register_provider + ProviderProfile in __init__.py)."""
    import hermes_cli.plugins_cmd as pc
    import plugins.memory as mem

    d = tmp_path / "myprov"
    d.mkdir()
    (d / "plugin.yaml").write_text("name: myprov\nversion: 1.0.0\n", encoding="utf-8")
    (d / "__init__.py").write_text(
        "from providers import register_provider, ProviderProfile\n\n"
        "def setup(ctx):\n"
        "    register_provider(ProviderProfile(name='x'))\n",
        encoding="utf-8",
    )
    entry = ("myprov", "1.0.0", "desc", "user", str(d), "providers/myprov")
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: [entry])
    monkeypatch.setattr(mem, "discover_memory_providers", lambda: [])

    assert pc._plugin_kind("providers/myprov") == "model-provider"
