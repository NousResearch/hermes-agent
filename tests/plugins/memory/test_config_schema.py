"""Tests for config-schema loading from memory provider plugin dirs."""

from plugins.memory import config_schema
from plugins.memory.config_schema import get_provider_config_schema


def test_unknown_provider_is_none():
    assert get_provider_config_schema("builtin") is None






def test_cache_keys_on_schema_path_not_name(monkeypatch, tmp_path):
    # User-installed plugins are per-profile; two profiles' plugins sharing a
    # name must not answer for each other.
    from plugins import memory

    schemas = {}
    for label in ("A", "B"):
        plugin_dir = tmp_path / label / "custom"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "config_schema.py").write_text(
            "from plugins.memory.config_schema import ProviderConfigSchema\n"
            f'CONFIG_SCHEMA = ProviderConfigSchema(name="custom", label="{label}")\n',
            encoding="utf-8",
        )
        schemas[label] = plugin_dir

    monkeypatch.setattr(config_schema, "_SCHEMA_CACHE", {})
    monkeypatch.setattr(memory, "find_provider_dir", lambda name: schemas["A"])
    assert get_provider_config_schema("custom").label == "A"

    monkeypatch.setattr(memory, "find_provider_dir", lambda name: schemas["B"])
    assert get_provider_config_schema("custom").label == "B"


def test_broken_schema_is_not_cached(monkeypatch, tmp_path):
    # A load failure must retry on the next request, not pin an empty panel.
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    schema_file = broken_dir / "config_schema.py"
    schema_file.write_text("this is not python(", encoding="utf-8")

    monkeypatch.setattr(config_schema, "_SCHEMA_CACHE", {})
    from plugins import memory

    monkeypatch.setattr(memory, "find_provider_dir", lambda name: broken_dir)

    assert get_provider_config_schema("broken") is None
    assert not config_schema._SCHEMA_CACHE

    schema_file.write_text(
        "from plugins.memory.config_schema import ProviderConfigSchema\n"
        'CONFIG_SCHEMA = ProviderConfigSchema(name="broken", label="Broken")\n',
        encoding="utf-8",
    )

    recovered = get_provider_config_schema("broken")
    assert recovered is not None
    assert recovered.label == "Broken"


def test_a_home_provider_schema_is_refused_in_a_hosted_process(monkeypatch, tmp_path):
    """A config schema is the provider plugin's own code, so the one hosted provenance rule
    covers it: in a hosted process a schema under the deployment home never imports (the
    panel simply has no schema), and the local behaviour above is unchanged."""
    from hermes_cli import hosted_mode
    from plugins import memory

    monkeypatch.setattr(hosted_mode, "is_hosted", lambda: True)
    monkeypatch.setattr(config_schema, "_SCHEMA_CACHE", {})
    provider_dir = tmp_path / "custom"
    provider_dir.mkdir()
    (provider_dir / "config_schema.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'schema-imported')!r}).write_text('x')\n"
        "from plugins.memory.config_schema import ProviderConfigSchema\n"
        'CONFIG_SCHEMA = ProviderConfigSchema(name="custom", label="Custom")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(memory, "find_provider_dir", lambda name: provider_dir)

    assert get_provider_config_schema("custom") is None
    assert not (tmp_path / "schema-imported").exists()  # the schema module never ran
