"""Explicit-profile canonical reads must not initialize or mutate profiles."""
import os
from concurrent.futures import ThreadPoolExecutor

from hermes_cli.config import DEFAULT_CONFIG, load_config_readonly
from hermes_constants import get_hermes_home


def test_explicit_readonly_home_is_canonical_isolated_and_does_not_initialize(tmp_path, monkeypatch):
    home = tmp_path / "requested"
    home.mkdir()
    path = home / "config.yaml"
    text = 'max_turns: 47\nplugins:\n  realms:\n    size: "${READONLY_SIZE}"\n'
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("READONLY_SIZE", "384x256")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        'plugins:\n  realms:\n    renderer: pixman\n', encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    ambient = get_hermes_home()
    before = dict(os.environ)
    missing = tmp_path / "missing"
    with ThreadPoolExecutor(max_workers=2) as executor:
        configured, defaults = list(executor.map(
            lambda profile: load_config_readonly(home=profile), [home, missing]
        ))
    assert configured["plugins"]["realms"]["size"] == "384x256"
    assert configured["plugins"]["realms"]["renderer"] == "pixman"
    assert configured["agent"]["max_turns"] == 47
    assert "max_turns" not in configured
    assert defaults["agent"]["max_turns"] == DEFAULT_CONFIG["agent"]["max_turns"]
    assert not missing.exists()
    assert list(home.iterdir()) == [path]
    assert path.read_text(encoding="utf-8") == text
    assert os.environ == before
    assert get_hermes_home() == ambient
    assert load_config_readonly(home=home) is configured


def test_explicit_readonly_rejects_corrupt_cached_fallback_without_writing(tmp_path):
    import pytest
    import yaml
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    home = tmp_path / "profile"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text('plugins:\n  realms:\n    size: 384x256\n', encoding="utf-8")
    token = set_hermes_home_override(home)
    try:
        good = load_config_readonly()
        path.write_text('plugins: [broken', encoding="utf-8")
        # The legacy API intentionally warns, backs up, and caches last-known-good.
        assert load_config_readonly() == good
    finally:
        reset_hermes_home_override(token)
    before = {p.relative_to(home): p.read_bytes() for p in home.rglob('*') if p.is_file()}
    for _ in range(2):
        with pytest.raises(yaml.YAMLError):
            load_config_readonly(home=home)
    assert {p.relative_to(home): p.read_bytes() for p in home.rglob('*') if p.is_file()} == before
