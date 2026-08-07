"""Regression test: HolographicMemoryProvider.save_config() must write the
shared main config.yaml atomically, like every sibling memory provider does.

Before this fix, save_config() bypassed hermes_cli.config's atomic
(utils.atomic_yaml_write-backed) round-trip and instead did a bare
``open(config_path, "w")`` truncating write directly against config.yaml —
the single file holding every provider's credentials and settings, not just
this plugin's. An interruption mid-write (crash, kill -9, power loss) left
the file truncated/corrupt; the read half then silently fell back to an
empty config on the next load, wiping every unrelated section.

hindsight/mem0/honcho route through utils.atomic_json_write for their own
separate files; supermemory and openviking route through
hermes_cli.config.save_config() (atomic_yaml_write-backed) for the shared
config.yaml. holographic was the only one of the six memory providers still
using the unsafe bare-write pattern.
"""

import yaml
import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.holographic import HolographicMemoryProvider


@pytest.fixture
def hermes_home(tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def test_save_config_preserves_unrelated_sections(hermes_home):
    """A read-modify-write must not drop sibling config sections — the exact
    failure mode a bare truncating write risks on interruption."""
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": "claude-fable-5",
                "providers": {"anthropic": {"api_key": "sk-existing-secret"}},
                "plugins": {"some-other-plugin": {"enabled": True}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    provider = HolographicMemoryProvider()
    provider.save_config({"db_path": "/tmp/x.db", "hrr_dim": 128}, str(hermes_home))

    on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert on_disk["providers"]["anthropic"]["api_key"] == "sk-existing-secret"
    assert on_disk["plugins"]["some-other-plugin"] == {"enabled": True}
    assert on_disk["plugins"]["hermes-memory-store"] == {
        "db_path": "/tmp/x.db",
        "hrr_dim": 128,
    }


def test_save_config_routes_through_atomic_yaml_write(hermes_home, monkeypatch):
    """The write must go through utils.atomic_yaml_write (temp file + fsync +
    atomic rename), not a bare truncating open()."""
    calls = []
    import utils

    real_atomic_yaml_write = utils.atomic_yaml_write

    def spy(path, data, *args, **kwargs):
        calls.append((path, data))
        return real_atomic_yaml_write(path, data, *args, **kwargs)

    monkeypatch.setattr(utils, "atomic_yaml_write", spy)

    provider = HolographicMemoryProvider()
    provider.save_config({"db_path": "/tmp/x.db"}, str(hermes_home))

    assert len(calls) == 1, "save_config must write via atomic_yaml_write exactly once"
    written_path, written_data = calls[0]
    assert written_data["plugins"]["hermes-memory-store"] == {"db_path": "/tmp/x.db"}


def test_save_config_creates_file_when_missing(hermes_home):
    """No pre-existing config.yaml — save_config must still succeed and
    create one (matches the prior bare-write behavior's happy path)."""
    provider = HolographicMemoryProvider()
    provider.save_config({"db_path": "/tmp/new.db"}, str(hermes_home))

    config_path = hermes_home / "config.yaml"
    assert config_path.exists()
    on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert on_disk["plugins"]["hermes-memory-store"] == {"db_path": "/tmp/new.db"}
