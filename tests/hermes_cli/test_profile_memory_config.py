"""cloned_memory_provider: a best-effort post-clone notice that never fails ``hermes profile create``."""

import pytest

from hermes_cli.profile_memory_config import cloned_memory_provider


@pytest.mark.parametrize("body", [
    "memory:\n  provider: honcho\nwhen: 2024-02-30\n",  # YAML constructor raises a bare ValueError
    "memory: [unclosed\n",                              # YAMLError
    b"memory:\n  provider: \xff\n",                     # UnicodeDecodeError
])
def test_unreadable_config_means_no_notice(tmp_path, body):
    (tmp_path / "honcho").mkdir()
    path = tmp_path / "config.yaml"
    if isinstance(body, bytes):
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    assert cloned_memory_provider(tmp_path) is None


def test_missing_config_means_no_notice(tmp_path):
    assert cloned_memory_provider(tmp_path) is None


def test_configured_provider_with_copied_state_is_named(tmp_path):
    (tmp_path / "honcho").mkdir()
    (tmp_path / "config.yaml").write_text("memory:\n  provider: honcho\n", encoding="utf-8")
    assert cloned_memory_provider(tmp_path) == "honcho"
