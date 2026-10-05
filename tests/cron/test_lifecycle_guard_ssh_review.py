"""Independent-review regressions; all lifecycle payloads are inert strings."""
from pathlib import Path

import pytest
from cron import lifecycle_guard as g


@pytest.fixture
def config(tmp_path, monkeypatch):
    (tmp_path / '.ssh').mkdir()
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setattr(g, '_SSH_SYSTEM_CONFIG', tmp_path / 'system')
    monkeypatch.setattr(g, '_local_host_identities', lambda: frozenset({'localhost', '127.0.0.1', '192.168.9.9'}))
    return tmp_path


@pytest.mark.parametrize('command', [
    "ssh -P 192.168.3.184 localhost 'hermes gateway restart'",
    "ssh -Yz 192.168.3.184 localhost 'hermes gateway restart'",
    "ssh --future-option 192.168.3.184 localhost 'hermes gateway restart'",
    "ssh 192.168.3.184 'echo hermes gateway restart' | /bin/sh",
    "ssh 192.168.3.184 'echo hermes.exe gateway stop' | env sh",
    "ssh 192.168.3.184 uptime; echo '#'; pkill python3",
    "ssh 192.168.3.184 uptime; echo '#'; taskkill /F /IM python.exe",
    "ssh 192.168.3.184 uptime; echo '#'; Stop-Process -Name python",
])
def test_new_option_and_pipeline_consumers_fail_closed(config, command):
    assert g.contains_gateway_lifecycle_command_or_referenced_script(command)


def test_system_read_resets_match_state(config):
    (config / '.ssh/config').write_text('Host peer\n HostName 192.168.3.184\nHost other\n')
    (config / 'system').write_text('ProxyCommand nc 127.0.0.1 22\n')
    assert g.contains_gateway_lifecycle_command_or_referenced_script("ssh peer 'hermes gateway restart'")


def test_include_restores_parent_match(config):
    (config / 'fragment').write_text('Host other\n')
    (config / '.ssh/config').write_text(f'Host peer\n HostName 192.168.3.184\n Include {config}/fragment\n ProxyCommand nc 127.0.0.1 22\n')
    assert g.contains_gateway_lifecycle_command_or_referenced_script("ssh peer 'hermes gateway restart'")


def test_inactive_include_cannot_activate_local_alias(config):
    (config / 'fragment').write_text('Host peer\n HostName 192.168.3.184\n')
    (config / '.ssh/config').write_text(f'Host other\n Include {config}/fragment\nHost peer\n HostName 127.0.0.1\n')
    assert g.contains_gateway_lifecycle_command_or_referenced_script("ssh peer 'hermes gateway restart'")


def test_config_reads_share_whole_walk_path_budget(config, monkeypatch):
    (config / '.ssh/config').write_text('Host peer\n HostName 192.168.3.184\n')
    monkeypatch.setattr(g, '_MAX_LIFECYCLE_SCAN_PATHS', 1)
    assert g.contains_gateway_lifecycle_command_or_referenced_script("ssh peer 'hermes gateway restart'")


def test_config_text_shares_whole_walk_budget(config, monkeypatch):
    (config / '.ssh/config').write_text('#' + 'x' * 1000 + '\nHost peer\n HostName 192.168.3.184\n')
    monkeypatch.setattr(g, '_MAX_LIFECYCLE_SCAN_BYTES', 256)
    assert g.contains_gateway_lifecycle_command_or_referenced_script("ssh peer 'hermes gateway restart'")


def test_config_include_directory_is_bounded(config, monkeypatch):
    directory = config / 'fragments'
    directory.mkdir()
    for i in range(50):
        (directory / str(i)).touch()
    (config / '.ssh/config').write_text(f'Include {directory}/*\nHost peer\n HostName 192.168.3.184\n')
    monkeypatch.setattr(g, '_MAX_LIFECYCLE_SCAN_PATHS', 16)
    assert g.contains_gateway_lifecycle_command_or_referenced_script("ssh peer 'hermes gateway restart'")


def test_safe_include_preserves_remote_alias(config):
    (config / 'fragment').write_text('Host other\n User nobody\n')
    (config / '.ssh/config').write_text(f'Host peer\n HostName 192.168.3.184\n Include {config}/fragment\n User peer\n')
    assert not g.contains_gateway_lifecycle_command_or_referenced_script("ssh peer 'hermes gateway restart'")
