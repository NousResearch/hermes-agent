"""Only the default profile runs the multiplexer; a named launch would split default's sessions."""
import pytest

from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE


def _root(tmp_path):
    root = tmp_path / '.hermes'
    root.mkdir()
    (root / 'config.yaml').write_text('model: {default: m}\n')
    (root / 'profiles' / 'foo').mkdir(parents=True)
    (root / 'profiles' / 'foo' / 'config.yaml').write_text('gateway:\n  multiplex_profiles: true\n')
    return root


def test_named_profile_launch_with_multiplex_is_a_config_verdict(tmp_path, monkeypatch):
    from gateway import run_bootstrap

    root = _root(tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root / 'profiles' / 'foo'))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    written = []
    monkeypatch.setattr('gateway.run._write_runtime_status_quiet', lambda **kw: written.append(kw))
    with pytest.raises(SystemExit) as exc:
        run_bootstrap._launch_home_may_multiplex()
    assert exc.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert written == [{'gateway_state': 'startup_failed', 'exit_reason': 'multiplex_requires_default_profile'}]


def test_default_profile_launch_may_multiplex(tmp_path, monkeypatch):
    from gateway import run_bootstrap

    root = _root(tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    assert run_bootstrap._launch_home_may_multiplex() is True


@pytest.mark.asyncio
async def test_verdict_precedes_replacing_a_running_gateway(tmp_path, monkeypatch):
    """``hermes gateway run --replace`` on a named profile with the flag must exit EX_CONFIG while the
    healthy gateway it would have replaced keeps running."""
    from gateway import run_bootstrap
    from gateway.config import GatewayConfig

    root = _root(tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root / 'profiles' / 'foo'))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr('hermes_cli.resource_limits.apply_nofile_soft_limit', lambda: None)
    monkeypatch.setattr('gateway.code_skew.record_boot_fingerprint', lambda: None)
    monkeypatch.setattr('gateway.status.get_running_pid', lambda: 4242)
    monkeypatch.setattr('gateway.run._write_runtime_status_quiet', lambda **kw: None)
    replaced = []

    async def fake_replace(existing_pid, replace):
        replaced.append(existing_pid)
        return False

    monkeypatch.setattr(run_bootstrap, '_start_gateway_replace_existing_instance', fake_replace)
    config = GatewayConfig()
    config.multiplex_profiles = True
    with pytest.raises(SystemExit) as exc:
        await run_bootstrap.start_gateway(config=config, replace=True)
    assert exc.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert replaced == []


def test_cloned_profile_does_not_inherit_the_multiplex_flag(tmp_path, monkeypatch):
    import yaml
    from hermes_cli.profiles import _strip_multiplex_flag

    nested = tmp_path / 'nested.yaml'
    nested.write_text('model: {default: m}\ngateway:\n  multiplex_profiles: true\n  service_install_choice: skip\n')
    _strip_multiplex_flag(nested)
    assert yaml.safe_load(nested.read_text()) == {'model': {'default': 'm'}, 'gateway': {'service_install_choice': 'skip'}}
    top = tmp_path / 'top.yaml'
    top.write_text('multiplex_profiles: true\nmodel: {default: m}\n')
    _strip_multiplex_flag(top)
    assert yaml.safe_load(top.read_text()) == {'model': {'default': 'm'}}
    untouched = tmp_path / 'plain.yaml'
    untouched.write_text('# keep my comment\nmodel: {default: m}\n')
    _strip_multiplex_flag(untouched)
    assert untouched.read_text() == '# keep my comment\nmodel: {default: m}\n'
