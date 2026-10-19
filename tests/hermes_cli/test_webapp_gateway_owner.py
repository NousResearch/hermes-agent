"""The Webapp finds or starts its profile's gateway the way local Desktop does (``webapp_gateway_owner``).

Fakes stand in only at the discovery/ensure boundary and the restart runner, so the ladder itself runs
for real: start an absent owner, refuse an owner too old for browser attach with its remedy, pass a
lasting verdict to the browser, and restart an owner running other code once, then attach as-is
(Electron's ``createStaleGatewayRestarter``).
"""
import asyncio
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import gateway_runtime, version_info, webapp_gateway_owner as owner
from hermes_cli.gateway_runtime import GatewayDiscovery, GatewayEndpoint

CAPABILITIES = frozenset({'session-authority-v1', 'durable-admission-v1', 'event-replay-v1', 'delegated-attach-v1'})


def _endpoint(home, **overrides):
    fields = dict(profile_id=str(home), instance_id='inst', authority_epoch=1, runtime_protocol=1,
                  api_origin='http://127.0.0.1:1', supervisor='systemd', capabilities=CAPABILITIES,
                  code_sha='same-code')
    return GatewayEndpoint(**{**fields, **overrides})


@pytest.fixture
def ladder(monkeypatch, tmp_path):
    """``(calls, set_owner)``: discovery answers with the owner set last, ensure starts one."""
    state = {'owner': None, 'started': None}
    calls = {'ensure': 0, 'restart': []}

    def discover(home, timeout=2.0):
        return GatewayDiscovery('ready', state['owner']) if state['owner'] else GatewayDiscovery('absent')

    def ensure(home, *, timeout=60.0, idle_exit=False):
        calls['ensure'] += 1
        assert idle_exit is False  # a Webapp-started gateway serves messaging too: never an idle exiter
        state['owner'] = state['started'].endpoint
        return state['started']

    def restart(profile, done):
        calls['restart'].append(profile)
        done.set()

    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint', discover)
    monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime', ensure)
    monkeypatch.setattr(version_info, 'get_code_identity', lambda refresh=False: {'sha': 'same-code'})
    monkeypatch.setattr(owner, '_run_restart', restart)
    monkeypatch.setattr(owner, '_restarts', {})
    monkeypatch.setattr(owner, '_warned_unsupervised', set())
    monkeypatch.setattr('gateway.status._profile_label_for_home', lambda home: 'default')
    return state, calls, tmp_path


def test_an_absent_owner_is_started_and_a_service_webapp_says_how_to_keep_it(ladder, monkeypatch, caplog):
    state, calls, home = ladder
    state['started'] = GatewayDiscovery('ready', _endpoint(home, supervisor='none'))
    monkeypatch.setattr('gateway.control_socket._detect_supervisor', lambda: 'systemd')
    with caplog.at_level(logging.WARNING, logger='hermes_cli.web_server'):
        assert asyncio.run(owner.owner_endpoint(home)).instance_id == 'inst'
        assert asyncio.run(owner.owner_endpoint(home)).instance_id == 'inst'  # now discovered: no start
    assert calls['ensure'] == 1
    warnings = [r.getMessage() for r in caplog.records if 'hermes gateway install' in r.getMessage()]
    assert len(warnings) == 1, warnings


def test_a_lasting_verdict_and_an_owner_too_old_to_attach_reach_the_browser(ladder):
    state, _calls, home = ladder
    state['started'] = GatewayDiscovery('incompatible', None, 'runtime_protocol', 'another install owns it')
    with pytest.raises(owner.GatewayAttachError) as refused:
        asyncio.run(owner.owner_endpoint(home))
    assert refused.value.verdict() == {'state': 'incompatible', 'reason_code': 'runtime_protocol',
                                       'detail': 'another install owns it'}
    state['owner'] = _endpoint(home, capabilities=CAPABILITIES - {'delegated-attach-v1'})
    with pytest.raises(owner.GatewayAttachError) as old:
        asyncio.run(owner.owner_endpoint(home))
    assert old.value.reason_code == 'delegated_attach_unavailable'
    assert 'hermes gateway restart' in old.value.detail


def test_an_owner_running_other_code_is_restarted_once_then_attached_as_is(ladder):
    state, calls, home = ladder
    state['owner'] = _endpoint(home, code_sha='old-code')
    for _ in range(3):
        assert asyncio.run(owner.owner_endpoint(home)).code_sha == 'old-code'
    assert calls['restart'] == ['default']
    # An owner that cannot name its code is never restarted on a guess.
    state['owner'] = _endpoint(home, code_sha=None)
    asyncio.run(owner.owner_endpoint(home))
    assert calls['restart'] == ['default']


def _canon(path):
    return os.path.normcase(str(Path(path).resolve()))


@pytest.fixture
def host(monkeypatch, tmp_path):
    """A Hermes root with a plain named profile, one whose folder is a link to storage outside the
    root, and one that opts out of the multiplexer. Discovery answers as main's does: the default
    gateway serves ``<root>/profiles/<name>`` judged by the folder's resolved path. Ensure records
    every home it is asked to start and blocks until ``release`` is set."""
    root = tmp_path / 'hermes'
    (root / 'profiles').mkdir(parents=True)
    (root / 'config.yaml').write_text('model:\n  default: x\n', encoding='utf-8')
    plain = root / 'profiles' / 'beta'
    plain.mkdir()
    outside = tmp_path / 'other-disk' / 'linked'
    outside.mkdir(parents=True)
    linked = root / 'profiles' / 'linked'
    if os.name == 'nt':
        import _winapi
        _winapi.CreateJunction(str(outside), str(linked))
    else:
        linked.symlink_to(outside, target_is_directory=True)
    standalone = root / 'profiles' / 'solo'
    standalone.mkdir()
    (standalone / 'config.yaml').write_text('gateway:\n  standalone: true\n', encoding='utf-8')

    running = set()
    ensured = []
    release = threading.Event()
    release.set()

    def served(home):
        canon = Path(_canon(home))
        return canon == Path(_canon(root)) or canon.parent == Path(_canon(root / 'profiles'))

    def discover(home, timeout=2.0):
        if _canon(home) in running or (_canon(root) in running and served(home) and _canon(home) != _canon(standalone)):
            return GatewayDiscovery('ready', _endpoint(home))
        return GatewayDiscovery('absent')

    def ensure(home, *, timeout=60.0, idle_exit=False):
        ensured.append(_canon(home))
        release.wait(5)
        running.add(_canon(home))
        return GatewayDiscovery('ready', _endpoint(home))

    monkeypatch.setattr(gateway_runtime, 'discover_gateway_endpoint', discover)
    monkeypatch.setattr(gateway_runtime, 'ensure_gateway_runtime', ensure)
    monkeypatch.setattr(version_info, 'get_code_identity', lambda refresh=False: {'sha': 'same-code'})
    monkeypatch.setattr(owner, '_restarts', {})
    monkeypatch.setattr(owner, '_warned_unsupervised', set())
    return SimpleNamespace(root=root, plain=plain, linked=linked, standalone=standalone,
                           ensured=ensured, release=release)


def test_a_profile_the_default_gateway_serves_never_gets_a_gateway_of_its_own(host):
    asyncio.run(owner.owner_endpoint(host.plain))
    # The linked folder resolves outside the root, so the default gateway cannot answer for it; that
    # is this profile's error, and starting a second owner for it would stop the default gateway.
    with pytest.raises(owner.GatewayAttachError):
        asyncio.run(owner.owner_endpoint(host.linked))
    assert set(host.ensured) == {_canon(host.root)}


def test_profiles_dialled_together_share_one_gateway_start(host):
    host.release.clear()

    async def boot():  # the browser's boot asks for the primary and every profile at once
        return await asyncio.gather(owner.owner_endpoint(host.root), owner.owner_endpoint(host.root),
                                    owner.owner_endpoint(host.plain))

    threading.Timer(0.5, host.release.set).start()
    assert len(asyncio.run(boot())) == 3
    assert host.ensured == [_canon(host.root)]


def test_a_standalone_profile_still_starts_its_own_gateway(host):
    asyncio.run(owner.owner_endpoint(host.standalone))
    assert host.ensured == [_canon(host.standalone)]
