"""A linked named profile (``profiles/<name>`` a symlink or Windows junction to a folder elsewhere) keeps the name
and root of its entry. Regression for #142944.

Clients resolve the home before discovery and ensure, which leaves ``profiles/``: the profile then had no root, so
its clients never asked the default gateway's control socket (``starting`` until the deadline), and a cold start ran
the link's target as its own root (``--profile default``). That gateway held the profile's lock and blocked the
default multiplexer's all-or-nothing reserve.
"""
import os
from pathlib import Path

from tests.gateway.test_hosted_group_members import _gateway, _link


def _layout(tmp_path, monkeypatch, *, standalone=False, folder='linked'):
    """``<root>/profiles/linked`` -> ``<tmp>/elsewhere/<folder>``; the root is this process's root."""
    root = tmp_path / '.hermes'
    (root / 'profiles').mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    target = tmp_path / 'elsewhere' / folder
    target.mkdir(parents=True, mode=0o700)
    (target / 'config.yaml').write_text('gateway:\n  standalone: true\n' if standalone else '{}\n', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(root))
    return root, target, _link(target, root / 'profiles' / 'linked')


def test_a_linked_home_keeps_its_entry_name_and_root(tmp_path, monkeypatch):
    from gateway.session_authorities import served_profile_name
    from hermes_cli.gateway_runtime import _canonical_home
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for

    root, target, link = _layout(tmp_path, monkeypatch)
    assert multiplexer_root_for(Path(_canonical_home(link))) == root.resolve()
    assert served_profile_name(link.resolve()) == 'linked'
    assert served_profile_name(target / 'logs') == 'linked'
    # A rename or delete of the link is seen at once.
    renamed = root / 'profiles' / 'renamed'
    os.rename(link, renamed)
    assert served_profile_name(target) == 'renamed'
    os.rmdir(renamed) if os.name == 'nt' else renamed.unlink()  # a directory link on Windows
    assert served_profile_name(target) == 'default'


def test_a_linked_profiles_dir_keeps_the_real_root(tmp_path, monkeypatch):
    """``profiles/`` itself linked into another tree: the target reads as another ``.hermes`` root."""
    from gateway.session_authorities import served_profile_name
    from hermes_cli.gateway_runtime import _canonical_home
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for

    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    vault = tmp_path / 'repo' / '.hermes' / 'profiles'
    (vault / 'work').mkdir(parents=True)
    _link(vault, root / 'profiles')
    monkeypatch.setenv('HERMES_HOME', str(root))
    canonical = Path(_canonical_home(root / 'profiles' / 'work'))
    assert multiplexer_root_for(canonical) == root.resolve()
    assert served_profile_name(canonical) == 'work'


def test_an_alias_never_gives_a_folder_a_second_name(tmp_path, monkeypatch):
    from gateway.session_authorities import served_profile_name

    root, target, _linked = _layout(tmp_path, monkeypatch)
    _link(target, root / 'profiles' / 'twin')  # two entries for one folder: neither name wins
    assert served_profile_name(target) == 'default'
    _link(target.parent, root / 'profiles' / 'outer')  # nor does a link to a folder enclosing it
    assert served_profile_name(target) == 'default'
    assert served_profile_name(target.parent / 'notes') == 'outer'
    plain = root / 'profiles' / 'alpha'
    plain.mkdir()
    _link(plain, root / 'profiles' / 'alias')  # a link to another profile's folder: still its owner's
    assert served_profile_name((root / 'profiles' / 'alias').resolve()) == 'alpha'


def test_a_profile_linked_into_another_hermes_tree_keeps_its_name_and_root(tmp_path, monkeypatch):
    """A moved or reused Hermes home (a folder named ``.hermes``, or one inside it) linked in as a profile."""
    from gateway.session_authorities import served_profile_name
    from hermes_cli.gateway_runtime import _canonical_home
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for

    root, _target, _linked = _layout(tmp_path, monkeypatch)
    for name, target in (('vault', tmp_path / 'vault' / '.hermes'), ('moved', tmp_path / 'backup' / '.hermes' / 'work')):
        target.mkdir(parents=True)
        _link(target, root / 'profiles' / name)
        assert served_profile_name(target / 'logs') == name
        assert multiplexer_root_for(Path(_canonical_home(root / 'profiles' / name))) == root.resolve()


def test_a_linked_profile_maps_again_once_its_offline_target_returns(tmp_path, monkeypatch):
    """A link to a removable or network volume that was offline when the map was built. Mounted, the
    volume may resolve the profile further, through a link on the volume itself."""
    from gateway.session_authorities import served_profile_name

    root, target, _linked = _layout(tmp_path, monkeypatch)
    drive = tmp_path / 'drive'
    _link(drive / 'work', root / 'profiles' / 'work')
    assert served_profile_name(target) == 'linked'  # builds the map while the drive is offline
    (drive / 'data' / 'work').mkdir(parents=True)  # mounted; profiles/ itself does not change
    _link(drive / 'data' / 'work', drive / 'work')
    assert served_profile_name(drive / 'data' / 'work') == 'work'


def test_a_home_nested_in_a_profile_is_not_that_profile(tmp_path, monkeypatch):
    """A custom home inside a profile's folder belongs to that profile as a path, never as its home."""
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    from hermes_constants import named_profile_home

    root, _target, _linked = _layout(tmp_path, monkeypatch)
    nested = root / 'profiles' / 'alpha' / 'task-home'
    nested.mkdir(parents=True)
    assert named_profile_home(nested) == root / 'profiles' / 'alpha'
    assert multiplexer_root_for(nested) is None


def test_clients_attach_to_the_multiplexer_serving_a_linked_profile(tmp_path, monkeypatch):
    from hermes_cli.gateway_runtime import discover_gateway_endpoint, ensure_gateway_runtime

    with _gateway(tmp_path, monkeypatch, served=('alpha',), linked=('linked',)) as (_runner, homes, _call):
        root, link = homes['default'], homes['linked']
        found = discover_gateway_endpoint(link, timeout=5)
        assert found.state == 'ready', found
        assert Path(found.endpoint.control_home).resolve() == root.resolve()
        assert found.endpoint.profile_id == str(link.resolve())
        assert ensure_gateway_runtime(link, timeout=5).state == 'ready'


def test_a_cold_linked_profile_starts_the_multiplexer_never_its_target(tmp_path, monkeypatch):
    from hermes_cli import gateway_runtime_service as service, gateway_runtime_start as start
    from hermes_cli.gateway_runtime import ensure_gateway_runtime

    root, _target, link = _layout(tmp_path, monkeypatch)
    spawned = []
    monkeypatch.setattr(service, 'discover_existing_gateway_service', lambda *a, **k: None)
    monkeypatch.setattr(start, 'spawn_unmanaged_gateway', lambda target, **k: spawned.append(Path(target)))
    ensure_gateway_runtime(link, timeout=0.5)
    assert spawned == [root.resolve()]
