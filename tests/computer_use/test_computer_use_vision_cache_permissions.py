"""At-rest permissions for the vision scratch cache created by ``computer_use``.

Scope: this file guards **``tools/computer_use/tool.py``** — its
``_vision_cache_dir`` / ``_write_private_bytes`` helpers, which materialise
desktop captures. The sibling suite
``tests/tools/test_vision_tools_media_cache_permissions.py`` guards the *other*
creator of the same ``cache/vision`` directory, ``tools/vision_tools.py``, and
owns the cross-module agreement test. Both suites are kept deliberately:
either module can win the race to create the shared directory, so each
creation site needs its own guard.

A desktop capture is as sensitive as whatever was on screen when it was
taken. These tests pin the *contract* — no group/other bits — rather than a
frozen octal, and they run the real creation path under a deliberately
permissive ``umask 022`` so a regression back to umask-derived modes fails
here instead of silently shipping.

POSIX-only: mode bits are advisory on Windows, where at-rest protection is
ACL-based and ``os.chmod`` only toggles the read-only flag.
"""

import os
import stat

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX mode bits; Windows at-rest protection is ACL-based",
)

GROUP_OTHER_BITS = (
    stat.S_IRGRP
    | stat.S_IWGRP
    | stat.S_IXGRP
    | stat.S_IROTH
    | stat.S_IWOTH
    | stat.S_IXOTH
)


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def permissive_umask():
    """Force a world-readable-by-default umask for the duration of a test."""
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Never let the ambient environment silently disable the hardening.
    # tests/conftest.py blanks HERMES_CONTAINER / HERMES_HOME_MODE /
    # HERMES_MANAGED, but NOT HERMES_SKIP_CHMOD. HERMES_MANAGED is repeated
    # here anyway so the fixture is self-contained: without it, a developer
    # shell on a managed install would make _secure_dir a no-op and the
    # healing test below would fail for the wrong reason.
    monkeypatch.delenv("HERMES_CONTAINER", raising=False)
    monkeypatch.delenv("HERMES_SKIP_CHMOD", raising=False)
    monkeypatch.delenv("HERMES_HOME_MODE", raising=False)
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    return home


@pytest.fixture
def neutralized_secure_dir(monkeypatch):
    """Make the post-creation reconciliation step a no-op.

    ``_vision_cache_dir`` hardens twice over: ``mode=`` on the ``mkdir`` call,
    then ``hermes_cli.config._secure_dir`` to reconcile policy. Either
    mechanism alone satisfies a plain "is it 0700?" assertion, so the two mask
    each other — with the reconciler in play every mode assertion in this file
    stays green even if ``mode=`` is deleted.

    Stubbing the reconciler out isolates the creation mode as the *only* thing
    that can produce the observed bits. Patched on ``hermes_cli.config``
    rather than on the caller because the production code imports it lazily
    inside the function, so the attribute is resolved at call time.
    """
    import hermes_cli.config as hermes_config

    calls = []

    def _recording_noop(path):
        calls.append(str(path))

    monkeypatch.setattr(hermes_config, "_secure_dir", _recording_noop)
    return calls


@pytest.fixture
def managed_nixos_home(hermes_home, monkeypatch):
    """Simulate a managed NixOS install that shares state via the hermes group.

    Reproduces the two conditions ``nix/nixosModules.nix`` actually creates:

    * ``systemd.tmpfiles`` pins ``stateDir/.hermes`` to ``2770`` — setgid,
      group-rwx — so the gateway and interactive ``hostUsers`` share it;
    * the service runs with ``UMask = "0007"``, commented "files created by
      the gateway should be group-writable so interactive users in the hermes
      group can read/write them".

    ``cache/vision`` is *not* in those tmpfiles rules, so it is created lazily
    at runtime under exactly this umask — which is why the creation mode, not
    just the reconciliation, has to honour the carve-out.
    """
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    os.chmod(hermes_home, 0o2770)
    previous = os.umask(0o007)
    try:
        yield hermes_home
    finally:
        os.umask(previous)


def test_fresh_vision_cache_dir_has_no_group_or_other_access(
    hermes_home, permissive_umask
):
    """A freshly created cache dir must not be readable by group/other."""
    from tools.computer_use.tool import _vision_cache_dir

    cache_dir = _vision_cache_dir()

    assert cache_dir.is_dir()
    mode = _mode(cache_dir)
    assert not (
        mode & GROUP_OTHER_BITS
    ), f"vision cache dir {cache_dir} is group/other-accessible: {oct(mode)}"


def test_vision_cache_dir_creation_is_not_umask_derived(hermes_home, permissive_umask):
    """The mode must come from the code, not from the ambient umask.

    Under ``umask 022`` a bare ``mkdir`` yields 0755. Asserting the owner
    keeps full access while group/other get nothing proves the mode was set
    deliberately at creation.
    """
    from tools.computer_use.tool import _vision_cache_dir

    mode = _mode(_vision_cache_dir())

    assert mode & stat.S_IRWXU == stat.S_IRWXU, "owner must retain full access"
    assert mode & GROUP_OTHER_BITS == 0, f"umask-derived mode leaked: {oct(mode)}"


def test_preexisting_world_readable_cache_dir_is_healed(hermes_home, permissive_umask):
    """An older Hermes left 0755 behind; the next call must tighten it."""
    from tools.computer_use.tool import _vision_cache_dir

    legacy = hermes_home / "cache" / "vision"
    legacy.mkdir(parents=True)
    os.chmod(legacy, 0o755)
    assert _mode(legacy) & stat.S_IROTH, "fixture precondition: starts world-readable"

    cache_dir = _vision_cache_dir()

    assert cache_dir == legacy
    assert not (
        _mode(cache_dir) & GROUP_OTHER_BITS
    ), f"pre-existing dir left exposed: {oct(_mode(cache_dir))}"


def test_captured_frame_is_written_owner_only(hermes_home, permissive_umask):
    """The screenshot bytes themselves must not land 0644."""
    from tools.computer_use.tool import _vision_cache_dir, _write_private_bytes

    target = _vision_cache_dir() / "capture.png"
    _write_private_bytes(target, b"\x89PNG\r\n\x1a\n fake frame")

    assert target.read_bytes().endswith(b"fake frame")
    mode = _mode(target)
    assert not (
        mode & GROUP_OTHER_BITS
    ), f"captured frame {target} is group/other-readable: {oct(mode)}"


def test_pre_fix_call_shapes_would_violate_the_contract(hermes_home, permissive_umask):
    """Guard against a vacuous suite.

    These are the exact pre-fix call shapes for the dir and the frame. If
    neither leaks group/other bits under ``umask 022`` any more, then the
    fixture — not the fix — is what the tests above are measuring, and they
    prove nothing.
    """
    from hermes_constants import get_hermes_dir

    pre_fix_dir = get_hermes_dir("cache/pre-fix-shape", "temp_pre_fix_shape")
    pre_fix_dir.mkdir(parents=True, exist_ok=True)
    dir_mode = _mode(pre_fix_dir)
    assert dir_mode & GROUP_OTHER_BITS, (
        "pre-fix mkdir shape no longer leaks group/other bits under umask 022; "
        f"got {oct(dir_mode)} — the dir-mode tests here would be vacuous"
    )

    pre_fix_frame = pre_fix_dir / "capture.png"
    pre_fix_frame.write_bytes(b"\x89PNG\r\n\x1a\n fake frame")
    frame_mode = _mode(pre_fix_frame)
    assert frame_mode & GROUP_OTHER_BITS, (
        "pre-fix write_bytes shape no longer leaks group/other bits under "
        f"umask 022; got {oct(frame_mode)} — the frame-mode test would be vacuous"
    )


def test_private_write_overwrites_and_stays_readable(hermes_home, permissive_umask):
    """A permission fix that breaks the read path is the wrong fix.

    The capture is written and then read straight back by
    ``_route_capture_through_aux_vision`` (it hands the path to
    ``vision_analyze_tool``), so truncating rewrite plus owner read must both
    keep working.
    """
    from tools.computer_use.tool import _vision_cache_dir, _write_private_bytes

    target = _vision_cache_dir() / "reused.png"
    _write_private_bytes(target, b"first-and-longer-frame")
    _write_private_bytes(target, b"second")

    assert target.read_bytes() == b"second", "O_TRUNC rewrite must not leave residue"
    assert _mode(target) & stat.S_IRUSR, "owner must still be able to read it back"


def test_managed_mode_leaves_existing_permissions_alone(
    hermes_home, permissive_umask, monkeypatch
):
    """Managed/NixOS installs own their own modes; we must not fight them.

    The NixOS activation script deliberately sets group-readable modes so
    interactive users in the hermes group can share state with the gateway
    service. The house helper skips tightening there, and so must we.
    """
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    legacy = hermes_home / "cache" / "vision"
    legacy.mkdir(parents=True)
    os.chmod(legacy, 0o750)

    from tools.computer_use.tool import _vision_cache_dir

    assert _mode(_vision_cache_dir()) == 0o750, "managed-mode permissions overridden"


def test_creation_mode_alone_yields_owner_only(hermes_home, neutralized_secure_dir):
    """``mode=`` on the ``mkdir`` call must be doing real work by itself.

    Two mechanisms harden this directory and either one satisfies a bare
    "is it 0700?" check, so they mask each other: with ``_secure_dir`` in play
    every mode assertion in this file stays green even if ``mode=`` is deleted.
    Here the reconciler is stubbed to a no-op, so the only thing that can
    produce owner-only bits is the mode passed at creation.

    This is the TOCTOU guarantee made observable. The window ``mode=`` closes —
    the instant between ``mkdir`` and a follow-up ``chmod`` where the captures
    directory sits world-readable — cannot be asserted by racing creation, but
    "correct without any chmod at all" is exactly equivalent and is
    deterministic.

    Deliberately not stacked with ``permissive_umask`` so a 0700 result cannot
    be an accident of a restrictive ambient umask on the machine running the
    suite; the sibling test below forces the umask wide open instead.
    """
    from tools.computer_use.tool import _vision_cache_dir

    cache_dir = _vision_cache_dir()

    assert cache_dir.is_dir()
    assert neutralized_secure_dir == [
        str(cache_dir)
    ], "reconciler stub was not exercised; this test is not isolating creation"
    mode = _mode(cache_dir)
    assert mode & GROUP_OTHER_BITS == 0, (
        f"with the reconciler neutralized {cache_dir} is group/other-accessible "
        f"({oct(mode)}) — the mode= on mkdir is not being applied, so the "
        "mkdir->chmod TOCTOU window is open"
    )
    assert mode & stat.S_IRWXU == stat.S_IRWXU, "owner must retain full access"


def test_creation_mode_is_not_inherited_from_a_permissive_umask(
    hermes_home, permissive_umask, neutralized_secure_dir
):
    """Same isolation, with the umask forced wide open.

    Together with the test above this pins the claim precisely: 0700 comes from
    the ``mode=`` argument, not from the ambient umask and not from the
    reconciler. ``Path.mkdir`` masks its ``mode`` with the umask, so ``mode=``
    of 0700 under ``umask 022`` still yields 0700 — but a *missing* ``mode=``
    yields 0755 and fails here.
    """
    from tools.computer_use.tool import _vision_cache_dir

    mode = _mode(_vision_cache_dir())

    assert neutralized_secure_dir, "reconciler stub was not exercised"
    assert (
        mode & GROUP_OTHER_BITS == 0
    ), f"umask-derived mode leaked with the reconciler neutralized: {oct(mode)}"


def test_managed_mode_fresh_cache_keeps_group_sharing(managed_nixos_home):
    """A *newly created* cache on NixOS must stay group-shareable.

    The sibling test above covers a pre-existing directory, where
    ``_secure_dir``'s ``is_managed()`` early return is enough. This is the
    other managed path: the directory does not exist yet, so the mode passed at
    *creation* is the only thing that decides it — and ``cache/vision`` is not
    among the directories ``nix/nixosModules.nix`` pre-creates via
    ``systemd.tmpfiles``, so on a managed host it is always this path.

    Forcing 0700 here would not merely be cosmetic. The module shares
    ``$HERMES_HOME`` between the gateway service and interactive ``hostUsers``
    through the hermes group (``2770`` + ``UMask = "0007"`` + a deliberate
    refusal to ``chown -R``, which strips setgid). A 0700 cache created by
    whichever side runs first makes the capture path fail with EACCES for the
    other.
    """
    from tools.computer_use.tool import _vision_cache_dir

    expected = managed_nixos_home / "cache" / "vision"
    assert not expected.exists(), "fixture precondition: dir must be absent"

    cache_dir = _vision_cache_dir()

    assert cache_dir.is_dir(), "managed mode must still create the cache dir"
    mode = _mode(cache_dir)
    assert mode & stat.S_IRWXG, (
        f"fresh managed-mode {cache_dir} dropped group access ({oct(mode)}); the "
        "NixOS module's hermes-group sharing (2770 + UMask=0007) is broken, so "
        "an interactive hostUsers CLI and the gateway can no longer share it"
    )


def test_home_mode_override_is_honored(hermes_home, permissive_umask, monkeypatch):
    """The documented web-server traversal hatch still applies.

    Deployments that need nginx/caddy to traverse HERMES_HOME set
    HERMES_HOME_MODE. Delegating to the house helper keeps one source of
    truth for that knob instead of hardcoding a mode this module invented.
    """
    monkeypatch.setenv("HERMES_HOME_MODE", "0701")
    from tools.computer_use.tool import _vision_cache_dir

    mode = _mode(_vision_cache_dir())

    assert mode == 0o701
    # Execute-only: traversal without directory listing. Frames stay 0600.
    assert not (mode & (stat.S_IRGRP | stat.S_IROTH)), "listing must stay closed"


def test_container_deployments_still_get_a_usable_directory(
    hermes_home, permissive_umask, monkeypatch
):
    """A container/volume-mount deployment must not crash or lose the dir.

    ``_secure_dir`` checks ``is_managed()`` only — unlike ``_secure_file`` it
    does **not** consult ``_is_container()``, so the cache is still tightened
    to 0700 here. That adds no new restriction: ``ensure_hermes_home`` already
    puts HERMES_HOME and every standard subdir at 0700 in a container, and
    multi-UID deployments are served by ``HERMES_UID``/``HERMES_GID`` or
    ``HERMES_HOME_MODE``, both of which ``_secure_dir`` honours. So this
    asserts the weaker, true contract: the directory exists and nothing raised.
    """
    monkeypatch.setenv("HERMES_CONTAINER", "1")
    monkeypatch.setenv("HERMES_SKIP_CHMOD", "1")
    from tools.computer_use.tool import _vision_cache_dir

    assert _vision_cache_dir().is_dir()
