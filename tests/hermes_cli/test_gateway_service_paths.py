from pathlib import Path
from unittest.mock import patch


def test_service_path_skips_nonexistent_node_modules(tmp_path):
    """Service PATH should not include node_modules/.bin if it doesn't exist."""
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    node_modules_bin = str(tmp_path / "node_modules" / ".bin")
    assert node_modules_bin not in dirs


def test_service_path_includes_node_modules_when_present(tmp_path):
    """Service PATH should include node_modules/.bin when it exists."""
    nm_bin = tmp_path / "node_modules" / ".bin"
    nm_bin.mkdir(parents=True)
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    assert str(nm_bin) in dirs


def test_service_path_includes_hermes_home_node_modules(tmp_path):
    """Service PATH should include ~/.hermes/node_modules/.bin when it exists."""
    hermes_nm = tmp_path / ".hermes" / "node_modules" / ".bin"
    hermes_nm.mkdir(parents=True)
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    assert str(hermes_nm) in dirs


def test_service_path_treats_permission_error_as_missing(tmp_path):
    """A ``PermissionError`` from ``is_dir()`` must not crash unit
    generation — treat the path as missing and continue.

    Reproduces the CI baseline (#26622 audit): on Ubuntu runners executing
    as root, ``stat('/root/.hermes/node/bin')`` can raise ``EACCES`` even
    though the calling user owns ``/root``. Before the guard,
    ``generate_systemd_unit()`` and ``generate_launchd_plist()`` would
    propagate the ``OSError`` and refuse to produce any unit at all.
    """
    from hermes_cli.gateway import _build_service_path_dirs

    real_is_dir = Path.is_dir

    def fake_is_dir(self):
        # Only the hermes_home node/bin probe trips EACCES; other paths
        # must keep their real behavior so the rest of the function is
        # exercised normally.
        if str(self).endswith("/.hermes/node/bin"):
            raise PermissionError(13, "Permission denied", str(self))
        return real_is_dir(self)

    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"), \
         patch.object(Path, "is_dir", fake_is_dir):
        dirs = _build_service_path_dirs(project_root=tmp_path)

    assert str(tmp_path / ".hermes" / "node" / "bin") not in dirs


def test_service_path_treats_oserror_as_missing(tmp_path):
    """Broken-symlink / unreachable network mount probes (``OSError``)
    must also be treated as "directory not present" rather than crashing.
    """
    from hermes_cli.gateway import _build_service_path_dirs

    real_is_dir = Path.is_dir

    def fake_is_dir(self):
        if str(self).endswith("/.hermes/node_modules/.bin"):
            raise OSError(5, "Input/output error", str(self))
        return real_is_dir(self)

    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"), \
         patch.object(Path, "is_dir", fake_is_dir):
        dirs = _build_service_path_dirs(project_root=tmp_path)

    assert str(tmp_path / ".hermes" / "node_modules" / ".bin") not in dirs
