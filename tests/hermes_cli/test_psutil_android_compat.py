import io
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import pytest

from hermes_cli import main as cli_main


PSUTIL_MARKER = 'LINUX = sys.platform.startswith("linux")'


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _fake_urlretrieve(source_archive: Path):
    def fake_urlretrieve(_url: str, filename: str | Path):
        Path(filename).write_bytes(source_archive.read_bytes())
        return str(filename), None

    return fake_urlretrieve


def test_install_psutil_android_compat_rejects_traversal_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Set up a controlled inner extraction directory under tmp_path so that a
    # "../" traversal member would escape to tmp_path (a path we can inspect).
    inner_dir = tmp_path / "extract_root"
    inner_dir.mkdir()
    escaped_target = tmp_path / "escaped.txt"

    source_archive = tmp_path / "evil.tar.gz"
    _write_tar(
        source_archive,
        {
            "psutil-7.2.2/psutil/_common.py": PSUTIL_MARKER.encode(),
            f"../{escaped_target.name}": b"pwned",
        },
    )

    # Monkeypatch tempfile.TemporaryDirectory so the function extracts into
    # inner_dir (a subdirectory of tmp_path).  A "../" traversal from inner_dir
    # would land in tmp_path, which we assert remains clean after the call.
    class _FakeTempDir:
        def __enter__(self):
            return str(inner_dir)

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _FakeTempDir)
    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve(source_archive))
    monkeypatch.setattr(cli_main, "_verify_file_sha256", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="Unsafe archive member path"):
        cli_main._install_psutil_android_compat(["python", "-m", "pip"])

    # escaped_target lives in tmp_path (the parent of inner_dir).  If the safe
    # extractor had failed to block the traversal, the file would exist here.
    assert not escaped_target.exists()


def test_install_psutil_android_compat_patches_and_installs_safe_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    source_archive = tmp_path / "psutil.tar.gz"
    _write_tar(
        source_archive,
        {"psutil-7.2.2/psutil/_common.py": PSUTIL_MARKER.encode()},
    )
    install_calls = []

    def fake_install(cmd, *, env=None):
        install_calls.append((cmd, env))
        common_py = Path(cmd[-1]) / "psutil" / "_common.py"
        assert 'sys.platform.startswith(("linux", "android"))' in common_py.read_text()

    monkeypatch.setattr(urllib.request, "urlretrieve", _fake_urlretrieve(source_archive))
    monkeypatch.setattr(cli_main, "_verify_file_sha256", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", fake_install)

    cli_main._install_psutil_android_compat(["python", "-m", "pip"], env={"X": "1"})

    assert len(install_calls) == 1
    cmd, env = install_calls[0]
    assert cmd[:5] == ["python", "-m", "pip", "install", "--no-build-isolation"]
    assert Path(cmd[-1]).name == "psutil-7.2.2"
    assert env == {"X": "1"}


def test_verify_file_sha256_rejects_mismatch(tmp_path: Path):
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="Downloaded archive hash mismatch"):
        cli_main._verify_file_sha256(archive, "0" * 64)
