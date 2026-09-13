"""Explicit pinned driver install; never changes the user's global cua-driver."""

import argparse
import hashlib
import json
from pathlib import Path
import os
import platform
import stat
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = "0.23.2"
ARCHIVE_SHA256 = "478e010d2b0426de9d8a07eb839802daa92137b33cb8affb93b991e91d76ce7e"
BINARY_SHA256 = "2aaad67b996d41cd909e4b7ac2ddd705653e8c7f9958010ac051838b02f64eeb"
URL = f"https://github.com/trycua/cua/releases/download/cua-driver-rs-v{VERSION}/cua-driver-rs-{VERSION}-linux-x86_64.tar.gz"
MEMBER = f"cua-driver-rs-{VERSION}-linux-x86_64/cua-driver"


def installed(target):
    target = Path(target)
    try:
        info = target.lstat()
        return (
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()  # windows-footgun: ok — runtime package rejects non-Linux hosts
            and os.access(target, os.X_OK)
            and hashlib.sha256(target.read_bytes()).hexdigest() == BINARY_SHA256
        )
    except OSError:
        return False


def _execution_identity(target):
    info = Path(target).stat()
    return [BINARY_SHA256, info.st_dev, info.st_ino, info.st_size, info.st_mode,
            info.st_uid, info.st_mtime_ns, info.st_ctime_ns]


def write_receipt(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".verified-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def execution_verified(target):
    try:
        target = Path(target).absolute()
        if target.parent.resolve() != target.parent:
            return False
        receipt = Path(target).with_name(".cua-driver-verified.json")
        return installed(target) and json.loads(receipt.read_text(encoding="utf-8")) == _execution_identity(target)
    except (OSError, ValueError):
        return False


def verify_execution(target):
    target = Path(target).absolute()
    receipt = target.with_name(".cua-driver-verified.json")
    receipt.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            [str(target), "--version"], capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Pinned Cua driver cannot run; check runtime libraries and executable filesystem permissions") from exc
    if result.returncode or VERSION not in result.stdout:
        raise ValueError("Pinned Cua driver cannot run; check runtime libraries and executable filesystem permissions")
    write_receipt(receipt, _execution_identity(target))


def profile_target(home):
    from .config import driver_path, effective_home

    target = driver_path(home)
    for parent in (target.parent, *target.parent.parents):
        if parent.is_symlink():
            raise ValueError("Realms driver directory must stay within this profile, without symlink redirects")
        if parent.exists():
            info = parent.stat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():  # windows-footgun: ok — runtime package rejects non-Linux hosts
                raise ValueError("Realms driver directory has unsafe profile ownership")
        if parent == effective_home(home):
            break
    return target


def install_archive(archive, target):
    archive, target = Path(archive), Path(target).absolute()
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Downloaded cua-driver release checksum mismatch")
    if installed(target):
        verify_execution(target)
        return str(target)
    with tarfile.open(archive, "r:gz") as package:
        member = package.getmember(MEMBER)
        if not member.isfile():
            raise ValueError("Driver archive member is not a regular file")
        stream = package.extractfile(member)
        if stream is None:
            raise ValueError("Missing driver archive payload")
        with stream:
            data = stream.read()
    if hashlib.sha256(data).hexdigest() != BINARY_SHA256:
        raise ValueError("Driver binary checksum mismatch")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".driver-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o700)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if not os.access(target, os.X_OK):
        raise PermissionError(
            "Installed driver is not executable on the target filesystem"
        )
    verify_execution(target)
    return str(target)


def configure_parser(parser):
    parser.add_argument(
        "--archive", type=Path, help="Use an already downloaded pinned archive"
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Override the default profile-local plugin-data/hermes-realms/bin/cua-driver path",
    )


def install(*, home=None, archive=None, target=None):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("This verified binary release is Linux x86_64 only")
    if target is None:
        target = profile_target(home)
    target = Path(target).absolute()
    if archive is not None:
        return install_archive(archive, target)
    if installed(target):
        verify_execution(target)
        return str(target)
    with tempfile.TemporaryDirectory(prefix="realms-driver-download-") as directory:
        archive = Path(directory) / "release.tar.gz"
        with (
            urllib.request.urlopen(URL, timeout=60) as response,
            archive.open("wb") as stream,
        ):
            import shutil

            shutil.copyfileobj(response, stream)
        return install_archive(archive, target)


def run(args):
    print(install(home=getattr(args, "home", None), archive=args.archive, target=args.target))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    try:
        run(parser.parse_args(argv))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
