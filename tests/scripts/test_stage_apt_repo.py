"""Tests for scripts/termux/stage_apt_repo.py — stdlib + pytest, no network.

GPG tests generate a throwaway key inside a temp GNUPGHOME and never touch
the invoking user's keyring; passphrase material never appears in test
output (no secret logging).
"""

import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from tests.termux_fixtures import build_deb

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "termux"
sys.path.insert(0, str(SCRIPTS))

import stage_apt_repo  # noqa: E402

GPG_PRESENT = shutil.which("gpg") is not None


def make_deb(path: Path, package: str, version: str, arch: str = "aarch64", compression: str = "gz") -> None:
    build_deb(path, {"Package": package, "Version": version, "Architecture": arch,
                     "Maintainer": "Test <test@example.com>", "Description": f"test package {package}"},
              compression=compression)


@pytest.fixture
def no_gpg(monkeypatch):
    """Make the script believe gpg is absent so signing is skipped (exit 3)."""
    monkeypatch.setattr(stage_apt_repo.shutil, "which", lambda _: None)


def test_canary_versions_below_stable():
    versions = ["1.2.3-1", "1.2.3~canary.20260831120000-1", "1.2.4~canary.1-1", "1.2.4-1"]
    ordered = sorted(versions, key=stage_apt_repo.deb_version_key)
    assert ordered == [
        "1.2.3~canary.20260831120000-1",
        "1.2.3-1",
        "1.2.4~canary.1-1",
        "1.2.4-1",
    ]


def test_unsigned_multiversion_publication_and_immutable_indexes(tmp_path, capsys):
    import hashlib

    pool = tmp_path / "pool"
    pool.mkdir()
    versions = ["1.2.3~canary.20260901000000-1", "1.2.3-1"]
    for filename, version, compression in zip(("a.deb", "b.deb"), versions, ("gz", "xz")):
        make_deb(pool / filename, "hermes-agent", version, compression=compression)
    out = tmp_path / "repo"
    args = ["--pool", str(pool), "--out", str(out), "--suite", "hermes-canary"]
    assert stage_apt_repo.main(args) == 3
    binary = out / "dists/hermes-canary/main/binary-aarch64"
    text = (binary / "Packages").read_text(encoding="utf-8")
    records = [dict(line.split(": ", 1) for line in stanza.splitlines()) for stanza in text.strip().split("\n\n")]
    assert [row["Version"] for row in records] == versions
    for row in records:
        assert row["Package"] == "hermes-agent" and row["Architecture"] == "aarch64"
        copied = out / row["Filename"]
        original = pool / copied.name
        assert copied.read_bytes() == original.read_bytes()
        assert row["SHA256"] == hashlib.sha256(copied.read_bytes()).hexdigest()
        assert row["Size"] == str(copied.stat().st_size)
    assert gzip.decompress((binary / "Packages.gz").read_bytes()).decode() == text
    release = (out / "dists/hermes-canary/Release").read_text(encoding="utf-8")
    assert "Suite: hermes-canary\n" in release and "Acquire-By-Hash: yes\n" in release
    assert "\n\n" not in release
    assert "Date: " in release.partition("SHA256:\n")[0]
    immutable = {}
    for name in ("Packages", "Packages.gz"):
        data = (binary / name).read_bytes()
        for algorithm in ("SHA256", "SHA512"):
            digest = hashlib.new(algorithm.lower(), data).hexdigest()
            assert [digest, str(len(data)), f"main/binary-aarch64/{name}"] in [line.split() for line in release.splitlines()]
            path = binary / "by-hash" / algorithm / digest
            assert path.read_bytes() == data
            immutable[path] = data
    assert stage_apt_repo.existing_published(out, "hermes-canary") == {("hermes-agent", v) for v in versions}
    with pytest.raises(SystemExit) as stopped:
        stage_apt_repo.main(args)
    assert stopped.value.code == 2 and "already published" in capsys.readouterr().err
    for old in pool.iterdir():
        old.unlink()
    make_deb(pool / "c.deb", "hermes-agent", "1.2.4-1")
    assert stage_apt_repo.main(args) == 3
    assert all(path.read_bytes() == data for path, data in immutable.items())


def test_unsigned_release_exit_3_without_gpg(tmp_path, no_gpg):
    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "hermes-agent_1.2.3-1_aarch64.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    code = stage_apt_repo.main(
        ["--pool", str(pool), "--out", str(out), "--suite", "hermes-canary"]
    )
    assert code == 3
    assert (out / "dists" / "hermes-canary" / "Release").exists()
    assert not (out / "dists" / "hermes-canary" / "InRelease").exists()
    assert not (out / "dists" / "hermes-canary" / "Release.gpg").exists()


def _generate_test_key(home: Path, passphrase: str = "") -> str:
    """Generate a throwaway ed25519 signing key inside `home` and return
    its fingerprint. Uses the production _gpg_run wrapper."""
    stage_apt_repo._gpg_run(
        home,
        ["--quick-generate-key", "Hermes APT Test <apt-test@example.invalid>",
         "ed25519", "sign", "never"],
        passphrase=passphrase,
    )
    listing = stage_apt_repo._gpg_run(
        home, ["--with-colons", "--list-secret-keys"]
    ).stdout.decode("utf-8", "replace")
    fprs = [line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:")]
    assert len(set(fprs)) == 1
    return fprs[0]


def _export_secret_key(home: Path, fpr: str, passphrase: str = "") -> bytes:
    return stage_apt_repo._gpg_run(
        home, ["--armor", "--export-secret-keys", fpr], passphrase=passphrase
    ).stdout


def _independent_gpgv_verify(keyring_home: Path, *args: Path) -> subprocess.CompletedProcess:
    """Verify with gpgv in a SEPARATE keyring that holds only the published
    public key — the same position a real device is in."""
    kr = stage_apt_repo._gpg_homedir_arg(keyring_home)
    return subprocess.run(
        ["gpgv", "--homedir", kr, "--keyring", f"{kr}/pubring.kbx",
         *[str(a) for a in args]],
        capture_output=True,
    )


@pytest.fixture
def short_home():
    """gpg homedirs must be SHORT: the agent's AF_UNIX socket lives inside
    the homedir and Windows AF_UNIX paths cap around ~107 chars — pytest's
    tmp_path tree is longer than that, so key/verify homes get their own
    mkdtemp at the temp root (this is also how production creates its
    staging home)."""
    made = []
    def make(prefix: str = "apt-test-gnupg-") -> Path:
        d = Path(tempfile.mkdtemp(prefix=prefix))
        made.append(d)
        return d
    yield make
    for d in made:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tracked_gpg_argv(monkeypatch):
    """Record every argv the stager hands to subprocess.run so tests can
    assert isolation (explicit --homedir) and no secret in argv."""
    real_run = stage_apt_repo.subprocess.run
    argvs = []
    def spy(args, **kwargs):
        argvs.append([str(a) for a in args])
        return real_run(args, **kwargs)
    monkeypatch.setattr(stage_apt_repo.subprocess, "run", spy)
    return argvs


@pytest.mark.skipif(not GPG_PRESENT, reason="gpg binary not available")
@pytest.mark.parametrize("secret_pass", ["", "correct-horse-battery-staple"])
def test_real_gpg_signs_and_published_public_key_verifies(tmp_path, monkeypatch, tracked_gpg_argv, short_home, secret_pass):
    """Full behavior: a staged repo signs in an isolated temp GNUPGHOME, and
    InRelease + detached Release.gpg verify as GOOD signatures using ONLY
    the published key.asc (independent gpgv keyring)."""
    monkeypatch.setenv("TERMUX_APT_GPG_PASSPHRASE", secret_pass)
    kh = short_home()
    fpr = _generate_test_key(kh, passphrase=secret_pass)
    stage_apt_repo._gpg_run(
        kh, ["--quick-add-key", fpr, "ed25519", "sign", "never"], passphrase=secret_pass,
    )
    keyfile = tmp_path / "signing.asc"
    keyfile.write_bytes(_export_secret_key(kh, fpr, passphrase=secret_pass))

    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "h.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    assert stage_apt_repo.main(
        ["--pool", str(pool), "--out", str(out),
         "--suite", "hermes-nightly", "--gpg-key-file", str(keyfile)]
    ) == 0

    dists = out / "dists" / "hermes-nightly"
    assert (dists / "InRelease").exists()
    assert (dists / "Release.gpg").exists()

    # Isolation: every gpg invocation carried an explicit --homedir inside
    # the system temp dir, never the user's default keyring.
    temp_root = stage_apt_repo._gpg_homedir_arg(Path(tempfile.gettempdir()))
    for argv in tracked_gpg_argv:
        if secret_pass:
            assert secret_pass not in " ".join(argv)
        assert "--homedir" in argv, f"gpg called without --homedir: {argv}"
        homedir = argv[argv.index("--homedir") + 1]
        assert homedir.startswith(temp_root), homedir

    vr = short_home(prefix="apt-test-verify-")
    stage_apt_repo._gpg_run(
        vr, ["--import"], stdin=(out / "key.asc").read_bytes()
    )
    r = _independent_gpgv_verify(vr, dists / "InRelease")
    assert r.returncode == 0, r.stderr.decode()
    assert b"Good signature" in r.stderr
    r = _independent_gpgv_verify(vr, dists / "Release.gpg", dists / "Release")
    assert r.returncode == 0, r.stderr.decode()
    assert b"Good signature" in r.stderr

    # The staging keyring was deleted afterwards.
    staging_homes = {
        argv[argv.index("--homedir") + 1]
        for argv in tracked_gpg_argv
        if "apt-stage-gnupg-" in argv[argv.index("--homedir") + 1]
    }
    assert staging_homes
    for home in staging_homes:
        native = Path(home)
        if os.name == "nt" and home.startswith("/") and home[2:3] == "/":
            native = Path(home[1] + ":/" + home[3:])
        assert not native.exists()


@pytest.mark.skipif(not GPG_PRESENT, reason="gpg binary not available")
def test_real_gpg_tampered_metadata_fails_closed(tmp_path, monkeypatch, short_home):
    """Fail-closed contract: verification of the signed artifacts is done
    with the signing key, and any post-sign mutation of the Release is
    rejected instead of published."""
    monkeypatch.delenv("TERMUX_APT_GPG_PASSPHRASE", raising=False)
    kh = short_home()
    fpr = _generate_test_key(kh)
    stage_apt_repo._gpg_run(
        kh, ["--quick-add-key", fpr, "ed25519", "sign", "never"], passphrase="",
    )
    keyfile = tmp_path / "signing.asc"
    keyfile.write_bytes(_export_secret_key(kh, fpr))

    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "h.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    real_gpg = stage_apt_repo._gpg_run

    def tamper_after_sign(home, args, **kwargs):
        result = real_gpg(home, args, **kwargs)
        if "--detach-sign" in args:
            release = Path(args[-1])
            release.write_bytes(release.read_bytes() + b"Architectures: amd64\n")
        return result

    monkeypatch.setattr(stage_apt_repo, "_gpg_run", tamper_after_sign)
    with pytest.raises(stage_apt_repo.StageError):
        stage_apt_repo.stage(pool, out, "hermes-stable", keyfile)
    assert not (out / "key.asc").exists(), "verification must precede public-key publication"
    dists = out / "dists/hermes-stable"
    vr = short_home(prefix="apt-test-verify-")
    public = real_gpg(kh, ["--armor", "--export", fpr]).stdout
    real_gpg(vr, ["--import"], stdin=public)
    assert _independent_gpgv_verify(vr, dists / "Release.gpg", dists / "Release").returncode != 0


@pytest.mark.skipif(not GPG_PRESENT, reason="gpg binary not available")
def test_multi_key_import_is_rejected_not_first_key_used(tmp_path, monkeypatch, capsys, short_home):
    """A supplied key file containing MORE THAN ONE secret key must fail
    closed — the stager must never silently sign with the first key."""
    monkeypatch.delenv("TERMUX_APT_GPG_PASSPHRASE", raising=False)
    kh = short_home()
    fpr1 = _generate_test_key(kh)
    kh2 = short_home()
    fpr2 = _generate_test_key(kh2)
    assert fpr1 != fpr2
    keyfile = tmp_path / "two-keys.asc"
    keyfile.write_bytes(
        _export_secret_key(kh, fpr1) + _export_secret_key(kh2, fpr2)
    )

    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "h.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    with pytest.raises(SystemExit) as ei:
        stage_apt_repo.main(
            ["--pool", str(pool), "--out", str(out),
             "--suite", "hermes-stable", "--gpg-key-file", str(keyfile)]
        )
    assert ei.value.code == 2
    assert "secret keys" in capsys.readouterr().err
    # nothing was signed or published
    dists = out / "dists" / "hermes-stable"
    if dists.exists():
        assert not (dists / "InRelease").exists()
        assert not (dists / "Release.gpg").exists()
