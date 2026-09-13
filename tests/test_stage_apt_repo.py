"""Tests for scripts/termux/stage_apt_repo.py — stdlib + pytest, no network.

GPG tests generate a throwaway key inside a temp GNUPGHOME and never touch
the invoking user's keyring; passphrase material never appears in test
output (no secret logging).
"""

import gzip
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts" / "termux"
sys.path.insert(0, str(SCRIPTS))

import stage_apt_repo  # noqa: E402

GPG_PRESENT = shutil.which("gpg") is not None


def make_deb(path: Path, package: str, version: str, arch: str = "aarch64", compression: str = "gz") -> None:
    """Build a minimal .deb (ar archive with control.tar.gz) using stdlib only."""
    control = (
        f"Package: {package}\n"
        f"Version: {version}\n"
        f"Architecture: {arch}\n"
        f"Maintainer: Test <test@example.com>\n"
        f"Description: test package {package}\n"
    )
    buf = io.BytesIO()
    mode = f"w:{compression}"
    member = f"control.tar.{compression}" if compression != "tar" else "control.tar"
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        data = control.encode("utf-8")
        ti = tarfile.TarInfo("control")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))

    ar = io.BytesIO()
    ar.write(b"!<arch>\n")
    payload = buf.getvalue()
    header = "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}".format(
        member, "0", "0", "0", "100644", str(len(payload))
    ).encode() + b"`\n"
    ar.write(header)
    ar.write(payload)
    if len(payload) % 2:
        ar.write(b"\n")
    path.write_bytes(ar.getvalue())


@pytest.fixture
def no_gpg(monkeypatch):
    """Make the script believe gpg is absent so signing is skipped (exit 3)."""
    monkeypatch.setattr(stage_apt_repo.shutil, "which", lambda _: None)


@pytest.fixture
def fake_gpg(monkeypatch, tmp_path):
    """Make the script believe gpg is present, but stub out signing."""
    monkeypatch.setattr(stage_apt_repo.shutil, "which", lambda _: "C:/fake/gpg.exe")
    monkeypatch.setattr(stage_apt_repo, "sign", lambda *a, **k: None)
    key = tmp_path / "signing.asc"
    key.write_text("stub-key\n")
    return key


def test_stages_xz_control_deb(tmp_path):
    """dpkg >= 1.21 emits xz/zst control members; our build uses -Zxz so the
    stager must read xz controls (gz is covered by every other test)."""
    pool = tmp_path / "pool"
    pool.mkdir()
    make_deb(pool / "hermes-agent_1.0-1_aarch64.deb", "hermes-agent", "1.0-1", compression="xz")
    out = tmp_path / "out"
    out.mkdir()
    rc = stage_apt_repo.stage(pool, out, "hermes-canary", None)
    assert rc == 3  # unsigned (no gpg key file) but staged


def test_control_field_extraction(tmp_path):
    deb = tmp_path / "pkg_a.deb"
    make_deb(deb, "hermes-agent", "1.2.3-1")
    fields = stage_apt_repo.deb_control_fields(deb)
    assert fields["Package"] == "hermes-agent"
    assert fields["Version"] == "1.2.3-1"
    assert fields["Architecture"] == "aarch64"


def test_canary_versions_below_stable():
    versions = ["1.2.3-1", "1.2.3~canary.20260831120000-1", "1.2.4~canary.1-1", "1.2.4-1"]
    ordered = sorted(versions, key=stage_apt_repo.deb_version_key)
    assert ordered == [
        "1.2.3~canary.20260831120000-1",
        "1.2.3-1",
        "1.2.4~canary.1-1",
        "1.2.4-1",
    ]


def test_dists_layout_and_pool_copy(tmp_path, fake_gpg):
    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "hermes-agent_1.2.3-1_aarch64.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    r = stage_apt_repo.main(
        [
            "--pool", str(pool), "--out", str(out), "--suite", "hermes-stable",
            "--gpg-key-file", str(fake_gpg),
        ]
    )
    assert r == 0

    dists = out / "dists" / "hermes-stable" / "main" / "binary-aarch64"
    assert (dists / "Packages").exists()
    assert (dists / "Packages.gz").exists()
    assert (out / "dists" / "hermes-stable" / "Release").exists()

    deb_out = out / "pool" / "h" / "hermes-agent_1.2.3-1_aarch64.deb"
    assert deb_out.exists()

    text = (dists / "Packages").read_text(encoding="utf-8")
    assert "Package: hermes-agent" in text
    assert "Version: 1.2.3-1" in text
    assert "Filename: pool/h/hermes-agent_1.2.3-1_aarch64.deb" in text
    assert "SHA256: " in text

    gz_text = gzip.decompress((dists / "Packages.gz").read_bytes()).decode()
    assert gz_text == text

    release = (out / "dists" / "hermes-stable" / "Release").read_text()
    assert "Suite: hermes-stable" in release
    assert "SHA256:" in release
    assert "SHA512:" in release
    # apt contract (learned from a real device rejecting our first repo):
    # Date is mandatory, and the checksum sections must live in the SAME
    # deb822 stanza as the header fields -- a blank line ends the record,
    # after which apt "provides only weak security information" and
    # disables the repository.
    assert "Date: " in release
    assert "\n\n" not in release, "blank line splits the Release stanza"
    head, _, checksums_block = release.partition("SHA256:\n")
    assert "Date: " in head, "Date must precede the checksum sections"


def test_immutability_refusal(tmp_path, fake_gpg, capsys):
    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "hermes-agent_1.2.3-1_aarch64.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    assert stage_apt_repo.main(
        [
            "--pool", str(pool), "--out", str(out), "--suite", "hermes-stable",
            "--gpg-key-file", str(fake_gpg),
        ]
    ) == 0
    with pytest.raises(SystemExit) as ei:
        stage_apt_repo.main(
            [
                "--pool", str(pool), "--out", str(out), "--suite", "hermes-stable",
                "--gpg-key-file", str(fake_gpg),
            ]
        )
    assert ei.value.code == 2
    assert "already published" in capsys.readouterr().err


def test_by_hash_indexes_match_release_and_survive_later_publication(tmp_path):
    import hashlib

    pool = tmp_path / "pool"
    pool.mkdir()
    package = pool / "hermes-agent.deb"
    make_deb(package, "hermes-agent", "1.0-1")
    out = tmp_path / "repo"
    assert stage_apt_repo.stage(pool, out, "hermes-canary", None) == 3
    binary = out / "dists/hermes-canary/main/binary-aarch64"
    original = {}
    for name in ("Packages", "Packages.gz"):
        data = (binary / name).read_bytes()
        for algorithm in ("SHA256", "SHA512"):
            digest = hashlib.new(algorithm.lower(), data).hexdigest()
            immutable = binary / "by-hash" / algorithm / digest
            assert immutable.read_bytes() == data
            original[immutable] = data
    release = (out / "dists/hermes-canary/Release").read_text()
    assert "Acquire-By-Hash: yes\n" in release
    make_deb(package, "hermes-agent", "1.1-1")
    assert stage_apt_repo.stage(pool, out, "hermes-canary", None) == 3
    for path, data in original.items():
        assert path.read_bytes() == data


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


def test_signing_invoked_when_gpg_and_key_present(tmp_path, monkeypatch):
    """No real gpg: assert sign() is called with the right dists dir/key file."""
    calls = []

    def fake_sign(dists, release_path, gpg_key_file):
        calls.append((str(dists), str(release_path), str(gpg_key_file)))
        (dists / "InRelease").write_text("stub", encoding="utf-8")
        (dists / "Release.gpg").write_text("stub", encoding="utf-8")

    monkeypatch.setattr(stage_apt_repo.shutil, "which", lambda _: "C:/fake/gpg.exe")
    monkeypatch.setattr(stage_apt_repo, "sign", fake_sign)

    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "hermes-agent_1.2.3-1_aarch64.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    keyfile = tmp_path / "signing.asc"
    keyfile.write_text("-----BEGIN PGP PRIVATE KEY BLOCK-----\n")
    code = stage_apt_repo.main(
        [
            "--pool", str(pool), "--out", str(out), "--suite", "hermes-stable",
            "--gpg-key-file", str(keyfile),
        ]
    )
    assert code == 0
    assert len(calls) == 1
    dists, release_path, kf = calls[0]
    assert dists == str(out / "dists" / "hermes-stable")
    assert release_path == str(out / "dists" / "hermes-stable" / "Release")
    assert kf == str(keyfile)
    assert (out / "dists" / "hermes-stable" / "InRelease").exists()


# ---------------------------------------------------------------------------
# deb822 record separation (multiversion Packages correctness)
# ---------------------------------------------------------------------------

def _stanza_count(packages_text: str) -> int:
    return len([s for s in packages_text.split("\n\n") if s.strip()])


def test_multiversion_packages_records_are_blank_line_separated(tmp_path):
    """Multiple versions of one package must be separate deb822 records:
    apt splits records on blank lines, so a missing blank line merges two
    versions into one garbled stanza and drops the later one."""
    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "a.deb", "hermes-agent", "1.2.3~canary.20260901000000-1")
    make_deb(pool / "b.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    assert stage_apt_repo.main(
        ["--pool", str(pool), "--out", str(out), "--suite", "hermes-canary"]
    ) == 3  # staged unsigned

    text = (out / "dists" / "hermes-canary" / "main" / "binary-aarch64" / "Packages").read_text()
    assert _stanza_count(text) == 2
    assert "Version: 1.2.3~canary.20260901000000-1\n" in text
    assert "Version: 1.2.3-1\n" in text
    # each stanza carries its own checksum
    assert text.count("SHA256: ") == 2
    # the repo's own published-set parser agrees (it feeds immutability)
    published = stage_apt_repo.existing_published(out, "hermes-canary")
    assert published == {
        ("hermes-agent", "1.2.3~canary.20260901000000-1"),
        ("hermes-agent", "1.2.3-1"),
    }


# ---------------------------------------------------------------------------
# Real-GPG behavioral tests (throwaway key in a temp GNUPGHOME)
# ---------------------------------------------------------------------------

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
def test_real_gpg_signs_and_published_public_key_verifies(tmp_path, monkeypatch, tracked_gpg_argv, short_home):
    """Full behavior: a staged repo signs in an isolated temp GNUPGHOME, and
    InRelease + detached Release.gpg verify as GOOD signatures using ONLY
    the published key.asc (independent gpgv keyring)."""
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
def test_real_gpg_passphrase_reaches_gpg_via_stdin_never_argv(tmp_path, monkeypatch, tracked_gpg_argv, short_home):
    """A passphrase-protected signing key works (env var -> stdin fd), and
    the passphrase never appears in any spawned argv."""
    secret_pass = "correct-horse-battery-staple"
    monkeypatch.setenv("TERMUX_APT_GPG_PASSPHRASE", secret_pass)
    kh = short_home()
    fpr = _generate_test_key(kh, passphrase=secret_pass)
    keyfile = tmp_path / "signing.asc"
    keyfile.write_bytes(_export_secret_key(kh, fpr, passphrase=secret_pass))

    pool = tmp_path / "pool-in"
    pool.mkdir()
    make_deb(pool / "h.deb", "hermes-agent", "1.2.3-1")
    out = tmp_path / "repo"
    assert stage_apt_repo.main(
        ["--pool", str(pool), "--out", str(out),
         "--suite", "hermes-canary", "--gpg-key-file", str(keyfile)]
    ) == 0

    for argv in tracked_gpg_argv:
        assert secret_pass not in " ".join(argv), "passphrase leaked into argv"

    dists = out / "dists" / "hermes-canary"
    vr = short_home(prefix="apt-test-verify-")
    stage_apt_repo._gpg_run(vr, ["--import"], stdin=(out / "key.asc").read_bytes())
    r = _independent_gpgv_verify(vr, dists / "InRelease")
    assert r.returncode == 0, r.stderr.decode()
    r = _independent_gpgv_verify(vr, dists / "Release.gpg", dists / "Release")
    assert r.returncode == 0, r.stderr.decode()


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
    assert stage_apt_repo.main(
        ["--pool", str(pool), "--out", str(out),
         "--suite", "hermes-stable", "--gpg-key-file", str(keyfile)]
    ) == 0

    dists = out / "dists" / "hermes-stable"
    # untouched artifacts verify with the exact signing fingerprint
    stage_apt_repo._verify_signature(kh, fpr, dists / "InRelease", None)
    stage_apt_repo._verify_signature(kh, fpr, dists / "Release.gpg", dists / "Release")

    # tamper with the signed Release -> detached sig no longer validates
    # (gpg exits non-zero during re-verification -> fail closed)
    release_path = dists / "Release"
    release_path.write_text(release_path.read_text() + "Architectures: amd64\n")
    with pytest.raises(stage_apt_repo.StageError):
        stage_apt_repo._verify_signature(kh, fpr, dists / "Release.gpg", release_path)

    # and gpgv agrees independently
    vr = short_home(prefix="apt-test-verify-")
    stage_apt_repo._gpg_run(vr, ["--import"], stdin=(out / "key.asc").read_bytes())
    assert _independent_gpgv_verify(vr, dists / "Release.gpg", release_path).returncode != 0


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
