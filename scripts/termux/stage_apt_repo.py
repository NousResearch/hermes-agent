#!/usr/bin/env python3
"""Stage a static APT repository layout (dists/ + pool/) from a pool of .debs.

Pure stdlib. Builds dists/<suite>/{Packages,Packages.gz,Release,InRelease,Release.gpg}
and copies .debs into pool/<first-char>/.

Usage:
  python stage_apt_repo.py --pool POOL_DIR --out OUT_DIR \
      --suite hermes-stable|hermes-canary [--gpg-key-file PATH]

Exit codes:
  0 - success
  2 - usage/IO error
  3 - Release emitted but not signed (gpg binary or key file missing);
      CI treats 3 as failure
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import NoReturn

ARCH = "aarch64"
COMPONENT = "main"

REQUIRED_CONTROL_FIELDS = ["Package", "Version", "Architecture"]


class StageError(Exception):
    """Fatal staging error."""


def die(msg: str, code: int = 2) -> NoReturn:
    print(f"stage_apt_repo: {msg}", file=sys.stderr)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# .deb control parsing (stdlib ar + tar, no dpkg-deb)
# ---------------------------------------------------------------------------

def read_ar_entries(data: bytes):
    """Yield (name, size, payload) for each member of an ar archive."""
    if data[:8] != b"!<arch>\n":
        raise StageError("not an ar archive")
    pos = 8
    while pos + 60 <= len(data):
        header = data[pos : pos + 60]
        name = header[0:16].decode("ascii", "replace").strip()
        size_field = header[48:58].decode("ascii", "replace").strip()
        try:
            size = int(size_field)
        except ValueError:
            raise StageError(f"bad ar member size {size_field!r}")
        pos += 60
        yield name, size, data[pos : pos + size]
        pos += size + (size % 2)  # members are 2-byte aligned


def deb_control_fields_and_bytes(deb_path: Path) -> tuple[dict, bytes]:
    """Parse a .deb's control member and return (fields, raw archive bytes).

    The caller gets the raw bytes too so hashing/pool-copy need no re-read.
    """
    data = deb_path.read_bytes()
    control_tar = None
    for name, size, payload in read_ar_entries(data):
        if name in ("control.tar", "control.tar.gz", "control.tar.xz", "control.tar.zst"):
            control_tar = (name, payload)
            break
    if control_tar is None:
        raise StageError(f"{deb_path.name}: no control.tar member found")
    name, payload = control_tar

    if name == "control.tar.gz":
        raw = gzip.decompress(payload)
    elif name == "control.tar.xz":
        raw = lzma.decompress(payload)
    elif name == "control.tar.zst":
        # zstd has no stdlib decoder; dpkg-deb -Zxz (our build default)
        # keeps control in xz, but tolerate zst debs from elsewhere when
        # the host has a zstd binary.
        try:
            raw = subprocess.run(
                ["zstd", "-d", "-c"], input=payload, check=True,
                capture_output=True,
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise StageError(
                f"{deb_path.name}: control member is zstd-compressed and no zstd binary is available"
            ) from e
    elif name == "control.tar":
        raw = payload
    else:
        raise StageError(f"{deb_path.name}: unsupported control compression {name}")

    fields: dict = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        for member in tf.getmembers():
            if member.name.lstrip("./") == "control":
                f = tf.extractfile(member)
                if f is None:
                    continue
                fields = _parse_debian_control(f.read().decode("utf-8", "replace"))
                break
    for req in REQUIRED_CONTROL_FIELDS:
        if req not in fields:
            raise StageError(f"{deb_path.name}: control missing {req}")
    return fields, data


def deb_control_fields(deb_path: Path) -> dict:
    """Parse Package/Version/Architecture/... from a .deb's control member."""
    fields, _ = deb_control_fields_and_bytes(deb_path)
    return fields


def _parse_debian_control(text: str) -> dict:
    fields: dict = {}
    last = None
    for line in text.splitlines():
        if not line.strip():
            last = None
            continue
        if line[0] in " \t" and last:
            fields[last] += " " + line.strip()
        elif ":" in line:
            key, _, val = line.partition(":")
            last = key.strip()
            fields[last] = val.strip()
    return fields


# ---------------------------------------------------------------------------
# dpkg version ordering: '~' sorts before end-of-version (and before empty)
# ---------------------------------------------------------------------------

def _order_char(ch: str) -> int:
    if ch == "~":
        return -1
    if ch.isdigit():
        return 0
    if ch.isalpha():
        return ord(ch)
    return ord(ch) + 256


def deb_version_key(version: str):
    """Sort key implementing dpkg version comparison for our versions."""
    epoch, _, rest = version.partition(":")
    epoch_num = int(epoch) if epoch.isdigit() else 0
    if ":" not in version:
        rest = version
    up, _, rev = rest.rpartition("-")
    if not up:
        up, rev = rest, ""

    def cmp_part(s: str):
        parts = []
        i = 0
        while i < len(s):
            if s[i].isdigit():
                j = i
                while j < len(s) and s[j].isdigit():
                    j += 1
                parts.append((0, int(s[i:j]), ""))
                i = j
            else:
                j = i
                while j < len(s) and not s[j].isdigit():
                    j += 1
                parts.append((1, tuple(_order_char(c) for c in s[i:j]), ""))
                i = j
        return parts

    # dpkg: end-of-part sorts after everything except '~'; padding the shorter
    # part with (2, ...) achieves that ('~' yields order char -1 < any pad).
    def padded(parts):
        return parts + [(2, (), "")] * 4

    return (epoch_num, padded(cmp_part(up)), padded(cmp_part(rev)))


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def existing_published(out_dir: Path, suite: str) -> set:
    """(package, version) pairs already published in dists/<suite>/Packages."""
    packages_file = out_dir / "dists" / suite / COMPONENT / f"binary-{ARCH}" / "Packages"
    published = set()
    if packages_file.exists():
        text = packages_file.read_text(encoding="utf-8")
        # deb822 records are separated by blank lines; never rely on field
        # order inside a record.
        for stanza in text.split("\n\n"):
            pkg = ver = None
            for line in stanza.splitlines():
                if line.startswith("Package: "):
                    pkg = line[len("Package: "):].strip()
                elif line.startswith("Version: "):
                    ver = line[len("Version: "):].strip()
            if pkg and ver:
                published.add((pkg, ver))
    return published


def stage(pool_dir: Path, out_dir: Path, suite: str, gpg_key_file: Path | None) -> int:
    debs = sorted(pool_dir.glob("*.deb"))
    if not debs:
        die(f"no .deb files found in pool {pool_dir}")

    published = existing_published(out_dir, suite)

    dists = out_dir / "dists" / suite
    binary_dir = dists / COMPONENT / f"binary-{ARCH}"
    binary_dir.mkdir(parents=True, exist_ok=True)

    stanzas = []
    for deb in debs:
        fields, raw = deb_control_fields_and_bytes(deb)
        key = (fields["Package"], fields["Version"])
        if key in published:
            die(
                f"refusing: {key[0]}_{key[1]} already published in dists/{suite} "
                "(published apt assets are immutable)"
            )
        arch = fields["Architecture"]
        target = out_dir / "pool" / deb.name[0].lower() / deb.name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Unconditional write: the pool file must always equal the source so
        # the stanza hashes below can never describe a stale/different file.
        target.write_bytes(raw)
        filename = f"pool/{deb.name[0].lower()}/{deb.name}"
        size = len(raw)
        sha256 = hashlib.sha256(raw).hexdigest()
        stanzas.append(
            {
                "Package": fields["Package"],
                "Version": fields["Version"],
                "Architecture": arch,
                "Maintainer": fields.get("Maintainer", "Hermes Agent <noreply@nousresearch.com>"),
                "Installed-Size": fields.get("Installed-Size", "0"),
                "Description": fields.get("Description", "Hermes Agent"),
                "Filename": filename,
                "Size": str(size),
                "SHA256": sha256,
            }
        )

    # Packages stanzas MUST be separated by blank lines: apt parses the file
    # as deb822 records and a missing blank line merges consecutive records
    # into one garbled stanza (biting exactly when the repo carries multiple
    # versions of the same package -- the first thing a canary+stable repo
    # has). existing_published() below relies on the same blank-line layout.
    stanzas.sort(
        key=lambda s: (s["Package"], deb_version_key(s["Version"]))
    )
    packages_text = "\n\n".join(
        "\n".join(f"{k}: {v}" for k, v in stanza.items()) for stanza in stanzas
    ) + ("\n" if stanzas else "")

    (binary_dir / "Packages").write_text(packages_text, encoding="utf-8")
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(binary_dir / "Packages.gz", "wb"), mtime=0) as gz:
        gz.write(packages_text.encode("utf-8"))

    # Date is REQUIRED by apt (it refuses a Release without one) and the
    # checksum sections must stay INSIDE the same deb822 stanza: a blank
    # line ends the record, and apt then never sees the hashes ("weak
    # security information"). One paragraph, no blank lines.
    release_fields = [
        "Origin: Hermes Agent",
        "Label: hermes-agent",
        f"Suite: {suite}",
        f"Codename: {suite}",
        f"Architectures: {ARCH}",
        f"Components: {COMPONENT}",
        "Acquire-By-Hash: yes",
        f"Description: Hermes Agent apt repository ({suite})",
        "Date: " + time.strftime("%a, %d %b %Y %H:%M:%S UTC", time.gmtime()),
    ]
    checksums = []
    sha512 = []
    for name in ("Packages", "Packages.gz"):
        p = binary_dir / name
        rel = f"{COMPONENT}/binary-{ARCH}/{name}"
        size = p.stat().st_size
        data = p.read_bytes()
        for algorithm, rows in (("SHA256", checksums), ("SHA512", sha512)):
            digest = hashlib.new(algorithm.lower(), data).hexdigest()
            rows.append(f" {digest} {size:8d} {rel}")
            immutable = binary_dir / "by-hash" / algorithm / digest
            immutable.parent.mkdir(parents=True, exist_ok=True)
            immutable.write_bytes(data)
    release = "\n".join(release_fields) + "\n"
    release += "SHA256:\n" + "\n".join(checksums) + "\n"
    release += "SHA512:\n" + "\n".join(sha512) + "\n"

    release_path = dists / "Release"
    release_path.write_text(release, encoding="utf-8")

    if gpg_key_file is not None and shutil.which("gpg"):
        sign(dists, release_path, gpg_key_file)
        return 0
    print("warning: gpg binary or key file unavailable; emitted unsigned Release", file=sys.stderr)
    return 3


def _gpg_homedir_arg(path: Path) -> str:
    """Format a homedir path for the host's gpg.

    On Windows the commonly available gpg is the MSYS/Git-for-Windows build,
    which rejects native ``C:\\...`` paths for --homedir and needs the
    ``/c/...`` form. POSIX gpg (CI's ubuntu runner) needs paths untouched,
    so the conversion applies only to Windows drive-letter paths.
    """
    s = str(path)
    if os.name == "nt" and len(s) >= 2 and s[1] == ":":
        return f"/{s[0].lower()}{s[2:].replace(chr(92), '/')}"
    return s


def _gpg_run(
    homedir: Path,
    args: list,
    *,
    stdin: bytes | None = None,
    passphrase: str | None = None,
) -> subprocess.CompletedProcess:
    """Run gpg with an explicit isolated homedir.

    The passphrase, when the key needs one, is fed on STDIN via
    ``--passphrase-fd 0`` -- it is never placed in argv (visible in
    /proc or error output) and never written to disk.
    """
    cmd = ["gpg", "--batch", "--yes", "--homedir", _gpg_homedir_arg(homedir)]
    if passphrase is not None:
        cmd += ["--pinentry-mode", "loopback", "--passphrase-fd", "0"]
        if stdin is not None:
            raise StageError("internal error: passphrase and data stdin are the same fd")
        stdin = passphrase.encode("utf-8")

    # File-backed capture, not pipes: gpg's spawned gpg-agent can outlive
    # (and inherit) the child's stdout/stderr pipes, and a reader waiting
    # for pipe EOF would then hang forever after gpg itself has exited.
    # A wait() on files returns as soon as gpg exits.
    # Also guarantee HOME: hardened CI runners execute with a scrubbed
    # environment (env -i), and the gpg-agent spawned by gpg fails to start
    # with no HOME set.
    env = dict(os.environ)
    if not env.get("HOME"):
        env["HOME"] = str(homedir)
    with tempfile.TemporaryDirectory(prefix="apt-stage-gpg-io-") as io_dir:
        out_path = Path(io_dir) / "stdout"
        err_path = Path(io_dir) / "stderr"
        with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
            result = subprocess.run(
                [*cmd, *args],
                input=stdin,
                stdin=subprocess.DEVNULL if stdin is None else None,
                stdout=fo, stderr=fe, env=env,
            )
        result.stdout = out_path.read_bytes()
        result.stderr = err_path.read_bytes()
    if result.returncode != 0:
        raise StageError(
            f"gpg {' '.join(args[:2])} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result


def _sole_secret_key_fingerprint(homedir: Path) -> str:
    """Fingerprint of the ONLY signing-capable secret key in the homedir.

    The homedir is a fresh temp dir into which exactly the supplied key file
    was imported, so whatever is here came from the import -- never a
    pre-existing unrelated key from a user keyring. If the file carried
    multiple keys we fail closed rather than silently signing with the
    first one.
    """
    listing = _gpg_run(
        homedir, ["--with-colons", "--list-secret-keys"]
    ).stdout.decode("utf-8", "replace")
    fingerprints = []
    primary = False
    for line in listing.splitlines():
        fields = line.split(":")
        if fields[0] in ("sec", "ssb"):
            primary = fields[0] == "sec"
        elif fields[0] == "fpr" and primary:
            fingerprints.append(fields[9])
            primary = False
    if not fingerprints:
        raise StageError("supplied key file produced no secret key after import")
    if len(set(fingerprints)) != 1:
        raise StageError(
            f"supplied key file contains {len(set(fingerprints))} secret keys; "
            "refusing to guess which one signs the repository"
        )
    return fingerprints[0]


def _verify_signature(
    homedir: Path, expected_fpr: str, sig_path: Path, signed_path: Path | None
) -> None:
    """Fail-closed check that sig_path is a GOOD signature by expected_fpr.

    Verified via gpg --status-fd VALIDSIG (not exit code alone), and the
    fingerprint must match the key that was selected from the import.
    """
    args = ["--with-colons", "--status-fd", "1", "--verify", str(sig_path)]
    if signed_path is not None:
        args.append(str(signed_path))
    result = _gpg_run(homedir, args)
    valid = []
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("[GNUPG:] VALIDSIG "):
            fields = line.split()
            valid.append(fields[2])
            if len(fields) > 11:
                valid.append(fields[11])  # Primary key when a signing subkey made the signature.
    if expected_fpr not in valid:
        raise StageError(
            f"post-sign verification failed for {sig_path.name}: no VALIDSIG "
            f"for {expected_fpr}; refusing to publish the repository"
        )


def sign(dists: Path, release_path: Path, gpg_key_file: Path) -> None:
    """Sign Release as InRelease + detached Release.gpg in an ISOLATED
    temp GNUPGHOME, verify the signatures fail-closed, then publish the
    signing key's public half at the repo root.

    Nothing here touches the invoking user's real keyring: the gpg homedir
    is a throwaway temp directory whose only content is the supplied key
    file, and it is removed afterwards. Real signing keys are usually
    passphrase-protected; TERMUX_APT_GPG_PASSPHRASE (set only when the key
    needs one) reaches gpg via stdin, never argv.
    """
    passphrase = os.environ.get("TERMUX_APT_GPG_PASSPHRASE", "")
    homedir = Path(tempfile.mkdtemp(prefix="apt-stage-gnupg-"))
    try:
        secret = gpg_key_file.read_bytes()
        _gpg_run(homedir, ["--import"], stdin=secret)
        key_id = _sole_secret_key_fingerprint(homedir)

        _gpg_run(
            homedir,
            [
                "--clearsign", "--local-user", key_id,
                "--output", str(dists / "InRelease"),
                str(release_path),
            ],
            passphrase=passphrase,
        )
        _gpg_run(
            homedir,
            [
                "--detach-sign", "--armor", "--local-user", key_id,
                "--output", str(dists / "Release.gpg"),
                str(release_path),
            ],
            passphrase=passphrase,
        )

        # Fail closed: the run is only a success if both artifacts verify as
        # GOOD signatures from the exact imported key. A signature that does
        # not verify is worse than none -- apt would reject the repo, but so
        # would any mirror that already cached it.
        _verify_signature(homedir, key_id, dists / "InRelease", None)
        _verify_signature(homedir, key_id, dists / "Release.gpg", release_path)

        # Publish the signing key's public half at the repo root. The key is
        # exported from the exact key that signed THIS suite, so the two can
        # never drift -- a key rotation re-publishes itself on the next run.
        # Users verify it by fingerprint (see website/docs/getting-started/
        # termux.md); the fingerprint is the trust anchor, not the URL.
        pub = _gpg_run(homedir, ["--armor", "--export", key_id]).stdout
        if not pub.strip():
            raise StageError("gpg exported an empty public key")
        (dists.parent.parent / "key.asc").write_bytes(pub)
    finally:
        shutil.rmtree(homedir, ignore_errors=True)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage a static APT repo layout.")
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    # hermes-nightly is the suite actually published today
    # (https://hermes-assets.nousresearch.com/releases/termux/nightly/,
    # verified 2026-09-06); hermes-stable/hermes-canary are what CI stages
    # for the stable/canary channels.
    ap.add_argument(
        "--suite", required=True,
        choices=["hermes-stable", "hermes-canary", "hermes-nightly"],
    )
    ap.add_argument("--gpg-key-file", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.pool.is_dir():
        die(f"pool dir not found: {args.pool}")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        return stage(args.pool, args.out, args.suite, args.gpg_key_file)
    except StageError as e:
        die(str(e))


if __name__ == "__main__":
    sys.exit(main())
