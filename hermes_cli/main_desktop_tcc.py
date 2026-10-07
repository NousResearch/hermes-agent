"""``hermes gui --setup-tcc-identity``: a persistent self-signed macOS code-signing identity.

Split out of ``hermes_cli/main_desktop.py``. The facade's build/sign helpers are imported lazily
inside the function that uses them (the facade late-imports this module from ``cmd_gui``).
"""

import contextlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _macos_codesigning_identity_valid(security: str, identity: str) -> bool:
    """True when `identity` is among VALID (``-v``) code-signing identities — the plain listing also
    shows untrusted certs codesign refuses. Idempotency probe + postcondition. Never raises."""
    try:
        result = subprocess.run(
            [security, "find-identity", "-v", "-p", "codesigning"], capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
    except Exception:
        return False

    return f'"{identity}"' in (result.stdout or "")


def _macos_create_signing_identity(
    openssl: str, security: str, codesign: str, keychain: str, identity: str) -> bool:
    """Create a self-signed code-signing cert (10 years), import it with codesign access, trust it for codeSign."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="hermes-tcc-"))
    try:
        key = tmp_dir / "sign.key"
        crt = tmp_dir / "sign.crt"
        p12 = tmp_dir / "sign.p12"
        subprocess.run(
            [
                openssl, "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key), "-out", str(crt),
                "-days", "3650", "-nodes",
                "-subj", f"/CN={identity}",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,digitalSignature,keyCertSign",
                "-addext", "extendedKeyUsage=codeSigning",
            ],
            capture_output=True, check=True)

        # OpenSSL 3 defaults to AES/SHA-2 PKCS#12 that `security import` rejects
        # with "MAC verification failed". `-legacy` restores the accepted
        # RC2/SHA-1 format but only exists on OpenSSL 3 — so try plain first and
        # fall back to `-legacy` when the IMPORT fails with that signature.
        # (Verified E2E on macOS 26.3.1 / OpenSSL 3.6.3 by @ctaylor86 on PR #77189.)
        def _export_p12(extra_args: list) -> None:
            subprocess.run(
                [
                    openssl, "pkcs12", "-export", *extra_args,
                    "-inkey", str(key), "-in", str(crt),
                    "-out", str(p12), "-passout", "pass:hermeslocal",
                ],
                capture_output=True, check=True)

        def _import_p12():
            return subprocess.run(
                [
                    security, "import", str(p12), "-k", keychain,
                    "-P", "hermeslocal",
                    "-T", codesign, "-T", "/usr/bin/codesign_allocate",
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)

        _export_p12([])
        imported = _import_p12()
        if imported.returncode != 0 and "MAC verification failed" in (imported.stderr or ""):
            # older OpenSSL without -legacy: keep the original failure
            with contextlib.suppress(subprocess.CalledProcessError):
                _export_p12(["-legacy"])
                imported = _import_p12()
        if imported.returncode != 0:
            print(f"  (could not import signing identity into keychain: {imported.stderr.strip()})")
            return False

        # Without explicit trust for the codeSign policy `find-identity -v`
        # reports 0 valid identities. This writes user trust settings, so macOS
        # may prompt for the login password ONCE — the one-time cost this
        # command exists to front-load.
        trusted = subprocess.run(
            [security, "add-trusted-cert", "-r", "trustRoot", "-p", "codeSign", "-k", keychain, str(crt)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if trusted.returncode != 0:
            print(
                "  (could not trust the certificate for code signing: "
                f"{(trusted.stderr or trusted.stdout).strip()})"
            )
            return False
        print(f"  → created, imported, and trusted self-signed identity: {identity!r}")
        return True
    except Exception as exc:
        print(f"  (certificate creation failed: {exc})")
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _desktop_macos_setup_tcc_identity(identity: str = "Hermes Local Signing") -> bool:
    """``--setup-tcc-identity``: create/import a self-signed code-signing cert, point
    ``desktop.macos_signing_identity`` at it and re-sign the packaged app. TCC grants follow the
    signing identity, so a certificate-anchored one is stable across rebuilds (the yabai/skhd
    mechanism). Idempotent; never raises."""
    from hermes_cli.main import PROJECT_ROOT
    from hermes_cli.main_desktop import _desktop_macos_relaunchable_fixup, _desktop_packaged_executable
    if sys.platform != "darwin":
        print("  (--setup-tcc-identity is macOS-only; skipping)")
        return False

    openssl = shutil.which("openssl")
    security = shutil.which("security")
    codesign = shutil.which("codesign")
    if not (openssl and security and codesign):
        print(
            "  (--setup-tcc-identity requires openssl, security, and codesign; "
            f"found openssl={bool(openssl)} security={bool(security)} codesign={bool(codesign)})"
        )
        return False

    keychain = str(Path.home() / "Library" / "Keychains" / "login.keychain-db")
    # Probe with `-v` (valid identities only) so a previously imported-but-
    # untrusted cert is repaired rather than reported as done.
    if _macos_codesigning_identity_valid(security, identity):
        print(f"  → identity {identity!r} already valid in keychain")
    elif not _macos_create_signing_identity(openssl, security, codesign, keychain, identity):
        return False

    # Postcondition gate: name-in-output checks pass for invalid identities;
    # only macOS agreeing the identity is usable counts.
    if not _macos_codesigning_identity_valid(security, identity):
        print(
            f"  (identity {identity!r} was imported but is not a VALID code-signing identity; "
            "run `security find-identity -v -p codesigning` to inspect, and see the manual "
            "Keychain Access steps in the desktop docs)"
        )
        return False

    # config.yaml, not .env — it's not a secret.
    try:
        from hermes_cli.config import set_config_value
        set_config_value("desktop.macos_signing_identity", identity)
        print(f"  → set desktop.macos_signing_identity = {identity!r}")
    except Exception as exc:
        print(f"  (could not write desktop.macos_signing_identity: {exc})")
        return False

    desktop_dir = PROJECT_ROOT / "apps" / "desktop"
    if _desktop_packaged_executable(desktop_dir) is not None:
        try:
            if _desktop_macos_relaunchable_fixup(desktop_dir):
                print(
                    "  → packaged app re-signed with certificate-anchored identity; "
                    "TCC grants persist across rebuilds"
                )
        except Exception as exc:
            print(f"  (could not re-sign packaged app: {exc})")

    print(
        "\n  Note: macOS will re-prompt for permissions ONE final time (the identity "
        "changed). Grant them and they persist from then on. If a permission gets "
        "stuck, reset it with:  tccutil reset All com.nousresearch.hermes"
    )
    return True
