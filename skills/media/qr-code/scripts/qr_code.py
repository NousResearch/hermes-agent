#!/usr/bin/env python3
"""qr_code — generate and decode QR codes using the best available local backend.

Encoders tried in order: ``qrencode`` (CLI), then the ``qrcode`` Python package.
Decoders tried in order: ``zbarimg`` (CLI), then the ``pyzbar`` Python package.
The script never makes a network call; all work is local.

Exit codes:
    0  success
    1  user error (bad arguments, missing payload)
    2  no usable backend found for the requested action
    3  backend ran but produced no result (e.g. image had no QR)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# --- backend detection -----------------------------------------------------

_EC_MAP = {"L": "L", "M": "M", "Q": "Q", "H": "H"}


def _have_qrencode() -> bool:
    return shutil.which("qrencode") is not None


def _have_zbarimg() -> bool:
    return shutil.which("zbarimg") is not None


def _have_pyqrcode() -> bool:
    try:
        import qrcode  # type: ignore  # optional dependency
        import PIL  # type: ignore  # optional dependency
    except ImportError:
        return False
    return True


def _have_pyzbar() -> bool:
    try:
        from pyzbar.pyzbar import decode  # type: ignore  # optional dependency, noqa: F401
        import PIL.Image  # type: ignore  # optional dependency  # noqa: F401
    except Exception:
        # libzbar shared lib may be missing even when the wheel is installed
        return False
    return True


def doctor() -> int:
    """Report which encoders/decoders are available. Always exits 0."""
    print("Encoders:")
    print(f"  qrencode (CLI)  : {'yes' if _have_qrencode() else 'no'}")
    print(f"  qrcode (Python) : {'yes' if _have_pyqrcode() else 'no'}")
    print("Decoders:")
    print(f"  zbarimg (CLI)   : {'yes' if _have_zbarimg() else 'no'}")
    print(f"  pyzbar (Python) : {'yes' if _have_pyzbar() else 'no'}")
    enc = _have_qrencode() or _have_pyqrcode()
    dec = _have_zbarimg() or _have_pyzbar()
    print()
    print(f"encode: {'ready' if enc else 'NO ENCODER — install qrencode or qrcode[pil]'}")
    print(f"decode: {'ready' if dec else 'NO DECODER — install zbarimg or pyzbar+pillow'}")
    return 0


# --- encoding --------------------------------------------------------------

def _encode_cli(payload: str, out: Optional[Path], terminal: bool, ec: str) -> int:
    cmd = ["qrencode", "-l", ec]
    if terminal:
        cmd.append("-t")  # ANSIUTF8 text art to stdout (default already stdout)
        cmd.append("UTF8")
        cmd.append("-o")  # qrencode uses - for stdout
        cmd.append("-")
        cmd.append(payload)
        completed = subprocess.run(cmd, input=None, capture_output=True, text=True)
        if completed.returncode != 0:
            sys.stderr.write(completed.stderr)
            return 3
        sys.stdout.write(completed.stdout)
        return 0
    if out is None:
        out = Path("qr.png")
    cmd += ["-o", str(out), "-s", "8", "-m", "2", payload]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr)
        return 3
    print(f"Wrote {out}")
    return 0


def _encode_python(payload: str, out: Optional[Path], terminal: bool, ec: str) -> int:
    import qrcode  # type: ignore  # optional dependency

    qr = qrcode.QRCode(error_correction=_ec_to_qrcode(ec))
    qr.add_data(payload)
    qr.make(fit=True)
    if terminal:
        qr.print_tty() if sys.stdout.isatty() else qr.print_ascii()
        return 0
    if out is None:
        out = Path("qr.png")
    img = qr.make_image()
    img.save(str(out))
    print(f"Wrote {out}")
    return 0


def _ec_to_qrcode(ec: str) -> int:
    import qrcode.constants as c  # type: ignore  # optional dependency

    return {
        "L": c.ERROR_CORRECT_L,
        "M": c.ERROR_CORRECT_M,
        "Q": c.ERROR_CORRECT_Q,
        "H": c.ERROR_CORRECT_H,
    }[ec]


def encode(payload: str, out: Optional[Path], terminal: bool, ec: str) -> int:
    if not payload:
        sys.stderr.write("encode: payload is empty\n")
        return 1
    if ec not in _EC_MAP:
        sys.stderr.write(f"encode: invalid error-correction level '{ec}' (L/M/Q/H)\n")
        return 1
    if _have_qrencode():
        return _encode_cli(payload, out, terminal, ec)
    if _have_pyqrcode():
        return _encode_python(payload, out, terminal, ec)
    sys.stderr.write(
        "encode: no encoder found. Install `qrencode` or `pip install qrcode[pil]`.\n"
    )
    return 2


# --- special payload builders ---------------------------------------------

def wifi_encode(ssid: str, password: str, security: str, hidden: bool,
                out: Optional[Path], terminal: bool, ec: str) -> int:
    if not ssid:
        sys.stderr.write("wifi: --ssid is required\n")
        return 1
    # Escape backslash, semicolon, comma per the WIFI: spec
    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace(":", "\\:")

    payload = f"WIFI:T:{security};S:{esc(ssid)};P:{esc(password)};;"
    if hidden:
        payload = payload.replace(";;", ";H:true;;", 1)
    return encode(payload, out, terminal, ec)


def _vcard_escape(value: str) -> str:
    """Escape a vCard property value per RFC 2426 (backslash, comma, semicolon, newline)."""
    return (
        value.replace("\\", "\\\\")
        .replace(",", r"\,")
        .replace(";", r"\;")
        .replace("\n", r"\n")
    )


def vcard_encode(name: str, phone: str, email: str,
                 out: Optional[Path], terminal: bool, ec: str) -> int:
    if not name:
        sys.stderr.write("vcard: --name is required\n")
        return 1
    escaped = _vcard_escape(name)
    # RFC 2426: VERSION, FN, and N are required; records are CRLF-delimited
    # with a trailing CRLF. N is "Family;Given;Additional;Prefix;Suffix".
    lines = [
        "BEGIN:VCARD",
        "VERSION:2.1",
        f"N:{escaped};;;;",
        f"FN:{escaped}",
    ]
    if phone:
        lines.append(f"TEL;CELL:{phone}")
    if email:
        lines.append(f"EMAIL:{email}")
    lines.append("END:VCARD")
    payload = "\r\n".join(lines) + "\r\n"
    return encode(payload, out, terminal, ec)


def batch_encode(infile: Path, outdir: Path, ec: str) -> int:
    if not infile.is_file():
        sys.stderr.write(f"batch: input file not found: {infile}\n")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    values = [ln.rstrip("\n") for ln in infile.read_text(encoding="utf-8").splitlines()]
    values = [v for v in values if v.strip()]
    if not values:
        sys.stderr.write("batch: input file is empty\n")
        return 1
    rc = 0
    for i, value in enumerate(values):
        # Stable, filesystem-safe stem derived from index + payload hash (stdlib only)
        import hashlib
        stem = f"{i:04d}_{hashlib.sha1(value.encode('utf-8')).hexdigest()[:8]}"
        out = outdir / f"{stem}.png"
        code = encode(value, out, False, ec)
        if code != 0:
            rc = code
    return rc


# --- decoding --------------------------------------------------------------

def _decode_cli(path: Path) -> Optional[str]:
    cmd = ["zbarimg", "--quiet", "--raw", str(path)]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode == 0 and completed.stdout:
        # zbarimg --raw emits a trailing newline; strip exactly one
        return completed.stdout.rstrip("\n")
    return None


def _decode_python(path: Path) -> Optional[str]:
    from pyzbar.pyzbar import decode  # type: ignore  # optional dependency
    import PIL.Image  # type: ignore  # optional dependency

    results = decode(PIL.Image.open(str(path)))
    if not results:
        return None
    data = results[0].data
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def decode(path: Path, raw: bool) -> int:
    if not path.is_file():
        sys.stderr.write(f"decode: image file not found: {path}\n")
        return 1
    payload = None
    if _have_zbarimg():
        payload = _decode_cli(path)
    if payload is None and _have_pyzbar():
        payload = _decode_python(path)
    if payload is None:
        if not (_have_zbarimg() or _have_pyzbar()):
            sys.stderr.write(
                "decode: no decoder found. Install `zbarimg` or `pip install pyzbar pillow`.\n"
            )
            return 2
        sys.stderr.write(f"decode: no QR code found in {path}\n")
        return 3
    print(payload if raw else f"Decoded: {payload}")
    return 0


# --- CLI -------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qr_code.py",
        description="Generate and decode QR codes locally (no network).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="show available encoders/decoders")

    p_enc = sub.add_parser("encode", help="encode text into a QR code")
    p_enc.add_argument("payload", help="text/URL to encode")
    p_enc.add_argument("-o", "--output", type=Path, default=None, help="output PNG path")
    p_enc.add_argument("-t", "--terminal", action="store_true",
                       help="render as text art to stdout instead of a file")
    p_enc.add_argument("--ec", default="M", choices=list(_EC_MAP),
                       help="error correction level (default M)")

    p_wifi = sub.add_parser("wifi", help="encode Wi-Fi credentials")
    p_wifi.add_argument("--ssid", required=True)
    p_wifi.add_argument("--password", default="")
    p_wifi.add_argument("--security", default="WPA", choices=["WPA", "WEP", "nopass"])
    p_wifi.add_argument("--hidden", action="store_true")
    p_wifi.add_argument("-o", "--output", type=Path, default=None)
    p_wifi.add_argument("-t", "--terminal", action="store_true")
    p_wifi.add_argument("--ec", default="M", choices=list(_EC_MAP))

    p_vcard = sub.add_parser("vcard", help="encode a contact (vCard)")
    p_vcard.add_argument("--name", required=True)
    p_vcard.add_argument("--phone", default="")
    p_vcard.add_argument("--email", default="")
    p_vcard.add_argument("-o", "--output", type=Path, default=None)
    p_vcard.add_argument("-t", "--terminal", action="store_true")
    p_vcard.add_argument("--ec", default="M", choices=list(_EC_MAP))

    p_dec = sub.add_parser("decode", help="decode a QR image to text")
    p_dec.add_argument("image", type=Path, help="path to the QR image")
    p_dec.add_argument("--raw", action="store_true", help="print only the payload")

    p_batch = sub.add_parser("batch", help="encode one PNG per line of an input file")
    p_batch.add_argument("infile", type=Path, help="text file, one payload per line")
    p_batch.add_argument("--outdir", type=Path, default=Path("qr"),
                         help="output directory (default ./qr)")
    p_batch.add_argument("--ec", default="M", choices=list(_EC_MAP))

    args = parser.parse_args(argv)

    if args.command == "doctor":
        return doctor()
    if args.command == "encode":
        return encode(args.payload, args.output, args.terminal, args.ec)
    if args.command == "wifi":
        return wifi_encode(args.ssid, args.password, args.security, args.hidden,
                           args.output, args.terminal, args.ec)
    if args.command == "vcard":
        return vcard_encode(args.name, args.phone, args.email,
                            args.output, args.terminal, args.ec)
    if args.command == "decode":
        return decode(args.image, args.raw)
    if args.command == "batch":
        return batch_encode(args.infile, args.outdir, args.ec)
    return 1  # unreachable; argparse enforces required subcommand


if __name__ == "__main__":
    sys.exit(main())
