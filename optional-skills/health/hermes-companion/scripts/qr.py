"""
Pure-Python QR Code generator with zero external dependencies.
Generates ISO/IEC 18004 compliant QR codes (Versions 1-6, Byte mode, ECC L and M)
and formats them for terminal ASCII/ANSI display.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Final, List, Optional, Sequence, Tuple, Union


# Galois Field GF(256) tables with primitive polynomial 0x11D (285)
_EXP_TABLE: List[int] = [0] * 512
_LOG_TABLE: List[int] = [0] * 256

def _init_gf256() -> None:
    x = 1
    for i in range(255):
        _EXP_TABLE[i] = x
        _EXP_TABLE[i + 255] = x
        _LOG_TABLE[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    _EXP_TABLE[510] = _EXP_TABLE[0]
    _EXP_TABLE[511] = _EXP_TABLE[1]

_init_gf256()


def _gf_mul(x: int, y: int) -> int:
    if x == 0 or y == 0:
        return 0
    return _EXP_TABLE[_LOG_TABLE[x] + _LOG_TABLE[y]]


def _rs_generator_poly(degree: int) -> List[int]:
    """Calculate Reed-Solomon generator polynomial for a given degree."""
    poly = [1]
    for i in range(degree):
        # Multiply poly by (x - alpha^i) = (x + alpha^i) in GF(256)
        factor = [_EXP_TABLE[i], 1]
        new_poly = [0] * (len(poly) + 1)
        for j, c in enumerate(poly):
            new_poly[j] ^= _gf_mul(c, factor[0])
            new_poly[j + 1] ^= c
        poly = new_poly
    return poly


def _rs_encode(data: Sequence[int], ec_len: int) -> List[int]:
    """Compute Reed-Solomon error correction codewords for data."""
    gen = _rs_generator_poly(ec_len)
    msg = list(data) + [0] * ec_len
    for i in range(len(data)):
        lead = msg[i]
        if lead != 0:
            for j in range(len(gen)):
                # Note: gen is highest degree first
                msg[i + j] ^= _gf_mul(lead, gen[len(gen) - 1 - j])
    return msg[len(data):]


# Version specifications: (version, ecc_level) -> (total_codewords, ec_codewords_per_block, num_blocks_g1, data_per_block_g1, num_blocks_g2, data_per_block_g2, alignment_positions)
# ECC Level: 0 = M (15% recovery), 1 = L (7% recovery)
_VERSION_SPECS: Final = {
    # V1: 21x21
    (1, 1): (26, 7, 1, 19, 0, 0, ()),
    (1, 0): (26, 10, 1, 16, 0, 0, ()),
    # V2: 25x25
    (2, 1): (44, 10, 1, 34, 0, 0, (6, 18)),
    (2, 0): (44, 16, 1, 28, 0, 0, (6, 18)),
    # V3: 29x29
    (3, 1): (70, 15, 1, 55, 0, 0, (6, 22)),
    (3, 0): (70, 26, 1, 44, 0, 0, (6, 22)),
    # V4: 33x33
    (4, 1): (100, 20, 1, 80, 0, 0, (6, 26)),
    (4, 0): (100, 18, 2, 32, 0, 0, (6, 26)),
    # V5: 37x37
    (5, 1): (134, 26, 1, 108, 0, 0, (6, 30)),
    (5, 0): (134, 24, 2, 43, 0, 0, (6, 30)),
    # V6: 41x41
    (6, 1): (172, 18, 2, 68, 0, 0, (6, 34)),
    (6, 0): (172, 16, 4, 27, 0, 0, (6, 34)),
}


def _select_version(data_len: int, prefer_ecc_m: bool = True) -> Tuple[int, int]:
    """Select the smallest QR version that fits data_len bytes."""
    ecc_order = [0, 1] if prefer_ecc_m else [1, 0]
    for v in range(1, 7):
        for ecc in ecc_order:
            spec = _VERSION_SPECS.get((v, ecc))
            if not spec:
                continue
            total_data = (spec[2] * spec[3]) + (spec[4] * spec[5])
            # Overhead in byte mode: 4 bits mode + 8 bits char count = 12 bits = 1.5 bytes -> need data_len + 3 bytes <= total_data
            if data_len + 3 <= total_data:
                return v, ecc
    return 6, 1  # Fallback to V6-L


class QRCode:
    def __init__(self, data: str, version: Optional[int] = None, ecc_level: Optional[int] = None) -> None:
        self.raw_data = data.encode("utf-8")
        if version is None or ecc_level is None:
            v, ecc = _select_version(len(self.raw_data))
            self.version = version or v
            self.ecc_level = ecc_level if ecc_level is not None else ecc
        else:
            self.version = version
            self.ecc_level = ecc_level

        spec = _VERSION_SPECS.get((self.version, self.ecc_level))
        if not spec:
            raise ValueError(f"Unsupported QR version/ECC combination: V{self.version} ECC {self.ecc_level}")
        total_data = (spec[2] * spec[3]) + (spec[4] * spec[5])
        if len(self.raw_data) + 2 > total_data:
            raise ValueError(
                f"Data length ({len(self.raw_data)} bytes) exceeds capacity of Version {self.version} ECC {self.ecc_level} ({total_data - 2} bytes)"
            )

        self.size = 17 + 4 * self.version
        self.modules: List[List[Optional[bool]]] = [[None] * self.size for _ in range(self.size)]
        self.is_function: List[List[bool]] = [[False] * self.size for _ in range(self.size)]

        self._build_qr()

    def _build_qr(self) -> None:
        self._place_finders()
        self._place_alignment()
        self._place_timing()
        self._place_dark_module()
        self._reserve_format_info()

        codewords = self._encode_data()
        self._place_data(codewords)

    def _set_module(self, row: int, col: int, is_dark: bool, is_func: bool = False) -> None:
        self.modules[row][col] = is_dark
        if is_func:
            self.is_function[row][col] = True

    def _place_finders(self) -> None:
        # Finder pattern is 7x7 at (0,0), (0, size-7), (size-7, 0)
        positions = [(0, 0), (0, self.size - 7), (self.size - 7, 0)]
        for r_orig, c_orig in positions:
            for r in range(7):
                for c in range(7):
                    is_border = (r == 0 or r == 6 or c == 0 or c == 6)
                    is_center = (2 <= r <= 4 and 2 <= c <= 4)
                    self._set_module(r_orig + r, c_orig + c, is_border or is_center, is_func=True)

            # Separator ring
            for r in range(-1, 8):
                for c in range(-1, 8):
                    rr = r_orig + r
                    cc = c_orig + c
                    if 0 <= rr < self.size and 0 <= cc < self.size:
                        if not self.is_function[rr][cc]:
                            self._set_module(rr, cc, False, is_func=True)

    def _place_alignment(self) -> None:
        spec = _VERSION_SPECS[(self.version, self.ecc_level)]
        coords = spec[6]
        if not coords:
            return

        for r_pos in coords:
            for c_pos in coords:
                # Skip if overlaps with finder patterns
                if (r_pos < 9 and c_pos < 9) or (r_pos < 9 and c_pos > self.size - 9) or (r_pos > self.size - 9 and c_pos < 9):
                    continue

                for r in range(-2, 3):
                    for c in range(-2, 3):
                        is_dark = (abs(r) == 2 or abs(c) == 2 or (r == 0 and c == 0))
                        self._set_module(r_pos + r, c_pos + c, is_dark, is_func=True)

    def _place_timing(self) -> None:
        for i in range(8, self.size - 8):
            val = (i % 2 == 0)
            if not self.is_function[6][i]:
                self._set_module(6, i, val, is_func=True)
            if not self.is_function[i][6]:
                self._set_module(i, 6, val, is_func=True)

    def _place_dark_module(self) -> None:
        # Dark module is always at (4 * version + 9, 8)
        self._set_module(4 * self.version + 9, 8, True, is_func=True)

    def _reserve_format_info(self) -> None:
        # Format info is placed around finders (15 bits)
        for i in range(9):
            if not self.is_function[8][i]:
                self.is_function[8][i] = True
            if not self.is_function[i][8]:
                self.is_function[i][8] = True
        for i in range(8):
            if not self.is_function[8][self.size - 1 - i]:
                self.is_function[8][self.size - 1 - i] = True
            if not self.is_function[self.size - 1 - i][8]:
                self.is_function[self.size - 1 - i][8] = True

    def _encode_data(self) -> List[int]:
        spec = _VERSION_SPECS[(self.version, self.ecc_level)]
        total_data = (spec[2] * spec[3]) + (spec[4] * spec[5])

        # Bit stream:
        # Mode: Byte (0100) = 4 bits
        # Char count: 8 bits for V1-V9
        bits = []
        # Mode 0100
        bits.extend([0, 1, 0, 0])
        # Char count (8 bits)
        count = len(self.raw_data)
        for b in range(7, -1, -1):
            bits.append((count >> b) & 1)
        # Data bytes
        for byte_val in self.raw_data:
            for b in range(7, -1, -1):
                bits.append((byte_val >> b) & 1)

        # Terminator: up to 4 zero bits
        capacity_bits = total_data * 8
        term_len = min(4, capacity_bits - len(bits))
        bits.extend([0] * term_len)

        # Pad to byte boundary
        while len(bits) % 8 != 0:
            bits.append(0)

        # Convert bits to codewords
        codewords = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            codewords.append(byte)

        # Pad with 0xEC and 0x11
        pad_bytes = [0xEC, 0x11]
        pad_idx = 0
        while len(codewords) < total_data:
            codewords.append(pad_bytes[pad_idx % 2])
            pad_idx += 1

        # Break into blocks and compute Reed-Solomon EC
        blocks_data: List[List[int]] = []
        blocks_ec: List[List[int]] = []
        ec_per_block = spec[1]
        offset = 0

        # Group 1
        for _ in range(spec[2]):
            block = codewords[offset:offset + spec[3]]
            offset += spec[3]
            blocks_data.append(block)
            blocks_ec.append(_rs_encode(block, ec_per_block))

        # Group 2
        for _ in range(spec[4]):
            block = codewords[offset:offset + spec[5]]
            offset += spec[5]
            blocks_data.append(block)
            blocks_ec.append(_rs_encode(block, ec_per_block))

        # Interleave data codewords
        interleaved: List[int] = []
        max_data_len = max(len(b) for b in blocks_data)
        for i in range(max_data_len):
            for b in blocks_data:
                if i < len(b):
                    interleaved.append(b[i])

        # Interleave EC codewords
        for i in range(ec_per_block):
            for b in blocks_ec:
                interleaved.append(b[i])

        return interleaved

    def _place_data(self, codewords: List[int]) -> None:
        # Convert codewords into bits
        bits: List[bool] = []
        for byte in codewords:
            for b in range(7, -1, -1):
                bits.append(bool((byte >> b) & 1))

        # Remainder bits (for V1-6, usually 0 or 7 bits)
        # Place bits in 2-column zig-zag from right to left
        row = self.size - 1
        col = self.size - 1
        direction = -1  # upwards
        bit_idx = 0

        # Choose standard mask 0: (row + col) % 2 == 0
        mask_fn = lambda r, c: (r + c) % 2 == 0

        while col > 0:
            if col == 6:  # Skip vertical timing pattern column
                col -= 1

            for c_offset in range(2):
                curr_col = col - c_offset
                if not self.is_function[row][curr_col]:
                    bit_val = bits[bit_idx] if bit_idx < len(bits) else False
                    bit_idx += 1
                    # Apply mask
                    masked_val = bit_val ^ mask_fn(row, curr_col)
                    self.modules[row][curr_col] = masked_val

            row += direction
            if row < 0:
                row = 0
                direction = 1
                col -= 2
            elif row >= self.size:
                row = self.size - 1
                direction = -1
                col -= 2

        self._write_format_info(mask_idx=0)

    def _write_format_info(self, mask_idx: int) -> None:
        # ECC level bits: M = 00_2, L = 01_2
        # In QR standard: L = 1, M = 0, Q = 3, H = 2
        ecc_bits = 0 if self.ecc_level == 0 else 1
        format_data = (ecc_bits << 3) | mask_idx

        # Calculate 10 BCH bits (generator 0x537 = 10100110111)
        rem = format_data << 10
        for i in range(14, 9, -1):
            if rem & (1 << i):
                rem ^= (0x537 << (i - 10))

        format_val = ((format_data << 10) | rem) ^ 0x5412

        # 15 bits of format info: Bit 14 is MSB, Bit 0 is LSB
        def get_bit(k: int) -> bool:
            return bool((format_val >> k) & 1)

        # Primary copy (Top-left around finders)
        # Bits 14..9 -> (8, 0..5)
        for i in range(6):
            self.modules[8][i] = get_bit(14 - i)
        # Bit 8 -> (8, 7)
        self.modules[8][7] = get_bit(8)
        # Bit 7 -> (8, 8)
        self.modules[8][8] = get_bit(7)
        # Bit 6 -> (7, 8)
        self.modules[7][8] = get_bit(6)
        # Bits 5..0 -> (5..0, 8)
        for i in range(6):
            self.modules[5 - i][8] = get_bit(5 - i)

        # Secondary copy (Split between bottom-left and top-right)
        # Bits 14..8 -> (size-1..size-7, 8) [7 bits, stopping before dark module at size-8]
        for i in range(7):
            self.modules[self.size - 1 - i][8] = get_bit(14 - i)
        # Bits 7..0 -> (8, size-8..size-1) [8 bits]
        for i in range(8):
            self.modules[8][self.size - 8 + i] = get_bit(7 - i)

    def to_ascii(self) -> str:
        """Render QR code as ASCII with half-blocks and explicit ANSI colors for terminal display."""
        border = 4
        full_size = self.size + border * 2
        lines = []

        grid: List[List[bool]] = [[False] * full_size for _ in range(full_size)]
        for r in range(self.size):
            for c in range(self.size):
                if self.modules[r][c]:
                    grid[r + border][c + border] = True

        # Render rows two-by-two using Unicode half-blocks.
        # With explicit ANSI background 40 (black) and foreground 37;1 (bright white),
        # dark modules render dark and light modules render light on ANY terminal background:
        # ' ' (space) = top dark, bot dark (black background)
        # '█' (full block) = top light, bot light (bright white foreground)
        # '▄' (lower half) = top dark, bot light (lower half is white)
        # '▀' (upper half) = top light, bot dark (upper half is white)
        for r in range(0, full_size, 2):
            line = ["\033[40;37;1m"]
            for c in range(full_size):
                top_dark = grid[r][c]
                bot_dark = grid[r + 1][c] if r + 1 < full_size else False

                if top_dark and bot_dark:
                    line.append(" ")
                elif not top_dark and not bot_dark:
                    line.append("█")
                elif top_dark and not bot_dark:
                    line.append("▄")
                else:
                    line.append("▀")
            line.append("\033[0m")
            lines.append("".join(line))

        return "\n".join(lines)

    def to_png(self, scale: int = 3, margin: int = 4) -> bytes:
        """Render QR code as a PNG binary image using stdlib zlib and struct."""
        full_size = self.size + margin * 2
        width = full_size * scale
        height = full_size * scale

        raw_scanlines: List[bytes] = []
        for r in range(full_size):
            row_pixels = bytearray()
            for c in range(full_size):
                if margin <= r < margin + self.size and margin <= c < margin + self.size:
                    is_dark = bool(self.modules[r - margin][c - margin])
                else:
                    is_dark = False
                pixel = 0 if is_dark else 255
                row_pixels.extend([pixel] * scale)
            scanline = b"\x00" + bytes(row_pixels)
            for _ in range(scale):
                raw_scanlines.append(scanline)

        raw_data = b"".join(raw_scanlines)
        compressed = zlib.compress(raw_data)

        # IHDR: width (4), height (4), bit_depth=8 (1), color_type=0 (grayscale) (1), comp=0 (1), filter=0 (1), interlace=0 (1)
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
        ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF)
        ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc

        idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF)
        idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

        iend_crc = struct.pack(">I", zlib.crc32(b"IEND") & 0xFFFFFFFF)
        iend_chunk = struct.pack(">I", 0) + b"IEND" + iend_crc

        return b"\x89PNG\r\n\x1a\n" + ihdr_chunk + idat_chunk + iend_chunk

    def to_svg(self, margin: int = 4) -> str:
        """Render QR code as dependency-free SVG with quiet zone."""
        full_size = self.size + margin * 2
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 {full_size} {full_size}" stroke="none">',
            '  <rect width="100%" height="100%" fill="#ffffff"/>',
        ]
        for r in range(self.size):
            for c in range(self.size):
                if self.modules[r][c]:
                    lines.append(f'  <rect x="{c + margin}" y="{r + margin}" width="1" height="1" fill="#000000"/>')
        lines.append("</svg>\n")
        return "\n".join(lines)


def generate_qr_ascii(data: str) -> str:
    """Convenience function to generate terminal ASCII QR code."""
    qr = QRCode(data)
    return qr.to_ascii()


def generate_qr_png(data: str, scale: int = 3, margin: int = 4) -> bytes:
    """Convenience function to generate PNG binary image for data."""
    qr = QRCode(data)
    return qr.to_png(scale=scale, margin=margin)


def generate_qr_svg(data: str, margin: int = 4) -> str:
    """Convenience function to generate SVG string for data."""
    qr = QRCode(data)
    return qr.to_svg(margin=margin)


def parse_qr_svg(svg_content: str) -> List[List[bool]]:
    """Parse an SVG QR code back into a boolean matrix suitable for decode_qr."""
    root = ET.fromstring(svg_content)
    viewbox = root.attrib.get("viewBox", "").split()
    if len(viewbox) >= 4:
        total_dim = int(float(viewbox[2]))
    else:
        width = int(float(root.attrib.get("width", 0)))
        total_dim = width

    rect_coords: List[Tuple[int, int]] = []
    for elem in root.iter():
        if elem.tag.endswith("rect") and "x" in elem.attrib and "y" in elem.attrib:
            rect_coords.append((int(float(elem.attrib["x"])), int(float(elem.attrib["y"]))))

    if not rect_coords:
        raise ValueError("No QR code modules found in SVG")

    min_x = min(x for x, y in rect_coords)
    min_y = min(y for x, y in rect_coords)
    margin = min(min_x, min_y)
    size = total_dim - margin * 2

    matrix = [[False] * size for _ in range(size)]
    for x, y in rect_coords:
        r = y - margin
        c = x - margin
        if 0 <= r < size and 0 <= c < size:
            matrix[r][c] = True

    return matrix


def write_qr_file(payload: str, path: Union[str, Path]) -> Path:
    """Write QR code to target path (.png or .svg) with mode 0600."""
    abs_str = os.path.abspath(os.path.expanduser(str(path)))
    target_path = Path(abs_str)
    ext = target_path.suffix.lower()
    if ext not in (".png", ".svg"):
        raise ValueError(f"Invalid QR file extension '{ext}': must be .png or .svg")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".png":
        written = False
        if shutil.which("qrencode"):
            try:
                subprocess.run(
                    ["qrencode", "-o", str(target_path), "-l", "L", "-m", "4", payload],
                    check=True,
                    capture_output=True,
                )
                written = True
            except Exception:
                pass
        if not written:
            qr = QRCode(payload)
            target_path.write_bytes(qr.to_png())
    elif ext == ".svg":
        qr = QRCode(payload)
        target_path.write_text(qr.to_svg(), encoding="utf-8")

    try:
        os.chmod(target_path, 0o600)
    except Exception:
        pass

    return target_path


def decode_qr(modules: Sequence[Sequence[bool]]) -> str:
    """Reference pure-Python QR decoder for ISO/IEC 18004 QR symbols (Versions 1-6, Byte mode)."""
    size = len(modules)
    if (size - 17) % 4 != 0:
        raise ValueError(f"Invalid QR dimension: {size}")
    version = (size - 17) // 4
    if not (1 <= version <= 6):
        raise ValueError(f"Unsupported QR version: {version}")

    # Read format info bits from primary top-left strip (bit 14 down to 0)
    val = 0
    coords = (
        [(8, c) for c in range(6)]
        + [(8, 7), (8, 8), (7, 8)]
        + [(r, 8) for r in range(5, -1, -1)]
    )
    for i, (r, c) in enumerate(coords):
        bit = 1 if modules[r][c] else 0
        val |= (bit << (14 - i))

    raw = val ^ 0x5412
    data = raw >> 10
    ecc_bits = (data >> 3) & 3
    mask_idx = data & 7
    ecc_level = 0 if ecc_bits == 0 else 1

    spec = _VERSION_SPECS.get((version, ecc_level))
    if not spec:
        raise ValueError(f"Unsupported QR version/ECC spec: V{version} ECC {ecc_level}")
    total_codewords = spec[0]

    # Reconstruct function module mask
    is_func = [[False] * size for _ in range(size)]
    for r_orig, c_orig in [(0, 0), (0, size - 7), (size - 7, 0)]:
        for r in range(-1, 8):
            for c in range(-1, 8):
                if 0 <= r_orig + r < size and 0 <= c_orig + c < size:
                    is_func[r_orig + r][c_orig + c] = True

    # Alignment patterns
    for r_pos in spec[6]:
        for c_pos in spec[6]:
            if (r_pos < 9 and c_pos < 9) or (r_pos < 9 and c_pos > size - 9) or (r_pos > size - 9 and c_pos < 9):
                continue
            for r in range(-2, 3):
                for c in range(-2, 3):
                    is_func[r_pos + r][c_pos + c] = True

    # Timing patterns
    for i in range(size):
        is_func[6][i] = True
        is_func[i][6] = True

    # Dark module & format info
    is_func[4 * version + 9][8] = True
    for i in range(9):
        is_func[8][i] = True
        is_func[i][8] = True
    for i in range(8):
        is_func[8][size - 1 - i] = True
        is_func[size - 1 - i][8] = True

    mask_fn = {
        0: lambda r, c: (r + c) % 2 == 0,
        1: lambda r, c: r % 2 == 0,
        2: lambda r, c: c % 3 == 0,
        3: lambda r, c: (r + c) % 3 == 0,
        4: lambda r, c: ((r // 2) + (c // 3)) % 2 == 0,
        5: lambda r, c: ((r * c) % 2) + ((r * c) % 3) == 0,
        6: lambda r, c: (((r * c) % 2) + ((r * c) % 3)) % 2 == 0,
        7: lambda r, c: (((r + c) % 2) + ((r * c) % 3)) % 2 == 0,
    }[mask_idx]

    # Read unmasked data bits in 2-column zig-zag
    bits: List[int] = []
    row = size - 1
    col = size - 1
    direction = -1
    while col > 0:
        if col == 6:
            col -= 1
        for c_offset in range(2):
            curr_col = col - c_offset
            if not is_func[row][curr_col]:
                masked_val = bool(modules[row][curr_col])
                orig_val = masked_val ^ mask_fn(row, curr_col)
                bits.append(1 if orig_val else 0)
        row += direction
        if row < 0:
            row = 0
            direction = 1
            col -= 2
        elif row >= size:
            row = size - 1
            direction = -1
            col -= 2

    # Group into codewords
    codewords: List[int] = []
    for i in range(0, total_codewords * 8, 8):
        byte = 0
        for b in range(8):
            byte = (byte << 1) | bits[i + b]
        codewords.append(byte)

    num_blocks = spec[2] + spec[4]
    ec_len = spec[1]

    # De-interleave data codewords
    data_blocks: List[List[int]] = [[] for _ in range(num_blocks)]
    max_data = max(spec[3], spec[5] if spec[4] else 0)
    idx = 0
    for i in range(max_data):
        for b in range(num_blocks):
            b_len = spec[3] if b < spec[2] else spec[5]
            if i < b_len:
                data_blocks[b].append(codewords[idx])
                idx += 1

    # De-interleave error correction codewords
    ec_blocks: List[List[int]] = [[] for _ in range(num_blocks)]
    for i in range(ec_len):
        for b in range(num_blocks):
            ec_blocks[b].append(codewords[idx])
            idx += 1

    # Verify Reed-Solomon checksum for each block
    gen = _rs_generator_poly(ec_len)
    for b in range(num_blocks):
        msg = list(data_blocks[b]) + list(ec_blocks[b])
        for i in range(len(data_blocks[b])):
            lead = msg[i]
            if lead != 0:
                for j in range(len(gen)):
                    msg[i + j] ^= _gf_mul(lead, gen[len(gen) - 1 - j])
        rem = msg[len(data_blocks[b]):]
        if any(r != 0 for r in rem):
            raise ValueError(f"Reed-Solomon check failed for block {b}")

    # Concatenate data bytes
    data_bytes: List[int] = []
    for b in data_blocks:
        data_bytes.extend(b)

    # Decode Byte mode stream
    data_bits: List[int] = []
    for byte in data_bytes:
        for b in range(7, -1, -1):
            data_bits.append((byte >> b) & 1)

    bit_pos = 0
    mode = 0
    for _ in range(4):
        mode = (mode << 1) | data_bits[bit_pos]
        bit_pos += 1
    if mode != 4:
        raise ValueError(f"Unsupported QR mode indicator: {mode} (only Byte mode 4 supported)")

    count = 0
    for _ in range(8):
        count = (count << 1) | data_bits[bit_pos]
        bit_pos += 1

    out = bytearray()
    for _ in range(count):
        char = 0
        for _ in range(8):
            char = (char << 1) | data_bits[bit_pos]
            bit_pos += 1
        out.append(char)

    return out.decode("utf-8")


def decode_qr_ascii(ascii_text: str) -> str:
    """Decode a QR code rendered as terminal ASCII half-blocks."""
    clean_lines = [
        re.sub(r"\x1b\[[0-9;]*m", "", line).strip("\r")
        for line in ascii_text.splitlines()
    ]
    clean_lines = [
        line for line in clean_lines
        if line and set(line) <= set(" █▄▀")
    ]
    if not clean_lines:
        raise ValueError("No QR code ASCII lines found")

    grid: List[List[bool]] = []
    for line in clean_lines:
        row_top = []
        row_bot = []
        for ch in line:
            if ch == " ":
                row_top.append(True)
                row_bot.append(True)
            elif ch == "█":
                row_top.append(False)
                row_bot.append(False)
            elif ch == "▄":
                row_top.append(True)
                row_bot.append(False)
            elif ch == "▀":
                row_top.append(False)
                row_bot.append(True)
        if row_top:
            grid.append(row_top)
            grid.append(row_bot)

    full_cols = len(grid[0])
    first_dark_col = 0
    while first_dark_col < full_cols and not any(grid[r][first_dark_col] for r in range(len(grid))):
        first_dark_col += 1

    first_dark_row = 0
    while first_dark_row < len(grid) and not any(grid[first_dark_row]):
        first_dark_row += 1

    border = min(first_dark_col, first_dark_row)
    size = full_cols - border * 2
    modules = [grid[border + r][border : border + size] for r in range(size)]
    return decode_qr(modules)

