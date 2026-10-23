"""
Pure-Python QR Code generator with zero external dependencies.
Generates ISO/IEC 18004 compliant QR codes (Versions 1-6, Byte mode, ECC L and M)
and formats them for terminal ASCII/ANSI display.
"""

from __future__ import annotations

from typing import Final, List, Optional, Sequence, Tuple


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
    (6, 1): (172, 36, 2, 68, 0, 0, (6, 34)),
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

        # Write 15 bits of format info
        # Top-left and Split around finder patterns
        format_bits = [(format_val >> i) & 1 for i in range(15)]

        # Position mapping
        # Bits 0-5 -> (8, 0..5)
        for i in range(6):
            self.modules[8][i] = bool(format_bits[i])
        # Bit 6 -> (8, 7)
        self.modules[8][7] = bool(format_bits[6])
        # Bit 7 -> (8, 8)
        self.modules[8][8] = bool(format_bits[7])
        # Bit 8 -> (7, 8)
        self.modules[7][8] = bool(format_bits[8])
        # Bits 9-14 -> (5..0, 8)
        for i in range(6):
            self.modules[5 - i][8] = bool(format_bits[9 + i])

        # Second copy:
        # Bits 0-7 -> (size-1..size-8, 8)
        for i in range(8):
            self.modules[self.size - 1 - i][8] = bool(format_bits[i])
        # Bits 8-14 -> (8, size-7..size-1)
        for i in range(7):
            self.modules[8][self.size - 7 + i] = bool(format_bits[8 + i])

    def to_ascii(self) -> str:
        """Render QR code as ASCII with double-blocks for terminal display."""
        border = 2
        full_size = self.size + border * 2
        lines = []

        # Terminal characters: use half blocks (▀, ▄, █, ' ') to make it square and compact
        # Two vertical QR modules per character row!
        grid: List[List[bool]] = [[False] * full_size for _ in range(full_size)]
        for r in range(self.size):
            for c in range(self.size):
                if self.modules[r][c]:
                    grid[r + border][c + border] = True

        # Render rows two-by-two
        # Black modules in QR should appear as dark, white as light.
        # In a standard dark terminal:
        # ' ' is black/background, '█' is white/foreground,
        # '▀' has top white/bottom black, '▄' has top black/bottom white.
        # Since QR codes require dark modules on white background, inverted rendering is best:
        # Background is white (quiet zone).
        for r in range(0, full_size, 2):
            line = []
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
            lines.append("".join(line))

        return "\n".join(lines)


def generate_qr_ascii(data: str) -> str:
    """Convenience function to generate terminal ASCII QR code."""
    qr = QRCode(data)
    return qr.to_ascii()
