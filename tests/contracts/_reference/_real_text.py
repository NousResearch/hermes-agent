"""SQLite's text spelling of a REAL, as a TEXT column stores a float bound into it, computed in Python.

SQLite renders a REAL with ``%!.17g``: its own approximate decimal digits, cut to the shortest of 17 significant digits that reads back as the same double when the cut drops at least three trailing zeros or nines, ``.0`` kept on a whole number, an exponent of at least two digits, ``Inf`` for infinity; NaN binds as NULL. The arithmetic below is SQLite's, step for step, so the spelling matches it digit for digit, last digit included.
"""

from __future__ import annotations

import math
import struct
from typing import Optional

__all__ = ["real_as_text"]

_U64 = (1 << 64) - 1
_BIT63 = 1 << 63

# The most significant 64 bits of 10**p for p in 0..26.
_BASE = (
    0x8000000000000000, 0xA000000000000000, 0xC800000000000000, 0xFA00000000000000, 0x9C40000000000000, 0xC350000000000000,
    0xF424000000000000, 0x9896800000000000, 0xBEBC200000000000, 0xEE6B280000000000, 0x9502F90000000000, 0xBA43B74000000000,
    0xE8D4A51000000000, 0x9184E72A00000000, 0xB5E620F480000000, 0xE35FA931A0000000, 0x8E1BC9BF04000000, 0xB1A2BC2EC5000000,
    0xDE0B6B3A76400000, 0x8AC7230489E80000, 0xAD78EBC5AC620000, 0xD8D726B7177A8000, 0x878678326EAC9000, 0xA968163F0A57B400,
    0xD3C21BCECCEDA100, 0x84595161401484A0, 0xA56FA5B99019A5C8,
)
# The most significant 64 bits, then the next 32, of 10**(27*g) for g in -13..12 (index g + 13; index 13 holds 10**-1).
_SCALE = (
    0x8049A4AC0C5811AE, 0xCF42894A5DCE35EA, 0xA76C582338ED2621, 0x873E4F75E2224E68, 0xDA7F5BF590966848, 0xB080392CC4349DEC,
    0x8E938662882AF53E, 0xE65829B3046B0AFA, 0xBA121A4650E4DDEB, 0x964E858C91BA2655, 0xF2D56790AB41C2A2, 0xC428D05AA4751E4C,
    0x9E74D1B791E07E48, 0xCCCCCCCCCCCCCCCC, 0xCECB8F27F4200F3A, 0xA70C3C40A64E6C51, 0x86F0AC99B4E8DAFD, 0xDA01EE641A708DE9,
    0xB01AE745B101E9E4, 0x8E41ADE9FBEBC27D, 0xE5D3EF282A242E81, 0xB9A74A0637CE2EE1, 0x95F83D0A1FB69CD9, 0xF24A01A73CF2DCCF,
    0xC3B8358109E84F07, 0x9E19DB92B4E31BA9,
)
_SCALE_LO = (
    0x205B896D, 0x52064CAD, 0xAF2AF2B8, 0x5A7744A7, 0xAF39A475, 0xBD8D794E, 0x547EB47B, 0x0CB4A5A3, 0x92F34D62, 0x3A6A07F9,
    0xFAE27299, 0xAA97E14C, 0x775EA265, 0xCCCCCCCC, 0x00000000, 0x999090B6, 0x69A028BB, 0xE80E6F48, 0x5EC05DD0, 0x14588F14,
    0x8F1668C9, 0x6D953E2C, 0x4ABDAF10, 0xBC633B39, 0x0A862F81, 0x6C07A2C2,
)


def _power_of_ten(p: int) -> tuple[int, int]:
    """The most significant 64 bits of 10**p, and the next 32."""
    if p == -1:
        return _SCALE[13], _SCALE_LO[13]
    if 0 <= p < 27:
        return _BASE[p], 0
    g = -(-p // 27) if p < 0 else p // 27
    n = p - 27 * g
    if n < 0:
        g, n = g - 1, n + 27
    scale, scale_lo = _SCALE[g + 13], _SCALE_LO[g + 13]
    if n == 0:
        return scale, scale_lo
    product = scale * _BASE[n] + ((scale_lo * _BASE[n]) >> 32)
    x, lo = product >> 64, (product >> 32) & 0xFFFFFFFF
    if not x & _BIT63:
        x, lo = ((x << 1) | ((lo >> 31) & 1)) & _U64, ((lo << 1) | 1) & 0xFFFFFFFF
    return x, lo


def _pow10_to_pow2(p: int) -> int:
    return (p * 108853) >> 15


def _pow2_to_pow10(p: int) -> int:
    return (p * 78913) >> 18


def _to_decimal(m: int, e: int) -> tuple[int, int]:
    """``(d, p)`` with ``m * 2**e`` about ``d * 10**p`` and ``d`` holding 18 significant digits; ``m`` has its top bit set."""
    p = 17 - _pow2_to_pow10(e + 63)
    h = (m * _power_of_ten(p)[0]) >> 64
    h >>= -(e + _pow10_to_pow2(p) + 2)
    return (h + ((h & 1) << 1)) >> 1, -p


def _to_double(d: int, p: int) -> float:
    """The double nearest ``d * 10**p`` as SQLite computes it."""
    if p < -348:
        return 0.0
    if p > 347:
        return math.inf
    b = d.bit_length()
    lp = _pow10_to_pow2(p)
    e = 53 - b - lp
    if e > 1074:
        if e >= 1130:
            return 0.0
        e = 1074
    s = -(e - (64 - b) + lp + 3)
    high, low = _power_of_ten(p)
    if low:
        high, low = (high + 1) & _U64, ~low & 0xFFFFFFFF
    x = (d << (64 - b)) & _U64
    product = x * high
    hi, mid1 = product >> 64, ((product & _U64) >> 32)
    sticky = 1
    if hi & ((1 << s) - 1) == 0:
        mid2 = ((x * (low << 32)) >> 96) & 0xFFFFFFFF
        sticky = 1 if (mid1 - mid2) & 0xFFFFFFFF > 1 else 0
        hi -= mid1 < mid2
    u = (hi >> s) | sticky
    if u >= (1 << 55) - 2:
        u, e = (u >> 1) | (u & 1), e - 1
    bits = (u + 1 + ((u >> 2) & 1)) >> 2
    if e <= -972:
        return math.inf
    if bits & (1 << 52):
        bits = (bits & ~(1 << 52)) | ((1075 - e) << 52)
    return struct.unpack("<d", struct.pack("<Q", bits & _U64))[0]


def _digits(r: float) -> tuple[str, int]:
    """The significant digits of a positive finite ``r`` at 17-digit precision, trailing zeros dropped, and the position of the decimal point."""
    bits = struct.unpack("<Q", struct.pack("<d", r))[0]
    e = (bits >> 52) & 0x7FF
    v = bits & 0x000FFFFFFFFFFFFF
    if e == 0:
        shift = 64 - v.bit_length()
        v, e = v << shift, -1074 - shift
    else:
        v, e = (v << 11) | _BIT63, e - 1086
    v, exp = _to_decimal(v, e)
    z = list(str(v))
    n = len(z)
    point = n + exp
    keep = 17
    if keep < n:
        if z[15] == "9" and z[14] == "9":
            jj = 14
            while jj > 0 and z[jj - 1] == "9":
                jj -= 1
            shorter = 1 if jj == 0 else int("".join(z[:jj])) + 1
            if r == _to_double(shorter, exp + n - jj):
                keep = jj + 1
        elif point >= n or z[13:16] == ["0", "0", "0"]:
            jj = 13
            while z[jj - 1] == "0":
                jj -= 1
            if r == _to_double(int("".join(z[:jj])), exp + n - jj):
                keep = jj + 1
        n = keep
        if z[keep] >= "5":
            j = keep - 1
            while True:
                if z[j] < "9":
                    z[j] = chr(ord(z[j]) + 1)
                    break
                z[j] = "0"
                if j == 0:
                    z.insert(0, "1")
                    n += 1
                    point += 1
                    break
                j -= 1
    while z[n - 1] == "0":
        n -= 1
    return "".join(z[:n]), point


def real_as_text(value: float) -> Optional[str]:
    """``value`` as SQLite stores a REAL in a TEXT column; ``None`` for NaN, which binds as NULL."""
    if math.isnan(value):
        return None
    sign = "-" if value < 0 else ""
    if math.isinf(value):
        return sign + "Inf"
    digits, point = _digits(-value if sign else value) if value else ("0", 1)
    exp = point - 1
    if exp < -4 or exp > 16:
        mantissa = digits[0] + "." + (digits[1:] or "0")
        return f"{sign}{mantissa}e{'-' if exp < 0 else '+'}{abs(exp):02d}"
    if point <= 0:
        whole, fraction = "0", "0" * -point + digits
    else:
        whole, fraction = digits[:point].ljust(point, "0"), digits[point:]
    return f"{sign}{whole}.{fraction or '0'}"
