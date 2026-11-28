"""Integrated loudness (ITU-R BS.1770-4, EBU R128 gating) for 48 kHz stereo."""

import math

# K-weighting at 48 kHz: high shelf, then high-pass (coefficients from BS.1770).
_SHELF = ((1.53512485958697, -2.69169618940638, 1.19839281085285), (-1.69065929318241, 0.73248077421585))
_HIGHPASS = ((1.0, -2.0, 1.0), (-1.99004745483398, 0.99007225036621))


def _biquad(x, coeffs):
    (b0, b1, b2), (a1, a2) = coeffs
    y = [0.0] * len(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, xi, y1, yi
        y[i] = yi
    return y


def _kernels(taps=8):
    out = []
    for frac in (0.25, 0.5, 0.75):
        k = []
        for j in range(-taps + 1, taps + 1):
            d = j - frac
            k.append((j, math.sin(math.pi * d) / (math.pi * d) * 0.5 * (1 + math.cos(math.pi * d / taps))))
        out.append(k)
    return out


_TP_KERNELS = _kernels()


def true_peak(x, near=0.7):
    """Linear true peak: 4x windowed-sinc interpolation around samples above near × sample peak."""
    peak = max(map(abs, x)) or 0.0
    lim = near * peak
    tp = peak
    for i in range(8, len(x) - 8):
        if abs(x[i]) < lim and abs(x[i + 1]) < lim:
            continue
        for kernel in _TP_KERNELS:
            v = abs(sum(c * x[i + j] for j, c in kernel))
            tp = max(tp, v)
    return tp


def integrated_lufs(left, right, sr=48000):
    """Integrated loudness in LUFS; -inf for silence."""
    if sr != 48000:
        raise ValueError("K-weighting coefficients here are for 48 kHz")
    kl = _biquad(_biquad(left, _SHELF), _HIGHPASS)
    kr = _biquad(_biquad(right, _SHELF), _HIGHPASS)

    cs = [0.0]
    acc = 0.0
    for a, b in zip(kl, kr):
        acc += a * a + b * b
        cs.append(acc)

    block, hop = int(0.4 * sr), int(0.1 * sr)
    z = [(cs[i + block] - cs[i]) / block for i in range(0, len(kl) - block + 1, hop)]
    lufs = lambda power: -0.691 + 10 * math.log10(power)

    above_abs = [p for p in z if p > 0 and lufs(p) > -70]
    if not above_abs:
        return float("-inf")
    relative = lufs(sum(above_abs) / len(above_abs)) - 10
    gated = [p for p in above_abs if lufs(p) > relative]
    return lufs(sum(gated) / len(gated))
