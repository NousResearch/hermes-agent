# RC and RLC Filter Math

Use these equations for first-pass calculations and comparison with simulation.
Unless stated otherwise, assume ideal components, zero source impedance,
and an unloaded output. Use ohms, farads, henries, and seconds throughout.

## RC Low-Pass

Series R feeding shunt C, with output across C:

```text
H(s) = 1 / (1 + sRC)
fc = 1 / (2*pi*R*C)
R = 1 / (2*pi*fc*C)
```

At fc, the magnitude is 1/sqrt(2), or about -3.0103 dB relative
to passband. Above cutoff, the slope approaches -20 dB/decade.

## RC High-Pass

Series C feeding shunt R, with output across R:

```text
H(s) = sRC / (1 + sRC)
fc = 1 / (2*pi*R*C)
```

At fc, the magnitude is about -3.0103 dB relative to passband.
Below cutoff, the magnitude rises at 20 dB/decade toward the passband.

For either ideal RC topology, R = 1000 ohms and C = 100 nF give:

```text
RC = 0.0001 seconds
fc = 1591.5494309 Hz
```

Finite source or load resistance changes the transfer function.
For a low-pass with series resistance Rs + R and shunt load RL:

```text
H(0) = RL / (Rs + R + RL)
fc = 1 / (2*pi*((Rs + R) || RL)*C)
A || B = A*B / (A+B)
```

Measure the cutoff relative to the actual passband gain.

## Series RLC Band-Pass

For a series RLC circuit driven by an ideal voltage source, measure
the output across R:

```text
H(s) = sRC / (s*s*L*C + sRC + 1)
omega0 = 1 / sqrt(L*C)                 radians/second
f0 = 1 / (2*pi*sqrt(L*C))              Hz
Q = sqrt(L/C) / R
delta_omega = R / L                   radians/second
BW = f_high - f_low = R / (2*pi*L)     Hz
Q = f0 / BW
f0 = sqrt(f_low*f_high)
```

The bandwidth endpoints are the half-power frequencies relative to the peak.
Include all series loss resistance when calculating Q; extra losses also
reduce peak gain when the output is measured across only the load resistor.

## RLC Notch

For series R feeding a shunt series LC branch, measure the junction voltage.
With ideal components and no additional load:

```text
H(s) = (s*s*L*C + 1) / (s*s*L*C + sRC + 1)
f0 = 1 / (2*pi*sqrt(L*C))
Q = sqrt(L/C) / R
```

The ideal numerator is zero at f0. Real notch depth depends on source/load
impedance, inductor winding resistance, capacitor ESR, and parasitics.
Other RLC connections require their own transfer function and Q definition.

## E24 Standard Values

One decade of nominal resistor values:

```text
10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30,
33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91
```

Multiply by powers of ten for other decades. Compare neighboring values,
including the next decade boundary, using the resulting frequency error.
Recompute after snapping; the closest resistance is not always the choice
with the lowest relative cutoff error.

## Frequency Scaling and Error

- Doubling R or C halves RC cutoff when the other value is fixed.
- Quadrupling L or C halves RLC resonance when the other value is fixed.
- Scaling L and C by the same factor preserves ideal series Q for fixed R.
- First-order RC responses approach 20 dB/decade per pole.
- For small RC variations, d(fc)/fc is approximately -(dR/R + dC/C).
- For small RLC variations, d(f0)/f0 is approximately -(dL/L + dC/C)/2.

```text
target_error_pct = 100 * abs(measured_hz - target_hz) / target_hz
theory_error_pct = 100 * abs(measured_hz - theory_hz) / theory_hz
```

Require a positive frequency target and positive component values.
Use tolerances and loading assumptions separately from nominal rounding error.
