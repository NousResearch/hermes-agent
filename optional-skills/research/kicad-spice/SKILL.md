---
name: kicad-spice
description: Design and verify passive RC and RLC analog filters.
version: 1.0.0
author: Snehal (Snehal707)
license: MIT
platforms: [linux, macos, windows]
tags: [analog-filters, circuit-verification, spice-simulation]
---

# KiCad SPICE Skill

Design passive RC and RLC filters by computing practical component values,
simulating their response, and checking frequency error.
Optional KiCad exports are starter Gerbers, not production-routed boards;
this procedure does not certify hardware for manufacture or safety.

## When to Use

- Select a low-pass, high-pass, band-pass, or notch topology.
- Translate a target frequency into practical resistor, capacitor, and inductor values.
- Verify a proposed filter with PySpice/Ngspice before creating KiCad artifacts.
- Review loading, component rounding, or tolerance effects on a passive design.

## Prerequisites

Use the Hermes `terminal` tool for local computation and simulation.
Provide a Python environment with PySpice and a compatible Ngspice shared library,
or a working public `ngspice` executable for batch netlist simulation.
Check the chosen Python interpreter can import PySpice before using it;
an arbitrary system Python may not have the required packages.

KiCad and `kicad-cli` are optional and needed only for board/Gerber export.
No API key or remote service is required.
Keep generated netlists, measurements, plots, and export files in a dedicated
project output directory, separate from the installed skill.

## How to Run

Load this skill, then use `terminal` to check the selected tools and execute
the calculation or simulation in the user's project directory.
A request such as "verify a 1 kΩ, 100 nF low-pass" needs no PCB export
unless the user requests it.

For a public Ngspice batch run, save this minimal example as
`outputs/rc_lowpass.cir`, then invoke
`ngspice -b -o outputs/rc_lowpass.log outputs/rc_lowpass.cir`
through `terminal`:

```spice
RC low-pass, unloaded output
V1 in 0 AC 1
R1 in out 1k
C1 out 0 100n
.temp 25
.control
ac dec 100 1 1Meg
wrdata outputs/rc_ac.dat v(out)
.endc
.end
```

Create the output directory before running the example from the project root.
Alternatively construct the same circuit with PySpice and run its AC analysis.
Preserve the frequency and complex response data for numerical verification;
a successful process exit alone does not prove the filter meets its target.

## Quick Reference

| Topology | Connection / measurement | Frequency |
| --- | --- | --- |
| RC low-pass | Series R, shunt C; output across C | `fc = 1/(2*pi*R*C)` |
| RC high-pass | Series C, shunt R; output across R | `fc = 1/(2*pi*R*C)` |
| Series RLC band-pass | Series RLC; output across R | `f0 = 1/(2*pi*sqrt(L*C))` |
| RLC notch | Series R feeding shunt series LC; output at junction | `f0 = 1/(2*pi*sqrt(L*C))` |

Read [filter_math.md](references/filter_math.md) for transfer functions,
E24 values, bandwidth units, and scaling before choosing or changing values.
Use SI units in calculations and explicit units in the report.
For `R = 1000 Ω` and `C = 100 nF`, the ideal RC cutoff is about `1591.5 Hz`.

## Procedure

1. Record the response type, target cutoff or center frequency, source and load
   impedances, signal amplitude, component constraints, and acceptable error.
   If the frequency target is missing, explain a proposed target and its
   application assumptions before treating it as a requirement.
2. Select the topology from the table. For RLC filters also specify bandwidth
   or Q: a resonance frequency alone does not determine all three components.
   Draw or describe the nodes so the measured output is unambiguous.
3. Compute ideal values. For RC, choose a practical capacitor and solve
   `R = 1/(2*pi*fc*C)`. Compare neighboring E24 resistors across decade
   boundaries and pick the value with the smallest acceptable frequency error.
4. Recompute theory with the selected values, including any rounded L or C.
   Account for source resistance, finite load impedance, capacitor ESR, and
   inductor winding resistance where they materially affect the response.
5. Build the chosen circuit in PySpice or a plain Ngspice netlist. Use a
   small-signal AC source and sweep at least a decade either side of the
   expected transition; widen it until the passband and stopband are visible.
   A 100-point-per-decade sweep from 1 Hz to 1 MHz is a useful initial range.
6. Measure RC cutoff at `20*log10(1/sqrt(2))` (about -3.0103 dB) relative
   to the passband gain. Bracket the crossing and interpolate in log frequency,
   then refine the sweep around it. Do not select the nearest sample blindly.
7. For band-pass, measure the peak and both half-power crossings; report
   center frequency, bandwidth in hertz, and Q. For notch, locate the minimum
   and report notch depth and its reference gain. Do not confuse a resonance
   or notch minimum with a single RC cutoff.
8. Calculate `error_pct = 100*abs(measured_hz-target_hz)/target_hz`.
   Separately compare simulation to theory so component rounding and model
   disagreement remain distinguishable. Use the requested tolerance for
   pass/fail; if absent, state an assumed tolerance, such as 5%.
9. Run a transient check when waveform behavior matters. Choose a timestep
   small enough to resolve the highest signal frequency and a duration long
   enough to settle; arbitrary fixed settings may hide high-Q ringing.
10. If requested, create or inspect a KiCad schematic/board with footprints,
    board outline, placement, and connectivity matching the simulated circuit.
    Run electrical/design-rule checks and inspect their reports before export.
    Through `terminal`, check `kicad-cli pcb export gerbers --help` and
    `kicad-cli pcb export drill --help`, then export the actual board with
    the required copper, mask, silkscreen, outline, and drill layers.
    Inspect exported files and report unrouted nets or unresolved violations.

## Pitfalls

- Sweep and Monte Carlo helpers, such as `sweep_optimizer.py` and
  `monte_carlo.py`-style tools, are separate CLIs when available.
  Invoke them explicitly through `terminal`; they are not called inside
  a `run()` pipeline, and this skill does not bundle or assume those scripts.
- E24 snapping changes the predicted frequency. Component tolerance and
  capacitor DC-bias derating can exceed the remaining nominal error.
- An unloaded formula can disagree with a loaded simulation for valid reasons.
  Document the model instead of adjusting results to force a passing verdict.
- Missing Ngspice/PySpice prevents simulation verification. Report the
  dependency failure and label calculated results as theory-only.
- Missing KiCad should skip optional export without discarding valid simulation.
  Gerber generation does not route a board or establish manufacturability.
- Mains, medical, high-voltage, or life-safety applications require qualified
  engineering and safety review; simulation is not a safety certification.

## Verification

Confirm node connections, units, passband reference, and measurement method.
Check the 1 kΩ / 100 nF RC case against approximately 1591.5 Hz when validating
an unloaded model; use the measured result rather than copying this number.
Report target, theory, simulated frequency, percentage error, tolerance, and
pass/fail, together with component values and source/load assumptions.

Link the saved simulation data and any plots, netlist, board, or Gerbers.
State which artifacts were actually generated and which checks were skipped.
Label any PCB output "starter Gerbers, not production-routed" and list
unresolved routing or design-rule issues before describing it as ready for review.
