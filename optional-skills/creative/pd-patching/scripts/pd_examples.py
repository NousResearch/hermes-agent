#!/usr/bin/env python3
"""A gallery of ready-to-run Pure Data patches for the dynamic-patching skill.

Every example is a signal-rate patch that starts making sound the moment DSP is
on -- no metro, toggle, or mouse click needed. Each returns a
:class:`pd_patch.PdPatch`, so it is auto-laid-out, safety-linted (master gain
<= 0.1, no signal object wired straight into ``[dac~]``), and ready to hand to
:class:`pd_client.PureDataClient`.

Try one (with Pd running the receiver stub):

    python3 pd_examples.py --list
    python3 pd_examples.py fm_bell            # print the messages
    python3 pd_examples.py wobble_bass --send # build, safety-check, and play it
    python3 pd_examples.py random_blips | python3 pd_client.py -

Standard library only.
"""
from __future__ import annotations

from pd_patch import PdPatch, build_simple_synth


# --- the gallery -----------------------------------------------------------

def sines(freqs=(220, 440, 660)) -> PdPatch:
    """Mix any number of sine waves (default a major triad).

    Handles the "play a 120/254/1201 Hz chord" style request directly: pass the
    frequencies and it wires ``osc~`` per tone, chain-sums them with ``+~``, and
    scales through a safe master ``[*~ 0.1]``.
    """
    freqs = list(freqs) or [440]
    p = PdPatch()
    p.clear()
    oscs = [p.obj("osc~", f) for f in freqs]
    summed = oscs[0]
    for o in oscs[1:]:                 # chain-sum: (((o0 + o1) + o2) + ...)
        nxt = p.obj("+~")
        p.connect(summed, 0, nxt, 0)
        p.connect(o, 0, nxt, 1)
        summed = nxt
    p.to_dac(summed, 0.1)              # hip~ 5 -> *~ 0.1 -> clip~ -1 1 -> dac~
    p.dsp(True)
    return p


def fm_bell(carrier: float = 220, ratio: float = 1.414, index: float = 260) -> PdPatch:
    """Metallic FM tone: a modulator bends the carrier's frequency.

    carrier*ratio sets the modulator pitch; ``index`` is the depth of the
    frequency swing. Irrational ratios (~1.414) give clangorous, bell-like
    timbres; whole-number ratios sound more harmonic.
    """
    p = PdPatch()
    p.clear()
    mod = p.obj("osc~", round(carrier * ratio, 3))  # modulator
    depth = p.obj("*~", index)                       # modulation index
    freq = p.obj("+~", carrier)                      # carrier base + swing
    car = p.obj("osc~")                              # carrier (freq via inlet)
    p.connect(mod, 0, depth, 0)
    p.connect(depth, 0, freq, 0)
    p.connect(freq, 0, car, 0)      # signal frequency into carrier's left inlet
    p.to_dac(car, 0.1)              # hip~ 5 -> *~ 0.1 -> clip~ -1 1 -> dac~
    p.dsp(True)
    return p


def wobble_bass(freq: float = 55, rate: float = 2, depth: float = 600,
                base: float = 700, q: float = 8) -> PdPatch:
    """Dubstep 'wob': a buzzy saw bass swept by an LFO-driven band-pass filter.

    ``rate`` is the wobble speed in Hz; ``depth``/``base`` set how far and where
    the filter's centre frequency sweeps. [vcf~] is used because its cutoff can
    be driven by a signal (unlike [lop~]).
    """
    p = PdPatch()
    p.clear()
    saw = p.obj("phasor~", freq)      # buzzy saw bass
    centred = p.obj("-~", 0.5)        # centre the ramp around 0
    lfo = p.obj("osc~", rate)         # wobble LFO
    swing = p.obj("*~", depth)
    cutoff = p.obj("+~", base)        # cutoff = base + LFO*depth
    filt = p.obj("vcf~", q)           # band-pass; centre freq via middle inlet
    p.connect(saw, 0, centred, 0)
    p.connect(centred, 0, filt, 0)    # audio into vcf~ left inlet
    p.connect(lfo, 0, swing, 0)
    p.connect(swing, 0, cutoff, 0)
    p.connect(cutoff, 0, filt, 1)     # cutoff signal into vcf~ middle inlet
    p.to_dac(filt, 0.1)               # hip~ 5 -> *~ 0.1 -> clip~ -1 1 -> dac~
    p.dsp(True)
    return p


def supersaw_drone(freq: float = 110, detune: float = 0.6) -> PdPatch:
    """Fat detuned drone: three saws a few cents apart, mixed and smoothed.

    The slight ``detune`` between the three [phasor~]s makes them drift in and
    out of phase, giving the thick, shimmering 'supersaw' character.
    """
    p = PdPatch()
    p.clear()
    a = p.obj("phasor~", freq)
    b = p.obj("phasor~", round(freq + detune, 3))
    c = p.obj("phasor~", round(freq - detune, 3))
    mix1 = p.obj("+~")
    mix2 = p.obj("+~")
    smooth = p.obj("lop~", 2500)      # tame the buzz
    p.connect(a, 0, mix1, 0)
    p.connect(b, 0, mix1, 1)
    p.connect(mix1, 0, mix2, 0)
    p.connect(c, 0, mix2, 1)
    p.connect(mix2, 0, smooth, 0)
    p.to_dac(smooth, 0.08)            # 3 voices -> keep it quiet; hip~/clip~ guard
    p.dsp(True)
    return p


def random_blips(speed: float = 5, low: float = 300, span: float = 500) -> PdPatch:
    """Generative blip melody -- a new random pitch ``speed`` times a second.

    [samphold~] latches a fresh sample of white noise on every falling edge of
    the [phasor~] clock, so the pitch jumps to a new random value with no metro
    or trigger. ``low``/``span`` map the -1..1 sample to a frequency range.
    """
    p = PdPatch()
    p.clear()
    clock = p.obj("phasor~", speed)   # falling edge = new note
    noise = p.obj("noise~")
    hold = p.obj("samphold~")         # sample noise on clock's falling edge
    scale = p.obj("*~", span / 2.0)
    offset = p.obj("+~", low + span / 2.0)
    tone = p.obj("osc~")              # pitch from the held signal
    p.connect(noise, 0, hold, 0)      # value to sample
    p.connect(clock, 0, hold, 1)      # sample-and-hold trigger
    p.connect(hold, 0, scale, 0)
    p.connect(scale, 0, offset, 0)
    p.connect(offset, 0, tone, 0)
    p.to_dac(tone, 0.1)               # hip~ 5 -> *~ 0.1 -> clip~ -1 1 -> dac~
    p.dsp(True)
    return p


def resonator(pitch: float = 220, feedback: float = 0.92) -> PdPatch:
    """Plucked/blown string drone: white noise poured into a tuned delay loop.

    A short [delwrite~]/[delread~] feedback loop tuned to ``1000/pitch`` ms acts
    as a Karplus-Strong-style resonator; the [lop~] in the loop rolls off the
    highs so it sings rather than buzzes. ``feedback`` stays below 1 to keep the
    loop stable.
    """
    delay_ms = round(1000.0 / pitch, 4)
    line = f"pdres{int(pitch)}"       # unique delay-line name
    p = PdPatch()
    p.clear()
    noise = p.obj("noise~")
    exc = p.obj("*~", 0.03)           # excitation level
    mix = p.obj("+~")                 # excitation + feedback
    damp = p.obj("lop~", 3000)        # loop damping
    write = p.obj("delwrite~", line, 50)
    read = p.obj("delread~", line, delay_ms)
    fb = p.obj("*~", feedback)        # feedback gain (<1)
    p.connect(noise, 0, exc, 0)
    p.connect(exc, 0, mix, 0)
    p.connect(mix, 0, damp, 0)
    p.connect(damp, 0, write, 0)
    p.connect(read, 0, fb, 0)
    p.connect(fb, 0, mix, 1)          # close the feedback loop
    p.to_dac(read, 0.1)               # hip~ 5 -> *~ 0.1 -> clip~ -1 1 -> dac~
    p.dsp(True)
    return p


# Registry: name -> (builder, one-line description)
EXAMPLES = {
    "sine": (build_simple_synth, "Plain 440 Hz sine -- the 'hello world' synth."),
    "sines": (sines, "Mix N sine waves; set them with --freqs 120 254 1201."),
    "fm_bell": (fm_bell, "Metallic FM bell tone."),
    "wobble_bass": (wobble_bass, "Dubstep LFO-swept band-pass bass."),
    "supersaw_drone": (supersaw_drone, "Fat detuned three-saw drone."),
    "random_blips": (random_blips, "Generative random-pitch blip melody."),
    "resonator": (resonator, "Karplus-Strong plucked/blown string drone."),
}


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and (optionally) play example Pure Data patches.",
    )
    parser.add_argument("name", nargs="?", help="Example to build (see --list)")
    parser.add_argument("--list", action="store_true", help="List all examples")
    parser.add_argument(
        "--freqs",
        type=float,
        nargs="+",
        help="Frequencies (Hz) for the 'sines' example, e.g. --freqs 120 254 1201",
    )
    parser.add_argument("--send", action="store_true", help="Send it to a running Pd")
    parser.add_argument(
        "--export", metavar="PATH",
        help="Write a standalone, shareable .pd file (no Pd needed) and exit",
    )
    parser.add_argument(
        "--wav", metavar="PATH",
        help="Record the output to a WAV (implies --send; needs Pd with audio on)",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="With --wav: record this many seconds, then stop and flush the file",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args(argv)

    if args.list or not args.name:
        width = max(len(k) for k in EXAMPLES)
        for name, (_, desc) in EXAMPLES.items():
            print(f"  {name:<{width}}  {desc}")
        return 0

    if args.name not in EXAMPLES:
        print(f"unknown example: {args.name!r} (try --list)")
        return 2

    import os
    import sys

    if args.name == "sines" and args.freqs:
        patch = sines(args.freqs)
    else:
        patch = EXAMPLES[args.name][0]()

    # --export writes a standalone .pd file; no Pd or safety-send needed.
    if args.export:
        path = os.path.abspath(args.export)
        patch.to_pd_file(path)
        print(f"exported {args.name} -> {path}")
        return 0

    if args.wav:
        patch.add_recorder()          # tap the limiter into [writesf~]

    violations = patch.safety_violations()
    if violations:
        print("SAFETY VIOLATIONS (not sending):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    if args.wav:
        import time
        from pd_client import PureDataClient

        if args.seconds is not None and args.seconds <= 0:
            print("--seconds must be a positive duration", file=sys.stderr)
            return 2
        client = PureDataClient(host=args.host, port=args.port)
        if not client.send_batch(patch.messages()):
            print(f"failed to send {args.name} to Pd", file=sys.stderr)
            return 1
        wav = os.path.abspath(args.wav)
        # If the patch uploaded but the open/start control fails to land, the
        # WAV never opens -- surface that instead of reporting success.
        if not client.send_batch(patch.record_messages(wav)):   # open + start
            print("failed to start recording (open/start not delivered)",
                  file=sys.stderr)
            return 1
        print(f"recording {args.name} -> {wav}")
        if args.seconds:
            time.sleep(args.seconds)
            if not client.send("pd-rec stop"):
                print("failed to send stop; the WAV may be truncated",
                      file=sys.stderr)
                return 1
            print(f"stopped after {args.seconds}s")
        else:
            print('stop with: python3 pd_client.py "pd-rec stop"')
        return 0

    if args.send:
        from pd_client import PureDataClient

        client = PureDataClient(host=args.host, port=args.port)
        ok = client.send_batch(patch.messages())
        print(f"sent {args.name}: {ok}")
        return 0 if ok else 1

    for line in patch.messages():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
