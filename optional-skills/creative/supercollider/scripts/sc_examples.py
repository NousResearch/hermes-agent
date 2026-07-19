#!/usr/bin/env python3
"""A gallery of ready-to-play SuperCollider synths for the dynamic-SynthDef skill.

Every example is a continuous synth that makes sound the moment it starts -- no
sequencer, gate or trigger needed. Each returns a :class:`sc_synthdef.SynthDef`,
so it is safety-linted (master gain <= 0.2, no raw signal wired straight to
``Out``) and ready to hand to :class:`sc_client.SCClient`.

Try one (with scsynth booted -- see templates/boot_scsynth.sh):

    python3 sc_examples.py --list
    python3 sc_examples.py wobble_bass --play        # load + start it
    python3 sc_examples.py sines --freqs 120 254 1201 --play
    python3 sc_examples.py fm_bell --export bell.scsyndef   # shareable file

Standard library only.
"""
from __future__ import annotations

from functools import reduce

from sc_synthdef import AR, KR, SynthDef, build_simple_synth


# --- the gallery -----------------------------------------------------------

def sines(freqs=(220, 440, 660), name: str = "hermes_sines") -> SynthDef:
    """Mix any number of sine waves (default a major triad).

    Handles the "play a 120/254/1201 Hz chord" request directly: one SinOsc per
    tone, summed with ``+``, scaled by a safe master gain.
    """
    freqs = list(freqs) or [440]
    sd = SynthDef(name)
    oscs = [sd.sinosc(f) for f in freqs]
    mixed = reduce(lambda a, b: a + b, oscs)   # (((o0 + o1) + o2) + ...)
    # N summed sines peak near N; scale the master gain down so the mix stays
    # well under full scale (the Limiter still guards the rest).
    sd.to_out(mixed, min(0.2, 0.8 / len(freqs)))
    return sd


def fm_bell(carrier: float = 220, ratio: float = 1.414, index: float = 260,
            name: str = "hermes_fm_bell") -> SynthDef:
    """Metallic FM tone: a modulator bends the carrier's frequency.

    ``carrier * ratio`` is the modulator pitch; ``index`` is how far the carrier
    frequency swings. Irrational ratios (~1.414) give clangorous, bell-like
    timbres; whole-number ratios sound more harmonic.
    """
    sd = SynthDef(name)
    freq = sd.control("freq", carrier)
    mod = sd.sinosc(freq * ratio) * index          # modulator * depth
    car = sd.sinosc(freq + mod)                     # carrier, frequency-modulated
    sd.to_out(car, 0.2)
    return sd


def wobble_bass(freq: float = 55, rate: float = 2, depth: float = 1200,
                base: float = 700, rq: float = 0.2,
                name: str = "hermes_wobble") -> SynthDef:
    """Dubstep 'wob': a buzzy saw bass swept by an LFO-driven resonant low-pass.

    ``rate`` is the wobble speed (Hz); ``base``/``depth`` set where and how far
    the filter cutoff sweeps. RLPF's cutoff is driven by a control-rate LFO.
    """
    sd = SynthDef(name)
    f = sd.control("freq", freq)
    wob = sd.control("rate", rate)
    saw = sd.saw(f)
    lfo = sd.sinosc(wob, rate=KR)                   # -1..1 wobble
    cutoff = lfo * depth + base                     # base +/- depth
    filt = sd.rlpf(saw, cutoff, rq)
    sd.to_out(filt, 0.2)
    return sd


def supersaw_drone(freq: float = 110, detune: float = 0.6,
                   name: str = "hermes_supersaw") -> SynthDef:
    """Fat detuned drone: three saws a few cents apart, mixed and smoothed.

    The slight ``detune`` makes the three Saws drift in and out of phase, giving
    the thick, shimmering 'supersaw' character. A LPF tames the buzz.
    """
    sd = SynthDef(name)
    f = sd.control("freq", freq)
    a = sd.saw(f)
    b = sd.saw(f + detune)
    c = sd.saw(f - detune)
    mixed = (a + b + c) * 0.4
    smooth = sd.lpf(mixed, 2500)
    sd.to_out(smooth, 0.16)
    return sd


def random_blips(speed: float = 5, low: float = 300, span: float = 500,
                 name: str = "hermes_blips") -> SynthDef:
    """Generative blip melody -- a new random pitch ``speed`` times a second.

    LFNoise0 latches a fresh random value ``speed`` times/sec (no sequencer); it
    is mapped from -1..1 onto ``[low, low+span]`` to pick the pitch. This is the
    SC idiom that Pd needs a whole samphold/phasor rig for.
    """
    sd = SynthDef(name)
    rate = sd.control("speed", speed)
    note = sd.lfnoise0(rate) * (span / 2.0) + (low + span / 2.0)
    tone = sd.sinosc(note)
    sd.to_out(tone, 0.2)
    return sd


def resonator(pitch: float = 220, decay: float = 3.0,
              name: str = "hermes_resonator") -> SynthDef:
    """Plucked/blown string drone: noise poured into a tuned comb resonator.

    A short CombL tuned to ``1/pitch`` seconds with a long ``decay`` acts as a
    Karplus-Strong-style resonator; a LPF in front rolls off the highs so it
    sings rather than hisses. Continuous noise keeps it droning.
    """
    sd = SynthDef(name)
    f = sd.control("freq", pitch)
    delay = 1.0 / pitch
    exc = sd.lpf(sd.whitenoise() * 0.15, 4000)
    string = sd.combl(exc, maxdelay=0.05, delay=delay, decay=decay)
    sd.to_out(string, 0.2)
    return sd


# Registry: name -> (builder, one-line description)
EXAMPLES = {
    "sine": (lambda: build_simple_synth(), "Plain 440 Hz sine -- the 'hello world' synth."),
    "sines": (sines, "Mix N sine waves; set them with --freqs 120 254 1201."),
    "fm_bell": (fm_bell, "Metallic FM bell tone."),
    "wobble_bass": (wobble_bass, "Dubstep LFO-swept resonant bass."),
    "supersaw_drone": (supersaw_drone, "Fat detuned three-saw drone."),
    "random_blips": (random_blips, "Generative random-pitch blip melody."),
    "resonator": (resonator, "Karplus-Strong plucked/blown string drone."),
}


def _main(argv=None) -> int:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description="Build and (optionally) play example SuperCollider synths.",
    )
    parser.add_argument("name", nargs="?", help="Example to build (see --list)")
    parser.add_argument("--list", action="store_true", help="List all examples")
    parser.add_argument("--freqs", type=float, nargs="+",
                        help="Frequencies (Hz) for 'sines', e.g. --freqs 120 254 1201")
    parser.add_argument("--play", action="store_true", help="Load + start it on a running scsynth")
    parser.add_argument("--node", type=int, default=1000, help="Node id to create (default 1000)")
    parser.add_argument("--export", metavar="PATH",
                        help="Write a standalone, loadable .scsyndef file and exit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=57110)
    args = parser.parse_args(argv)

    if args.list or not args.name:
        width = max(len(k) for k in EXAMPLES)
        for nm, (_, desc) in EXAMPLES.items():
            print(f"  {nm:<{width}}  {desc}")
        return 0

    if args.name not in EXAMPLES:
        print(f"unknown example: {args.name!r} (try --list)", file=sys.stderr)
        return 2

    if args.name == "sines" and args.freqs:
        sd = sines(args.freqs)
    else:
        sd = EXAMPLES[args.name][0]()

    violations = sd.safety_violations()
    if violations:
        print("SAFETY VIOLATIONS (not playing):", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    if args.export:
        path = os.path.abspath(args.export)
        sd.to_scsyndef_file(path)
        print(f"exported {args.name} -> {path}")
        return 0

    if args.play:
        from sc_client import SCClient
        client = SCClient(host=args.host, port=args.port)
        try:
            nid = client.play(sd, node_id=args.node)
        except OSError as e:
            print(f"failed to reach scsynth at {args.host}:{args.port}: {e}",
                  file=sys.stderr)
            return 1
        ok, err = client.verify()
        if not ok:
            print(f"scsynth did not confirm playback ({err}); the SynthDef may "
                  "have been rejected, or the server is not booted "
                  "(see templates/boot_scsynth.sh)", file=sys.stderr)
            return 1
        print(f"playing {args.name} as node {nid} "
              f"(stop: python3 sc_client.py --free-all)")
        return 0

    # default: emit the compiled .scsyndef bytes on stdout
    sys.stdout.buffer.write(sd.compile())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
