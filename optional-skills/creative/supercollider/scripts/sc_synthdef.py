#!/usr/bin/env python3
"""Pure-Python SuperCollider SynthDef compiler + graph builder + safety linter.

SuperCollider splits into two programs: the language ``sclang`` (which usually
*compiles* SynthDefs) and the audio server ``scsynth`` (which plays them). The
usual path routes everything through ``sclang``. This module skips it: it emits
the **SynthDef v2 binary format** directly from Python, so a compiled synth graph
can be shipped to a bare ``scsynth`` over OSC (see ``sc_client.py``) with **no
sclang, no external packages -- standard library only.**

Build a graph the way you would patch modular gear::

    sd = SynthDef("hello")
    freq = sd.control("freq", 440)       # a live-settable parameter
    sig  = sd.sinosc(freq)               # SinOsc.ar(freq)
    sd.to_out(sig, 0.2)                  # LeakDC -> *0.2 -> Limiter -> Out.ar(0)
    blob = sd.compile()                  # -> bytes of a .scsyndef, ready for /d_recv

``OutRef`` overloads ``+ - * /`` so signal math reads naturally
(``sig = sd.sinosc(200) * 0.5 + sd.sinosc(300) * 0.5``); each operator emits a
``BinaryOpUGen`` behind the scenes.

The safety linter (:meth:`SynthDef.safety_violations`) enforces the same class of
ear/speaker protections the Pure Data skill does: every ``Out`` must be reached
through ``LeakDC -> [* <= MAX_MASTER_GAIN] -> Limiter``. Call :meth:`is_safe`
before you play.

Standard library only.
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple, Union

# -- calculation rates (SynthDef byte values) -------------------------------
IR = 0   # scalar   (a constant, computed once)
KR = 1   # control  (block rate, ~ every 64 samples)
AR = 2   # audio    (sample rate)
DR = 3   # demand

# -- BinaryOpUGen operator -> special index (from the SC operator table) ----
#   These indices are what scsynth uses to pick +, -, *, / at run time.
BINOP = {"+": 0, "-": 1, "*": 2, "/": 4}

# -- master-output safety (speaker / headphone / ear protection) ------------
# Every Out must be reached through this protective chain:
#   source -> LeakDC.ar -> [* <= MAX_MASTER_GAIN] -> Limiter.ar -> Out.ar
# LeakDC   removes DC offset / subsonics that waste headroom and over-excurse
#          woofers (the analog of Pd's [hip~ 5]).
# Limiter  is a look-ahead brick wall so transient overshoots never reach the
#          DAC as full-scale bursts (the analog of Pd's [clip~ -1 1]).
MAX_MASTER_GAIN = 0.2
DCBLOCK_OBJECT = "LeakDC"
LIMITER_OBJECTS = ("Limiter", "Clip")
OUT_OBJECT = "Out"
GAIN_OBJECT = "BinaryOpUGen"   # a '*' BinaryOpUGen is the master gain
LIMITER_LOOKAHEAD = 0.01       # seconds, Limiter's fixed (scalar) window


Number = Union[int, float]


class OutRef:
    """A reference to one output of one UGen; the currency of graph wiring.

    Returned by every UGen-creating method. Passing it as an input wires the two
    UGens together. Arithmetic operators build ``BinaryOpUGen`` nodes so signal
    math reads like ordinary Python: ``sd.sinosc(440) * 0.2``.
    """

    __slots__ = ("ugen", "out", "rate", "_def")

    def __init__(self, ugen: int, out: int, rate: int, sdef: "SynthDef"):
        self.ugen = ugen      # index of the source UGen in the graph
        self.out = out        # which output of that UGen
        self.rate = rate      # calc rate of this output (IR/KR/AR)
        self._def = sdef

    def _binop(self, other: "InputLike", op: str, reverse: bool = False) -> "OutRef":
        rate = max(self.rate, _rate_of(other))
        inputs = [other, self] if reverse else [self, other]
        return self._def._add("BinaryOpUGen", rate, inputs, special=BINOP[op])

    def __mul__(self, o: "InputLike") -> "OutRef":
        return self._binop(o, "*")

    __rmul__ = __mul__          # multiplication is commutative

    def __add__(self, o: "InputLike") -> "OutRef":
        return self._binop(o, "+")

    __radd__ = __add__          # addition is commutative

    def __sub__(self, o: "InputLike") -> "OutRef":
        return self._binop(o, "-")

    def __rsub__(self, o: "InputLike") -> "OutRef":
        return self._binop(o, "-", reverse=True)

    def __truediv__(self, o: "InputLike") -> "OutRef":
        return self._binop(o, "/")

    def __rtruediv__(self, o: "InputLike") -> "OutRef":
        return self._binop(o, "/", reverse=True)

    def __neg__(self) -> "OutRef":
        return self._binop(-1.0, "*")


InputLike = Union[OutRef, Number]


def _rate_of(x: "InputLike") -> int:
    return x.rate if isinstance(x, OutRef) else IR


class _UGen:
    __slots__ = ("name", "rate", "inputs", "num_outputs", "special", "output_rates")

    def __init__(self, name: str, rate: int, inputs: List[InputLike],
                 num_outputs: int, special: int, output_rates: List[int]):
        self.name = name
        self.rate = rate
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.special = special
        self.output_rates = output_rates


# -- little-endian? no: SynthDef files are big-endian --------------------------
def _pstr(s: str) -> bytes:
    """A Pascal string: one length byte followed by the raw bytes."""
    data = s.encode("ascii")
    if len(data) > 255:
        raise ValueError(f"symbol too long for a SynthDef pstring: {s!r}")
    return bytes([len(data)]) + data


class SynthDef:
    """Accumulates a UGen graph and compiles it to SynthDef v2 bytes."""

    def __init__(self, name: str):
        self.name = name
        self.ugens: List[_UGen] = []
        self.params: List[Tuple[str, float]] = []   # (name, default)
        self._control_ugen: Optional[int] = None     # index of the lone Control

    # -- low-level construction --------------------------------------------
    def _add(self, name: str, rate: int, inputs: List[InputLike],
             num_outputs: int = 1, special: int = 0,
             output_rates: Optional[List[int]] = None) -> Union[OutRef, List[OutRef]]:
        """Append a UGen. Returns an OutRef (or a list, for multi-output UGens)."""
        idx = len(self.ugens)
        rates = output_rates if output_rates is not None else [rate] * num_outputs
        self.ugens.append(_UGen(name, rate, list(inputs), num_outputs, special, rates))
        if num_outputs == 1:
            return OutRef(idx, 0, rates[0], self)
        return [OutRef(idx, k, rates[k], self) for k in range(num_outputs)]

    def ugen(self, name: str, rate: int, inputs: List[InputLike],
             num_outputs: int = 1, special: int = 0) -> Union[OutRef, List[OutRef]]:
        """Public escape hatch: add any UGen the helpers below don't cover."""
        return self._add(name, rate, inputs, num_outputs, special)

    def control(self, name: str, default: Number = 0.0) -> OutRef:
        """Declare a named, live-settable control (a synth parameter).

        Reference the returned OutRef anywhere a signal is expected; at run time
        send ``/n_set <node> <name> <value>`` (``SCClient.set``) to change it.
        All controls share one ``Control`` UGen, created on first call.
        """
        if self._control_ugen is None:
            self._control_ugen = len(self.ugens)
            self.ugens.append(_UGen("Control", KR, [], 0, 0, []))
        ctl = self.ugens[self._control_ugen]
        out_index = ctl.num_outputs
        ctl.num_outputs += 1
        ctl.output_rates.append(KR)
        self.params.append((name, float(default)))
        return OutRef(self._control_ugen, out_index, KR, self)

    # -- curated UGen palette ----------------------------------------------
    # Oscillators / sources
    def sinosc(self, freq: InputLike = 440, phase: InputLike = 0, rate: int = AR) -> OutRef:
        return self._add("SinOsc", rate, [freq, phase])

    def saw(self, freq: InputLike = 440, rate: int = AR) -> OutRef:
        return self._add("Saw", rate, [freq])

    def pulse(self, freq: InputLike = 440, width: InputLike = 0.5, rate: int = AR) -> OutRef:
        return self._add("Pulse", rate, [freq, width])

    def lfsaw(self, freq: InputLike = 1, iphase: InputLike = 0, rate: int = KR) -> OutRef:
        return self._add("LFSaw", rate, [freq, iphase])

    def lftri(self, freq: InputLike = 1, iphase: InputLike = 0, rate: int = KR) -> OutRef:
        return self._add("LFTri", rate, [freq, iphase])

    def impulse(self, freq: InputLike = 1, phase: InputLike = 0, rate: int = AR) -> OutRef:
        return self._add("Impulse", rate, [freq, phase])

    def whitenoise(self, rate: int = AR) -> OutRef:
        return self._add("WhiteNoise", rate, [])

    def pinknoise(self, rate: int = AR) -> OutRef:
        return self._add("PinkNoise", rate, [])

    def brownnoise(self, rate: int = AR) -> OutRef:
        return self._add("BrownNoise", rate, [])

    def dust(self, density: InputLike = 100, rate: int = AR) -> OutRef:
        return self._add("Dust", rate, [density])

    def lfnoise0(self, freq: InputLike = 10, rate: int = KR) -> OutRef:
        """Stepped random (sample-and-hold noise) -- a new value ``freq`` times/sec."""
        return self._add("LFNoise0", rate, [freq])

    def lfnoise1(self, freq: InputLike = 10, rate: int = KR) -> OutRef:
        """Ramped (linearly interpolated) random noise."""
        return self._add("LFNoise1", rate, [freq])

    # Filters
    def lpf(self, sig: InputLike, freq: InputLike = 1000, rate: int = AR) -> OutRef:
        return self._add("LPF", rate, [sig, freq])

    def hpf(self, sig: InputLike, freq: InputLike = 1000, rate: int = AR) -> OutRef:
        return self._add("HPF", rate, [sig, freq])

    def bpf(self, sig: InputLike, freq: InputLike = 1000, rq: InputLike = 1, rate: int = AR) -> OutRef:
        return self._add("BPF", rate, [sig, freq, rq])

    def rlpf(self, sig: InputLike, freq: InputLike = 1000, rq: InputLike = 1, rate: int = AR) -> OutRef:
        return self._add("RLPF", rate, [sig, freq, rq])

    def rhpf(self, sig: InputLike, freq: InputLike = 1000, rq: InputLike = 1, rate: int = AR) -> OutRef:
        return self._add("RHPF", rate, [sig, freq, rq])

    def resonz(self, sig: InputLike, freq: InputLike = 440, bwr: InputLike = 1, rate: int = AR) -> OutRef:
        return self._add("Resonz", rate, [sig, freq, bwr])

    # Delay / resonator
    def combl(self, sig: InputLike, maxdelay: Number = 0.2,
              delay: InputLike = 0.2, decay: InputLike = 1.0, rate: int = AR) -> OutRef:
        """CombL: interpolating comb delay with feedback set by ``decay`` (seconds).

        A short delay tuned to ``1/pitch`` seconds with a long decay behaves like a
        Karplus-Strong resonator. ``maxdelay`` is a scalar (fixed at build time).
        """
        return self._add("CombL", rate, [sig, float(maxdelay), delay, decay])

    # Utility / dynamics
    def leakdc(self, sig: InputLike, coef: InputLike = 0.995, rate: int = AR) -> OutRef:
        return self._add("LeakDC", rate, [sig, coef])

    def limiter(self, sig: InputLike, level: InputLike = 1.0,
                dur: Number = LIMITER_LOOKAHEAD, rate: int = AR) -> OutRef:
        return self._add("Limiter", rate, [sig, level, float(dur)])

    def clip(self, sig: InputLike, lo: InputLike = -1, hi: InputLike = 1, rate: int = AR) -> OutRef:
        return self._add("Clip", rate, [sig, lo, hi])

    def pan2(self, sig: InputLike, pos: InputLike = 0, level: InputLike = 1) -> List[OutRef]:
        """Equal-power stereo pan; returns ``[left, right]`` OutRefs."""
        return self._add("Pan2", AR, [sig, pos, level], num_outputs=2)  # type: ignore[return-value]

    # -- output stage + safety ---------------------------------------------
    def out(self, bus: InputLike, *channels: InputLike) -> None:
        """Raw ``Out.ar(bus, *channels)``. Prefer :meth:`to_out` for safety."""
        self._add(OUT_OBJECT, AR, [bus, *channels], num_outputs=0)

    def to_out(self, source: InputLike, gain: Number = MAX_MASTER_GAIN,
               bus: InputLike = 0) -> OutRef:
        """Wire ``source`` to hardware out through the protective master chain.

        Builds ``LeakDC -> [* gain] -> Limiter`` and sends the result to both
        output channels (dual-mono), so every synth that ends here is DC-blocked,
        gain-bounded and hard-limited before it reaches the speakers. Returns the
        Limiter's OutRef. This is exactly what :meth:`safety_violations` checks
        for -- always end a synth with ``to_out(...)``.
        """
        dc = self.leakdc(source)                 # DC / subsonic blocker
        amp = dc * float(gain)                   # bounded master gain (BinaryOp *)
        lim = self.limiter(amp, 1.0)             # brick-wall limiter
        self.out(bus, lim, lim)                  # L and R (dual-mono)
        return lim

    # -- compilation -------------------------------------------------------
    def compile(self) -> bytes:
        """Serialize the graph to SynthDef v2 bytes (a valid ``.scsyndef``)."""
        # 1. Gather the constant pool (every numeric, non-OutRef input), deduped.
        consts: List[float] = []
        cindex: Dict[float, int] = {}

        def register(v: Number) -> int:
            f = float(v)
            if f not in cindex:
                cindex[f] = len(consts)
                consts.append(f)
            return cindex[f]

        for u in self.ugens:
            for inp in u.inputs:
                if not isinstance(inp, OutRef):
                    register(inp)

        out = bytearray()
        out += b"SCgf"                       # file type id
        out += struct.pack(">i", 2)          # file version 2
        out += struct.pack(">h", 1)          # number of synthdefs in this file

        out += _pstr(self.name)

        out += struct.pack(">i", len(consts))
        for c in consts:
            out += struct.pack(">f", c)

        out += struct.pack(">i", len(self.params))          # initial param values
        for _, default in self.params:
            out += struct.pack(">f", default)

        out += struct.pack(">i", len(self.params))          # param name -> index
        for i, (pname, _) in enumerate(self.params):
            out += _pstr(pname)
            out += struct.pack(">i", i)

        out += struct.pack(">i", len(self.ugens))
        for u in self.ugens:
            out += _pstr(u.name)
            out += struct.pack(">b", u.rate)
            out += struct.pack(">i", len(u.inputs))
            out += struct.pack(">i", u.num_outputs)
            out += struct.pack(">h", u.special)
            for inp in u.inputs:
                if isinstance(inp, OutRef):
                    out += struct.pack(">i", inp.ugen)   # source UGen index
                    out += struct.pack(">i", inp.out)    # source output index
                else:
                    out += struct.pack(">i", -1)         # -1 == constant
                    out += struct.pack(">i", cindex[float(inp)])
            for r in u.output_rates:
                out += struct.pack(">b", r)

        out += struct.pack(">h", 0)          # number of variants
        return bytes(out)

    def to_scsyndef_file(self, path: str) -> str:
        """Write the compiled graph as a standalone, loadable ``.scsyndef`` file.

        Any scsynth can load it (``/d_load`` / ``/d_loadDir``) with no Python and
        no sclang -- the natural shareable artifact for a synth built here.
        """
        with open(path, "wb") as fh:
            fh.write(self.compile())
        return path

    # -- safety linting -----------------------------------------------------
    def _incoming(self) -> Dict[int, List[int]]:
        edges: Dict[int, List[int]] = {}
        for i, u in enumerate(self.ugens):
            for inp in u.inputs:
                if isinstance(inp, OutRef):
                    edges.setdefault(i, []).append(inp.ugen)
        return edges

    def _ancestors(self, idx: int, incoming: Dict[int, List[int]]) -> set:
        seen: set = set()
        stack = list(incoming.get(idx, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(incoming.get(cur, []))
        return seen

    def safety_violations(self) -> List[str]:
        """Return human-readable master-output safety violations (empty == safe).

        For every connected ``Out``:

        1. **Limiter at the output.** Each signal input of ``Out`` must be a
           ``Limiter`` (or ``Clip``); a raw signal wired to ``Out`` lets
           overshoots reach the DAC as full-scale bursts.
        2. **Bounded master gain.** That limiter must be fed by a ``*``
           ``BinaryOpUGen`` whose constant multiplier is <= ``MAX_MASTER_GAIN``.
        3. **DC / subsonic block.** A ``LeakDC`` must sit somewhere upstream of
           the ``Out``.

        :meth:`to_out` builds exactly this chain.
        """
        violations: List[str] = []
        incoming = self._incoming()
        outs = [i for i, u in enumerate(self.ugens) if u.name == OUT_OBJECT]

        for oi in outs:
            out_u = self.ugens[oi]
            signal_inputs = out_u.inputs[1:]     # input 0 is the bus number
            connected = [s for s in signal_inputs if isinstance(s, OutRef)]
            if not connected:
                continue  # Out with no signal makes no sound; nothing to check

            for s in connected:
                feeder = self.ugens[s.ugen]
                # (1) limiter directly before Out
                if feeder.name not in LIMITER_OBJECTS:
                    violations.append(
                        f"[{feeder.name}] (ugen {s.ugen}) feeds Out (ugen {oi}) "
                        f"without a Limiter/Clip; overshoots can hit the DAC as "
                        f"full-scale bursts"
                    )
                    continue
                # (2) bounded master gain feeding that limiter
                if not feeder.inputs:
                    continue
                gain_in = feeder.inputs[0]
                gain_ok = False
                if isinstance(gain_in, OutRef):
                    g = self.ugens[gain_in.ugen]
                    if g.name == GAIN_OBJECT and g.special == BINOP["*"]:
                        const_operands = [x for x in g.inputs if not isinstance(x, OutRef)]
                        for value in const_operands:
                            # Bound by the documented range: a negative gain has
                            # magnitude too, so `value <= MAX` alone would wrongly
                            # pass -10.0. Require 0 <= gain <= MAX_MASTER_GAIN.
                            if 0 <= float(value) <= MAX_MASTER_GAIN:
                                gain_ok = True
                            else:
                                violations.append(
                                    f"master gain [* {value}] (ugen {gain_in.ugen}) "
                                    f"is outside 0..{MAX_MASTER_GAIN}; set it within "
                                    f"that range to protect ears/gear"
                                )
                        if not const_operands:
                            violations.append(
                                f"master gain [*] (ugen {gain_in.ugen}) before the "
                                f"Limiter has no constant multiplier <= "
                                f"{MAX_MASTER_GAIN}"
                            )
                            gain_ok = True  # reported; don't double-report below
                if not gain_ok:
                    violations.append(
                        f"Limiter (ugen {s.ugen}) before Out has no bounded master "
                        f"gain [* <= {MAX_MASTER_GAIN}] feeding it"
                    )

            # (3) DC / subsonic blocker somewhere upstream
            anc_names = {self.ugens[i].name for i in self._ancestors(oi, incoming)}
            if DCBLOCK_OBJECT not in anc_names:
                violations.append(
                    f"Out (ugen {oi}) chain has no DC/subsonic blocker "
                    f"[{DCBLOCK_OBJECT}]; DC offset can over-excurse speakers"
                )
        # Out is fed dual-mono (the same limiter twice), so the same problem can
        # surface more than once; report each distinct violation only once.
        deduped: List[str] = []
        for v in violations:
            if v not in deduped:
                deduped.append(v)
        return deduped

    def is_safe(self) -> bool:
        return not self.safety_violations()


def build_simple_synth(name: str = "hermes_sine", freq: Number = 440,
                       gain: Number = MAX_MASTER_GAIN) -> SynthDef:
    """The canonical safe reference synth: a single sine through the master chain.

    ``SinOsc.ar(freq) -> LeakDC -> [* gain] -> Limiter -> Out.ar(0)`` via
    :meth:`SynthDef.to_out`.
    """
    sd = SynthDef(name)
    f = sd.control("freq", freq)
    sd.to_out(sd.sinosc(f), gain)
    return sd


def build_recorder(name: str = "hermes_recorder", channels: int = 2) -> SynthDef:
    """A synth that streams the output bus to disk via ``DiskOut``.

    Reads the hardware output bus (``In.ar(0, channels)``) -- i.e. the exact,
    post-limiter audio you hear -- and writes it to the buffer named by the
    ``buf`` control. Play it at the tail of the node tree while your synths run,
    then free it to finish the file. See ``SCClient.record``.
    """
    sd = SynthDef(name)
    buf = sd.control("buf", 0)
    sig = sd.ugen("In", AR, [0.0], num_outputs=channels)
    if channels == 1:
        sig = [sig]  # type: ignore[list-item]
    sd.ugen("DiskOut", AR, [buf, *sig], num_outputs=0)
    return sd


if __name__ == "__main__":
    import sys

    patch = build_simple_synth()
    blob = patch.compile()
    sys.stderr.write(f"{patch.name}: {len(blob)} bytes, "
                     f"{len(patch.ugens)} ugens, {len(patch.params)} params\n")
    problems = patch.safety_violations()
    if problems:
        sys.stderr.write("SAFETY VIOLATIONS:\n")
        for p in problems:
            sys.stderr.write(f"  - {p}\n")
        sys.exit(1)
    sys.stdout.buffer.write(blob)   # emit raw .scsyndef bytes on stdout
