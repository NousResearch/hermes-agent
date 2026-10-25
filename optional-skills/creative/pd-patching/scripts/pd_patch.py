#!/usr/bin/env python3
"""Structured builder + safety linter for Pure Data dynamic patches.

The agent can emit raw ``pd-<canvas> ...`` strings by hand, but building a
patch through :class:`PdPatch` gives three things for free:

* correct, monotonically increasing layout coordinates,
* stable 0-based object indices for wiring, and
* a safety linter that enforces the DSP rules in ``SKILL.md`` (no signal object
  wired straight into ``[dac~]``; master gain kept at or below 0.1).

The output of :meth:`PdPatch.messages` is a list of command strings ready to
hand to :class:`pd_client.PureDataClient.send_batch`.

Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Vertical / horizontal spacing used when coordinates are auto-assigned. Matches
# the layout guidance in SKILL.md (rows ~50px apart, columns ~140px apart).
ROW_STEP = 50
COL_STEP = 140

# Maximum safe master gain multiplier feeding [dac~].
MAX_MASTER_GAIN = 0.1

# Protective master-output stage (speaker / headphone / ear safety):
#   source -> [hip~ 5] -> [*~ gain] -> [clip~ -1 1] -> [dac~]
# [hip~ 5]   removes DC offset and subsonic (<~5 Hz) energy that wastes
#            headroom and can over-excurse woofers.
# [clip~ -1 1] is a brick-wall limiter so transient overshoots never reach
#            the DAC as full-scale clipping bursts.
DCBLOCK_CUTOFF = 5      # Hz, for [hip~]
CLIP_LOW = -1
CLIP_HIGH = 1

SIGNAL_SUFFIX = "~"
GAIN_OBJECT = "*~"
DAC_OBJECT = "dac~"
DCBLOCK_OBJECT = "hip~"
LIMITER_OBJECT = "clip~"


@dataclass
class PdObject:
    index: int
    name: str
    args: Tuple[str, ...]
    x: int
    y: int


@dataclass
class PdConnection:
    src: int
    outlet: int
    dst: int
    inlet: int


@dataclass
class PdPatch:
    """Accumulates objects/connections targeting a single Pd canvas."""

    canvas: str = "target_canvas"
    x0: int = 100
    y0: int = 50
    _objects: List[PdObject] = field(default_factory=list)
    _connections: List[PdConnection] = field(default_factory=list)
    _messages: List[str] = field(default_factory=list)
    _cursor_x: int = field(init=False)
    _cursor_y: int = field(init=False)

    def __post_init__(self) -> None:
        self._cursor_x = self.x0
        self._cursor_y = self.y0

    @property
    def target(self) -> str:
        """The Pd receiver symbol for this canvas, e.g. ``pd-target_canvas``."""
        return f"pd-{self.canvas}"

    # -- construction -------------------------------------------------------
    def clear(self) -> "PdPatch":
        """Emit a canvas clear and reset object/connection tracking."""
        self._messages.append(f"{self.target} clear")
        self._objects = []
        self._connections = []
        self._cursor_x = self.x0
        self._cursor_y = self.y0
        return self

    def obj(
        self,
        name: str,
        *args,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> int:
        """Create an object and return its 0-based index for wiring."""
        index = len(self._objects)
        px = self._cursor_x if x is None else x
        py = self._cursor_y if y is None else y
        str_args = tuple(str(a) for a in args)
        line = f"{self.target} obj {px} {py} {name}"
        if str_args:
            line += " " + " ".join(str_args)
        self._messages.append(line)
        self._objects.append(PdObject(index, name, str_args, px, py))
        # advance the auto-layout cursor to the next row
        self._cursor_y += ROW_STEP
        return index

    def new_column(self) -> "PdPatch":
        """Move the auto-layout cursor to the top of the next column."""
        self._cursor_x += COL_STEP
        self._cursor_y = self.y0
        return self

    def connect(self, src: int, outlet: int, dst: int, inlet: int) -> "PdPatch":
        """Wire ``src`` outlet -> ``dst`` inlet (0-based indices)."""
        self._messages.append(
            f"{self.target} connect {src} {outlet} {dst} {inlet}"
        )
        self._connections.append(PdConnection(src, outlet, dst, inlet))
        return self

    def to_dac(
        self, source: int, gain: float = MAX_MASTER_GAIN, record: bool = False
    ) -> int:
        """Wire ``source`` to ``[dac~]`` through the protective master chain.

        Builds ``[hip~ 5] -> [*~ gain] -> [clip~ -1 1] -> [dac~]`` (stereo) so
        every patch that ends here is DC-blocked, gain-bounded, and hard-limited
        before it reaches the speakers. Returns the ``[dac~]`` index. This is the
        recommended way to reach the output -- it is exactly what
        :meth:`safety_violations` checks for.

        When ``record=True`` the same limited signal is also tapped to a
        ``[writesf~ 2]`` fed by ``[r pd-rec]``, so the exact audio you hear can
        be captured to a WAV. Control it at runtime by sending messages to the
        ``pd-rec`` receiver (the stub routes it): ``pd-rec open <abs.wav>`` then
        ``pd-rec start``, and later ``pd-rec stop`` (see :meth:`record_messages`).
        """
        hp = self.obj(DCBLOCK_OBJECT, DCBLOCK_CUTOFF)   # DC / subsonic blocker
        amp = self.obj(GAIN_OBJECT, gain)               # bounded master gain
        lim = self.obj(LIMITER_OBJECT, CLIP_LOW, CLIP_HIGH)  # brick-wall limiter
        dac = self.obj(DAC_OBJECT)
        self.connect(source, 0, hp, 0)
        self.connect(hp, 0, amp, 0)
        self.connect(amp, 0, lim, 0)
        self.connect(lim, 0, dac, 0)  # left
        self.connect(lim, 0, dac, 1)  # right
        if record:
            writer = self.obj("writesf~", 2)            # stereo soundfile writer
            rec = self.obj("r", "pd-rec")               # control: open/start/stop
            self.connect(lim, 0, writer, 0)             # record L (post-limiter)
            self.connect(lim, 0, writer, 1)             # record R
            self.connect(rec, 0, writer, 0)             # control into left inlet
        return dac

    def add_recorder(self, channels: int = 2) -> int:
        """Tap the existing limiter (``[clip~]`` before ``[dac~]``) to a recorder.

        Adds ``[writesf~ channels]`` fed by the limiter, plus ``[r pd-rec]`` for
        control, to a patch already ended with :meth:`to_dac`. Lets a canned
        example be recorded without rebuilding it. Returns the writer index.
        """
        incoming: Dict[int, List[int]] = {}
        for c in self._connections:
            incoming.setdefault(c.dst, []).append(c.src)
        by_index = {o.index: o for o in self._objects}
        lim_idx: Optional[int] = None
        for o in self._objects:
            if o.name != DAC_OBJECT:
                continue
            for s in incoming.get(o.index, []):
                if by_index.get(s) and by_index[s].name == LIMITER_OBJECT:
                    lim_idx = s
                    break
            if lim_idx is not None:
                break
        if lim_idx is None:
            raise ValueError("no [clip~] limiter before [dac~]; call to_dac() first")
        writer = self.obj("writesf~", channels)
        rec = self.obj("r", "pd-rec")
        for ch in range(channels):
            self.connect(lim_idx, 0, writer, ch)
        self.connect(rec, 0, writer, 0)
        return writer

    @staticmethod
    def record_messages(path: str, start: bool = True) -> List[str]:
        """Return the messages that open (and optionally start) WAV recording.

        ``path`` should be absolute (``writesf~`` resolves relative paths against
        Pd's cwd). Pair with ``["pd-rec stop"]`` to finish and flush the file.
        """
        msgs = [f"pd-rec open {path}"]
        if start:
            msgs.append("pd-rec start")
        return msgs

    def to_pd_file(self, path: str, width: int = 560, height: int = 420) -> str:
        """Write the current objects/connections as a standalone ``.pd`` patch.

        Produces a normal Pd file (openable and shareable, no socket needed);
        ``clear``/``dsp`` control messages are not part of a saved patch, so they
        are omitted. Returns ``path``.
        """
        lines = [f"#N canvas 0 22 {width} {height} 12;"]
        for o in self._objects:
            argstr = (" " + " ".join(o.args)) if o.args else ""
            lines.append(f"#X obj {o.x} {o.y} {o.name}{argstr};")
        for c in self._connections:
            lines.append(f"#X connect {c.src} {c.outlet} {c.dst} {c.inlet};")
        text = "\n".join(lines) + "\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def dsp(self, on: bool = True) -> "PdPatch":
        """Toggle the global DSP engine (addressed to the ``pd`` receiver)."""
        self._messages.append(f"pd dsp {1 if on else 0}")
        return self

    # -- output -------------------------------------------------------------
    def messages(self) -> List[str]:
        """Return the ordered list of command strings emitted so far."""
        return list(self._messages)

    @property
    def objects(self) -> List[PdObject]:
        return list(self._objects)

    # -- safety linting -----------------------------------------------------
    def _gain_value(self, obj: PdObject) -> Optional[float]:
        if obj.name != GAIN_OBJECT or not obj.args:
            return None
        try:
            return float(obj.args[0])
        except (TypeError, ValueError):
            return None

    def safety_violations(self) -> List[str]:
        """Return human-readable DSP safety violations (empty == safe).

        Enforces the protective master-output chain that reaches every
        connected ``[dac~]`` (build it with :meth:`to_dac`):

        1. **Limiter at the DAC.** The object feeding ``[dac~]`` must be a
           ``[clip~]`` -- a raw signal object (``osc~``/``vcf~``/``*~`` ...) wired
           straight to the DAC lets overshoots through as full-scale bursts.
        2. **Bounded master gain.** A ``[*~]`` feeding that ``[clip~]`` must have
           a numeric multiplier at or below ``MAX_MASTER_GAIN`` (0.1).
        3. **DC / subsonic block.** A ``[hip~]`` must sit somewhere upstream of
           the ``[dac~]`` so DC offset can't over-excurse speakers.
        """
        violations: List[str] = []
        by_index = {o.index: o for o in self._objects}
        incoming: Dict[int, List[int]] = {}
        for conn in self._connections:
            incoming.setdefault(conn.dst, []).append(conn.src)

        def ancestors(idx: int) -> set:
            seen: set = set()
            stack = list(incoming.get(idx, []))
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                stack.extend(incoming.get(cur, []))
            return seen

        dac_indices = [o.index for o in self._objects if o.name == DAC_OBJECT]
        for dac in dac_indices:
            feeders = incoming.get(dac, [])
            if not feeders:
                continue  # unconnected [dac~] makes no sound; nothing to check

            # (1) limiter directly before [dac~]
            for f in feeders:
                src = by_index.get(f)
                if src is None:
                    violations.append(
                        f"[dac~] (index {dac}) has unknown source index {f}"
                    )
                    continue
                if src.name != LIMITER_OBJECT:
                    violations.append(
                        f"[{src.name}] (index {f}) feeds [dac~] (index {dac}) "
                        f"without a [clip~ {CLIP_LOW} {CLIP_HIGH}] limiter; "
                        f"overshoots can hit the DAC as full-scale bursts"
                    )
                    continue
                # (2) bounded master gain feeding that limiter
                gains = [
                    by_index[g]
                    for g in incoming.get(f, [])
                    if g in by_index and by_index[g].name == GAIN_OBJECT
                ]
                if not gains:
                    violations.append(
                        f"[clip~] (index {f}) before [dac~] has no bounded master "
                        f"gain [*~ <= {MAX_MASTER_GAIN}] feeding it"
                    )
                for m in gains:
                    gv = self._gain_value(m)
                    if gv is None:
                        violations.append(
                            f"master [*~] (index {m.index}) has no numeric gain; "
                            f"set it to <= {MAX_MASTER_GAIN}"
                        )
                    elif gv > MAX_MASTER_GAIN:
                        violations.append(
                            f"master [*~ {m.args[0]}] (index {m.index}) exceeds "
                            f"{MAX_MASTER_GAIN}; lower it to protect ears/gear"
                        )

            # (3) DC / subsonic blocker somewhere upstream of [dac~]
            anc_names = {
                by_index[i].name for i in ancestors(dac) if i in by_index
            }
            if DCBLOCK_OBJECT not in anc_names:
                violations.append(
                    f"[dac~] (index {dac}) output chain has no DC/subsonic "
                    f"blocker [{DCBLOCK_OBJECT} {DCBLOCK_CUTOFF}]; DC offset can "
                    f"damage speakers"
                )
        return violations

    def is_safe(self) -> bool:
        return not self.safety_violations()


def build_simple_synth(freq: float = 440, gain: float = MAX_MASTER_GAIN) -> PdPatch:
    """Build the canonical safe sine synth.

    ``osc~ -> [hip~ 5] -> [*~ gain] -> [clip~ -1 1] -> dac~`` (stereo) via
    :meth:`PdPatch.to_dac`. Returns a :class:`PdPatch` whose :meth:`messages`
    are ready to send; the canonical safe reference graph.
    """
    p = PdPatch()
    p.clear()
    osc = p.obj("osc~", freq)
    p.to_dac(osc, gain)          # hip~ 5 -> *~ gain -> clip~ -1 1 -> dac~
    p.dsp(True)
    return p


if __name__ == "__main__":
    patch = build_simple_synth()
    for line in patch.messages():
        print(line)
    problems = patch.safety_violations()
    if problems:
        import sys

        print("\nSAFETY VIOLATIONS:", file=sys.stderr)
        for prob in problems:
            print(f"  - {prob}", file=sys.stderr)
        sys.exit(1)
