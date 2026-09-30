"""Probe registry. Pure data + decorators; no I/O at import time."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

TIERS = ("T0", "T1", "T2", "T3")
COLLECT = ("core", "extended", "deep")
LEVELS = ("L1", "L2")


@dataclass
class Probe:
    id: str
    level: str                 # L1 detector | L2 extractor
    family: str                # category, e.g. "browser", "gaming"
    tier: str                  # privacy tier of the VALUE this probe returns
    collect: str               # core | extended | deep
    gate: Optional[str]        # id of a probe whose value must be truthy; None = OS gate only
    os: str                    # "windows" | "darwin" | "linux" | "any"
    timeout_ms: int
    fn: Optional[Callable] = None      # python probe: fn(h, facts) -> value | None
    ps: Optional[str] = None           # powershell probe: script whose output object becomes the value
    needs_admin: bool = False
    doc: str = ""


@dataclass
class Insight:
    id: str
    inputs: list
    fn: Callable                       # fn(facts) -> dict(claim=, strength=, value=?) | None
    doc: str = ""


REGISTRY: dict = {}
INSIGHTS: dict = {}


def _key(p: Probe) -> str:
    """The same signal id may be registered once per OS; the first registration keeps the bare id."""
    if p.id not in REGISTRY:
        return p.id
    return f"{p.id}@{p.os}"


def _check(p: Probe) -> None:
    for q in REGISTRY.values():
        if q.id == p.id and (q.os == p.os or "any" in (q.os, p.os)):
            raise ValueError(f"duplicate probe id {p.id} for os {p.os}")
    if p.tier not in TIERS or p.collect not in COLLECT or p.level not in LEVELS:
        raise ValueError(f"bad tier/collect/level on {p.id}")
    if p.level == "L2" and not p.gate:
        raise ValueError(f"L2 probe {p.id} needs a gate")


def probe(id: str, *, level: str, family: str, tier: str = "T0", collect: str = "core",
          gate: Optional[str] = None, os: str = "windows", timeout_ms: int = 2000,
          needs_admin: bool = False):
    """Register a python probe. Return None / False / {} for 'absent'."""
    def deco(fn):
        p = Probe(id, level, family, tier, collect, gate, os, timeout_ms, fn=fn,
                  needs_admin=needs_admin, doc=(fn.__doc__ or "").strip())
        _check(p)
        REGISTRY[_key(p)] = p
        return fn
    return deco


def ps_probe(id: str, script: str, *, level: str = "L2", family: str, tier: str = "T0",
             collect: str = "extended", gate: Optional[str] = None, timeout_ms: int = 8000,
             needs_admin: bool = False, doc: str = ""):
    """Register a PowerShell snippet. All ps probes run in ONE batched powershell.exe.
    The snippet must leave its result object on the pipeline (it is captured with ConvertTo-Json)."""
    p = Probe(id, level, family, tier, collect, gate, "windows", timeout_ms, ps=script,
              needs_admin=needs_admin, doc=doc)
    _check(p)
    REGISTRY[_key(p)] = p


def insight(id: str, inputs: list):
    """Register an L3 derivation over facts. fn(facts) gets {id: value} for ok facts only."""
    def deco(fn):
        if id in INSIGHTS:
            raise ValueError(f"duplicate insight {id}")
        INSIGHTS[id] = Insight(id, list(inputs), fn, (fn.__doc__ or "").strip())
        return fn
    return deco
