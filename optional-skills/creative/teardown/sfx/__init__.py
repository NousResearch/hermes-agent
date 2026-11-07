"""Procedural sound design for motion graphics. Pure Python stdlib, 48 kHz stereo.

    python run_sfx.py sfx/cues/effort.py   render a cue sheet to WAV
    python run_sfx.py --list                             list the sound generators
"""

from .mix import Mix
from .synth import SR

__all__ = ["SR", "Mix"]
