"""
Lightweight SRA Core — extracted from Hermes-Skill-View for in-process use.
Provides fast skill matching without the daemon overhead.
"""
from .advisor import SkillAdvisor

__all__ = ["SkillAdvisor"]
