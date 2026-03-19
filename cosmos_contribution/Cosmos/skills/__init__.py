"""
Cosmos Skills System
========================

"Good news, everyone! I've assimilated new capabilities!"

Skills are modular capabilities that the collective can load and use.
Based on HermesAgent's Skill Document format but adapted for the Cosmos swarm.

Skills can:
- Add new MCP tools
- Enable browser automation
- Provide payment capabilities
- Add specialized knowledge
"""

from .skill_loader import SkillLoader, Skill, load_skill
from .cosmos_skills import COSMOS_SKILLS

__all__ = ['SkillLoader', 'Skill', 'load_skill', 'COSMOS_SKILLS']
