"""
Sister Registry - Plugin-style registry for the 12-Sister Personality System.
Source of truth for sister definitions, matching, and prompt loading.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path

@dataclass
class Sister:
    """Represents a sister agent with her identity and capabilities."""
    id: str
    name: str
    role: str
    archetype: str
    domain: str
    delegation_scope: List[str]
    risk_level: str  # low, standard, high, production
    system_prompt: str
    description: str
    core_directive: str = "" # The primary mission statement
    personality_style: str = "" # Tone and behavioral guidelines
    legacy_aliases: List[str] = field(default_factory=list)
    enabled: bool = True
    model_preference: Optional[str] = None  # HARP hint only, not hardcoded

    def matches_keywords(self, query: str) -> float:
        """Score how well this sister matches a query based on keywords."""
        query_lower = query.lower()
        query_terms = set(query_lower.replace("_", " ").replace("-", " ").split())
        score = 0.0

        def _matches(term: str) -> bool:
            normalized = term.lower().replace("_", " ").replace("-", " ")
            if not normalized:
                return False
            return normalized in query_lower or any(part in query_terms for part in normalized.split())

        # Direct ID/name match
        if _matches(self.id) or _matches(self.name):
            score += 10.0

        # Role/archetype/domain match
        for term in [self.role, self.archetype, self.domain]:
            if _matches(term):
                score += 5.0

        # Delegation scope match
        for scope in self.delegation_scope:
            if _matches(scope):
                score += 3.0

        # Legacy alias match
        for alias in self.legacy_aliases:
            if _matches(alias):
                score += 8.0

        return score


class SisterRegistry:
    """
    Central registry for all sister agents.
    Reads from sisters.yaml and provides matching, loading, and delegation support.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.sisters: Dict[str, Sister] = {}
        self._alias_map: Dict[str, str] = {}
        self._config_path = config_path or self._default_config_path()
        self._load_registry()

    def _default_config_path(self) -> str:
        """Find the sisters.yaml config file."""
        candidates = [
            os.path.expanduser("~/.hermes/config/sisters.yaml"),
            os.path.expanduser("~/.hermes/sisters.yaml"),
            os.path.join(os.path.dirname(__file__), "..", "config", "sisters.yaml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    def _load_registry(self):
        """Load sister definitions from YAML config."""
        import yaml

        if not os.path.exists(self._config_path):
            self._create_default_registry()
            return

        with open(self._config_path, 'r') as f:
            data = yaml.safe_load(f)

        if not data or 'sisters' not in data:
            self._create_default_registry()
            return

        for s_data in data['sisters']:
            sister = Sister(
                id=s_data['id'],
                name=s_data['name'],
                role=s_data['role'],
                archetype=s_data.get('archetype', ''),
                domain=s_data.get('domain', ''),
                delegation_scope=s_data.get('delegation_scope', []),
                risk_level=s_data.get('risk_level', 'standard'),
                system_prompt=s_data.get('system_prompt', ''),
                description=s_data.get('description', ''),
                core_directive=s_data.get('core_directive', ''),
                personality_style=s_data.get('personality_style', ''),
                legacy_aliases=s_data.get('legacy_aliases', []),
                enabled=s_data.get('enabled', True),
                model_preference=s_data.get('model_preference'),
            )
            self.sisters[sister.id] = sister
            for alias in sister.legacy_aliases:
                self._alias_map[alias.lower()] = sister.id

        # Ensure Astra is always present as orchestrator
        if 'astra' not in self.sisters:
            self._add_astra()

    def _create_default_registry(self):
        """Create the default 12-sister roster if no config exists."""
        default_sisters = [
            Sister(
                id="astra",
                name="Astra",
                role="Main Orchestrator",
                archetype="Strategic Coordinator",
                domain="Core Intelligence",
                delegation_scope=["routing", "delegation", "synthesis", "quality_control"],
                risk_level="production",
                core_directive="You are Astra, the primary AI orchestrator. You coordinate, delegate, and ensure the right sister handles each task.",
                personality_style="Sharp, decisive, professional, and concise.",
                system_prompt="You are Astra, the primary AI orchestrator. You coordinate, delegate, and ensure the right sister handles each task.",
                description="Primary orchestrator - routes tasks to specialized sisters",
                legacy_aliases=[],
            ),
            Sister(
                id="novus",
                name="Novus",
                role="Local Code Specialist",
                archetype="Private Code Engineer",
                domain="Core Intelligence",
                delegation_scope=["local_code", "private_repos", "zero_cost_tasks", "ollama_tasks"],
                risk_level="standard",
                core_directive="Specializing in local, private code work using Ollama. Zero cost, 32k context, fully private.",
                personality_style="Methodical, focused on privacy and efficiency.",
                system_prompt="You are Novus, specializing in local, private code work using Ollama. Zero cost, 32k context, fully private.",
                description="Local/private code work (Ollama, zero cost, 32k context)",
                legacy_aliases=[],
            ),
            Sister(
                id="nova",
                name="Nova",
                role="Browser & Vision Specialist",
                archetype="Web Research Agent",
                domain="Core Intelligence",
                delegation_scope=["browser", "screenshots", "vision", "paid_research", "web_automation"],
                risk_level="standard",
                core_directive="Specializing in browser automation, screenshots, vision tasks, and paid research.",
                personality_style="Observant, detailed, and resourceful.",
                system_prompt="You are Nova, specializing in browser automation, screenshots, vision tasks, and paid research.",
                description="Browser, screenshots, vision, paid research",
                legacy_aliases=[],
            ),
            Sister(
                id="luna",
                name="Luna",
                role="Deep Researcher",
                archetype="Research Synthesis Specialist",
                domain="Core Intelligence",
                delegation_scope=["deep_research", "synthesis", "literature_review", "market_intelligence"],
                risk_level="standard",
                core_directive="Specializing in deep research, synthesis, and comprehensive analysis.",
                personality_style="Intellectual, analytical, and comprehensive.",
                system_prompt="You are Luna, specializing in deep research, synthesis, and comprehensive analysis.",
                description="Deep research and synthesis",
                legacy_aliases=[],
            ),
            Sister(
                id="maya",
                name="Maya",
                role="Implementation & Shipping",
                archetype="Builder",
                domain="Core Intelligence",
                delegation_scope=["implementation", "shipping", "deployment", "production_code"],
                risk_level="production",
                core_directive="Specializing in implementation, shipping, and getting things to production.",
                personality_style="Pragmatic, result-oriented, and disciplined.",
                system_prompt="You are Maya, specializing in implementation, shipping, and getting things to production.",
                description="Implementation and shipping",
                legacy_aliases=[],
            ),
            Sister(
                id="helena",
                name="Helena",
                role="Legal Advisor",
                archetype="Compliance Specialist",
                domain="Business & Law",
                delegation_scope=["legal", "compliance", "contracts", "regulatory"],
                risk_level="high",
                core_directive="Specializing in legal questions, compliance, and regulatory matters.",
                personality_style="Formal, precise, and risk-averse.",
                system_prompt="You are Helena, specializing in legal questions, compliance, and regulatory matters.",
                description="Legal questions, compliance",
                legacy_aliases=[],
            ),
            Sister(
                id="larissa",
                name="Larissa",
                role="Customer Success Specialist",
                archetype="SAC/Escalation Expert",
                domain="Business & Law",
                delegation_scope=["customer_service", "follow_up", "escalation", "satisfaction"],
                risk_level="standard",
                core_directive="Specializing in customer service, follow-up, and escalation handling.",
                personality_style="Empathetic, professional, and diplomatic.",
                system_prompt="You are Larissa, specializing in customer service, follow-up, and escalation handling.",
                description="Customer service and follow-up",
                legacy_aliases=["larissinha"],
            ),
            Sister(
                id="clara",
                name="Clara",
                role="Sales Development",
                archetype="Lead Qualification Expert",
                domain="Business & Law",
                delegation_scope=["sales", "lead_qualification", "outreach", "pipeline"],
                risk_level="standard",
                core_directive="Specializing in sales, lead qualification, and pipeline management.",
                personality_style="Persuasive, outgoing, and strategic.",
                system_prompt="You are Clara, specializing in sales, lead qualification, and pipeline management.",
                description="Sales and lead qualification",
                legacy_aliases=[],
            ),
            Sister(
                id="bia",
                name="Bia",
                role="Signal Monitoring",
                archetype="Risk Intelligence Analyst",
                domain="Creative & Support",
                delegation_scope=["signal_monitoring", "risk_intelligence", "threat_detection", "anomaly_detection"],
                risk_level="high",
                core_directive="Specializing in signal monitoring, risk intelligence, and threat detection.",
                personality_style="Alert, skeptical, and highly observant.",
                system_prompt="You are Bia, specializing in signal monitoring, risk intelligence, and threat detection.",
                description="Signal monitoring and risk intelligence",
                legacy_aliases=["fofoqueiro"],
            ),
            Sister(
                id="vitoria",
                name="Vitoria",
                role="Creative Director",
                archetype="Visual Content Specialist",
                domain="Creative & Support",
                delegation_scope=["creative", "visual_content", "brand_assets", "design"],
                risk_level="standard",
                core_directive="Specializing in creative and visual content, brand assets, and design.",
                personality_style="Inspiring, artistic, and intuitive.",
                system_prompt="You are Vitoria, specializing in creative and visual content, brand assets, and design.",
                description="Creative and visual content",
                legacy_aliases=["vini"],
            ),
            Sister(
                id="daine",
                name="Daine",
                role="Data Analyst",
                archetype="Analytics Specialist",
                domain="Creative & Support",
                delegation_scope=["analytics", "data_reports", "metrics", "visualization"],
                risk_level="standard",
                core_directive="Specializing in analytics, data reports, and metrics visualization.",
                personality_style="Analytical, objective, and data-driven.",
                system_prompt="You are Daine, specializing in analytics, data reports, and metrics visualization.",
                description="Analytics and data reports",
                legacy_aliases=["daiane"],
            ),
            Sister(
                id="ada",
                name="Ada",
                role="Code Review & Generation",
                archetype="Senior Developer",
                domain="Creative & Support",
                delegation_scope=["code_review", "code_generation", "debugging", "refactoring"],
                risk_level="standard",
                core_directive="Specializing in code review, generation, debugging, and refactoring.",
                personality_style="Correct, rigorous, and mentorship-oriented.",
                system_prompt="You are Ada, specializing in code review, generation, debugging, and refactoring.",
                description="Code review, generation, debugging",
                legacy_aliases=[],
            ),
        ]

        for sister in default_sisters:
            self.sisters[sister.id] = sister
            for alias in sister.legacy_aliases:
                self._alias_map[alias.lower()] = sister.id

        # Save default config
        self._save_registry()

    def _add_astra(self):
        """Ensure Astra exists as the orchestrator."""
        astra = Sister(
            id="astra",
            name="Astra",
            role="Main Orchestrator",
            archetype="Strategic Coordinator",
            domain="Core Intelligence",
            delegation_scope=["routing", "delegation", "synthesis", "quality_control"],
            risk_level="production",
            core_directive="You are Astra, the primary AI orchestrator. You coordinate, delegate, and ensure the right sister handles each task.",
            personality_style="Sharp, decisive, professional, and concise.",
            system_prompt="You are Astra, the primary AI orchestrator. You coordinate, delegate, and ensure the right sister handles each task.",
            description="Primary orchestrator - routes tasks to specialized sisters",
            legacy_aliases=[],
        )
        self.sisters['astra'] = astra
        self._save_registry()

    def _save_registry(self):
        """Save current registry to YAML config."""
        import yaml

        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)

        data = {
            'sisters': [
                {
                    'id': s.id,
                    'name': s.name,
                    'role': s.role,
                    'archetype': s.archetype,
                    'domain': s.domain,
                    'delegation_scope': s.delegation_scope,
                    'risk_level': s.risk_level,
                    'system_prompt': s.system_prompt,
                    'description': s.description,
                    'core_directive': s.core_directive,
                    'personality_style': s.personality_style,
                    'legacy_aliases': s.legacy_aliases,
                    'enabled': s.enabled,
                    'model_preference': s.model_preference,
                }
                for s in self.sisters.values()
            ]
        }

        with open(self._config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get(self, sister_id: str) -> Optional[Sister]:
        """Get a sister by ID (resolves legacy aliases)."""
        resolved_id = self._alias_map.get(sister_id.lower(), sister_id.lower())
        return self.sisters.get(resolved_id)

    def get_all(self) -> List[Sister]:
        """Get all enabled sisters."""
        return [s for s in self.sisters.values() if s.enabled]

    def get_by_domain(self, domain: str) -> List[Sister]:
        """Get sisters filtered by domain."""
        return [s for s in self.get_all() if s.domain.lower() == domain.lower()]

    def match_task(self, query: str, top_k: int = 3) -> List[Sister]:
        """
        Match a task query to the best sisters.
        Returns top_k sisters sorted by relevance score.
        """
        scored = []
        for sister in self.get_all():
            score = sister.matches_keywords(query)
            if score > 0:
                scored.append((score, sister))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_k]]

    def match_task_single(self, query: str) -> Optional[Sister]:
        """Get the single best sister match for a query."""
        matches = self.match_task(query, top_k=1)
        return matches[0] if matches else None

    def list_sisters(self) -> List[Dict[str, Any]]:
        """List all sisters as dicts for API/CLI consumption."""
        return [
            {
                'id': s.id,
                'name': s.name,
                'role': s.role,
                'archetype': s.archetype,
                'domain': s.domain,
                'delegation_scope': s.delegation_scope,
                'risk_level': s.risk_level,
                'description': s.description,
                'legacy_aliases': s.legacy_aliases,
                'enabled': s.enabled,
                'model_preference': s.model_preference,
            }
            for s in self.get_all()
        ]

    def reload(self):
        """Reload registry from config file."""
        self.sisters.clear()
        self._alias_map.clear()
        self._load_registry()


# Global registry instance
_registry: Optional[SisterRegistry] = None


def get_registry() -> SisterRegistry:
    """Get the global sister registry instance."""
    global _registry
    if _registry is None:
        _registry = SisterRegistry()
    return _registry


def reload_registry():
    """Force reload the global registry."""
    global _registry
    _registry = SisterRegistry()
    return _registry
