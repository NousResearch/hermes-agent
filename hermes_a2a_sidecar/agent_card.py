from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
)

from .config import DEFAULT_SKILLS, SidecarConfig

_SKILL_DESCRIPTIONS = {
    "delegate_task_to_sasha_hermes": "Delegate a reviewed, policy-gated task to Sasha's Hermes Kanban queue.",
    "delegate_engineering_task": "Submit an engineering task for a Hermes engineering worker.",
    "delegate_research_task": "Submit a research/synthesis task for a Hermes worker.",
    "request_summary": "Request a concise summary or status artifact from Hermes.",
    "submit_artifact_for_review": "Submit a remote artifact handle for checksum/policy review before use.",
}


def _security_requirement(*scopes: str) -> SecurityRequirement:
    req = SecurityRequirement()
    req.schemes["bearer"].list.extend(scopes)
    return req


def build_agent_card(config: SidecarConfig, *, extended: bool = False) -> AgentCard:
    """Build a standards-compliant A2A Agent Card for the sidecar."""

    all_skills = []
    seen: set[str] = set()
    for peer in config.peers.values():
        for skill in peer.allowed_skills:
            if skill and skill not in seen:
                seen.add(skill)
                all_skills.append(skill)
    if not all_skills:
        all_skills = list(DEFAULT_SKILLS)
    if not extended:
        # Public card stays intentionally sparse. Per-peer fine-grained scopes
        # are enforced after auth, not advertised to everyone on the internet.
        all_skills = [s for s in all_skills if s in DEFAULT_SKILLS]

    card = AgentCard(
        name=config.agent_name,
        description=config.description,
        version=config.version,
        provider=AgentProvider(organization=config.provider_name, url=config.provider_url),
        capabilities=AgentCapabilities(streaming=True, push_notifications=False, extended_agent_card=True),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        supported_interfaces=[
            AgentInterface(
                url=config.rpc_url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        documentation_url="https://hermes-agent.nousresearch.com/docs/user-guide/features/a2a-sidecar",
    )
    card.security_schemes["bearer"].CopyFrom(
        SecurityScheme(
            http_auth_security_scheme=HTTPAuthSecurityScheme(
                scheme="bearer",
                bearer_format="opaque-or-perimeter-token",
                description="Bearer token, Cloudflare Access/OIDC service token, or mTLS/Tailscale-authenticated reverse proxy identity.",
            )
        )
    )
    card.security_requirements.append(_security_requirement("a2a:delegate"))

    for skill in all_skills:
        card.skills.append(
            AgentSkill(
                id=skill,
                name=skill.replace("_", " ").title(),
                description=_SKILL_DESCRIPTIONS.get(skill, "Hermes A2A delegated task skill."),
                tags=["hermes", "a2a", "delegation"],
                examples=["Send a task message with metadata.skill set to this skill id."],
                input_modes=["text/plain", "application/json"],
                output_modes=["text/plain", "application/json"],
                security_requirements=[_security_requirement(f"a2a:{skill}")],
            )
        )
    return card
