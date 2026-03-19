"""
Skill Registry Routes

Endpoints:
- GET /api/skills - list all registered skills
- GET /api/skills/search - Search skills by query
- GET /api/skills/{name} - Get specific skill details
- GET /api/skills/agent/{agent} - Get skills for a specific agent
- GET /api/skills/category/{category} - Get skills by category
- GET /api/skills/prompt-context - Get skill context for agent prompts
- POST /api/skills/register - Register a custom skill
"""

from fastapi import APIRouter, Request, Query
from loguru import logger

router = APIRouter(tags=["skills"])


@router.get("/api/skills")
async def list_skills():
    """list all registered skills in the swarm."""
    from Cosmos.core.skill_registry import get_skill_registry, initialize_registry
    registry = get_skill_registry()
    if not registry._discovered:
        await initialize_registry()
    return registry.to_dict()


@router.get("/api/skills/search")
async def search_skills(q: str = Query(..., description="Search query")):
    """Search skills by natural language query."""
    from Cosmos.core.skill_registry import get_skill_registry, initialize_registry
    registry = get_skill_registry()
    if not registry._discovered:
        await initialize_registry()
    results = registry.find_skills(q)
    return {
        "query": q,
        "results": [
            {"name": s.name, "description": s.description, "category": s.category.value, "agents": s.agents}
            for s in results
        ],
    }


@router.get("/api/skills/prompt-context")
async def get_prompt_context(agent: str = Query(None)):
    """Get skill context formatted for injection into agent prompts."""
    from Cosmos.core.skill_registry import get_skill_registry, initialize_registry
    registry = get_skill_registry()
    if not registry._discovered:
        await initialize_registry()
    return {"context": registry.get_prompt_context(agent=agent)}


@router.get("/api/skills/agent/{agent}")
async def get_agent_skills(agent: str):
    """Get all skills available to a specific agent."""
    from Cosmos.core.skill_registry import get_skill_registry, initialize_registry
    registry = get_skill_registry()
    if not registry._discovered:
        await initialize_registry()
    skills = registry.get_agent_skills(agent)
    return {
        "agent": agent,
        "skill_count": len(skills),
        "skills": [{"name": s.name, "description": s.description, "category": s.category.value} for s in skills],
    }


@router.get("/api/skills/category/{category}")
async def get_category_skills(category: str):
    """Get all skills in a specific category."""
    from Cosmos.core.skill_registry import get_skill_registry, initialize_registry, SkillCategory
    registry = get_skill_registry()
    if not registry._discovered:
        await initialize_registry()
    try:
        cat = SkillCategory(category)
    except ValueError:
        return {"error": f"Invalid category. Valid: {[c.value for c in SkillCategory]}"}
    skills = registry.get_category_skills(cat)
    return {
        "category": category,
        "skill_count": len(skills),
        "skills": [{"name": s.name, "description": s.description, "agents": s.agents} for s in skills],
    }


@router.get("/api/skills/{name}")
async def get_skill(name: str):
    """Get detailed information about a specific skill."""
    from Cosmos.core.skill_registry import get_skill_registry, initialize_registry
    registry = get_skill_registry()
    if not registry._discovered:
        await initialize_registry()
    skill = registry.get_skill(name)
    if not skill:
        return {"error": f"Skill '{name}' not found"}
    return {
        "name": skill.name,
        "description": skill.description,
        "category": skill.category.value,
        "module_path": skill.module_path,
        "function_name": skill.function_name,
        "agents": skill.agents,
        "keywords": skill.keywords,
        "parameters": skill.parameters,
        "requires_api_key": skill.requires_api_key,
        "cooldown_seconds": skill.cooldown_seconds,
        "enabled": skill.enabled,
        "source": skill.source,
        "usage_count": skill.usage_count,
        "success_rate": skill.success_rate,
        "last_used": skill.last_used,
    }


@router.post("/api/skills/register")
async def register_custom_skill(request: Request):
    """Register a custom skill in the registry."""
    from Cosmos.core.skill_registry import get_skill_registry, initialize_registry, Skill, SkillCategory
    registry = get_skill_registry()
    if not registry._discovered:
        await initialize_registry()
    try:
        body = await request.json()
        skill = Skill(
            name=body["name"],
            description=body["description"],
            category=SkillCategory(body.get("category", "custom")),
            module_path=body.get("module_path", ""),
            function_name=body.get("function_name", ""),
            agents=body.get("agents", []),
            keywords=body.get("keywords", []),
            parameters=body.get("parameters", {}),
            source="custom",
        )
        registry.register_skill(skill)
        registry.save_custom_skills()
        return {"status": "registered", "name": skill.name}
    except KeyError as e:
        return {"error": f"Missing required field: {e}"}
    except Exception as e:
        logger.error(f"Failed to register custom skill: {e}")
        return {"error": str(e)}
