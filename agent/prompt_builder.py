"""System prompt assembly — façade.

All logic moved to agent/prompts/. This file re-exports the public API
for backwards compatibility.
"""

from agent.prompts.constants import (
    DEFAULT_AGENT_IDENTITY,
    MEMORY_GUIDANCE,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
    OPENAI_MODEL_EXECUTION_GUIDANCE,
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE,
    DEVELOPER_ROLE_MODELS,
    PLATFORM_HINTS,
    WSL_ENVIRONMENT_HINT,
    CONTEXT_FILE_MAX_CHARS,
    CONTEXT_TRUNCATE_HEAD_RATIO,
    CONTEXT_TRUNCATE_TAIL_RATIO,
)

from agent.prompts.environment import (
    build_environment_hints,
)

from agent.prompts.context_files import (
    _CONTEXT_THREAT_PATTERNS,
    _CONTEXT_INVISIBLE_CHARS,
    _scan_context_content,
    _find_git_root,
    _find_hermes_md,
    _strip_yaml_frontmatter,
    _truncate_content,
    _load_hermes_md,
    _load_agents_md,
    _load_claude_md,
    _load_cursorrules,
    load_soul_md,
    build_context_files_prompt,
)

from agent.prompts.skills_prompt import (
    _SKILLS_PROMPT_CACHE_MAX,
    _SKILLS_PROMPT_CACHE_LOCK,
    _SKILLS_SNAPSHOT_VERSION,
    _skills_prompt_snapshot_path,
    clear_skills_system_prompt_cache,
    _build_skills_manifest,
    _load_skills_snapshot,
    _write_skills_snapshot,
    _build_snapshot_entry,
    _parse_skill_file,
    _skill_should_show,
    build_skills_system_prompt,
    build_nous_subscription_prompt,
)

__all__ = [
    # constants
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "TOOL_USE_ENFORCEMENT_GUIDANCE",
    "TOOL_USE_ENFORCEMENT_MODELS",
    "OPENAI_MODEL_EXECUTION_GUIDANCE",
    "GOOGLE_MODEL_OPERATIONAL_GUIDANCE",
    "DEVELOPER_ROLE_MODELS",
    "PLATFORM_HINTS",
    "WSL_ENVIRONMENT_HINT",
    "CONTEXT_FILE_MAX_CHARS",
    "CONTEXT_TRUNCATE_HEAD_RATIO",
    "CONTEXT_TRUNCATE_TAIL_RATIO",
    # environment
    "build_environment_hints",
    # context_files
    "_CONTEXT_THREAT_PATTERNS",
    "_CONTEXT_INVISIBLE_CHARS",
    "_scan_context_content",
    "_find_git_root",
    "_find_hermes_md",
    "_strip_yaml_frontmatter",
    "_truncate_content",
    "_load_hermes_md",
    "_load_agents_md",
    "_load_claude_md",
    "_load_cursorrules",
    "load_soul_md",
    "build_context_files_prompt",
    # skills_prompt
    "_SKILLS_PROMPT_CACHE_MAX",
    "_SKILLS_PROMPT_CACHE_LOCK",
    "_SKILLS_SNAPSHOT_VERSION",
    "_skills_prompt_snapshot_path",
    "clear_skills_system_prompt_cache",
    "_build_skills_manifest",
    "_load_skills_snapshot",
    "_write_skills_snapshot",
    "_build_snapshot_entry",
    "_parse_skill_file",
    "_skill_should_show",
    "build_skills_system_prompt",
    "build_nous_subscription_prompt",
]
