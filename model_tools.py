#!/usr/bin/env python3
"""
Model Tools Module

This module constructs tool schemas and handlers for AI model API calls.
It imports tools from various toolset modules and provides a unified interface
for defining tools and executing function calls.

Currently supports:
- Web tools (search, extract, crawl) from web_tools.py
- Terminal tools (simple command execution, no session persistence) from simple_terminal_tool.py
- Vision tools (image analysis) from vision_tools.py
- Mixture of Agents tools (collaborative multi-model reasoning) from mixture_of_agents_tool.py
- Image generation tools (text-to-image with upscaling) from image_generation_tool.py

Usage:
    from model_tools import get_tool_definitions, handle_function_call
    
    # Get all available tool definitions for model API
    tools = get_tool_definitions()
    
    # Get specific toolsets
    web_tools = get_tool_definitions(enabled_toolsets=['web_tools'])
    
    # Handle function calls from model
    result = handle_function_call("web_search", {"query": "Python"})
"""

import json
import asyncio
import os
from typing import Dict, Any, List, Optional, Tuple

from tools.web_tools import web_search_tool, web_extract_tool, web_crawl_tool, check_firecrawl_api_key
from tools.terminal_tool import terminal_tool, check_terminal_requirements, TERMINAL_TOOL_DESCRIPTION, cleanup_vm
# File manipulation tools (read, write, patch, search)
from tools.file_tools import read_file_tool, write_file_tool, patch_tool, search_tool
from tools import check_file_requirements
# Hecate/MorphCloud terminal tool (cloud VMs) - available as alternative backend
from tools.terminal_hecate import terminal_hecate_tool, check_hecate_requirements, TERMINAL_HECATE_DESCRIPTION
from tools.vision_tools import vision_analyze_tool, check_vision_requirements
from tools.mixture_of_agents_tool import mixture_of_agents_tool, check_moa_requirements
from tools.image_generation_tool import image_generate_tool, check_image_generation_requirements
from tools.skills_tool import skills_list, skill_view, check_skills_requirements, SKILLS_TOOL_DESCRIPTION
# Agent-managed skill creation/editing
from tools.skill_manager_tool import skill_manage, check_skill_manage_requirements, SKILL_MANAGE_SCHEMA
# RL Training tools (Tinker-Atropos)
from tools.rl_training_tool import (
    rl_list_environments,
    rl_select_environment,
    rl_get_current_config,
    rl_edit_config,
    rl_start_training,
    rl_check_status,
    rl_stop_training,
    rl_get_results,
    rl_list_runs,
    rl_test_inference,
    check_rl_api_keys,
)
# Cronjob management tools (CLI-only)
from tools.cronjob_tools import (
    schedule_cronjob,
    list_cronjobs,
    remove_cronjob,
    check_cronjob_requirements,
    get_cronjob_tool_definitions,
    SCHEDULE_CRONJOB_SCHEMA,
    LIST_CRONJOBS_SCHEMA,
    REMOVE_CRONJOB_SCHEMA
)
# Browser automation tools (agent-browser + Browserbase)
from tools.browser_tool import (
    browser_navigate,
    browser_snapshot,
    browser_click,
    browser_type,
    browser_scroll,
    browser_back,
    browser_press,
    browser_close,
    browser_get_images,
    browser_vision,
    cleanup_browser,
    check_browser_requirements,
    BROWSER_TOOL_SCHEMAS
)
# Text-to-speech tool (Edge TTS / ElevenLabs / OpenAI)
from tools.tts_tool import text_to_speech_tool, check_tts_requirements
# Planning & task management tool
from tools.todo_tool import todo_tool, check_todo_requirements, TODO_SCHEMA
# Persistent memory tool
from tools.memory_tool import memory_tool, check_memory_requirements, MEMORY_SCHEMA
# Session search tool (past conversation recall with summarization)
from tools.session_search_tool import session_search, check_session_search_requirements, SESSION_SEARCH_SCHEMA
from toolsets import (
    get_toolset, resolve_toolset, resolve_multiple_toolsets,
    get_all_toolsets, get_toolset_names, validate_toolset,
    get_toolset_info, print_toolset_tree
)


# =============================================================================
# Tool Availability Checking
# =============================================================================

# Maps toolsets to their required API keys/environment variables
TOOLSET_REQUIREMENTS = {
    "web": {
        "name": "Web Search & Extract",
        "env_vars": ["FIRECRAWL_API_KEY"],
        "check_fn": check_firecrawl_api_key,
        "setup_url": "https://firecrawl.dev/",
        "tools": ["web_search", "web_extract"],
    },
    "vision": {
        "name": "Vision (Image Analysis)",
        "env_vars": ["OPENROUTER_API_KEY"],
        "check_fn": check_vision_requirements,
        "setup_url": "https://openrouter.ai/keys",
        "tools": ["vision_analyze"],
    },
    "moa": {
        "name": "Mixture of Agents",
        "env_vars": ["OPENROUTER_API_KEY"],
        "check_fn": check_moa_requirements,
        "setup_url": "https://openrouter.ai/keys",
        "tools": ["mixture_of_agents"],
    },
    "image_gen": {
        "name": "Image Generation",
        "env_vars": ["FAL_KEY"],
        "check_fn": check_image_generation_requirements,
        "setup_url": "https://fal.ai/",
        "tools": ["image_generate"],
    },
    "browser": {
        "name": "Browser Automation",
        "env_vars": ["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID"],
        "check_fn": check_browser_requirements,
        "setup_url": "https://browserbase.com/",
        "tools": ["browser_navigate", "browser_snapshot", "browser_click", "browser_type"],
    },
    "terminal": {
        "name": "Terminal/Command Execution",
        "env_vars": [],  # No API key required, just system dependencies
        "check_fn": check_terminal_requirements,
        "setup_url": None,
        "tools": ["terminal"],
    },
    "skills": {
        "name": "Skills Knowledge Base",
        "env_vars": [],  # Just needs skills directory
        "check_fn": check_skills_requirements,
        "setup_url": None,
        "tools": ["skills_list", "skill_view", "skill_manage"],
    },
    "rl": {
        "name": "RL Training (Tinker-Atropos)",
        "env_vars": ["TINKER_API_KEY", "WANDB_API_KEY"],
        "check_fn": check_rl_api_keys,
        "setup_url": "https://wandb.ai/authorize",
        "tools": [
            "rl_list_environments", "rl_select_environment",
            "rl_get_current_config", "rl_edit_config",
            "rl_start_training", "rl_check_status",
            "rl_stop_training", "rl_get_results",
            "rl_list_runs", "rl_test_inference",
        ],
    },
    "file": {
        "name": "File Operations (read, write, patch, search)",
        "env_vars": [],  # Uses terminal backend, no additional requirements
        "check_fn": check_file_requirements,
        "setup_url": None,
        "tools": ["read_file", "write_file", "patch", "search"],
    },
    "tts": {
        "name": "Text-to-Speech",
        "env_vars": [],  # Edge TTS needs no key; premium providers checked at runtime
        "check_fn": check_tts_requirements,
        "setup_url": None,
        "tools": ["text_to_speech"],
    },
    "todo": {
        "name": "Planning & Task Management",
        "env_vars": [],  # Pure in-memory, no external deps
        "check_fn": check_todo_requirements,
        "setup_url": None,
        "tools": ["todo"],
    },
    "memory": {
        "name": "Persistent Memory",
        "env_vars": [],  # File-based, no external deps
        "check_fn": check_memory_requirements,
        "setup_url": None,
        "tools": ["memory"],
    },
    "session_search": {
        "name": "Session History Search",
        "env_vars": ["OPENROUTER_API_KEY"],  # Needs summarizer model
        "check_fn": check_session_search_requirements,
        "setup_url": "https://openrouter.ai/keys",
        "tools": ["session_search"],
    },
}


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Check which tool categories are available based on API keys and requirements.
    
    Returns:
        Tuple containing:
        - List of available toolset names
        - List of dicts with info about unavailable toolsets and what's missing
    """
    available = []
    unavailable = []
    
    for toolset_id, info in TOOLSET_REQUIREMENTS.items():
        if info["check_fn"]():
            available.append(toolset_id)
        else:
            # Figure out what's missing
            missing_vars = [var for var in info["env_vars"] if not os.getenv(var)]
            unavailable.append({
                "id": toolset_id,
                "name": info["name"],
                "missing_vars": missing_vars,
                "setup_url": info["setup_url"],
                "tools": info["tools"],
            })
    
    return available, unavailable


def print_tool_availability_warnings(unavailable: List[Dict[str, Any]], prefix: str = ""):
    """Print warnings about unavailable tools."""
    if not unavailable:
        return
    
    # Filter to only those missing API keys (not system dependencies)
    api_key_missing = [u for u in unavailable if u["missing_vars"]]
    
    if api_key_missing:
        print(f"{prefix}⚠️  Some tools are disabled due to missing API keys:")
        for item in api_key_missing:
            vars_str = ", ".join(item["missing_vars"])
            print(f"{prefix}   • {item['name']}: missing {vars_str}")
            if item["setup_url"]:
                print(f"{prefix}     Get key at: {item['setup_url']}")
        print(f"{prefix}   Run 'hermes setup' to configure API keys")
        print()


def get_tool_availability_summary() -> Dict[str, Any]:
    """
    Get a summary of tool availability for display in status/doctor commands.
    
    Returns:
        Dict with 'available' and 'unavailable' lists of tool info
    """
    available, unavailable = check_tool_availability()
    
    return {
        "available": [
            {"id": tid, "name": TOOLSET_REQUIREMENTS[tid]["name"], "tools": TOOLSET_REQUIREMENTS[tid]["tools"]}
            for tid in available
        ],
        "unavailable": unavailable,
    }


def get_web_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for web tools in OpenAI's expected format.
    
    Returns:
        List[Dict]: List of web tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information on any topic. Returns up to 5 relevant results with titles and URLs. Uses advanced search depth for comprehensive results. PREFERRED over browser tools for finding information - faster and more cost-effective. Use browser tools only when you need to interact with pages (click, fill forms, handle dynamic content).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_extract",
                "description": "Extract and read the full content from specific web page URLs. Useful for getting detailed information from webpages found through search. The content returned will be excerpts and key points summarized with an LLM to reduce impact on the context window. PREFERRED over browser tools for reading page content - faster and more cost-effective. Use browser tools only when pages require interaction or have dynamic content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of URLs to extract content from (max 5 URLs per call)",
                            "maxItems": 5
                        }
                    },
                    "required": ["urls"]
                }
            }
        },
    ]

def get_terminal_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for terminal tools in OpenAI's expected format.
    
    Uses mini-swe-agent backend (local/docker/modal) by default.

    Returns:
        List[Dict]: List of terminal tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": TERMINAL_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute on the VM"
                        },
                        "background": {
                            "type": "boolean",
                            "description": "Whether to run the command in the background (default: false)",
                            "default": False
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds (optional)",
                            "minimum": 1
                        },
                        "workdir": {
                            "type": "string",
                            "description": "Working directory for this command (absolute path). Defaults to the session working directory."
                        },
                        "check_interval": {
                            "type": "integer",
                            "description": "Seconds between automatic status checks for background processes (gateway/messaging only, minimum 30). When set, I'll proactively report progress.",
                            "minimum": 30
                        },
                        "pty": {
                            "type": "boolean",
                            "description": "Run in pseudo-terminal (PTY) mode for interactive CLI tools like Codex, Claude Code, or Python REPL. Only works with local and SSH backends. Default: false.",
                            "default": False
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ]


def get_vision_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for vision tools in OpenAI's expected format.
    
    Returns:
        List[Dict]: List of vision tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "vision_analyze",
                "description": "Analyze images using AI vision. Accepts HTTP/HTTPS URLs or local file paths (e.g. from the image cache). Provides comprehensive image description and answers specific questions about the image content. Perfect for understanding visual content, reading text in images, identifying objects, analyzing scenes, and extracting visual information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "The URL or local file path of the image to analyze. Accepts publicly accessible HTTP/HTTPS URLs or local file paths (e.g. /home/user/.hermes/image_cache/abc123.jpg)."
                        },
                        "question": {
                            "type": "string",
                            "description": "Your specific question or request about the image to resolve. The AI will automatically provide a complete image description AND answer your specific question."
                        }
                    },
                    "required": ["image_url", "question"]
                }
            }
        }
    ]


def get_moa_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for Mixture-of-Agents tools in OpenAI's expected format.
    
    Returns:
        List[Dict]: List of MoA tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "mixture_of_agents",
                "description": "Process extremely difficult problems requiring intense reasoning using a Mixture-of-Agents. This tool leverages multiple frontier language models to collaboratively solve complex tasks that single models struggle with. Uses a fixed 2-layer architecture: reference models generate diverse responses, then an aggregator synthesizes the best solution. Best for: complex mathematical proofs, advanced coding problems, multi-step analytical reasoning, precise and complex STEM problems, algorithm design, and problems requiring diverse domain expertise.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_prompt": {
                            "type": "string",
                            "description": "The complex query or problem to solve using multiple AI models. Should be a challenging problem that benefits from diverse perspectives and collaborative reasoning."
                        }
                    },
                    "required": ["user_prompt"]
                }
            }
        }
    ]


def get_image_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for image generation tools in OpenAI's expected format.
    
    Returns:
        List[Dict]: List of image generation tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "image_generate",
                "description": "Generate high-quality images from text prompts using FLUX 2 Pro model with automatic 2x upscaling. Creates detailed, artistic images that are automatically upscaled for hi-rez results. Returns a single upscaled image URL. Display it using markdown: ![description](URL)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt describing the desired image. Be detailed and descriptive."
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "enum": ["landscape", "square", "portrait"],
                            "description": "The aspect ratio of the generated image. 'landscape' is 16:9 wide, 'portrait' is 16:9 tall, 'square' is 1:1.",
                            "default": "landscape"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ]


def get_skills_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for skills tools in OpenAI's expected format.
    
    Returns:
        List[Dict]: List of skills tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "skills_list",
                "description": "List available skills (name + description). Use skill_view(name) to load full content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Optional category filter to narrow results"
                        }
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "skill_view",
                "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The skill name (use skills_list to see available skills)"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content."
                        }
                    },
                    "required": ["name"]
                }
            }
        }
    ]


def get_skill_manage_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for the skill management tool.
    
    Returns:
        List[Dict]: List containing the skill_manage tool definition compatible with OpenAI API
    """
    return [{"type": "function", "function": SKILL_MANAGE_SCHEMA}]


def get_browser_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for browser automation tools in OpenAI's expected format.
    
    Uses agent-browser CLI with Browserbase cloud execution.
    
    Returns:
        List[Dict]: List of browser tool definitions compatible with OpenAI API
    """
    return [{"type": "function", "function": schema} for schema in BROWSER_TOOL_SCHEMAS]


def get_cronjob_tool_definitions_formatted() -> List[Dict[str, Any]]:
    """
    Get tool definitions for cronjob management tools in OpenAI's expected format.
    
    These tools are only available in the hermes-cli toolset (interactive CLI mode).
    
    Returns:
        List[Dict]: List of cronjob tool definitions compatible with OpenAI API
    """
    return [{"type": "function", "function": schema} for schema in [
        SCHEDULE_CRONJOB_SCHEMA,
        LIST_CRONJOBS_SCHEMA,
        REMOVE_CRONJOB_SCHEMA
    ]]


def get_rl_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for RL training tools in OpenAI's expected format.
    
    These tools enable running RL training through Tinker-Atropos.
    
    Returns:
        List[Dict]: List of RL tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "rl_list_environments",
                "description": "List all available RL environments. Returns environment names, paths, and descriptions. TIP: Read the file_path with file tools to understand how each environment works (verifiers, data loading, rewards).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_select_environment",
                "description": "Select an RL environment for training. Loads the environment's default configuration. After selecting, use rl_get_current_config() to see settings and rl_edit_config() to modify them.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name of the environment to select (from rl_list_environments)"
                        }
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_get_current_config",
                "description": "Get the current environment configuration. Returns only fields that can be modified: group_size, max_token_length, total_steps, steps_per_eval, use_wandb, wandb_name, max_num_workers.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_edit_config",
                "description": "Update a configuration field. Use rl_get_current_config() first to see all available fields for the selected environment. Each environment has different configurable options. Infrastructure settings (tokenizer, URLs, lora_rank, learning_rate) are locked.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "description": "Name of the field to update (get available fields from rl_get_current_config)"
                        },
                        "value": {
                            "description": "New value for the field"
                        }
                    },
                    "required": ["field", "value"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_start_training",
                "description": "Start a new RL training run with the current environment and config. Most training parameters (lora_rank, learning_rate, etc.) are fixed. Use rl_edit_config() to set group_size, batch_size, wandb_project before starting. WARNING: Training takes hours.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_check_status",
                "description": "Get status and metrics for a training run. RATE LIMITED: enforces 30-minute minimum between checks for the same run. Returns WandB metrics: step, state, reward_mean, loss, percent_correct.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "The run ID from rl_start_training()"
                        }
                    },
                    "required": ["run_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_stop_training",
                "description": "Stop a running training job. Use if metrics look bad, training is stagnant, or you want to try different settings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "The run ID to stop"
                        }
                    },
                    "required": ["run_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_get_results",
                "description": "Get final results and metrics for a completed training run. Returns final metrics and path to trained weights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "run_id": {
                            "type": "string",
                            "description": "The run ID to get results for"
                        }
                    },
                    "required": ["run_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_list_runs",
                "description": "List all training runs (active and completed) with their status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rl_test_inference",
                "description": "Quick inference test for any environment. Runs a few steps of inference + scoring using OpenRouter. Default: 3 steps × 16 completions = 48 rollouts per model, testing 3 models = 144 total. Tests environment loading, prompt construction, inference parsing, and verifier logic. Use BEFORE training to catch issues.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "num_steps": {
                            "type": "integer",
                            "description": "Number of steps to run (default: 3, recommended max for testing)",
                            "default": 3
                        },
                        "group_size": {
                            "type": "integer",
                            "description": "Completions per step (default: 16, like training)",
                            "default": 16
                        },
                        "models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of OpenRouter model IDs. Default: qwen/qwen3-8b, z-ai/glm-4.7-flash, minimax/minimax-m2.1"
                        }
                    },
                    "required": []
                }
            }
        }
    ]


def get_file_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for file manipulation tools in OpenAI's expected format.
    
    File tools operate via the terminal backend and support any environment
    (local, docker, singularity, ssh, modal).
    
    Returns:
        List[Dict]: List of file tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read a file with pagination support. Preferred over 'cat' in the terminal because it "
                    "provides line numbers, handles binary/image files, and suggests similar filenames if "
                    "the file is not found.\n\n"
                    "**Output format:** Each line is returned as 'LINE_NUM|CONTENT' for easy reference.\n"
                    "**Binary files:** Detected automatically; images (png/jpg/gif/webp) are returned as base64 with MIME type and dimensions.\n"
                    "**Large files:** Use offset and limit to paginate. The response includes total line count and a hint for the next page.\n"
                    "**Paths:** Supports absolute paths, relative paths (from working directory), and ~ expansion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (absolute, relative, or ~/path)"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (1-indexed, default: 1)",
                            "default": 1,
                            "minimum": 1
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default: 500, max: 2000)",
                            "default": 500,
                            "maximum": 2000
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write content to a file, completely replacing any existing content. Creates parent "
                    "directories automatically if they don't exist. Preferred over 'echo' or heredoc in the "
                    "terminal because it safely handles special characters, newlines, and shell metacharacters "
                    "without escaping issues.\n\n"
                    "**Important:** This OVERWRITES the entire file. To make targeted edits to an existing file, "
                    "use the 'patch' tool instead.\n"
                    "**Paths:** Supports absolute paths, relative paths, and ~ expansion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write (will be created if it doesn't exist, overwritten if it does)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete content to write to the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "patch",
                "description": (
                    "Modify existing files using targeted edits. Preferred over 'sed' or manual rewriting because "
                    "it uses intelligent fuzzy matching that tolerates minor whitespace and indentation differences, "
                    "and auto-runs syntax checks (Python, JS, TS, Go, Rust) after editing.\n\n"
                    "**Replace mode (recommended):** Find a unique string in the file and replace it. Uses a "
                    "9-strategy fuzzy matching chain (exact → line-trimmed → whitespace-normalized → "
                    "indentation-flexible → context-aware) so small formatting differences won't cause failures. "
                    "Returns a unified diff showing exactly what changed.\n\n"
                    "**Patch mode:** Apply multi-file changes using V4A patch format for large-scale edits across "
                    "multiple files in one call.\n\n"
                    "**Auto-lint:** After every edit, automatically runs syntax checks and reports errors so you "
                    "can fix them immediately."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["replace", "patch"],
                            "description": "Edit mode: 'replace' for targeted find-and-replace, 'patch' for V4A multi-file patches",
                            "default": "replace"
                        },
                        "path": {
                            "type": "string",
                            "description": "File path to edit (required for 'replace' mode)"
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Text to find in the file (required for 'replace' mode). Must be unique in the file unless replace_all=true. Include enough surrounding context to ensure uniqueness."
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text (required for 'replace' mode). Can be empty string to delete the matched text."
                        },
                        "replace_all": {
                            "type": "boolean",
                            "description": "Replace all occurrences instead of requiring a unique match (default: false)",
                            "default": False
                        },
                        "patch": {
                            "type": "string",
                            "description": "V4A format patch content (required for 'patch' mode). Format:\n*** Begin Patch\n*** Update File: path/to/file\n@@ context hint @@\n context line\n-removed line\n+added line\n*** End Patch"
                        }
                    },
                    "required": ["mode"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Search for content inside files or find files by name. Preferred over 'grep' or 'find' "
                    "in the terminal because it uses ripgrep (fast) with automatic fallback to grep, handles "
                    "pagination, and returns structured results sorted by modification time (newest first).\n\n"
                    "**Content search (target='content'):** Regex-powered search inside files with optional "
                    "file type filtering and context lines. Three output modes: full matches with line numbers, "
                    "file paths only, or match counts per file.\n\n"
                    "**File search (target='files'):** Find files by glob pattern (e.g., '*.py', '*config*'). "
                    "Results sorted by modification time so recently changed files appear first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "For target='content': regex pattern to search for. For target='files': glob pattern (e.g., '*.py', '*config*')"
                        },
                        "target": {
                            "type": "string",
                            "enum": ["content", "files"],
                            "description": "Search mode: 'content' searches inside files (like grep/rg), 'files' searches for files by name (like find/glob)",
                            "default": "content"
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory or file to search in (default: current working directory)",
                            "default": "."
                        },
                        "file_glob": {
                            "type": "string",
                            "description": "Filter files by pattern when target='content' (e.g., '*.py' to only search Python files)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 50)",
                            "default": 50
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Skip first N results for pagination (default: 0)",
                            "default": 0
                        },
                        "output_mode": {
                            "type": "string",
                            "enum": ["content", "files_only", "count"],
                            "description": "Output format for content search: 'content' shows matching lines with line numbers, 'files_only' lists file paths, 'count' shows match counts per file",
                            "default": "content"
                        },
                        "context": {
                            "type": "integer",
                            "description": "Number of lines to show before and after each match (only for target='content', output_mode='content')",
                            "default": 0
                        }
                    },
                    "required": ["pattern"]
                }
            }
        }
    ]


def get_tts_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for text-to-speech tools in OpenAI's expected format.
    
    Returns:
        List[Dict]: List of TTS tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "text_to_speech",
                "description": "Convert text to speech audio. Returns a MEDIA: path that the platform delivers as a voice message. On Telegram it plays as a voice bubble, on Discord/WhatsApp as an audio attachment. In CLI mode, saves to ~/voice-memos/. Voice and provider are user-configured, not model-selected.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to convert to speech. Keep under 4000 characters."
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Optional custom file path to save the audio. Defaults to ~/voice-memos/<timestamp>.mp3"
                        }
                    },
                    "required": ["text"]
                }
            }
        }
    ]


def get_todo_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for the todo (planning/task management) tool.
    
    Returns:
        List[Dict]: List containing the todo tool definition compatible with OpenAI API
    """
    return [{"type": "function", "function": TODO_SCHEMA}]


def get_memory_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for the persistent memory tool.
    
    Returns:
        List[Dict]: List containing the memory tool definition compatible with OpenAI API
    """
    return [{"type": "function", "function": MEMORY_SCHEMA}]


def get_session_search_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for the session history search tool.
    
    Returns:
        List[Dict]: List containing the session_search tool definition compatible with OpenAI API
    """
    return [{"type": "function", "function": SESSION_SEARCH_SCHEMA}]


def get_send_message_tool_definitions():
    """Tool definitions for cross-channel messaging."""
    return [
        {
            "type": "function",
            "function": {
                "name": "send_message",
                "description": "Send a message to a user or channel on any connected messaging platform. Use this when the user asks you to send something to a different platform, or when delivering notifications/alerts to a specific destination.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Delivery target. Format: 'platform' (uses home channel) or 'platform:chat_id' (specific chat). Examples: 'telegram', 'discord:123456789', 'slack:C01234ABCDE'"
                        },
                        "message": {
                            "type": "string",
                            "description": "The message text to send"
                        }
                    },
                    "required": ["target", "message"]
                }
            }
        }
    ]


def get_process_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for the process management tool.

    The process tool manages background processes started with terminal(background=true).
    Actions: list, poll, log, wait, kill.  Phase 2 adds: write, submit.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "process",
                "description": (
                    "Manage background processes started with terminal(background=true). "
                    "Actions: 'list' (show all), 'poll' (check status + new output), "
                    "'log' (full output with pagination), 'wait' (block until done or timeout), "
                    "'kill' (terminate), 'write' (send raw data to stdin), 'submit' (send data + Enter). "
                    "Use 'wait' when you have nothing else to do and want "
                    "to block until a background process finishes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["list", "poll", "log", "wait", "kill", "write", "submit"],
                            "description": "Action to perform on background processes"
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Process session ID (from terminal background output). Required for poll/log/wait/kill."
                        },
                        "data": {
                            "type": "string",
                            "description": "Text to send to process stdin (for 'write' and 'submit' actions)"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max seconds to block for 'wait' action. Returns partial output on timeout.",
                            "minimum": 1
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line offset for 'log' action (default: last 200 lines)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max lines to return for 'log' action",
                            "minimum": 1
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    ]


def get_all_tool_names() -> List[str]:
    """
    Get the names of all available tools across all toolsets.
    
    Returns:
        List[str]: List of all tool names
    """
    tool_names = []
    
    # Web tools
    if check_firecrawl_api_key():
        tool_names.extend(["web_search", "web_extract"])

    # Terminal tools (mini-swe-agent backend)
    if check_terminal_requirements():
        tool_names.extend(["terminal", "process"])

    # Vision tools
    if check_vision_requirements():
        tool_names.extend(["vision_analyze"])
    
    # MoA tools
    if check_moa_requirements():
        tool_names.extend(["mixture_of_agents"])
    
    # Image generation tools
    if check_image_generation_requirements():
        tool_names.extend(["image_generate"])
    
    # Skills tools
    if check_skills_requirements():
        tool_names.extend(["skills_list", "skill_view", "skill_manage"])
    
    # Browser automation tools
    if check_browser_requirements():
        tool_names.extend([
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_close", "browser_get_images",
            "browser_vision"
        ])
    
    # Cronjob management tools (CLI-only, checked at runtime)
    if check_cronjob_requirements():
        tool_names.extend([
            "schedule_cronjob", "list_cronjobs", "remove_cronjob"
        ])
    
    # RL Training tools
    if check_rl_api_keys():
        tool_names.extend([
            "rl_list_environments", "rl_select_environment",
            "rl_get_current_config", "rl_edit_config",
            "rl_start_training", "rl_check_status",
            "rl_stop_training", "rl_get_results",
            "rl_list_runs", "rl_test_inference"
        ])
    
    # File manipulation tools (use terminal backend)
    if check_file_requirements():
        tool_names.extend([
            "read_file", "write_file", "patch", "search"
        ])
    
    # Text-to-speech tools
    if check_tts_requirements():
        tool_names.extend(["text_to_speech"])
    
    # Planning & task management (always available)
    if check_todo_requirements():
        tool_names.extend(["todo"])
    
    # Persistent memory (always available)
    if check_memory_requirements():
        tool_names.extend(["memory"])
    
    # Session history search
    if check_session_search_requirements():
        tool_names.extend(["session_search"])
    
    # Cross-channel messaging (always available on messaging platforms)
    tool_names.extend(["send_message"])
    
    return tool_names


# Master mapping of every tool name → its toolset.
# This is the single source of truth for all valid tool names in the system.
# Import TOOL_TO_TOOLSET_MAP from here whenever you need to check valid tools.
TOOL_TO_TOOLSET_MAP = {
    "web_search": "web_tools",
    "web_extract": "web_tools",
    "terminal": "terminal_tools",
    "process": "terminal_tools",
    "vision_analyze": "vision_tools",
    "mixture_of_agents": "moa_tools",
    "image_generate": "image_tools",
    # Skills tools
    "skills_list": "skills_tools",
    "skill_view": "skills_tools",
    "skill_manage": "skills_tools",
    # Browser automation tools
    "browser_navigate": "browser_tools",
    "browser_snapshot": "browser_tools",
    "browser_click": "browser_tools",
    "browser_type": "browser_tools",
    "browser_scroll": "browser_tools",
    "browser_back": "browser_tools",
    "browser_press": "browser_tools",
    "browser_close": "browser_tools",
    "browser_get_images": "browser_tools",
    "browser_vision": "browser_tools",
    # Cronjob management tools
    "schedule_cronjob": "cronjob_tools",
    "list_cronjobs": "cronjob_tools",
    "remove_cronjob": "cronjob_tools",
    # RL Training tools
    "rl_list_environments": "rl_tools",
    "rl_select_environment": "rl_tools",
    "rl_get_current_config": "rl_tools",
    "rl_edit_config": "rl_tools",
    "rl_start_training": "rl_tools",
    "rl_check_status": "rl_tools",
    "rl_stop_training": "rl_tools",
    "rl_get_results": "rl_tools",
    "rl_list_runs": "rl_tools",
    "rl_test_inference": "rl_tools",
    # Text-to-speech tools
    "text_to_speech": "tts_tools",
    # File manipulation tools
    "read_file": "file_tools",
    "write_file": "file_tools",
    "patch": "file_tools",
    "search": "file_tools",
    # Cross-channel messaging
    "send_message": "messaging_tools",
    # Planning & task management
    "todo": "todo_tools",
    # Persistent memory
    "memory": "memory_tools",
    # Session history search
    "session_search": "session_search_tools",
}


def get_toolset_for_tool(tool_name: str) -> str:
    """
    Get the toolset that a tool belongs to.
    
    Args:
        tool_name (str): Name of the tool
        
    Returns:
        str: Name of the toolset, or "unknown" if not found
    """
    return TOOL_TO_TOOLSET_MAP.get(tool_name, "unknown")


def get_tool_definitions(
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    quiet_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get tool definitions for model API calls with toolset-based filtering.
    
    This function aggregates tool definitions from available toolsets.
    All tools must be part of a toolset to be accessible. Individual tool
    selection is not supported - use toolsets to organize and select tools.
    
    Args:
        enabled_toolsets (List[str]): Only include tools from these toolsets.
                                     If None, all available tools are included.
        disabled_toolsets (List[str]): Exclude tools from these toolsets.
                                      Applied only if enabled_toolsets is None.
    
    Returns:
        List[Dict]: Filtered list of tool definitions
    
    Examples:
        # Use predefined toolsets
        tools = get_tool_definitions(enabled_toolsets=["research"])
        tools = get_tool_definitions(enabled_toolsets=["development"])
        
        # Combine multiple toolsets
        tools = get_tool_definitions(enabled_toolsets=["web", "vision"])
        
        # All tools except those in terminal toolset
        tools = get_tool_definitions(disabled_toolsets=["terminal"])
        
        # Default - all available tools
        tools = get_tool_definitions()
    """
    # Collect all available tool definitions
    all_available_tools_map = {}
    
    # Map tool names to their definitions
    if check_firecrawl_api_key():
        for tool in get_web_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool

    if check_terminal_requirements():
        for tool in get_terminal_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
        # Process management tool (paired with terminal)
        for tool in get_process_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool

    if check_vision_requirements():
        for tool in get_vision_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    if check_moa_requirements():
        for tool in get_moa_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    if check_image_generation_requirements():
        for tool in get_image_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    if check_skills_requirements():
        for tool in get_skills_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
        for tool in get_skill_manage_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    if check_browser_requirements():
        for tool in get_browser_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # Cronjob management tools (CLI-only)
    if check_cronjob_requirements():
        for tool in get_cronjob_tool_definitions_formatted():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # RL Training tools
    if check_rl_api_keys():
        for tool in get_rl_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # File manipulation tools (use terminal backend)
    if check_file_requirements():
        for tool in get_file_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # Text-to-speech tools
    if check_tts_requirements():
        for tool in get_tts_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # Planning & task management tool
    if check_todo_requirements():
        for tool in get_todo_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # Persistent memory tool
    if check_memory_requirements():
        for tool in get_memory_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # Session history search tool
    if check_session_search_requirements():
        for tool in get_session_search_tool_definitions():
            all_available_tools_map[tool["function"]["name"]] = tool
    
    # Cross-channel messaging (always available on messaging platforms)
    for tool in get_send_message_tool_definitions():
        all_available_tools_map[tool["function"]["name"]] = tool
    
    # Determine which tools to include based on toolsets
    tools_to_include = set()
    
    if enabled_toolsets:
        # Only include tools from enabled toolsets
        for toolset_name in enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved_tools = resolve_toolset(toolset_name)
                tools_to_include.update(resolved_tools)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved_tools) if resolved_tools else 'no tools'}")
            else:
                # Try legacy compatibility
                if toolset_name in ["web_tools", "terminal_tools", "vision_tools", "moa_tools", "image_tools", "skills_tools", "browser_tools", "cronjob_tools"]:
                    # Map legacy names to new system
                    legacy_map = {
                        "web_tools": ["web_search", "web_extract"],
                        "terminal_tools": ["terminal"],
                        "vision_tools": ["vision_analyze"],
                        "moa_tools": ["mixture_of_agents"],
                        "image_tools": ["image_generate"],
                        "skills_tools": ["skills_list", "skill_view", "skill_manage"],
                        "browser_tools": [
                            "browser_navigate", "browser_snapshot", "browser_click",
                            "browser_type", "browser_scroll", "browser_back",
                            "browser_press", "browser_close", "browser_get_images",
                            "browser_vision"
                        ],
                        "cronjob_tools": ["schedule_cronjob", "list_cronjobs", "remove_cronjob"],
                        "rl_tools": [
                            "rl_list_environments", "rl_select_environment",
                            "rl_get_current_config", "rl_edit_config",
                            "rl_start_training", "rl_check_status",
                            "rl_stop_training", "rl_get_results",
                            "rl_list_runs", "rl_test_inference"
                        ],
                        "file_tools": ["read_file", "write_file", "patch", "search"],
                        "tts_tools": ["text_to_speech"]
                    }
                    legacy_tools = legacy_map.get(toolset_name, [])
                    tools_to_include.update(legacy_tools)
                    if not quiet_mode:
                        print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
                else:
                    if not quiet_mode:
                        print(f"⚠️  Unknown toolset: {toolset_name}")
    elif disabled_toolsets:
        # Start with all tools from all toolsets, then remove disabled ones
        # Note: Only tools that are part of toolsets are accessible
        # We need to get all tools from all defined toolsets
        from toolsets import get_all_toolsets
        all_toolset_tools = set()
        for toolset_name in get_all_toolsets():
            resolved_tools = resolve_toolset(toolset_name)
            all_toolset_tools.update(resolved_tools)
        
        # Start with all tools from toolsets
        tools_to_include = all_toolset_tools
        
        # Remove tools from disabled toolsets
        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                resolved_tools = resolve_toolset(toolset_name)
                tools_to_include.difference_update(resolved_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved_tools) if resolved_tools else 'no tools'}")
            else:
                # Try legacy compatibility
                if toolset_name in ["web_tools", "terminal_tools", "vision_tools", "moa_tools", "image_tools", "skills_tools", "browser_tools", "cronjob_tools"]:
                    legacy_map = {
                        "web_tools": ["web_search", "web_extract"],
                        "terminal_tools": ["terminal"],
                        "vision_tools": ["vision_analyze"],
                        "moa_tools": ["mixture_of_agents"],
                        "image_tools": ["image_generate"],
                        "skills_tools": ["skills_list", "skill_view", "skill_manage"],
                        "browser_tools": [
                            "browser_navigate", "browser_snapshot", "browser_click",
                            "browser_type", "browser_scroll", "browser_back",
                            "browser_press", "browser_close", "browser_get_images",
                            "browser_vision"
                        ],
                        "cronjob_tools": ["schedule_cronjob", "list_cronjobs", "remove_cronjob"],
                        "rl_tools": [
                            "rl_list_environments", "rl_select_environment",
                            "rl_get_current_config", "rl_edit_config",
                            "rl_start_training", "rl_check_status",
                            "rl_stop_training", "rl_get_results",
                            "rl_list_runs", "rl_test_inference"
                        ],
                        "file_tools": ["read_file", "write_file", "patch", "search"],
                        "tts_tools": ["text_to_speech"]
                    }
                    legacy_tools = legacy_map.get(toolset_name, [])
                    tools_to_include.difference_update(legacy_tools)
                    if not quiet_mode:
                        print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
                else:
                    if not quiet_mode:
                        print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        # No filtering - include all tools from all defined toolsets
        from toolsets import get_all_toolsets
        for toolset_name in get_all_toolsets():
            resolved_tools = resolve_toolset(toolset_name)
            tools_to_include.update(resolved_tools)
    
    # Build final tool list (only include tools that are available)
    filtered_tools = []
    for tool_name in tools_to_include:
        if tool_name in all_available_tools_map:
            filtered_tools.append(all_available_tools_map[tool_name])
    
    # Sort tools for consistent ordering
    filtered_tools.sort(key=lambda t: t["function"]["name"])
    
    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")
    
    return filtered_tools

def handle_web_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Handle function calls for web tools.
    
    Args:
        function_name (str): Name of the web function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "web_search":
        query = function_args.get("query", "")
        # Always use fixed limit of 5
        limit = 5
        return web_search_tool(query, limit)
    
    elif function_name == "web_extract":
        urls = function_args.get("urls", [])
        # Limit URLs to prevent abuse
        urls = urls[:5] if isinstance(urls, list) else []
        # Run async function -- use existing loop if available (Atropos),
        # otherwise create one (normal CLI)
        try:
            loop = asyncio.get_running_loop()
            # Already in an async context (Atropos) -- run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    lambda: asyncio.run(web_extract_tool(urls, "markdown"))
                ).result(timeout=120)
        except RuntimeError:
            # No running loop (normal CLI) -- use asyncio.run directly
            return asyncio.run(web_extract_tool(urls, "markdown"))
    
    else:
        return json.dumps({"error": f"Unknown web function: {function_name}"}, ensure_ascii=False)

def handle_terminal_function_call(function_name: str, function_args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    """
    Handle function calls for terminal tools.
    
    Uses mini-swe-agent backend (local/docker/modal) by default.

    Args:
        function_name (str): Name of the terminal function to call
        function_args (Dict): Arguments for the function
        task_id (str): Unique identifier for this task to isolate environments between concurrent tasks (optional)

    Returns:
        str: Function result as JSON string
    """
    if function_name == "terminal":
        command = function_args.get("command")
        background = function_args.get("background", False)
        timeout = function_args.get("timeout")
        workdir = function_args.get("workdir")
        check_interval = function_args.get("check_interval")
        pty = function_args.get("pty", False)

        return terminal_tool(command=command, background=background, timeout=timeout, task_id=task_id, workdir=workdir, check_interval=check_interval, pty=pty)

    else:
        return json.dumps({"error": f"Unknown terminal function: {function_name}"}, ensure_ascii=False)


def handle_process_function_call(function_name: str, function_args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    """
    Handle function calls for the process management tool.

    Routes actions (list, poll, log, wait, kill) to the ProcessRegistry.
    """
    from tools.process_registry import process_registry

    action = function_args.get("action", "")
    session_id = function_args.get("session_id", "")

    if action == "list":
        sessions = process_registry.list_sessions(task_id=task_id)
        return json.dumps({"processes": sessions}, ensure_ascii=False)

    elif action == "poll":
        if not session_id:
            return json.dumps({"error": "session_id is required for poll"}, ensure_ascii=False)
        return json.dumps(process_registry.poll(session_id), ensure_ascii=False)

    elif action == "log":
        if not session_id:
            return json.dumps({"error": "session_id is required for log"}, ensure_ascii=False)
        offset = function_args.get("offset", 0)
        limit = function_args.get("limit", 200)
        return json.dumps(process_registry.read_log(session_id, offset=offset, limit=limit), ensure_ascii=False)

    elif action == "wait":
        if not session_id:
            return json.dumps({"error": "session_id is required for wait"}, ensure_ascii=False)
        timeout = function_args.get("timeout")
        return json.dumps(process_registry.wait(session_id, timeout=timeout), ensure_ascii=False)

    elif action == "kill":
        if not session_id:
            return json.dumps({"error": "session_id is required for kill"}, ensure_ascii=False)
        return json.dumps(process_registry.kill_process(session_id), ensure_ascii=False)

    elif action == "write":
        if not session_id:
            return json.dumps({"error": "session_id is required for write"}, ensure_ascii=False)
        data = function_args.get("data", "")
        return json.dumps(process_registry.write_stdin(session_id, data), ensure_ascii=False)

    elif action == "submit":
        if not session_id:
            return json.dumps({"error": "session_id is required for submit"}, ensure_ascii=False)
        data = function_args.get("data", "")
        return json.dumps(process_registry.submit_stdin(session_id, data), ensure_ascii=False)

    else:
        return json.dumps({"error": f"Unknown process action: {action}. Use: list, poll, log, wait, kill, write, submit"}, ensure_ascii=False)


def handle_vision_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Handle function calls for vision tools.
    
    Args:
        function_name (str): Name of the vision function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "vision_analyze":
        image_url = function_args.get("image_url", "")
        question = function_args.get("question", "")

        full_prompt = f"Fully describe and explain everything about this image, then answer the following question:\n\n{question}"
        
        # Run async function in event loop
        return asyncio.run(vision_analyze_tool(image_url, full_prompt, "google/gemini-3-flash-preview"))
    
    else:
        return json.dumps({"error": f"Unknown vision function: {function_name}"}, ensure_ascii=False)


def handle_moa_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Handle function calls for Mixture-of-Agents tools.
    
    Args:
        function_name (str): Name of the MoA function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "mixture_of_agents":
        user_prompt = function_args.get("user_prompt", "")
        
        if not user_prompt:
            return json.dumps({"error": "user_prompt is required for MoA processing"}, ensure_ascii=False)
        
        # Run async function in event loop
        return asyncio.run(mixture_of_agents_tool(user_prompt=user_prompt))
    
    else:
        return json.dumps({"error": f"Unknown MoA function: {function_name}"}, ensure_ascii=False)


def handle_image_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Handle function calls for image generation tools.
    
    Args:
        function_name (str): Name of the image generation function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "image_generate":
        prompt = function_args.get("prompt", "")
        
        if not prompt:
            return json.dumps({"success": False, "image": None}, ensure_ascii=False)
        
        aspect_ratio = function_args.get("aspect_ratio", "landscape")
        
        # Use fixed internal defaults for all other parameters (not exposed to model)
        num_inference_steps = 50
        guidance_scale = 4.5
        num_images = 1
        output_format = "png"
        seed = None
        
        # Run async function in event loop with proper handling for multiprocessing
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                # If closed, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the coroutine in the event loop
        result = loop.run_until_complete(image_generate_tool(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            output_format=output_format,
            seed=seed
        ))
        
        return result
    
    else:
        return json.dumps({"error": f"Unknown image generation function: {function_name}"}, ensure_ascii=False)


def handle_skills_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Handle function calls for skills tools (read-only and management).
    
    Args:
        function_name (str): Name of the skills function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "skills_list":
        category = function_args.get("category")
        return skills_list(category=category)
    
    elif function_name == "skill_view":
        name = function_args.get("name", "")
        if not name:
            return json.dumps({"error": "Skill name is required"}, ensure_ascii=False)
        file_path = function_args.get("file_path")
        return skill_view(name, file_path=file_path)
    
    elif function_name == "skill_manage":
        action = function_args.get("action", "")
        name = function_args.get("name", "")
        if not action:
            return json.dumps({"error": "action is required"}, ensure_ascii=False)
        if not name:
            return json.dumps({"error": "name is required"}, ensure_ascii=False)
        return skill_manage(
            action=action,
            name=name,
            content=function_args.get("content"),
            category=function_args.get("category"),
            file_path=function_args.get("file_path"),
            file_content=function_args.get("file_content"),
            old_string=function_args.get("old_string"),
            new_string=function_args.get("new_string"),
            replace_all=function_args.get("replace_all", False),
        )
    
    else:
        return json.dumps({"error": f"Unknown skills function: {function_name}"}, ensure_ascii=False)


# Browser tool handlers mapping
BROWSER_HANDLERS = {
    "browser_navigate": browser_navigate,
    "browser_click": browser_click,
    "browser_type": browser_type,
    "browser_scroll": browser_scroll,
    "browser_back": browser_back,
    "browser_press": browser_press,
    "browser_close": browser_close,
    "browser_get_images": browser_get_images,
    "browser_vision": browser_vision,
}


def handle_browser_function_call(
    function_name: str, 
    function_args: Dict[str, Any], 
    task_id: Optional[str] = None,
    user_task: Optional[str] = None
) -> str:
    """
    Handle function calls for browser automation tools.
    
    Args:
        function_name (str): Name of the browser function to call
        function_args (Dict): Arguments for the function
        task_id (str): Task identifier for session isolation
        user_task (str): User's current task (for task-aware extraction in snapshots)
    
    Returns:
        str: Function result as JSON string
    """
    # Special handling for browser_snapshot which needs user_task for extraction
    if function_name == "browser_snapshot":
        full = function_args.get("full", False)
        return browser_snapshot(full=full, task_id=task_id, user_task=user_task)
    
    # Handle other browser tools
    if function_name in BROWSER_HANDLERS:
        handler = BROWSER_HANDLERS[function_name]
        # Add task_id to args
        return handler(**function_args, task_id=task_id)
    
    return json.dumps({"error": f"Unknown browser function: {function_name}"}, ensure_ascii=False)


def handle_cronjob_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None
) -> str:
    """
    Handle function calls for cronjob management tools.
    
    These tools are only available in interactive CLI mode (hermes-cli toolset).
    
    Args:
        function_name (str): Name of the cronjob function to call
        function_args (Dict): Arguments for the function
        task_id (str): Task identifier (unused, for API consistency)
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "schedule_cronjob":
        return schedule_cronjob(
            prompt=function_args.get("prompt", ""),
            schedule=function_args.get("schedule", ""),
            name=function_args.get("name"),
            repeat=function_args.get("repeat"),
            task_id=task_id
        )
    
    elif function_name == "list_cronjobs":
        return list_cronjobs(
            include_disabled=function_args.get("include_disabled", False),
            task_id=task_id
        )
    
    elif function_name == "remove_cronjob":
        return remove_cronjob(
            job_id=function_args.get("job_id", ""),
            task_id=task_id
        )
    
    return json.dumps({"error": f"Unknown cronjob function: {function_name}"}, ensure_ascii=False)


def handle_rl_function_call(
    function_name: str,
    function_args: Dict[str, Any]
) -> str:
    """
    Handle function calls for RL training tools.
    
    These tools communicate with the RL API server to manage training runs.
    
    Args:
        function_name (str): Name of the RL function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    # Run async functions in event loop
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if function_name == "rl_list_environments":
        return loop.run_until_complete(rl_list_environments())
    
    elif function_name == "rl_select_environment":
        return loop.run_until_complete(
            rl_select_environment(name=function_args.get("name", ""))
        )
    
    elif function_name == "rl_get_current_config":
        return loop.run_until_complete(rl_get_current_config())
    
    elif function_name == "rl_edit_config":
        return loop.run_until_complete(
            rl_edit_config(
                field=function_args.get("field", ""),
                value=function_args.get("value")
            )
        )
    
    elif function_name == "rl_start_training":
        return loop.run_until_complete(rl_start_training())
    
    elif function_name == "rl_check_status":
        return loop.run_until_complete(
            rl_check_status(run_id=function_args.get("run_id", ""))
        )
    
    elif function_name == "rl_stop_training":
        return loop.run_until_complete(
            rl_stop_training(run_id=function_args.get("run_id", ""))
        )
    
    elif function_name == "rl_get_results":
        return loop.run_until_complete(
            rl_get_results(run_id=function_args.get("run_id", ""))
        )
    
    elif function_name == "rl_list_runs":
        return loop.run_until_complete(rl_list_runs())
    
    elif function_name == "rl_test_inference":
        return loop.run_until_complete(
            rl_test_inference(
                num_steps=function_args.get("num_steps", 3),
                group_size=function_args.get("group_size", 16),
                models=function_args.get("models"),
            )
        )
    
    return json.dumps({"error": f"Unknown RL function: {function_name}"}, ensure_ascii=False)


def handle_file_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None
) -> str:
    """
    Handle function calls for file manipulation tools.
    
    These tools use the terminal backend for all operations, supporting
    local, docker, singularity, ssh, and modal environments.
    
    Args:
        function_name (str): Name of the file function to call
        function_args (Dict): Arguments for the function
        task_id (str): Task identifier for environment isolation
    
    Returns:
        str: Function result as JSON string
    """
    # Determine task_id to use
    tid = task_id or "default"
    
    if function_name == "read_file":
        return read_file_tool(
            path=function_args.get("path", ""),
            offset=function_args.get("offset", 1),
            limit=function_args.get("limit", 500),
            task_id=tid
        )
    
    elif function_name == "write_file":
        return write_file_tool(
            path=function_args.get("path", ""),
            content=function_args.get("content", ""),
            task_id=tid
        )
    
    elif function_name == "patch":
        return patch_tool(
            mode=function_args.get("mode", "replace"),
            path=function_args.get("path"),
            old_string=function_args.get("old_string"),
            new_string=function_args.get("new_string"),
            replace_all=function_args.get("replace_all", False),
            patch=function_args.get("patch"),
            task_id=tid
        )
    
    elif function_name == "search":
        return search_tool(
            pattern=function_args.get("pattern", ""),
            target=function_args.get("target", "content"),
            path=function_args.get("path", "."),
            file_glob=function_args.get("file_glob"),
            limit=function_args.get("limit", 50),
            offset=function_args.get("offset", 0),
            output_mode=function_args.get("output_mode", "content"),
            context=function_args.get("context", 0),
            task_id=tid
        )
    
    return json.dumps({"error": f"Unknown file function: {function_name}"}, ensure_ascii=False)


def handle_tts_function_call(
    function_name: str,
    function_args: Dict[str, Any]
) -> str:
    """
    Handle function calls for text-to-speech tools.
    
    Args:
        function_name (str): Name of the TTS function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "text_to_speech":
        text = function_args.get("text", "")
        output_path = function_args.get("output_path")
        return text_to_speech_tool(text=text, output_path=output_path)
    
    return json.dumps({"error": f"Unknown TTS function: {function_name}"}, ensure_ascii=False)


def handle_send_message_function_call(function_name, function_args):
    """Handle cross-channel send_message tool calls.

    Sends a message directly to the target platform using its API.
    Works in both CLI and gateway contexts -- does not require the
    gateway to be running.  Loads credentials from the gateway config
    (env vars / ~/.hermes/gateway.json).
    """
    import json
    import asyncio

    target = function_args.get("target", "")
    message = function_args.get("message", "")
    if not target or not message:
        return json.dumps({"error": "Both 'target' and 'message' are required"})

    # Parse target: "platform" or "platform:chat_id"
    parts = target.split(":", 1)
    platform_name = parts[0].strip().lower()
    chat_id = parts[1].strip() if len(parts) > 1 else None

    try:
        from gateway.config import load_gateway_config, Platform
        config = load_gateway_config()
    except Exception as e:
        return json.dumps({"error": f"Failed to load gateway config: {e}"})

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
    }
    platform = platform_map.get(platform_name)
    if not platform:
        avail = ", ".join(platform_map.keys())
        return json.dumps({"error": f"Unknown platform: {platform_name}. Available: {avail}"})

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        return json.dumps({"error": f"Platform '{platform_name}' is not configured. Set up credentials in ~/.hermes/gateway.json or environment variables."})

    if not chat_id:
        home = config.get_home_channel(platform)
        if home:
            chat_id = home.chat_id
        else:
            return json.dumps({"error": f"No chat_id specified and no home channel configured for {platform_name}. Use format 'platform:chat_id'."})

    try:
        result = _run_async(_send_to_platform(platform, pconfig, chat_id, message))
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Send failed: {e}"})


def _run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g. inside
    the gateway's async stack), we spin up a disposable thread so
    asyncio.run() can create its own loop without conflicting.
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=30)
    return asyncio.run(coro)


async def _send_to_platform(platform, pconfig, chat_id, message):
    """Route a message to the appropriate platform sender."""
    from gateway.config import Platform
    if platform == Platform.TELEGRAM:
        return await _send_telegram(pconfig.token, chat_id, message)
    elif platform == Platform.DISCORD:
        return await _send_discord(pconfig.token, chat_id, message)
    elif platform == Platform.SLACK:
        return await _send_slack(pconfig.token, chat_id, message)
    return {"error": f"Direct sending not yet implemented for {platform.value}"}


async def _send_telegram(token, chat_id, message):
    """Send via Telegram Bot API (one-shot, no polling needed)."""
    try:
        from telegram import Bot
        bot = Bot(token=token)
        msg = await bot.send_message(chat_id=int(chat_id), text=message)
        return {"success": True, "platform": "telegram", "chat_id": chat_id, "message_id": str(msg.message_id)}
    except ImportError:
        return {"error": "python-telegram-bot not installed. Run: pip install python-telegram-bot"}
    except Exception as e:
        return {"error": f"Telegram send failed: {e}"}


async def _send_discord(token, chat_id, message):
    """Send via Discord REST API (no websocket client needed)."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        url = f"https://discord.com/api/v10/channels/{chat_id}/messages"
        headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
        chunks = [message[i:i+2000] for i in range(0, len(message), 2000)]
        message_ids = []
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                async with session.post(url, headers=headers, json={"content": chunk}) as resp:
                    if resp.status not in (200, 201):
                        body = await resp.text()
                        return {"error": f"Discord API error ({resp.status}): {body}"}
                    data = await resp.json()
                    message_ids.append(data.get("id"))
        return {"success": True, "platform": "discord", "chat_id": chat_id, "message_ids": message_ids}
    except Exception as e:
        return {"error": f"Discord send failed: {e}"}


async def _send_slack(token, chat_id, message):
    """Send via Slack Web API."""
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        url = "https://slack.com/api/chat.postMessage"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"channel": chat_id, "text": message}) as resp:
                data = await resp.json()
                if data.get("ok"):
                    return {"success": True, "platform": "slack", "chat_id": chat_id, "message_id": data.get("ts")}
                return {"error": f"Slack API error: {data.get('error', 'unknown')}"}
    except Exception as e:
        return {"error": f"Slack send failed: {e}"}


def handle_function_call(
    function_name: str, 
    function_args: Dict[str, Any], 
    task_id: Optional[str] = None,
    user_task: Optional[str] = None
) -> str:
    """
    Main function call dispatcher that routes calls to appropriate toolsets.

    This function determines which toolset a function belongs to and dispatches
    the call to the appropriate handler. This makes it easy to add new toolsets
    without changing the main calling interface.

    Args:
        function_name (str): Name of the function to call
        function_args (Dict): Arguments for the function
        task_id (str): Unique identifier for this task to isolate VMs/sessions between concurrent tasks (optional)
        user_task (str): The user's original task/query (used for task-aware content extraction) (optional)

    Returns:
        str: Function result as JSON string

    Raises:
        None: Returns error as JSON string instead of raising exceptions
    """
    try:
        # Route web tools
        if function_name in ["web_search", "web_extract"]:
            return handle_web_function_call(function_name, function_args)

        # Route terminal tools
        elif function_name in ["terminal"]:
            return handle_terminal_function_call(function_name, function_args, task_id)

        # Route process management tools
        elif function_name in ["process"]:
            return handle_process_function_call(function_name, function_args, task_id)

        # Route vision tools
        elif function_name in ["vision_analyze"]:
            return handle_vision_function_call(function_name, function_args)

        # Route MoA tools
        elif function_name in ["mixture_of_agents"]:
            return handle_moa_function_call(function_name, function_args)

        # Route image generation tools
        elif function_name in ["image_generate"]:
            return handle_image_function_call(function_name, function_args)

        # Route skills tools
        elif function_name in ["skills_list", "skill_view", "skill_manage"]:
            return handle_skills_function_call(function_name, function_args)

        # Route browser automation tools
        elif function_name in [
            "browser_navigate", "browser_snapshot", "browser_click",
            "browser_type", "browser_scroll", "browser_back",
            "browser_press", "browser_close", "browser_get_images",
            "browser_vision"
        ]:
            return handle_browser_function_call(function_name, function_args, task_id, user_task)

        # Route cronjob management tools
        elif function_name in ["schedule_cronjob", "list_cronjobs", "remove_cronjob"]:
            return handle_cronjob_function_call(function_name, function_args, task_id)

        # Route RL training tools
        elif function_name in [
            "rl_list_environments", "rl_select_environment",
            "rl_get_current_config", "rl_edit_config",
            "rl_start_training", "rl_check_status",
            "rl_stop_training", "rl_get_results",
            "rl_list_runs", "rl_test_inference"
        ]:
            return handle_rl_function_call(function_name, function_args)

        # Route file manipulation tools
        elif function_name in ["read_file", "write_file", "patch", "search"]:
            return handle_file_function_call(function_name, function_args, task_id)

        # Route text-to-speech tools
        elif function_name in ["text_to_speech"]:
            return handle_tts_function_call(function_name, function_args)

        # Route cross-channel messaging
        elif function_name == "send_message":
            return handle_send_message_function_call(function_name, function_args)

        # Todo tool -- handled by the agent loop (needs TodoStore instance).
        # This fallback should never execute in practice; run_agent.py intercepts first.
        elif function_name == "todo":
            return json.dumps({"error": "todo must be handled by the agent loop"})

        # Memory tool -- handled by the agent loop (needs MemoryStore instance).
        elif function_name == "memory":
            return json.dumps({"error": "Memory is not available. It may be disabled in config or this environment."})

        # Session search -- handled by the agent loop (needs SessionDB instance).
        elif function_name == "session_search":
            return json.dumps({"error": "Session search is not available. The session database may not be initialized."})

        else:
            error_msg = f"Unknown function: {function_name}"
            print(f"❌ {error_msg}")
            
            return json.dumps({"error": error_msg}, ensure_ascii=False)
    
    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg}, ensure_ascii=False)

def get_available_toolsets() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available toolsets and their status.
    
    Returns:
        Dict: Information about each toolset including availability and tools
    """
    toolsets = {
        "web_tools": {
            "available": check_firecrawl_api_key(),
            "tools": ["web_search_tool", "web_extract_tool"],
            "description": "Web search and content extraction tools",
            "requirements": ["FIRECRAWL_API_KEY environment variable"]
        },
        "terminal_tools": {
            "available": check_terminal_requirements(),
            "tools": ["terminal_tool"],
            "description": "Execute commands using mini-swe-agent (local/docker/modal)",
            "requirements": ["mini-swe-agent package, TERMINAL_ENV to select backend"]
        },
        "vision_tools": {
            "available": check_vision_requirements(),
            "tools": ["vision_analyze_tool"],
            "description": "Analyze images from URLs using AI vision for comprehensive understanding",
            "requirements": ["NOUS_API_KEY environment variable"]
        },
        "moa_tools": {
            "available": check_moa_requirements(),
            "tools": ["mixture_of_agents_tool"],
            "description": "Process extremely difficult problems using Mixture-of-Agents methodology with multiple frontier models collaborating for enhanced reasoning. Best for complex math, coding, and analytical tasks.",
            "requirements": ["NOUS_API_KEY environment variable"]
        },
        "image_tools": {
            "available": check_image_generation_requirements(),
            "tools": ["image_generate_tool"],
            "description": "Generate high-quality images from text prompts using FAL.ai's FLUX.1 Krea model with automatic 2x upscaling for enhanced quality",
            "requirements": ["FAL_KEY environment variable", "fal-client package"]
        },
        "skills_tools": {
            "available": check_skills_requirements(),
            "tools": ["skills_list", "skill_view", "skill_manage"],
            "description": "Access, create, edit, and manage skill documents that provide specialized instructions, guidelines, or knowledge the agent can load on demand",
            "requirements": ["~/.hermes/skills/ directory (seeded from bundled skills on install)"]
        },
        "browser_tools": {
            "available": check_browser_requirements(),
            "tools": [
                "browser_navigate", "browser_snapshot", "browser_click",
                "browser_type", "browser_scroll", "browser_back",
                "browser_press", "browser_close", "browser_get_images",
                "browser_vision"
            ],
            "description": "Browser automation for web interaction using agent-browser CLI with Browserbase cloud execution",
            "requirements": ["BROWSERBASE_API_KEY", "BROWSERBASE_PROJECT_ID", "agent-browser npm package"]
        },
        "cronjob_tools": {
            "available": check_cronjob_requirements(),
            "tools": ["schedule_cronjob", "list_cronjobs", "remove_cronjob"],
            "description": "Schedule and manage automated tasks (cronjobs) - only available in interactive CLI mode",
            "requirements": ["HERMES_INTERACTIVE=1 (set automatically by cli.py)"]
        },
        "file_tools": {
            "available": check_file_requirements(),
            "tools": ["read_file", "write_file", "patch", "search"],
            "description": "File manipulation tools: read/write files, search content/files, patch with fuzzy matching",
            "requirements": ["Terminal backend available (local/docker/ssh/singularity/modal)"]
        },
        "tts_tools": {
            "available": check_tts_requirements(),
            "tools": ["text_to_speech"],
            "description": "Text-to-speech: convert text to audio (Edge TTS free, ElevenLabs, OpenAI)",
            "requirements": ["edge-tts package (free) or ELEVENLABS_API_KEY or OPENAI_API_KEY"]
        },
        "todo_tools": {
            "available": check_todo_requirements(),
            "tools": ["todo"],
            "description": "Planning & task management: in-memory todo list for multi-step work",
            "requirements": []
        },
        "memory_tools": {
            "available": check_memory_requirements(),
            "tools": ["memory"],
            "description": "Persistent memory: bounded MEMORY.md + USER.md injected into system prompt",
            "requirements": []
        },
        "session_search_tools": {
            "available": check_session_search_requirements(),
            "tools": ["session_search"],
            "description": "Session history search: FTS5 search + Gemini Flash summarization of past conversations",
            "requirements": ["OPENROUTER_API_KEY", "~/.hermes/state.db"]
        }
    }
    
    return toolsets

def check_toolset_requirements() -> Dict[str, bool]:
    """
    Check if all requirements for available toolsets are met.

    Returns:
        Dict: Status of each toolset's requirements
    """
    return {
        "web_tools": check_firecrawl_api_key(),
        "terminal_tools": check_terminal_requirements(),
        "vision_tools": check_vision_requirements(),
        "moa_tools": check_moa_requirements(),
        "image_tools": check_image_generation_requirements(),
        "skills_tools": check_skills_requirements(),
        "browser_tools": check_browser_requirements(),
        "cronjob_tools": check_cronjob_requirements(),
        "file_tools": check_file_requirements(),
        "tts_tools": check_tts_requirements()
    }

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🛠️  Model Tools Module")
    print("=" * 40)
    
    # Check toolset requirements
    requirements = check_toolset_requirements()
    print("📋 Toolset Requirements:")
    for toolset, available in requirements.items():
        status = "✅" if available else "❌"
        print(f"  {status} {toolset}: {'Available' if available else 'Missing requirements'}")
    
    # Show all available tool names
    all_tool_names = get_all_tool_names()
    print(f"\n🔧 Available Tools ({len(all_tool_names)} total):")
    for tool_name in all_tool_names:
        toolset = get_toolset_for_tool(tool_name)
        print(f"  📌 {tool_name} (from {toolset})")
    
    # Show available tools with full definitions
    tools = get_tool_definitions()
    print(f"\n📝 Tool Definitions ({len(tools)} loaded):")
    for tool in tools:
        func_name = tool["function"]["name"]
        desc = tool["function"]["description"]
        print(f"  🔹 {func_name}: {desc[:60]}{'...' if len(desc) > 60 else ''}")
    
    # Show toolset info
    toolsets = get_available_toolsets()
    print(f"\n📦 Toolset Information:")
    for name, info in toolsets.items():
        status = "✅" if info["available"] else "❌"
        print(f"  {status} {name}: {info['description']}")
        if not info["available"]:
            print(f"    Requirements: {', '.join(info['requirements'])}")
    
    print("\n💡 Usage Examples:")
    print("  from model_tools import get_tool_definitions, handle_function_call")
    print("  # All tools")
    print("  tools = get_tool_definitions()")
    print("  # Only web tools")
    print("  tools = get_tool_definitions(enabled_toolsets=['web_tools'])")
    print("  # Specific tools only")
    print("  tools = get_tool_definitions(enabled_tools=['web_search', 'terminal'])")
    print("  # All except terminal")
    print("  tools = get_tool_definitions(disabled_tools=['terminal'])")
    
    # Example filtering
    print(f"\n🧪 Filtering Examples:")
    web_only = get_tool_definitions(enabled_toolsets=["web_tools"])
    print(f"  Web tools only: {len(web_only)} tools")
    
    if len(all_tool_names) > 1:
        specific_tools = get_tool_definitions(enabled_tools=["web_search"])
        print(f"  Only web_search: {len(specific_tools)} tool(s)")
        
        if "terminal" in all_tool_names:
            no_terminal = get_tool_definitions(disabled_tools=["terminal"])
            print(f"  All except terminal: {len(no_terminal)} tools")
