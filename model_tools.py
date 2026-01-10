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
from typing import Dict, Any, List, Optional

from tools.web_tools import web_search_tool, web_extract_tool, web_crawl_tool, check_firecrawl_api_key
from tools.simple_terminal_tool import simple_terminal_tool, check_requirements as check_simple_terminal_requirements, SIMPLE_TERMINAL_TOOL_DESCRIPTION
# Keep old terminal tool for backwards compatibility if needed
# from tools.terminal_tool import terminal_tool, check_hecate_requirements, TERMINAL_TOOL_DESCRIPTION
from tools.vision_tools import vision_analyze_tool, check_vision_requirements
from tools.mixture_of_agents_tool import mixture_of_agents_tool, check_moa_requirements
from tools.image_generation_tool import image_generate_tool, check_image_generation_requirements
from toolsets import (
    get_toolset, resolve_toolset, resolve_multiple_toolsets,
    get_all_toolsets, get_toolset_names, validate_toolset,
    get_toolset_info, print_toolset_tree
)

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
                "description": "Search the web for information on any topic. Returns up to 5 relevant results with titles and URLs. Uses advanced search depth for comprehensive results.",
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
                "description": "Extract and read the full content from specific web page URLs. Useful for getting detailed information from webpages found through search. The content returned will be excerpts and key points summarized with an LLM to reduce impact on the context window.",
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
        {
            "type": "function",
            "function": {
                "name": "web_crawl",
                "description": "Crawl a website with specific instructions to find and extract targeted content. Uses AI to intelligently navigate and extract relevant information from across the site. The content returned will be excerpts and key points summarized with an LLM to reduce impact on the context window.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The base URL to crawl (can include or exclude https://)"
                        },
                        "instructions": {
                            "type": "string",
                            "description": "Specific instructions for what to crawl/extract using AI intelligence (e.g., 'Find pricing information', 'Get documentation pages', 'Extract contact details')"
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    ]

def get_terminal_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get tool definitions for terminal tools in OpenAI's expected format.

    Returns:
        List[Dict]: List of terminal tool definitions compatible with OpenAI API
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": SIMPLE_TERMINAL_TOOL_DESCRIPTION,
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
                "description": "Analyze images from URLs using AI vision. Provides comprehensive image description and answers specific questions about the image content. Perfect for understanding visual content, reading text in images, identifying objects, analyzing scenes, and extracting visual information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_url": {
                            "type": "string",
                            "description": "The URL of the image to analyze (must be publicly accessible HTTP/HTTPS URL)"
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
                "description": "Generate high-quality images from text prompts using FLUX Krea model with automatic 2x upscaling. Creates detailed, artistic images that are automatically enhanced for superior quality. Returns a single upscaled image URL that can be displayed using <img src=\"{URL}\"></img> tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt describing the desired image. Be detailed and descriptive."
                        },
                        "image_size": {
                            "type": "string",
                            "enum": ["square","portrait_16_9", "landscape_16_9"],
                            "description": "The size/aspect ratio of the generated image (default: landscape_4_3)",
                            "default": "landscape_16_9"
                        }
                    },
                    "required": ["prompt"]
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
        tool_names.extend(["web_search", "web_extract", "web_crawl"])

    # Terminal tools
    if check_simple_terminal_requirements():
        tool_names.extend(["terminal"])

    # Vision tools
    if check_vision_requirements():
        tool_names.extend(["vision_analyze"])
    
    # MoA tools
    if check_moa_requirements():
        tool_names.extend(["mixture_of_agents"])
    
    # Image generation tools
    if check_image_generation_requirements():
        tool_names.extend(["image_generate"])
    
    return tool_names


def get_toolset_for_tool(tool_name: str) -> str:
    """
    Get the toolset that a tool belongs to.
    
    Args:
        tool_name (str): Name of the tool
        
    Returns:
        str: Name of the toolset, or "unknown" if not found
    """
    toolset_mapping = {
        "web_search": "web_tools",
        "web_extract": "web_tools", 
        "web_crawl": "web_tools",
        "terminal": "terminal_tools",
        "vision_analyze": "vision_tools",
        "mixture_of_agents": "moa_tools",
        "image_generate": "image_tools"
    }
    
    return toolset_mapping.get(tool_name, "unknown")


def get_tool_definitions(
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None
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

    if check_simple_terminal_requirements():
        for tool in get_terminal_tool_definitions():
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
    
    # Determine which tools to include based on toolsets
    tools_to_include = set()
    
    if enabled_toolsets:
        # Only include tools from enabled toolsets
        for toolset_name in enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved_tools = resolve_toolset(toolset_name)
                tools_to_include.update(resolved_tools)
                print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved_tools) if resolved_tools else 'no tools'}")
            else:
                # Try legacy compatibility
                if toolset_name in ["web_tools", "terminal_tools", "vision_tools", "moa_tools", "image_tools"]:
                    # Map legacy names to new system
                    legacy_map = {
                        "web_tools": ["web_search", "web_extract", "web_crawl"],
                        "terminal_tools": ["terminal"],
                        "vision_tools": ["vision_analyze"],
                        "moa_tools": ["mixture_of_agents"],
                        "image_tools": ["image_generate"]
                    }
                    legacy_tools = legacy_map.get(toolset_name, [])
                    tools_to_include.update(legacy_tools)
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
                else:
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
                print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved_tools) if resolved_tools else 'no tools'}")
            else:
                # Try legacy compatibility
                if toolset_name in ["web_tools", "terminal_tools", "vision_tools", "moa_tools", "image_tools"]:
                    legacy_map = {
                        "web_tools": ["web_search", "web_extract", "web_crawl"],
                        "terminal_tools": ["terminal"],
                        "vision_tools": ["vision_analyze"],
                        "moa_tools": ["mixture_of_agents"],
                        "image_tools": ["image_generate"]
                    }
                    legacy_tools = legacy_map.get(toolset_name, [])
                    tools_to_include.difference_update(legacy_tools)
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
                else:
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
        # Run async function in event loop
        return asyncio.run(web_extract_tool(urls, "markdown"))
    
    elif function_name == "web_crawl":
        url = function_args.get("url", "")
        instructions = function_args.get("instructions")
        # Run async function in event loop
        return asyncio.run(web_crawl_tool(url, instructions, "basic"))
    
    else:
        return json.dumps({"error": f"Unknown web function: {function_name}"}, ensure_ascii=False)

def handle_terminal_function_call(function_name: str, function_args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    """
    Handle function calls for terminal tools.

    Args:
        function_name (str): Name of the terminal function to call
        function_args (Dict): Arguments for the function
        task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional)

    Returns:
        str: Function result as JSON string
    """
    if function_name == "terminal":
        command = function_args.get("command")
        background = function_args.get("background", False)
        timeout = function_args.get("timeout")

        return simple_terminal_tool(command=command, background=background, timeout=timeout, task_id=task_id)

    else:
        return json.dumps({"error": f"Unknown terminal function: {function_name}"}, ensure_ascii=False)


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
        return asyncio.run(vision_analyze_tool(image_url, full_prompt, "gemini-2.5-flash"))
    
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
        
        image_size = function_args.get("image_size", "landscape_16_9")
        
        # Use fixed internal defaults for all other parameters (not exposed to model)
        num_inference_steps = 50
        guidance_scale = 4.5
        num_images = 1
        enable_safety_checker = True
        output_format = "png"
        acceleration = "none"
        allow_nsfw_images = True
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
            image_size=image_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images=num_images,
            enable_safety_checker=enable_safety_checker,
            output_format=output_format,
            acceleration=acceleration,
            allow_nsfw_images=allow_nsfw_images,
            seed=seed
        ))
        
        return result
    
    else:
        return json.dumps({"error": f"Unknown image generation function: {function_name}"}, ensure_ascii=False)


def handle_function_call(function_name: str, function_args: Dict[str, Any], task_id: Optional[str] = None) -> str:
    """
    Main function call dispatcher that routes calls to appropriate toolsets.

    This function determines which toolset a function belongs to and dispatches
    the call to the appropriate handler. This makes it easy to add new toolsets
    without changing the main calling interface.

    Args:
        function_name (str): Name of the function to call
        function_args (Dict): Arguments for the function
        task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional)

    Returns:
        str: Function result as JSON string

    Raises:
        None: Returns error as JSON string instead of raising exceptions
    """
    try:
        # Route web tools
        if function_name in ["web_search", "web_extract", "web_crawl"]:
            return handle_web_function_call(function_name, function_args)

        # Route terminal tools
        elif function_name in ["terminal"]:
            return handle_terminal_function_call(function_name, function_args, task_id)

        # Route vision tools
        elif function_name in ["vision_analyze"]:
            return handle_vision_function_call(function_name, function_args)

        # Route MoA tools
        elif function_name in ["mixture_of_agents"]:
            return handle_moa_function_call(function_name, function_args)

        # Route image generation tools
        elif function_name in ["image_generate"]:
            return handle_image_function_call(function_name, function_args)

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
            "tools": ["web_search_tool", "web_extract_tool", "web_crawl_tool"],
            "description": "Web search, content extraction, and website crawling tools",
            "requirements": ["FIRECRAWL_API_KEY environment variable"]
        },
        "terminal_tools": {
            "available": check_simple_terminal_requirements(),
            "tools": ["simple_terminal_tool"],
            "description": "Execute commands on secure Linux VMs without session persistence",
            "requirements": ["MORPH_API_KEY environment variable"]
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
        "terminal_tools": check_simple_terminal_requirements(),
        "vision_tools": check_vision_requirements(),
        "moa_tools": check_moa_requirements(),
        "image_tools": check_image_generation_requirements()
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
