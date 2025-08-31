#!/usr/bin/env python3
"""
Model Tools Module

This module constructs tool schemas and handlers for AI model API calls.
It imports tools from various toolset modules and provides a unified interface
for defining tools and executing function calls.

Currently supports:
- Web tools (search, extract, crawl) from web_tools.py

Usage:
    from model_tools import get_tool_definitions, handle_function_call
    
    # Get tool definitions for model API
    tools = get_tool_definitions()
    
    # Handle function calls from model
    result = handle_function_call("web_search_tool", {"query": "Python", "limit": 3})
"""

import json
import asyncio
from typing import Dict, Any, List

# Import toolsets
from web_tools import web_search_tool, web_extract_tool, web_crawl_tool, check_firecrawl_api_key
from terminal_tool import terminal_tool, check_hecate_requirements, TERMINAL_TOOL_DESCRIPTION
from vision_tools import vision_analyze_tool, check_vision_requirements
from mixture_of_agents_tool import mixture_of_agents_tool, check_moa_requirements
from image_generation_tool import image_generate_tool, check_image_generation_requirements

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
                "description": "Search the web for information on any topic. Returns relevant results with titles and URLs. Uses advanced search depth for comprehensive results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results to return (default: 5, max: 10)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
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
                        },
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "html"],
                            "description": "Desired output format for extracted content (optional)"
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
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["basic", "advanced"],
                            "description": "Depth of extraction - 'basic' for surface content, 'advanced' for deeper analysis (default: basic)",
                            "default": "basic"
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
                "description": TERMINAL_TOOL_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute on the VM"
                        },
                        "input_keys": {
                            "type": "string",
                            "description": "Keystrokes to send to the most recent interactive session (e.g., 'hello\\n' for typing hello + Enter). If no active session exists, this will be ignored."
                        },
                        "background": {
                            "type": "boolean",
                            "description": "Whether to run the command in the background (default: false)",
                            "default": False
                        },
                        "idle_threshold": {
                            "type": "number",
                            "description": "Seconds to wait for output before considering session idle (default: 5.0)",
                            "default": 5.0,
                            "minimum": 0.1
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Command timeout in seconds (optional)",
                            "minimum": 1
                        }
                    },
                    "required": []
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
                            "description": "Your specific question or request about the image to resolve. The AI will automatically provide a complete image description AND answer your specific question. Examples: 'What text can you read?', 'What architectural style is this?', 'Describe the mood and emotions', 'What safety hazards do you see?'"
                        },
                        "model": {
                            "type": "string",
                            "description": "The vision model to use for analysis (optional, default: gemini-2.5-flash)",
                            "default": "gemini-2.5-flash"
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
                "description": "Process extremely difficult problems requiring intense reasoning using the Mixture-of-Agents methodology. This tool leverages multiple frontier language models to collaboratively solve complex tasks that single models struggle with. Uses a fixed 2-layer architecture: reference models (claude-opus-4, gemini-2.5-pro, o4-mini, deepseek-r1) generate diverse responses, then an aggregator synthesizes the best solution. Best for: complex mathematical proofs, advanced coding problems, multi-step analytical reasoning, precise and complex STEM problems, algorithm design, and problems requiring diverse domain expertise.",
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
                "description": "Generate high-quality images from text prompts using FAL.ai's FLUX.1 Krea model with automatic 2x upscaling. Creates detailed, artistic images that are automatically enhanced for superior quality. Returns a single upscaled image URL that can be displayed using <img src=\"{URL}\"></img> tags.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt describing the desired image. Be detailed and descriptive for best results."
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
    if check_hecate_requirements():
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
    
    # Future toolsets can be added here:
    # if check_file_tools():
    #     tool_names.extend(["file_read", "file_write"])
    
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
        # Future tools can be added here
    }
    
    return toolset_mapping.get(tool_name, "unknown")


def get_tool_definitions(
    enabled_tools: List[str] = None, 
    disabled_tools: List[str] = None,
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Get tool definitions for model API calls with optional filtering.
    
    This function aggregates tool definitions from all available toolsets
    and applies filtering based on the provided parameters.
    
    Filter Priority (higher priority overrides lower):
    1. enabled_tools (highest priority - only these tools, overrides everything)
    2. disabled_tools (applied after toolset filtering)
    3. enabled_toolsets (only tools from these toolsets)
    4. disabled_toolsets (exclude tools from these toolsets)
    
    Args:
        enabled_tools (List[str]): Only include these specific tools. If provided, 
                                  ONLY these tools will be included (overrides all other filters)
        disabled_tools (List[str]): Exclude these specific tools (applied after toolset filtering)
        enabled_toolsets (List[str]): Only include tools from these toolsets
        disabled_toolsets (List[str]): Exclude tools from these toolsets
    
    Returns:
        List[Dict]: Filtered list of tool definitions
    
    Examples:
        # Only web tools
        tools = get_tool_definitions(enabled_toolsets=["web_tools"])
        
        # All tools except terminal
        tools = get_tool_definitions(disabled_tools=["terminal"])
        
        # Only specific tools (overrides toolset filters)
        tools = get_tool_definitions(enabled_tools=["web_search", "web_extract"])
        
        # Conflicting filters (enabled_tools wins)
        tools = get_tool_definitions(enabled_toolsets=["web_tools"], enabled_tools=["terminal"])
        # Result: Only terminal tool (enabled_tools overrides enabled_toolsets)
    """
    # Detect and warn about potential conflicts
    conflicts_detected = False
    
    if enabled_tools and (enabled_toolsets or disabled_toolsets or disabled_tools):
        print("⚠️  enabled_tools overrides all other filters")
        conflicts_detected = True
    
    if enabled_toolsets and disabled_toolsets:
        # Check for overlap
        enabled_set = set(enabled_toolsets)
        disabled_set = set(disabled_toolsets)
        overlap = enabled_set & disabled_set
        if overlap:
            print(f"⚠️  Conflicting toolsets: {overlap} in both enabled and disabled")
            print(f"   → enabled_toolsets takes priority")
            conflicts_detected = True
    
    if enabled_tools and disabled_tools:
        # Check for overlap
        enabled_set = set(enabled_tools)
        disabled_set = set(disabled_tools)
        overlap = enabled_set & disabled_set
        if overlap:
            print(f"⚠️  Conflicting tools: {overlap} in both enabled and disabled")
            print(f"   → enabled_tools takes priority")
            conflicts_detected = True
    
    all_tools = []
    
    # Collect all available tools from each toolset
    toolset_tools = {
        "web_tools": get_web_tool_definitions() if check_firecrawl_api_key() else [],
        "terminal_tools": get_terminal_tool_definitions() if check_hecate_requirements() else [],
        "vision_tools": get_vision_tool_definitions() if check_vision_requirements() else [],
        "moa_tools": get_moa_tool_definitions() if check_moa_requirements() else [],
        "image_tools": get_image_tool_definitions() if check_image_generation_requirements() else []
        # Future toolsets can be added here:
        # "file_tools": get_file_tool_definitions() if check_file_tools() else [],
    }
    
    # HIGHEST PRIORITY: enabled_tools (overrides everything)
    if enabled_tools:
        if conflicts_detected:
            print(f"🎯 Using only enabled_tools: {enabled_tools}")
        
        # Collect all available tools first
        all_available_tools = []
        for tools in toolset_tools.values():
            all_available_tools.extend(tools)
        
        # Only include specifically enabled tools
        tool_names_to_include = set(enabled_tools)
        filtered_tools = [
            tool for tool in all_available_tools 
            if tool["function"]["name"] in tool_names_to_include
        ]
        
        # Warn about requested tools that aren't available
        found_tools = {tool["function"]["name"] for tool in filtered_tools}
        missing_tools = tool_names_to_include - found_tools
        if missing_tools:
            print(f"⚠️  Requested tools not available: {missing_tools}")
        
        return filtered_tools
    
    # Apply toolset-level filtering first
    if enabled_toolsets:
        # Only include tools from enabled toolsets
        for toolset_name in enabled_toolsets:
            if toolset_name in toolset_tools:
                all_tools.extend(toolset_tools[toolset_name])
            else:
                print(f"⚠️  Unknown toolset: {toolset_name}")
    elif disabled_toolsets:
        # Include all tools except from disabled toolsets
        for toolset_name, tools in toolset_tools.items():
            if toolset_name not in disabled_toolsets:
                all_tools.extend(tools)
    else:
        # Include all available tools
        for tools in toolset_tools.values():
            all_tools.extend(tools)
    
    # Apply tool-level filtering (disabled_tools)
    if disabled_tools:
        tool_names_to_exclude = set(disabled_tools)
        original_tools = [tool["function"]["name"] for tool in all_tools]
        
        all_tools = [
            tool for tool in all_tools 
            if tool["function"]["name"] not in tool_names_to_exclude
        ]
        
        # Show what was actually filtered out
        remaining_tools = {tool["function"]["name"] for tool in all_tools}
        actually_excluded = set(original_tools) & tool_names_to_exclude
        if actually_excluded:
            print(f"🚫 Excluded tools: {actually_excluded}")
    
    return all_tools

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
        limit = function_args.get("limit", 5)
        # Ensure limit is within bounds
        limit = max(1, min(10, limit))
        return web_search_tool(query, limit)
    
    elif function_name == "web_extract":
        urls = function_args.get("urls", [])
        # Limit URLs to prevent abuse
        urls = urls[:5] if isinstance(urls, list) else []
        format = function_args.get("format")
        # Run async function in event loop
        return asyncio.run(web_extract_tool(urls, format))
    
    elif function_name == "web_crawl":
        url = function_args.get("url", "")
        instructions = function_args.get("instructions")
        depth = function_args.get("depth", "basic")
        # Run async function in event loop
        return asyncio.run(web_crawl_tool(url, instructions, depth))
    
    else:
        return json.dumps({"error": f"Unknown web function: {function_name}"})

def handle_terminal_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Handle function calls for terminal tools.
    
    Args:
        function_name (str): Name of the terminal function to call
        function_args (Dict): Arguments for the function
    
    Returns:
        str: Function result as JSON string
    """
    if function_name == "terminal":
        command = function_args.get("command")
        input_keys = function_args.get("input_keys")
        background = function_args.get("background", False)
        idle_threshold = function_args.get("idle_threshold", 5.0)
        timeout = function_args.get("timeout")
        snapshot_id = function_args.get("snapshot_id")
        # Session management is handled internally - don't pass session_id from model
        return terminal_tool(command, input_keys, None, background, idle_threshold, timeout, snapshot_id=snapshot_id)
    
    else:
        return json.dumps({"error": f"Unknown terminal function: {function_name}"})


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
        model = function_args.get("model", "gemini-2.5-flash")
        
        # Automatically prepend full description request to user's question
        full_prompt = f"Fully describe and explain everything about this image\n\n{question}"
        
        # Run async function in event loop
        return asyncio.run(vision_analyze_tool(image_url, full_prompt, model))
    
    else:
        return json.dumps({"error": f"Unknown vision function: {function_name}"})


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
            return json.dumps({"error": "user_prompt is required for MoA processing"})
        
        # Run async function in event loop
        return asyncio.run(mixture_of_agents_tool(user_prompt=user_prompt))
    
    else:
        return json.dumps({"error": f"Unknown MoA function: {function_name}"})


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
            return json.dumps({"success": False, "image": None})
        
        # Extract only the exposed parameters
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
        
        # Run async function in event loop
        return asyncio.run(image_generate_tool(
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
    
    else:
        return json.dumps({"error": f"Unknown image generation function: {function_name}"})


def handle_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """
    Main function call dispatcher that routes calls to appropriate toolsets.
    
    This function determines which toolset a function belongs to and dispatches
    the call to the appropriate handler. This makes it easy to add new toolsets
    without changing the main calling interface.
    
    Args:
        function_name (str): Name of the function to call
        function_args (Dict): Arguments for the function
    
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
            return handle_terminal_function_call(function_name, function_args)
        
        # Route vision tools
        elif function_name in ["vision_analyze"]:
            return handle_vision_function_call(function_name, function_args)
        
        # Route MoA tools
        elif function_name in ["mixture_of_agents"]:
            return handle_moa_function_call(function_name, function_args)
        
        # Route image generation tools
        elif function_name in ["image_generate"]:
            return handle_image_function_call(function_name, function_args)
        
        # Future toolsets can be routed here:
        # elif function_name in ["file_read_tool", "file_write_tool"]:
        #     return handle_file_function_call(function_name, function_args)
        # elif function_name in ["code_execute_tool", "code_analyze_tool"]:
        #     return handle_code_function_call(function_name, function_args)
        
        else:
            error_msg = f"Unknown function: {function_name}"
            print(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})
    
    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return json.dumps({"error": error_msg})

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
            "available": check_hecate_requirements(),
            "tools": ["terminal_tool"],
            "description": "Execute commands with optional interactive session support on Linux VMs",
            "requirements": ["MORPH_API_KEY environment variable", "hecate package"]
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
        # Future toolsets can be added here
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
        "terminal_tools": check_hecate_requirements(),
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
