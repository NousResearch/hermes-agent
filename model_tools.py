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
from typing import Dict, Any, List

# Import toolsets
from web_tools import web_search_tool, web_extract_tool, web_crawl_tool, check_tavily_api_key
from terminal_tool import terminal_tool, check_hecate_requirements, TERMINAL_TOOL_DESCRIPTION

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
                "description": "Search the web for information on any topic. Returns relevant results with titles, URLs, content snippets, and answers. Uses advanced search depth for comprehensive results.",
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
                "description": "Extract and read the full content from specific web page URLs. Useful for getting detailed information from webpages found through search.",
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
                "description": "Crawl a website with specific instructions to find and extract targeted content. Uses AI to intelligently navigate and extract relevant information from across the site.",
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

def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get all available tool definitions for model API calls.
    
    This function aggregates tool definitions from all available toolsets.
    Currently includes web tools, but can be extended to include other toolsets.
    
    Returns:
        List[Dict]: Complete list of all available tool definitions
    """
    tools = []
    
    # Add web tools
    tools.extend(get_web_tool_definitions())
    
    # Add terminal tools
    tools.extend(get_terminal_tool_definitions())
    
    # Future toolsets can be added here:
    # tools.extend(get_file_tool_definitions())
    # tools.extend(get_code_tool_definitions())
    # tools.extend(get_database_tool_definitions())
    
    return tools

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
        return web_extract_tool(urls, format)
    
    elif function_name == "web_crawl":
        url = function_args.get("url", "")
        instructions = function_args.get("instructions")
        depth = function_args.get("depth", "basic")
        return web_crawl_tool(url, instructions, depth)
    
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
        # Session management is handled internally - don't pass session_id from model
        return terminal_tool(command, input_keys, None, background, idle_threshold, timeout)
    
    else:
        return json.dumps({"error": f"Unknown terminal function: {function_name}"})

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
            "available": check_tavily_api_key(),
            "tools": ["web_search_tool", "web_extract_tool", "web_crawl_tool"],
            "description": "Web search, content extraction, and website crawling tools",
            "requirements": ["TAVILY_API_KEY environment variable"]
        },
        "terminal_tools": {
            "available": check_hecate_requirements(),
            "tools": ["terminal_tool"],
            "description": "Execute commands with optional interactive session support on Linux VMs",
            "requirements": ["MORPH_API_KEY environment variable", "hecate package"]
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
        "web_tools": check_tavily_api_key(),
        "terminal_tools": check_hecate_requirements()
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
    
    # Show available tools
    tools = get_tool_definitions()
    print(f"\n🔧 Available Tools ({len(tools)} total):")
    for tool in tools:
        func_name = tool["function"]["name"]
        desc = tool["function"]["description"]
        print(f"  📌 {func_name}: {desc[:80]}{'...' if len(desc) > 80 else ''}")
    
    # Show toolset info
    toolsets = get_available_toolsets()
    print(f"\n📦 Toolset Information:")
    for name, info in toolsets.items():
        status = "✅" if info["available"] else "❌"
        print(f"  {status} {name}: {info['description']}")
        if not info["available"]:
            print(f"    Requirements: {', '.join(info['requirements'])}")
    
    print("\n💡 Usage Example:")
    print("  from model_tools import get_tool_definitions, handle_function_call")
    print("  tools = get_tool_definitions()")
    print("  result = handle_function_call('web_search_tool', {'query': 'Python'})")
