#!/usr/bin/env python3
"""
Terminal Tools Module

This module provides terminal/command execution tools using Hecate's VM infrastructure.
It wraps Hecate's functionality to provide a simple interface for executing commands
on Morph VMs with automatic lifecycle management.

Available tools:
- terminal_execute_tool: Execute a single command and get output
- terminal_session_tool: Execute a command in a persistent session

Usage:
    from terminal_tool import terminal_execute_tool, terminal_session_tool
    
    # Execute a single command
    result = terminal_execute_tool("ls -la")
    
    # Execute in a session (for interactive commands)
    result = terminal_session_tool("python", input_keys="print('hello')\\nexit()\\n")
"""

import json
import os
from typing import Optional
from hecate import run_tool_with_lifecycle_management
from morphcloud._llm import ToolCall

def terminal_execute_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None
) -> str:
    """
    Execute a command on a Morph VM and return the output.
    
    This tool uses Hecate's VM lifecycle management to automatically create
    and manage VMs. VMs are reused within the configured lifetime window
    and automatically cleaned up after inactivity.
    
    Args:
        command: The command to execute
        background: Whether to run the command in the background (default: False)
        timeout: Command timeout in seconds (optional)
    
    Returns:
        str: JSON string containing the command output, exit code, and any errors
    
    Example:
        >>> result = terminal_execute_tool("ls -la /tmp")
        >>> print(json.loads(result))
        {
            "output": "total 8\\ndrwxrwxrwt 2 root root 4096 ...",
            "exit_code": 0,
            "error": null
        }
    """
    try:
        # Create tool call for Hecate
        tool_input = {
            "command": command,
            "background": background
        }
        
        if timeout is not None:
            tool_input["timeout"] = timeout
        
        tool_call = ToolCall(
            name="run_command",
            input=tool_input
        )
        
        # Execute with lifecycle management
        result = run_tool_with_lifecycle_management(tool_call)
        
        # Format the result
        formatted_result = {
            "output": result.get("output", ""),
            "exit_code": result.get("returncode", result.get("exit_code", -1)),
            "error": result.get("error")
        }
        
        # Add session info if present (for interactive sessions)
        if "session_id" in result:
            formatted_result["session_id"] = result["session_id"]
        if "screen" in result:
            formatted_result["screen"] = result["screen"]
        
        return json.dumps(formatted_result)
        
    except Exception as e:
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "error": f"Failed to execute command: {str(e)}"
        })

def terminal_session_tool(
    command: Optional[str] = None,
    input_keys: Optional[str] = None,
    session_id: Optional[str] = None,
    idle_threshold: float = 5.0
) -> str:
    """
    Execute a command in an interactive terminal session.
    
    This tool is useful for:
    - Running interactive programs (vim, python REPL, etc.)
    - Maintaining state between commands
    - Sending keystrokes to running programs
    
    Args:
        command: Command to start a new session (optional if continuing existing session)
        input_keys: Keystrokes to send to the session (e.g., "hello\\n" for typing hello + Enter)
        session_id: ID of existing session to continue (optional)
        idle_threshold: Seconds to wait for output before considering session idle (default: 5.0)
    
    Returns:
        str: JSON string containing session info, screen content, and any errors
    
    Example:
        # Start a Python REPL session
        >>> result = terminal_session_tool("python")
        >>> session_data = json.loads(result)
        >>> session_id = session_data["session_id"]
        
        # Send commands to the session
        >>> result = terminal_session_tool(
        ...     input_keys="print('Hello, World!')\\n",
        ...     session_id=session_id
        ... )
    """
    try:
        tool_input = {}
        
        if command:
            tool_input["command"] = command
        if input_keys:
            tool_input["input_keys"] = input_keys
        if session_id:
            tool_input["session_id"] = session_id
        if idle_threshold != 5.0:
            tool_input["idle_threshold"] = idle_threshold
        
        tool_call = ToolCall(
            name="run_command",
            input=tool_input
        )
        
        # Execute with lifecycle management
        result = run_tool_with_lifecycle_management(tool_call)
        
        # Format the result for session tools
        formatted_result = {
            "session_id": result.get("session_id"),
            "screen": result.get("screen", ""),
            "exit_code": result.get("returncode", result.get("exit_code", 0)),
            "error": result.get("error"),
            "status": "active" if result.get("session_id") else "ended"
        }
        
        # Include output if present (for non-interactive commands)
        if "output" in result:
            formatted_result["output"] = result["output"]
        
        return json.dumps(formatted_result)
        
    except Exception as e:
        return json.dumps({
            "session_id": None,
            "screen": "",
            "exit_code": -1,
            "error": f"Failed to manage session: {str(e)}",
            "status": "error"
        })

def check_hecate_requirements() -> bool:
    """
    Check if all requirements for terminal tools are met.
    
    Returns:
        bool: True if all requirements are met, False otherwise
    """
    # Check for required environment variables
    required_vars = ["MORPH_API_KEY"]
    optional_vars = ["OPENAI_API_KEY"]  # Needed for Hecate's LLM features
    
    missing_required = [var for var in required_vars if not os.getenv(var)]
    missing_optional = [var for var in optional_vars if not os.getenv(var)]
    
    if missing_required:
        print(f"Missing required environment variables: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"Warning: Missing optional environment variables: {', '.join(missing_optional)}")
        print("   (Some Hecate features may be limited)")
    
    # Check if Hecate is importable
    try:
        import hecate
        return True
    except ImportError:
        print("Hecate is not installed. Please install it with: pip install hecate")
        return False

# Module-level initialization check
_requirements_met = check_hecate_requirements()

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("Terminal Tools Module")
    print("=" * 40)
    
    if not _requirements_met:
        print("Requirements not met. Please check the messages above.")
        exit(1)
    
    print("All requirements met!")
    print("\nAvailable Tools:")
    print("  - terminal_execute_tool: Execute single commands")
    print("  - terminal_session_tool: Interactive terminal sessions")
    
    print("\nUsage Examples:")
    print("  # Execute a command")
    print("  result = terminal_execute_tool('ls -la')")
    print("  ")
    print("  # Start an interactive session")
    print("  result = terminal_session_tool('python')")
    print("  session_data = json.loads(result)")
    print("  session_id = session_data['session_id']")
    print("  ")
    print("  # Send input to the session")
    print("  result = terminal_session_tool(")
    print("      input_keys='print(\"Hello\")\\\\n',")
    print("      session_id=session_id")
    print("  )")
    
    print("\nEnvironment Variables:")
    print(f"  MORPH_API_KEY: {'Set' if os.getenv('MORPH_API_KEY') else 'Not set'}")
    print(f"  OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set (optional)'}")
    print(f"  HECATE_VM_LIFETIME_SECONDS: {os.getenv('HECATE_VM_LIFETIME_SECONDS', '300')} (default: 300)")