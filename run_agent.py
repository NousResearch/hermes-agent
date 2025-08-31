#!/usr/bin/env python3
"""
AI Agent Runner with Tool Calling

This module provides a clean, standalone agent that can execute AI models
with tool calling capabilities. It handles the conversation loop, tool execution,
and response management.

Features:
- Automatic tool calling loop until completion
- Configurable model parameters
- Error handling and recovery
- Message history management
- Support for multiple model providers

Usage:
    from run_agent import AIAgent
    
    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime

# Import our tool system
from model_tools import get_tool_definitions, handle_function_call, check_toolset_requirements


class AIAgent:
    """
    AI Agent with tool calling capabilities.
    
    This class manages the conversation flow, tool execution, and response handling
    for AI models that support function calling.
    """
    
    def __init__(
        self, 
        base_url: str = None, 
        api_key: str = None, 
        model: str = "gpt-4",
        max_iterations: int = 10,
        tool_delay: float = 1.0,
        enabled_tools: List[str] = None,
        disabled_tools: List[str] = None,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        morph_snapshot_id: str | None = None,
    ):
        """
        Initialize the AI Agent.
        
        Args:
            base_url (str): Base URL for the model API (optional)
            api_key (str): API key for authentication (optional, uses env var if not provided)
            model (str): Model name to use (default: "gpt-4")
            max_iterations (int): Maximum number of tool calling iterations (default: 10)
            tool_delay (float): Delay between tool calls in seconds (default: 1.0)
            enabled_tools (List[str]): Only enable these specific tools (optional)
            disabled_tools (List[str]): Disable these specific tools (optional)
            enabled_toolsets (List[str]): Only enable tools from these toolsets (optional)
            disabled_toolsets (List[str]): Disable tools from these toolsets (optional)
            save_trajectories (bool): Whether to save conversation trajectories to JSONL files (default: False)
            morph_snapshot_id (str | None): Morph Cloud snapshot id from which to start terminal tool
        """
        self.model = model
        self.max_iterations = max_iterations
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.morph_snapshot_id = morph_snapshot_id
        
        # Store tool filtering options
        self.enabled_tools = enabled_tools
        self.disabled_tools = disabled_tools
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # Initialize OpenAI client
        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            client_kwargs["api_key"] = os.getenv("ANTHROPIC_API_KEY", "dummy-key")
        
        try:
            self.client = OpenAI(**client_kwargs)
            print(f"🤖 AI Agent initialized with model: {self.model}")
            if base_url:
                print(f"🔗 Using custom base URL: {base_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # Get available tools with filtering
        self.tools = get_tool_definitions(
            enabled_tools=enabled_tools,
            disabled_tools=disabled_tools,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets
        )
        
        # Show tool configuration
        if self.tools:
            tool_names = [tool["function"]["name"] for tool in self.tools]
            print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")
            
            # Show filtering info if applied
            if enabled_tools:
                print(f"   ✅ Enabled tools: {', '.join(enabled_tools)}")
            if disabled_tools:
                print(f"   ❌ Disabled tools: {', '.join(disabled_tools)}")
            if enabled_toolsets:
                print(f"   ✅ Enabled toolsets: {', '.join(enabled_toolsets)}")
            if disabled_toolsets:
                print(f"   ❌ Disabled toolsets: {', '.join(disabled_toolsets)}")
        else:
            print("🛠️  No tools loaded (all tools filtered out or unavailable)")
        
        # Check tool requirements
        if self.tools:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  Some tools may not work due to missing requirements: {missing_reqs}")
        
        # Show trajectory saving status
        if self.save_trajectories:
            print("📝 Trajectory saving enabled")
    
    def _format_tools_for_system_message(self) -> str:
        """
        Format tool definitions for the system message in the trajectory format.
        
        Returns:
            str: JSON string representation of tool definitions
        """
        if not self.tools:
            return "[]"
        
        # Convert tool definitions to the format expected in trajectories
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # Match the format in the example
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools)
    
    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        Convert internal message format to trajectory format for saving.
        
        Args:
            messages (List[Dict]): Internal message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
            
        Returns:
            List[Dict]: Messages in trajectory format
        """
        trajectory = []
        
        # Add system message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # Add the initial user message
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # Process remaining messages
        i = 1  # Skip the first user message as we already added it
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                # Check if this message has tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Format assistant message with tool calls
                    content = ""
                    if msg.get("content") and msg["content"].strip():
                        content = msg["content"] + "\n"
                    
                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json)}\n</tool_call>\n"
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # Collect all subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # Format tool response with XML tags
                        tool_response = f"<tool_response>\n"
                        
                        # Try to parse tool content as JSON if it looks like JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # Keep as string if not valid JSON
                        
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": msg["tool_calls"][len(tool_responses)]["function"]["name"] if len(tool_responses) < len(msg["tool_calls"]) else "unknown",
                            "content": tool_content
                        })
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # Add all tool responses as a single message
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # Skip the tool messages we just processed
                
                else:
                    # Regular assistant message without tool calls
                    trajectory.append({
                        "from": "gpt",
                        "value": msg["content"] or ""
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory
    
    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        Save conversation trajectory to JSONL file.
        
        Args:
            messages (List[Dict]): Complete message history
            user_query (str): Original user query
            completed (bool): Whether the conversation completed successfully
        """
        if not self.save_trajectories:
            return
        
        # Convert messages to trajectory format
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        
        # Determine which file to save to
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"
        
        # Create trajectory entry
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "completed": completed
        }
        
        # Append to JSONL file
        try:
            with open(filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"💾 Trajectory saved to {filename}")
        except Exception as e:
            print(f"⚠️ Failed to save trajectory: {e}")
    
    def run_conversation(
        self, 
        user_message: str, 
        system_message: str = None, 
        conversation_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.
        
        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            
        Returns:
            Dict: Complete conversation result with final response and message history
        """
        # Initialize conversation
        messages = conversation_history or []
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        # Main conversation loop
        api_call_count = 0
        final_response = None
        
        while api_call_count < self.max_iterations:
            api_call_count += 1
            print(f"\n🔄 Making API call #{api_call_count}...")
            
            try:
                # Make API call with tools
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools if self.tools else None
                )
                
                assistant_message = response.choices[0].message
                
                # Handle assistant response
                if assistant_message.content:
                    print(f"🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    print(f"🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    # Add assistant message with tool calls to conversation
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": tool_call.type,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                            for tool_call in assistant_message.tool_calls
                        ]
                    })
                    
                    # Execute each tool call
                    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
                        function_name = tool_call.function.name
                        
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            print(f"❌ Invalid JSON in tool call arguments: {e}")
                            function_args = {}
                        
                        print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")

                        if function_name == "terminal" and self.morph_snapshot_id is not None:
                            function_args["snapshot_id"] = self.morph_snapshot_id
                        
                        # Execute the tool
                        function_result = handle_function_call(function_name, function_args)
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "content": function_result,
                            "tool_call_id": tool_call.id
                        })
                        
                        print(f"  ✅ Tool {i} completed")
                        
                        # Delay between tool calls
                        if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                            time.sleep(self.tool_delay)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""
                    
                    # Add final assistant message
                    messages.append({
                        "role": "assistant", 
                        "content": final_response
                    })
                    
                    print(f"🎉 Conversation completed after {api_call_count} API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during API call #{api_call_count}: {str(e)}"
                print(f"❌ {error_msg}")
                
                # Add error to conversation and try to continue
                messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {error_msg}. Let me try a different approach."
                })
                
                # If we're near the limit, break to avoid infinite loops
                if api_call_count >= self.max_iterations - 1:
                    final_response = f"I apologize, but I encountered repeated errors: {error_msg}"
                    break
        
        # Handle max iterations reached
        if api_call_count >= self.max_iterations:
            print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Stopping to prevent infinite loop.")
            if final_response is None:
                final_response = "I've reached the maximum number of iterations. Here's what I found so far."
        
        # Determine if conversation completed successfully
        completed = final_response is not None and api_call_count < self.max_iterations
        
        # Save trajectory if enabled
        self._save_trajectory(messages, user_message, completed)
        
        return {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed
        }
    
    def chat(self, message: str) -> str:
        """
        Simple chat interface that returns just the final response.
        
        Args:
            message (str): User message
            
        Returns:
            str: Final assistant response
        """
        result = self.run_conversation(message)
        return result["final_response"]


def main(
    query: str = None,
    model: str = "claude-opus-4-20250514", 
    api_key: str = None,
    base_url: str = "https://api.anthropic.com/v1/",
    max_turns: int = 10,
    enabled_tools: str = None,
    disabled_tools: str = None,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    morph_snapshot_id: str | None = None,
):
    """
    Main function for running the agent directly.
    
    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use. Defaults to claude-opus-4-20250514.
        api_key (str): API key for authentication. Uses ANTHROPIC_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://api.anthropic.com/v1/
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_tools (str): Comma-separated list of tools to enable (e.g., "web_search,terminal")
        disabled_tools (str): Comma-separated list of tools to disable (e.g., "terminal")
        enabled_toolsets (str): Comma-separated list of toolsets to enable (e.g., "web_tools")
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal_tools")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files. Defaults to False.
        morph_snapshot_id (str | None): Morph Cloud snapshot id to start terminal tool from
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    
    # Handle tool listing
    if list_tools:
        from model_tools import get_all_tool_names, get_toolset_for_tool, get_available_toolsets
        
        print("📋 Available Tools & Toolsets:")
        print("-" * 30)
        
        # Show toolsets
        toolsets = get_available_toolsets()
        print("📦 Toolsets:")
        for name, info in toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        # Show individual tools
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in all_tools:
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print(f"\n💡 Usage Examples:")
        print(f"  # Run with only web tools")
        print(f"  python run_agent.py --enabled_toolsets=web_tools --query='search for Python news'")
        print(f"  # Run with specific tools only")
        print(f"  python run_agent.py --enabled_tools=web_search,web_extract --query='research topic'")
        print(f"  # Run without terminal tools")
        print(f"  python run_agent.py --disabled_tools=terminal --query='web research only'")
        print(f"  # Run with trajectory saving enabled")
        print(f"  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    # Parse tool selection arguments
    enabled_tools_list = None
    disabled_tools_list = None
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
    if enabled_tools:
        enabled_tools_list = [t.strip() for t in enabled_tools.split(",")]
        print(f"🎯 Enabled tools: {enabled_tools_list}")
    
    if disabled_tools:
        disabled_tools_list = [t.strip() for t in disabled_tools.split(",")]
        print(f"🚫 Disabled tools: {disabled_tools_list}")
    
    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")
    
    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")
    
    if save_trajectories:
        print(f"💾 Trajectory saving: ENABLED")
        print(f"   - Successful conversations → trajectory_samples.jsonl")
        print(f"   - Failed conversations → failed_trajectories.jsonl")
    
    # Initialize agent with provided parameters
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_tools=enabled_tools_list,
            disabled_tools=disabled_tools_list,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            morph_snapshot_id=morph_snapshot_id
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Use provided query or default to Python 3.13 example
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)
    
    # Run conversation
    result = agent.run_conversation(user_query)
    
    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")
    
    if result['final_response']:
        print(f"\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])
    
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    fire.Fire(main)
