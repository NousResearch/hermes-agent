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
        disabled_toolsets: List[str] = None
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
        """
        self.model = model
        self.max_iterations = max_iterations
        self.tool_delay = tool_delay
        
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
    
    def create_system_message(self, custom_system: str = None) -> str:
        """
        Create the system message for the agent.
        
        Args:
            custom_system (str): Custom system message (optional)
            
        Returns:
            str: System message content
        """
        if custom_system:
            return custom_system
        
        return (
            "You are an AI assistant that provides helpful responses. You may use extremely long chains of thought "
            "to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help "
            "come to a correct solution prior to answering. You should enclose your thoughts and internal monologue "
            "inside <thinking> tags.\n\n"
            "You are equipped with web research tools that allow you to search the web, extract content from web pages, "
            "and crawl websites. Use these tools to gather current information and provide accurate, well-researched responses. "
            "You can call multiple tools in parallel if they are not reliant on each other's results. You can also use "
            "sequential tool calls to build on data you've collected from previous tool calls. Continue using tools until "
            "you feel confident you have enough information to provide a comprehensive answer."
        )
    
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
        
        # Add system message if not already present
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {
                "role": "system",
                "content": self.create_system_message(system_message)
            })
        
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
        
        return {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": final_response is not None
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
    list_tools: bool = False
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
            disabled_toolsets=disabled_toolsets_list
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
