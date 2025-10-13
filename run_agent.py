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
import logging
import os
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime

# Import our tool system
from model_tools import get_tool_definitions, handle_function_call, check_toolset_requirements
from mock_web_tools import MOCK_TOOL_FUNCTIONS, MOCK_WEB_TOOLS

# Import WebSocket connection pool (optional dependency)
# Use synchronous API to avoid event loop management in agent layer
try:
    from api_endpoint.websocket_connection_pool import connect_sync, send_event_sync, is_connected
    WEBSOCKET_LOGGER_AVAILABLE = True
except ImportError:
    WEBSOCKET_LOGGER_AVAILABLE = False
    connect_sync = None
    send_event_sync = None
    is_connected = None
    print("⚠️  WebSocket logger not available (missing websockets package)")


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
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        enable_websocket_logging: bool = False,
        websocket_server: str = "ws://localhost:8000/ws",
        mock_web_tools: bool = False,
        mock_delay: int = 60
    ):
        """
        Initialize the AI Agent.
        
        Args:
            base_url (str): Base URL for the model API (optional)
            api_key (str): API key for authentication (optional, uses env var if not provided)
            model (str): Model name to use (default: "gpt-4")
            max_iterations (int): Maximum number of tool calling iterations (default: 10)
            tool_delay (float): Delay between tool calls in seconds (default: 1.0)
            enabled_toolsets (List[str]): Only enable tools from these toolsets (optional)
            disabled_toolsets (List[str]): Disable tools from these toolsets (optional)
            save_trajectories (bool): Whether to save conversation trajectories to JSONL files (default: False)
            verbose_logging (bool): Enable verbose logging for debugging (default: False)
            enable_websocket_logging (bool): Enable real-time WebSocket logging (default: False)
            websocket_server (str): WebSocket server URL (default: ws://localhost:8000/ws)
            mock_web_tools (bool): Use mock web tools for testing (no API calls, configurable delays) (default: False)
            mock_delay (int): Delay in seconds for mock web_extract to test timeout (default: 60)
        """
        self.model = model
        self.max_iterations = max_iterations
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.enable_websocket_logging = enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE
        self.websocket_server = websocket_server
        self.mock_web_tools = mock_web_tools
        self.mock_delay = mock_delay
        # Note: We use global ws_pool instead of per-instance connection
        
        # Store toolset filtering options
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # Configure logging
        if self.verbose_logging:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            # Also set OpenAI client logging to debug
            logging.getLogger('openai').setLevel(logging.DEBUG)
            logging.getLogger('httpx').setLevel(logging.DEBUG)
            print("🔍 Verbose logging enabled")
        else:
            # Set logging to INFO level for important messages only
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            # Reduce OpenAI client logging
            logging.getLogger('openai').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)
        
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
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets
        )
        
        # Show tool configuration
        if self.tools:
            tool_names = [tool["function"]["name"] for tool in self.tools]
            print(f"🛠️  Loaded {len(self.tools)} tools: {', '.join(tool_names)}")
            
            # Show filtering info if applied
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
        
        # Show mock tools status
        if self.mock_web_tools:
            print(f"🧪 MOCK MODE ENABLED - Web tools will use fake data (delay: {self.mock_delay}s)")
            print(f"   Perfect for testing WebSocket reconnection without API costs!")
    
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
    
    def _init_websocket_connection(self, session_id: str):
        """
        Initialize WebSocket connection pool if enabled.
        
        Connects to logging server using persistent connection pool.
        Connection is shared across all agent runs - no per-run overhead!
        
        Uses synchronous API - no event loop management needed in agent layer.
        """
        if self.enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE and connect_sync:
            try:
                # Connect to server (idempotent - safe if already connected)
                # API layer handles all event loop management internally
                connect_sync(self.websocket_server)
                
                # Send session_start event for this specific session
                send_event_sync("session_start", session_id, {
                    "session_id": session_id,
                    "start_time": datetime.now().isoformat()
                })
                
                print(f"📡 WebSocket logging enabled (session: {session_id[:8]}...)")
            except Exception as e:
                print(f"⚠️  Failed to initialize WebSocket connection: {e}")
                self.enable_websocket_logging = False
    
    def run_conversation(
        self, 
        user_message: str, 
        system_message: str = None, 
        conversation_history: List[Dict[str, Any]] = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.
        
        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            session_id (str): Optional session ID (generated if not provided)
            
        Returns:
            Dict: Complete conversation result with final response and message history
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        # Uses synchronous API - no event loop management in agent layer
        if self.enable_websocket_logging:
            try:
                # Connect to logging server and log initial query
                # All event loop management is handled inside the API layer
                self._init_websocket_connection(session_id)
                send_event_sync("query", session_id, {
                    "query": user_message,
                    "model": self.model,
                    "toolsets": self.enabled_toolsets
                })
            except Exception as e:
                print(f"⚠️  WebSocket logging initialization failed: {e}")
                import traceback
                if self.verbose_logging:
                    traceback.print_exc()
        
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
            
            # ============================================================
            # WEBSOCKET LOGGING: API Call Start
            # ============================================================
            # Log that we're about to make an API call to the model
            # Captures: which call number, how many messages, whether tools available
            if self.enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE and send_event_sync:
                try:
                    send_event_sync("api_call", session_id, {
                        "call_number": api_call_count,
                        "message_count": len(messages),
                        "has_tools": bool(self.tools)
                    })
                except Exception as e:
                    if self.verbose_logging:
                        print(f"⚠️  WebSocket logging error: {e}")
            
            # Log request details if verbose
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            
            while retry_count <= max_retries:
                try:
                    # Make API call with tools
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.tools if self.tools else None,
                        timeout=60.0  # Add explicit timeout
                    )

                    print(f"🔧 Response: {response}")
                    
                    api_duration = time.time() - api_start_time
                    print(f"⏱️  API call completed in {api_duration:.2f}s")
                    
                    # ============================================================
                    # WEBSOCKET LOGGING: API Response
                    # ============================================================
                    # Log the response we got back from the AI model
                    # Captures: what the model said, whether it wants tools, how long it took
                    if self.enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE and send_event_sync:
                        try:
                            assistant_msg = response.choices[0].message
                            send_event_sync("response", session_id, {
                                "call_number": api_call_count,
                                "content": assistant_msg.content if hasattr(assistant_msg, 'content') else None,
                                "has_tool_calls": hasattr(assistant_msg, 'tool_calls') and bool(assistant_msg.tool_calls),
                                "tool_call_count": len(assistant_msg.tool_calls) if hasattr(assistant_msg, 'tool_calls') and assistant_msg.tool_calls else 0,
                                "duration": api_duration
                            })
                        except Exception as e:
                            if self.verbose_logging:
                                print(f"⚠️  WebSocket logging error: {e}")
                    
                    if self.verbose_logging:
                        logging.debug(f"API Response received - Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    break  # Success, exit retry loop
                    
                except Exception as api_error:
                    retry_count += 1
                    if retry_count > max_retries:
                        raise api_error
                    
                    wait_time = min(2 ** retry_count, 10)  # Exponential backoff, max 10s
                    print(f"⚠️  API call failed (attempt {retry_count}/{max_retries}): {str(api_error)[:100]}")
                    print(f"⏳ Retrying in {wait_time}s...")
                    logging.warning(f"API retry {retry_count}/{max_retries} after error: {api_error}")
                    time.sleep(wait_time)
            
            try:
                assistant_message = response.choices[0].message
                
                # Handle assistant response
                if assistant_message.content:
                    print(f"🤖 Assistant: {assistant_message.content}")
                
                # Check for tool calls
                if assistant_message.tool_calls:

                    print(f"🔧 Tool calls: {assistant_message.tool_calls}")
                    print(f"🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
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
                        
                        # ============================================================
                        # WEBSOCKET LOGGING: Tool Call (Before Execution)
                        # ============================================================
                        # Log which tool we're about to execute and with what parameters
                        # This happens BEFORE tool runs, so we know what was requested
                        if self.enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE and send_event_sync:
                            try:
                                send_event_sync("tool_call", session_id, {
                                    "call_number": api_call_count,
                                    "tool_index": i,
                                    "tool_name": function_name,
                                    "parameters": function_args,  # E.g., {"query": "Python", "limit": 5}
                                    "tool_call_id": tool_call.id
                                })
                            except Exception as e:
                                if self.verbose_logging:
                                    print(f"⚠️  WebSocket logging error: {e}")
                        
                        tool_start_time = time.time()
                        
                        # Execute the tool (mock or real based on configuration)
                        if self.mock_web_tools and function_name in MOCK_TOOL_FUNCTIONS:
                            # Use mock implementation (no API calls, configurable delay)
                            mock_function = MOCK_TOOL_FUNCTIONS[function_name]
                            # Inject mock_delay for web_extract if not provided
                            if function_name == "web_extract" and "delay" not in function_args:
                                function_args["delay"] = self.mock_delay
                            function_result = mock_function(**function_args)
                        else:
                            # Use real tool implementation
                            function_result = handle_function_call(function_name, function_args)
                        
                        tool_duration = time.time() - tool_start_time
                        result_preview = function_result[:200] if len(function_result) > 200 else function_result
                        
                        if self.verbose_logging:
                            logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                            logging.debug(f"Tool result preview: {result_preview}...")
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "content": function_result,
                            "tool_call_id": tool_call.id
                        })
                        
                        print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                        
                        # ============================================================
                        # WEBSOCKET LOGGING: Tool Result (After Execution)
                        # ============================================================
                        # Log the result we got back from the tool
                        # IMPORTANT: Logs BOTH truncated preview AND full raw result
                        # 
                        # Why both?
                        # - result: Truncated to 1000 chars for quick preview in UI
                        # - raw_result: FULL untruncated output for verification
                        #
                        # This is crucial for web tools where you want to see:
                        # - What the scraper actually returned (raw_result)
                        # - What the LLM processed it into (compare against raw)
                        # - Verify the LLM isn't losing important information
                        if self.enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE and send_event_sync:
                            try:
                                send_event_sync("tool_result", session_id, {
                                    "call_number": api_call_count,
                                    "tool_index": i,
                                    "tool_name": function_name,
                                    "result": function_result[:1000] if function_result else None,  # Truncated preview
                                    "raw_result": function_result,  # Full untruncated result (can be 100KB+)
                                    "error": None,
                                    "duration": tool_duration,
                                    "tool_call_id": tool_call.id
                                })
                            except Exception as e:
                                if self.verbose_logging:
                                    print(f"⚠️  WebSocket logging error: {e}")
                        
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
                
                # ============================================================
                # WEBSOCKET LOGGING: Error Event
                # ============================================================
                # Log any errors that occur during API calls or tool execution
                # Helps track failures and debug issues
                if self.enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE and send_event_sync:
                    try:
                        send_event_sync("error", session_id, {
                            "error_message": error_msg,
                            "call_number": api_call_count
                        })
                    except Exception as ws_error:
                        if self.verbose_logging:
                            print(f"⚠️  WebSocket logging error: {ws_error}")
                
                if self.verbose_logging:
                    logging.exception("Detailed error information:")
                
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
        
        # ============================================================
        # WEBSOCKET LOGGING: Session Complete
        # ============================================================
        # Log final completion event for this session
        # Note: WebSocket connection stays open for future runs (persistent pool)
        # Uses synchronous API - no event loop management in agent layer
        if self.enable_websocket_logging and WEBSOCKET_LOGGER_AVAILABLE and send_event_sync:
            try:
                # Log completion with summary information
                # API layer handles event loop management internally
                send_event_sync("complete", session_id, {
                    "final_response": final_response,      # What the agent finally answered
                    "total_calls": api_call_count,          # How many API calls were made
                    "completed": completed                   # Did it finish successfully?
                })
                # Connection persists automatically - agent has no control over lifecycle
            except Exception as e:
                if self.verbose_logging:
                    print(f"⚠️  WebSocket logging error: {e}")
                    import traceback
                    traceback.print_exc()
        
        return {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "session_id": session_id if self.enable_websocket_logging else None
        }
    
    def chat(self, message: str) -> str: # After we connect the UI we can put whatever we want here
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
    model: str = "claude-sonnet-4-5-20250929", 
    api_key: str = None,
    base_url: str = "https://api.anthropic.com/v1/",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    verbose: bool = False,
    enable_websocket_logging: bool = False,
    websocket_server: str = "ws://localhost:8000/ws",
    mock_web_tools: bool = False,
    mock_delay: int = 60
):
    """
    Main function for running the agent directly.
    
    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use. Defaults to claude-opus-4-20250514.
        api_key (str): API key for authentication. Uses ANTHROPIC_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://api.anthropic.com/v1/
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined 
                              toolsets (e.g., "research", "development", "safe"). 
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files. Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        enable_websocket_logging (bool): Enable real-time WebSocket logging. Defaults to False.
        websocket_server (str): WebSocket server URL. Defaults to ws://localhost:8000/ws.
        mock_web_tools (bool): Use mock web tools for testing (no API calls, configurable delays). Defaults to False.
        mock_delay (int): Delay in seconds for mock web_extract (default: 60s to test timeout). Defaults to 60.
        
    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
    
    Mock Tools (Testing):
        Use --mock_web_tools to test WebSocket reconnection without API calls:
        - web_search: Returns fake results after 2s
        - web_extract: Returns fake content after 60s (tests timeout)
        - web_crawl: Returns fake pages after 30s
    
    WebSocket Logging:
        1. Start logging server: python logging_server.py
        2. Run agent with --enable_websocket_logging flag
        3. View logs in realtime at http://localhost:8000
    """
    print("🤖 AI Agent with Tool Calling")
    print("=" * 50)
    
    # Handle tool listing
    if list_tools:
        from model_tools import get_all_tool_names, get_toolset_for_tool, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info
        
        print("📋 Available Tools & Toolsets:")
        print("-" * 50)
        
        # Show new toolsets system
        print("\n🎯 Predefined Toolsets (New System):")
        print("-" * 40)
        all_toolsets = get_all_toolsets()
        
        # Group by category
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []
        
        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in ["research", "development", "analysis", "content_creation", "full_stack"]:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)
        
        # Print basic toolsets
        print("\n📌 Basic Toolsets:")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Tools: {tools_str}")
        
        # Print composite toolsets
        print("\n📂 Composite Toolsets (built from other toolsets):")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    Includes: {includes_str}")
            print(f"    Total tools: {info['tool_count']}")
        
        # Print scenario-specific toolsets
        print("\n🎭 Scenario-Specific Toolsets:")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    Total tools: {info['tool_count']}")
        
        
        # Show legacy toolset compatibility
        print("\n📦 Legacy Toolsets (for backward compatibility):")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        # Show individual tools
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print(f"\n💡 Usage Examples:")
        print(f"  # Use predefined toolsets")
        print(f"  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print(f"  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print(f"  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print(f"  ")
        print(f"  # Combine multiple toolsets")
        print(f"  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print(f"  ")
        print(f"  # Disable toolsets")
        print(f"  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print(f"  ")
        print(f"  # Run with trajectory saving enabled")
        print(f"  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    # Parse toolset selection arguments
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
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
    
    if enable_websocket_logging:
        print(f"📡 WebSocket logging: ENABLED")
        print(f"   - Server: {websocket_server}")
        print(f"   - Make sure logging server is running: python logging_server.py")
    
    # Initialize agent with provided parameters
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            enable_websocket_logging=enable_websocket_logging,
            websocket_server=websocket_server,
            mock_web_tools=mock_web_tools,
            mock_delay=mock_delay
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

    # There needs to be a multi-turn conversation here
    # Hermes Agent needs to be multi-turn to be useful
    
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


# Order of operations:
# First track the ways in which information flows through the agent in realtime
# Create a FastAPI endpoint that is first able to listen for the logging through sockets
# Create the UI through there and now you have you have a pretty UI. CHECKPOINT 1
# Now that you have better visualization write out the chat interface and allow it to be controlled through the UI as well as the main program
# Now decide how the information flows through the agent you may need to do some trial and error to get this part right
# Implement multiturn conversation now and then CHECKPOINT 2 is now done with multiturn conversations