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
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ Loaded environment variables from {env_path}")
else:
    print(f"ℹ️  No .env file found at {env_path}. Using system environment variables.")

# Import our tool system
from model_tools import get_tool_definitions, handle_function_call, check_toolset_requirements
from tools.terminal_tool import cleanup_vm
from tools.browser_tool import cleanup_browser


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
        model: str = "anthropic/claude-sonnet-4-20250514",  # OpenRouter format
        max_iterations: int = 10,
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
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
            ephemeral_system_prompt (str): System prompt used during agent execution but NOT saved to trajectories (optional)
            log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses (default: 20)
            log_prefix (str): Prefix to add to all log messages for identification in parallel processing (default: "")
            providers_allowed (List[str]): OpenRouter providers to allow (optional)
            providers_ignored (List[str]): OpenRouter providers to ignore (optional)
            providers_order (List[str]): OpenRouter providers to try in order (optional)
            provider_sort (str): Sort providers by price/throughput/latency (optional)
        """
        self.model = model
        self.max_iterations = max_iterations
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        self.base_url = base_url or ""  # Store for OpenRouter detection
        
        # Store OpenRouter provider preferences
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort

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
            # Keep OpenAI and httpx at WARNING level to reduce noise
            # We have our own retry and error logging that's more informative
            logging.getLogger('openai').setLevel(logging.WARNING)
            logging.getLogger('openai._base_client').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)
            logging.getLogger('httpcore').setLevel(logging.WARNING)
            logging.getLogger('asyncio').setLevel(logging.WARNING)  # Suppress asyncio debug
            print("🔍 Verbose logging enabled (OpenAI/httpx/asyncio internal logs suppressed)")
        else:
            # Set logging to INFO level for important messages only
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            # Suppress noisy library logging
            logging.getLogger('openai').setLevel(logging.ERROR)
            logging.getLogger('openai._base_client').setLevel(logging.ERROR)
            logging.getLogger('httpx').setLevel(logging.ERROR)
            logging.getLogger('httpcore').setLevel(logging.ERROR)
        
        # Initialize OpenAI client - defaults to OpenRouter
        client_kwargs = {}
        
        # Default to OpenRouter if no base_url provided
        if base_url:
            client_kwargs["base_url"] = base_url
        else:
            client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
        
        # Handle API key - OpenRouter is the primary provider
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            # Primary: OPENROUTER_API_KEY, fallback to direct provider keys
            client_kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY", "")
        
        try:
            self.client = OpenAI(**client_kwargs)
            print(f"🤖 AI Agent initialized with model: {self.model}")
            if base_url:
                print(f"🔗 Using custom base URL: {base_url}")
            # Always show API key info (masked) for debugging auth issues
            key_used = client_kwargs.get("api_key", "none")
            if key_used and key_used != "dummy-key" and len(key_used) > 12:
                print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
            else:
                print(f"⚠️  Warning: API key appears invalid or missing (got: '{key_used[:20] if key_used else 'none'}...')")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
        # Get available tools with filtering
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets
        )
        
        # Show tool configuration and store valid tool names for validation
        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
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
        
        # Show ephemeral system prompt status
        if self.ephemeral_system_prompt:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 Ephemeral system prompt: '{prompt_preview}' (not saved to trajectories)")
    
    def _has_content_after_think_block(self, content: str) -> bool:
        """
        Check if content has actual text after any <think></think> blocks.
        
        This detects cases where the model only outputs reasoning but no actual
        response, which indicates an incomplete generation that should be retried.
        
        Args:
            content: The assistant message content to check
            
        Returns:
            True if there's meaningful content after think blocks, False otherwise
        """
        if not content:
            return False
        
        import re
        # Remove all <think>...</think> blocks (including nested ones, non-greedy)
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())
    
    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        Get messages up to (but not including) the last assistant turn.
        
        This is used when we need to "roll back" to the last successful point
        in the conversation, typically when the final assistant message is
        incomplete or malformed.
        
        Args:
            messages: Full message list
            
        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []
        
        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # No assistant message found, return all messages
            return messages.copy()
        
        # Return everything up to (not including) the last assistant message
        return messages[:last_assistant_idx]
    
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
        
        return json.dumps(formatted_tools, ensure_ascii=False)
    
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
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        content += msg["content"] + "\n"
                    
                    # Add tool calls wrapped in XML tags
                    for tool_call in msg["tool_calls"]:
                        # Parse arguments - should always succeed since we validate during conversation
                        # but keep try-except as safety net
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # This shouldn't happen since we validate and retry during conversation,
                            # but if it does, log warning and use empty dict
                            logging.warning(f"Unexpected invalid JSON in trajectory conversion: {tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
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
                        }, ensure_ascii=False)
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
                    # Add <think> tags around reasoning for trajectory storage
                    content = ""
                    
                    # Prepend reasoning in <think> tags if available
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    content += msg["content"] or ""
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
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
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None
    ) -> Dict[str, Any]:
        """
        Run a complete conversation with tool calling until completion.

        Args:
            user_message (str): The user's message/question
            system_message (str): Custom system message (optional, overrides ephemeral_system_prompt if provided)
            conversation_history (List[Dict]): Previous conversation messages (optional)
            task_id (str): Unique identifier for this task to isolate VMs between concurrent tasks (optional, auto-generated if not provided)

        Returns:
            Dict: Complete conversation result with final response and message history
        """
        # Generate unique task_id if not provided to isolate VMs between concurrent tasks
        import uuid
        effective_task_id = task_id or str(uuid.uuid4())
        
        # Reset retry counters at the start of each conversation to prevent state leakage
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        
        # Initialize conversation
        messages = conversation_history or []
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        print(f"💬 Starting conversation: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        # Determine which system prompt to use for API calls (ephemeral)
        # Priority: explicit system_message > ephemeral_system_prompt > None
        active_system_prompt = system_message if system_message is not None else self.ephemeral_system_prompt
        
        # Main conversation loop
        api_call_count = 0
        final_response = None
        
        while api_call_count < self.max_iterations:
            api_call_count += 1
            
            # Prepare messages for API call
            # If we have an ephemeral system prompt, prepend it to the messages
            # Note: Reasoning is embedded in content via <think> tags for trajectory storage.
            # However, providers like Moonshot AI require a separate 'reasoning_content' field
            # on assistant messages with tool_calls. We handle both cases here.
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                
                # For assistant messages with tool_calls, providers require 'reasoning_content' field
                # Extract reasoning from our stored 'reasoning' field and add it as 'reasoning_content'
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    reasoning_text = msg.get("reasoning")
                    if reasoning_text:
                        # Add reasoning_content for API compatibility (Moonshot AI, Novita, etc.)
                        api_msg["reasoning_content"] = reasoning_text
                
                # Remove 'reasoning' field - it's for trajectory storage only
                # The reasoning is already in the content via <think> tags AND
                # we've added reasoning_content for API compatibility above
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # Remove 'reasoning_details' if present - we use reasoning_content instead
                if "reasoning_details" in api_msg:
                    api_msg.pop("reasoning_details")
                api_messages.append(api_msg)
            
            if active_system_prompt:
                # Insert system message at the beginning
                api_messages = [{"role": "system", "content": active_system_prompt}] + api_messages
            
            # Calculate approximate request size for logging
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = total_chars // 4  # Rough estimate: 4 chars per token
            
            print(f"\n{self.log_prefix}🔄 Making API call #{api_call_count}/{self.max_iterations}...")
            print(f"{self.log_prefix}   📊 Request size: {len(api_messages)} messages, ~{approx_tokens:,} tokens (~{total_chars:,} chars)")
            print(f"{self.log_prefix}   🔧 Available tools: {len(self.tools) if self.tools else 0}")
            
            # Log request details if verbose
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 6  # Increased to allow longer backoff periods

            while retry_count <= max_retries:
                try:
                    # Build OpenRouter provider preferences if specified
                    provider_preferences = {}
                    if self.providers_allowed:
                        provider_preferences["only"] = self.providers_allowed
                    if self.providers_ignored:
                        provider_preferences["ignore"] = self.providers_ignored
                    if self.providers_order:
                        provider_preferences["order"] = self.providers_order
                    if self.provider_sort:
                        provider_preferences["sort"] = self.provider_sort
                    
                    # Make API call with tools - increased timeout for long responses
                    api_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                        "tools": self.tools if self.tools else None,
                        "timeout": 600.0  # 10 minute timeout for very long responses
                    }
                    
                    # Add extra_body for OpenRouter (provider preferences + reasoning)
                    extra_body = {}
                    
                    # Add provider preferences if specified
                    if provider_preferences:
                        extra_body["provider"] = provider_preferences
                    
                    # Enable reasoning with xhigh effort for OpenRouter
                    if "openrouter" in self.base_url.lower():
                        extra_body["reasoning"] = {
                            "enabled": True,
                            "effort": "xhigh"
                        }
                    
                    if extra_body:
                        api_kwargs["extra_body"] = extra_body
                    
                    response = self.client.chat.completions.create(**api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    print(f"{self.log_prefix}⏱️  API call completed in {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # Log response with provider info if available
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")

                    # Validate response has valid choices before proceeding
                    if response is None or not hasattr(response, 'choices') or response.choices is None or len(response.choices) == 0:
                        # This is often rate limiting or provider returning malformed response
                        retry_count += 1
                        error_details = []
                        if response is None:
                            error_details.append("response is None")
                        elif not hasattr(response, 'choices'):
                            error_details.append("response has no 'choices' attribute")
                        elif response.choices is None:
                            error_details.append("response.choices is None")
                        else:
                            error_details.append("response.choices is empty")
                        
                        # Check for error field in response (some providers include this)
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, 'error') and response.error:
                            error_msg = str(response.error)
                            # Try to extract provider from error metadata
                            if hasattr(response.error, 'metadata') and response.error.metadata:
                                provider_name = response.error.metadata.get('provider_name', 'Unknown')
                        elif response and hasattr(response, 'message') and response.message:
                            error_msg = str(response.message)
                        
                        # Try to get provider from model field (OpenRouter often returns actual model used)
                        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                            provider_name = f"model={response.model}"
                        
                        # Check for x-openrouter-provider or similar metadata
                        if provider_name == "Unknown" and response:
                            # Log all response attributes for debugging
                            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                            if self.verbose_logging:
                                logging.debug(f"Response attributes for invalid response: {resp_attrs}")
                        
                        print(f"{self.log_prefix}⚠️  Invalid API response (attempt {retry_count}/{max_retries}): {', '.join(error_details)}")
                        print(f"{self.log_prefix}   🏢 Provider: {provider_name}")
                        print(f"{self.log_prefix}   📝 Provider message: {error_msg[:200]}")
                        print(f"{self.log_prefix}   ⏱️  Response time: {api_duration:.2f}s (fast response often indicates rate limiting)")
                        
                        if retry_count > max_retries:
                            print(f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded for invalid responses. Giving up.")
                            logging.error(f"{self.log_prefix}Invalid API response after {max_retries} retries.")
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"Invalid API response (choices is None/empty). Likely rate limited by provider.",
                                "failed": True  # Mark as failure for filtering
                            }
                        
                        # Longer backoff for rate limiting (likely cause of None choices)
                        wait_time = min(5 * (2 ** (retry_count - 1)), 120)  # 5s, 10s, 20s, 40s, 80s, 120s
                        print(f"{self.log_prefix}⏳ Retrying in {wait_time}s (extended backoff for possible rate limit)...")
                        logging.warning(f"Invalid API response (retry {retry_count}/{max_retries}): {', '.join(error_details)} | Provider: {provider_name}")
                        time.sleep(wait_time)
                        continue  # Retry the API call

                    # Check finish_reason before proceeding
                    finish_reason = response.choices[0].finish_reason
                    
                    # Handle "length" finish_reason - response was truncated
                    if finish_reason == "length":
                        print(f"{self.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens")
                        
                        # If we have prior messages, roll back to last complete state
                        if len(messages) > 1:
                            print(f"{self.log_prefix}   ⏪ Rolling back to last complete assistant turn")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                            
                            # Clean up VM and browser
                            try:
                                cleanup_vm(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup VM for task {effective_task_id}: {e}")
                            try:
                                cleanup_browser(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup browser for task {effective_task_id}: {e}")
                            
                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit"
                            }
                        else:
                            # First message was truncated - mark as failed
                            print(f"{self.log_prefix}❌ First response truncated - cannot recover")
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "First response truncated due to output length limit"
                            }
                    
                    break  # Success, exit retry loop

                except Exception as api_error:
                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    
                    # Enhanced error logging
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    
                    print(f"{self.log_prefix}⚠️  API call failed (attempt {retry_count}/{max_retries}): {error_type}")
                    print(f"{self.log_prefix}   ⏱️  Time elapsed before failure: {elapsed_time:.2f}s")
                    print(f"{self.log_prefix}   📝 Error: {str(api_error)[:200]}")
                    print(f"{self.log_prefix}   📊 Request context: {len(api_messages)} messages, ~{approx_tokens:,} tokens, {len(self.tools) if self.tools else 0} tools")
                    
                    # Check for non-retryable errors (context length exceeded)
                    is_context_length_error = any(phrase in error_msg for phrase in [
                        'context length', 'maximum context', 'token limit', 
                        'too many tokens', 'reduce the length', 'exceeds the limit'
                    ])
                    
                    if is_context_length_error:
                        print(f"{self.log_prefix}❌ Context length exceeded - this error cannot be resolved by retrying.")
                        print(f"{self.log_prefix}   💡 The conversation has accumulated too much content from tool responses.")
                        logging.error(f"{self.log_prefix}Context length exceeded: {approx_tokens:,} tokens. Cannot continue.")
                        # Return a partial result instead of crashing
                        return {
                            "messages": messages,
                            "completed": False,
                            "api_calls": api_call_count,
                            "error": f"Context length exceeded ({approx_tokens:,} tokens). Conversation terminated early.",
                            "partial": True
                        }
                    
                    if retry_count > max_retries:
                        print(f"{self.log_prefix}❌ Max retries ({max_retries}) exceeded. Giving up.")
                        logging.error(f"{self.log_prefix}API call failed after {max_retries} retries. Last error: {api_error}")
                        logging.error(f"{self.log_prefix}Request details - Messages: {len(api_messages)}, Approx tokens: {approx_tokens:,}")
                        raise api_error

                    wait_time = min(2 ** retry_count, 60)  # Exponential backoff: 2s, 4s, 8s, 16s, 32s, 60s, 60s
                    print(f"⚠️  OpenAI-compatible API call failed (attempt {retry_count}/{max_retries}): {str(api_error)[:100]}")
                    print(f"⏳ Retrying in {wait_time}s...")
                    logging.warning(f"API retry {retry_count}/{max_retries} after error: {api_error}")
                    time.sleep(wait_time)
            
            try:
                assistant_message = response.choices[0].message
                
                # Handle assistant response
                if assistant_message.content:
                    print(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    print(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # Validate tool call names - detect model hallucinations
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls 
                        if tc.function.name not in self.valid_tool_names
                    ]
                    
                    if invalid_tool_calls:
                        # Track retries for invalid tool calls
                        if not hasattr(self, '_invalid_tool_retries'):
                            self._invalid_tool_retries = 0
                        self._invalid_tool_retries += 1
                        
                        invalid_preview = invalid_tool_calls[0][:80] + "..." if len(invalid_tool_calls[0]) > 80 else invalid_tool_calls[0]
                        print(f"{self.log_prefix}⚠️  Invalid tool call detected: '{invalid_preview}'")
                        print(f"{self.log_prefix}   Valid tools: {sorted(self.valid_tool_names)}")
                        
                        if self._invalid_tool_retries < 3:
                            print(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_tool_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            print(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.")
                            # Return partial result - don't include the bad tool call in messages
                            self._invalid_tool_retries = 0  # Reset for next conversation
                            return {
                                "final_response": None,
                                "messages": messages,  # Messages up to last valid point
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }
                    
                    # Reset retry counter on successful tool call validation
                    if hasattr(self, '_invalid_tool_retries'):
                        self._invalid_tool_retries = 0
                    
                    # Validate tool call arguments are valid JSON
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        try:
                            json.loads(tc.function.arguments)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))
                    
                    if invalid_json_args:
                        # Track retries for invalid JSON arguments
                        self._invalid_json_retries += 1
                        
                        tool_name, error_msg = invalid_json_args[0]
                        print(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")
                        
                        if self._invalid_json_retries < 3:
                            print(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            print(f"{self.log_prefix}❌ Max retries (3) for invalid JSON arguments exceeded. Stopping as partial.")
                            self._invalid_json_retries = 0  # Reset for next conversation
                            return {
                                "final_response": None,
                                "messages": messages,  # Messages up to last valid point
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid JSON arguments for tool '{tool_name}': {error_msg}"
                            }
                    
                    # Reset retry counter on successful JSON validation
                    self._invalid_json_retries = 0
                    
                    # Extract reasoning from response if available (for reasoning models like minimax, kimi, etc.)
                    # Extract reasoning from response for storage
                    # The reasoning_content field will be added when preparing API messages
                    reasoning_text = None
                    if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
                        reasoning_text = assistant_message.reasoning
                    elif hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
                        reasoning_text = assistant_message.reasoning_content
                    
                    # Build assistant message with tool calls
                    # Content stays as-is; reasoning is stored separately and will be passed
                    # to the API via reasoning_content field when preparing api_messages
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "reasoning": reasoning_text,  # Stored for trajectory extraction & API calls
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
                    }
                    
                    messages.append(assistant_msg)
                    
                    # Execute each tool call
                    for i, tool_call in enumerate(assistant_message.tool_calls, 1):
                        function_name = tool_call.function.name
                        
                        # Parse arguments - should always succeed since we validated above
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            # This shouldn't happen since we validate and retry above
                            logging.warning(f"Unexpected JSON error after validation: {e}")
                            function_args = {}
                        
                        # Preview tool call arguments
                        args_str = json.dumps(function_args, ensure_ascii=False)
                        args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                        print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

                        tool_start_time = time.time()

                        # Execute the tool with task_id to isolate VMs between concurrent tasks
                        function_result = handle_function_call(function_name, function_args, effective_task_id)

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

                        # Preview tool response
                        response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                        print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")
                        
                        # Delay between tool calls
                        if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                            time.sleep(self.tool_delay)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""
                    
                    # Check if response only has think block with no actual content after it
                    if not self._has_content_after_think_block(final_response):
                        # Track retries for empty-after-think responses
                        if not hasattr(self, '_empty_content_retries'):
                            self._empty_content_retries = 0
                        self._empty_content_retries += 1
                        
                        content_preview = final_response[:80] + "..." if len(final_response) > 80 else final_response
                        print(f"{self.log_prefix}⚠️  Response only contains think block with no content after it")
                        print(f"{self.log_prefix}   Content: '{content_preview}'")
                        
                        if self._empty_content_retries < 3:
                            print(f"{self.log_prefix}🔄 Retrying API call ({self._empty_content_retries}/3)...")
                            # Don't add the incomplete message, just retry
                            continue
                        else:
                            # Max retries exceeded - roll back to last complete assistant turn
                            print(f"{self.log_prefix}❌ Max retries (3) for empty content exceeded. Rolling back to last complete turn.")
                            self._empty_content_retries = 0  # Reset for next conversation
                            
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                            
                            # Clean up VM and browser
                            try:
                                cleanup_vm(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup VM for task {effective_task_id}: {e}")
                            try:
                                cleanup_browser(effective_task_id)
                            except Exception as e:
                                if self.verbose_logging:
                                    logging.warning(f"Failed to cleanup browser for task {effective_task_id}: {e}")
                            
                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Model generated only think blocks with no actual response after 3 retries"
                            }
                    
                    # Reset retry counter on successful content
                    if hasattr(self, '_empty_content_retries'):
                        self._empty_content_retries = 0
                    
                    # Extract reasoning from response if available
                    reasoning_text = None
                    if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
                        reasoning_text = assistant_message.reasoning
                    elif hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
                        reasoning_text = assistant_message.reasoning_content
                    
                    # Build final assistant message
                    # Content stays as-is; reasoning stored separately for trajectory extraction
                    final_msg = {
                        "role": "assistant", 
                        "content": final_response,
                        "reasoning": reasoning_text  # Stored for trajectory extraction
                    }
                    
                    messages.append(final_msg)
                    
                    print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
                    break
                
            except Exception as e:
                error_msg = f"Error during OpenAI-compatible API call #{api_call_count}: {str(e)}"
                print(f"❌ {error_msg}")
                
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

        # Clean up VM and browser for this task after conversation completes
        try:
            cleanup_vm(effective_task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {effective_task_id}: {e}")
        
        try:
            cleanup_browser(effective_task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {effective_task_id}: {e}")

        return {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False  # True only when stopped due to invalid tool calls
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
    model: str = "anthropic/claude-sonnet-4-20250514",
    api_key: str = None,
    base_url: str = "https://openrouter.ai/api/v1",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """
    Main function for running the agent directly.

    Args:
        query (str): Natural language query for the agent. Defaults to Python 3.13 example.
        model (str): Model name to use (OpenRouter format: provider/model). Defaults to anthropic/claude-sonnet-4-20250514.
        api_key (str): API key for authentication. Uses OPENROUTER_API_KEY env var if not provided.
        base_url (str): Base URL for the model API. Defaults to https://openrouter.ai/api/v1
        max_turns (int): Maximum number of API call iterations. Defaults to 10.
        enabled_toolsets (str): Comma-separated list of toolsets to enable. Supports predefined
                              toolsets (e.g., "research", "development", "safe").
                              Multiple toolsets can be combined: "web,vision"
        disabled_toolsets (str): Comma-separated list of toolsets to disable (e.g., "terminal")
        list_tools (bool): Just list available tools and exit
        save_trajectories (bool): Save conversation trajectories to JSONL files (appends to trajectory_samples.jsonl). Defaults to False.
        save_sample (bool): Save a single trajectory sample to a UUID-named JSONL file for inspection. Defaults to False.
        verbose (bool): Enable verbose logging for debugging. Defaults to False.
        log_prefix_chars (int): Number of characters to show in log previews for tool calls/responses. Defaults to 20.

    Toolset Examples:
        - "research": Web search, extract, crawl + vision tools
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
            log_prefix_chars=log_prefix_chars
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
    
    # Save sample trajectory to UUID-named file if requested
    if save_sample:
        import uuid
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.jsonl"
        
        # Convert messages to trajectory format (same as batch_runner)
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )
        
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }
        
        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")
    
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    fire.Fire(main)
