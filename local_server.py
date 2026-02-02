"""
Local OpenAI-compatible server implementation for Hermes-Agent (Atropos integration).

Extends the Atropos APIServer to work with local OpenAI-compatible APIs (e.g. vLLM, SGLang),
providing tokens_and_logprobs_completion support via client-side tokenization.
"""

import asyncio
import os
import warnings
from typing import Any, List, Optional

import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion

from atroposlib.envs.server_handling.server_baseline import (
    APIServer,
    APIServerConfig,
    ReasoningConfig,
)


class LocalServer(APIServer):
    """
    OpenAI-compatible local server with tokens_and_logprobs support.
    
    Uses an OpenAI-compatible API (typically at a /v1 endpoint) and handles
    token extraction via client-side tokenization.
    
    Note: Many local servers don't return per-token logprobs in the standard API,
    so this implementation uses placeholder logprobs (0.0) for PoC purposes.
    For production training, use vLLM/SGLang servers that return real logprobs.
    """

    def __init__(
        self,
        config: APIServerConfig,
        tokenizer: Optional[Any] = None,
        tokenizer_name: str = "gpt2",
        reasoning_config: Optional[ReasoningConfig] = None,
    ):
        """
        Initialize the local server.
        
        Args:
            config: Server configuration
            tokenizer: Pre-initialized tokenizer (optional)
            tokenizer_name: Name of tokenizer to load if tokenizer not provided
            reasoning_config: Optional reasoning configuration
        """
        # Build the OpenAI client pointing to the server's /v1 endpoint
        base_url = config.base_url
        if base_url and not base_url.endswith("/v1"):
            base_url = f"{base_url.rstrip('/')}/v1"
        
        self.openai = openai.AsyncClient(
            api_key=config.api_key or "local",  # Local servers often ignore auth
            base_url=base_url,
            timeout=config.timeout,
        )
        
        # Initialize tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                from transformers import AutoTokenizer  # type: ignore
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Missing optional dependency 'transformers'. Pass a tokenizer instance to LocalServer, "
                    "or install transformers to enable `tokenizer_name` auto-loading."
                ) from exc
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
        # Add a simple chat template if the tokenizer doesn't have one
        # This is needed for ManagedServer's chat_completion to work
        if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
            # Simple ChatML-style template
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )
        
        super().__init__(config, reasoning_config=reasoning_config)
        # Local servers are treated as always-healthy unless a status task is enabled.
        self.server_healthy = True

    @classmethod
    def from_env(
        cls,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        **kwargs,
    ) -> "LocalServer":
        """
        Create a LocalServer from environment variables (or explicit overrides).
        
        Env vars (checked in order):
        - base URL: ATROPOS_SERVER_BASE_URL, OPENAI_BASE_URL, LOCAL_LLM_BASE_URL, LLM_BASE_URL
        - model:    ATROPOS_SERVER_MODEL,    LLM_MODEL,       LOCAL_LLM_MODEL
        - api key:  ATROPOS_SERVER_API_KEY,  OPENAI_API_KEY,  LOCAL_LLM_API_KEY, LLM_API_KEY
        """
        from dotenv import load_dotenv
        load_dotenv()
        
        base_url = (
            base_url
            or os.getenv("ATROPOS_SERVER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LOCAL_LLM_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or "http://localhost:11434"
        )
        model = (
            model
            or os.getenv("ATROPOS_SERVER_MODEL")
            or os.getenv("LLM_MODEL")
            or os.getenv("LOCAL_LLM_MODEL")
            or "hermes3:8b"
        )
        api_key = (
            api_key
            or os.getenv("ATROPOS_SERVER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LOCAL_LLM_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        
        config = APIServerConfig(
            model_name=model,
            base_url=base_url,
            api_key=api_key or "local",
            timeout=kwargs.get("timeout", 120),
            num_max_requests_at_once=kwargs.get("num_max_requests_at_once", 4),
            num_requests_for_eval=kwargs.get("num_requests_for_eval", 4),
            health_check=False,  # Local dev servers often lack /health
        )
        
        return cls(config, tokenizer_name=tokenizer_name)

    async def check_server_status_task(self, chat_completion: bool = True):
        """
        Check if the server is healthy.
        
        For local development, we generally assume the server is healthy.
        """
        while True:
            try:
                # Simple health check via a minimal completion
                if chat_completion:
                    await self.openai.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1,
                    )
                else:
                    await self.openai.completions.create(
                        model=self.config.model_name,
                        prompt="hi",
                        max_tokens=1,
                    )
                self.server_healthy = True
            except Exception:
                self.server_healthy = False
            await asyncio.sleep(5)

    async def _chat_completion_wrapper(self, **kwargs) -> ChatCompletion:
        """
        Wrapper for chat completion using an OpenAI-compatible API.
        """
        assert kwargs.get("model") is not None, "Model is required!"
        assert kwargs.get("messages") is not None, "Messages are required!"
        
        n = kwargs.get("n", 1)
        
        # Some OpenAI-compatible servers don't support n > 1, so we make multiple requests.
        if n > 1:
            completion_list = await asyncio.gather(
                *[self.openai.chat.completions.create(**{**kwargs, "n": 1}) for _ in range(n)]
            )
            # Merge completions
            completions = completion_list[0]
            for c in completion_list[1:]:
                for choice in c.choices:
                    choice.index = len(completions.choices)
                    completions.choices.append(choice)
            return completions
        else:
            return await self.openai.chat.completions.create(**kwargs)

    async def _completion_wrapper(self, **kwargs) -> Completion:
        """
        Wrapper for completion using an OpenAI-compatible API.
        """
        assert kwargs.get("model") is not None, "Model is required!"
        assert kwargs.get("prompt") is not None, "Prompt is required!"
        
        n = kwargs.get("n", 1)
        
        # Some OpenAI-compatible servers don't support n > 1.
        if n > 1:
            completion_list = await asyncio.gather(
                *[self.openai.completions.create(**{**kwargs, "n": 1}) for _ in range(n)]
            )
            completions = completion_list[0]
            for c in completion_list[1:]:
                for choice in c.choices:
                    choice.index = len(completions.choices)
                    completions.choices.append(choice)
            return completions
        else:
            return await self.openai.completions.create(**kwargs)

    async def _tokens_and_logprobs_completion_wrapper(
        self, **kwargs
    ) -> tuple[List[int], List[List[int]], List[List[float]], List[str]]:
        """
        Wrapper for tokens and logprobs completion.
        
        Returns:
            Tuple of (prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons)
        
        Note: Many OpenAI-compatible local servers don't return per-token logprobs,
        so we use placeholder logprobs (0.0). For real training, use vLLM/SGLang.
        """
        model = kwargs.get("model")
        assert model is not None, "Model is required!"
        
        # Handle input_ids (from ManagedServer) or prompt
        if "input_ids" in kwargs:
            prompt_tokens = kwargs.pop("input_ids")
            prompt = self.tokenizer.decode(prompt_tokens)
            kwargs.pop("prompt", None)
        else:
            prompt = kwargs.pop("prompt", "")
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        n = kwargs.pop("n", 1)
        max_tokens = kwargs.pop("max_tokens", 256)
        temperature = kwargs.pop("temperature", 0.7)
        stop = kwargs.pop("stop", None)
        
        # Make completion requests
        completions = []
        for _ in range(n):
            try:
                response = await self.openai.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                completions.append(response)
            except Exception as e:
                # Fallback to chat completion if completion endpoint not supported
                warnings.warn(f"Completion API failed, trying chat: {e}")
                response = await self.openai.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
                # Convert to completion-like response
                completions.append(response)
        
        output_tokens_list = []
        output_logprobs_list = []
        finish_reasons = []
        
        for completion in completions:
            # Extract text from response
            if hasattr(completion.choices[0], "text"):
                # Completion API response
                text = completion.choices[0].text
                finish_reason = completion.choices[0].finish_reason or "stop"
            else:
                # Chat completion API response
                text = completion.choices[0].message.content or ""
                finish_reason = completion.choices[0].finish_reason or "stop"
            
            # Tokenize output
            output_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Placeholder logprobs (many local servers don't provide per-token logprobs).
            # In production, use vLLM/SGLang which return real logprobs
            output_logprobs = [0.0] * len(output_tokens)
            
            output_tokens_list.append(output_tokens)
            output_logprobs_list.append(output_logprobs)
            finish_reasons.append(finish_reason)
        
        return prompt_tokens, output_tokens_list, output_logprobs_list, finish_reasons

    def managed_server(self, tokenizer=None, track_tree: bool = False):
        """
        Create a ManagedServer context manager for this server.
        
        Args:
            tokenizer: Optional tokenizer override
            track_tree: Whether to maintain tree structure for multi-turn
            
        Returns:
            ManagedServer context manager
        """
        from atroposlib.envs.server_handling.managed_server import ManagedServer
        
        return ManagedServerContext(
            self,
            tokenizer=tokenizer or self.tokenizer,
            track_tree=track_tree,
        )


class ManagedServerContext:
    """
    Context manager wrapper for ManagedServer.
    
    Usage:
        async with server.managed_server(tokenizer=tokenizer) as managed:
            response = await managed.chat_completion(...)
            state = managed.get_state()
    """
    
    def __init__(self, server: LocalServer, tokenizer, track_tree: bool = False):
        self.server = server
        self.tokenizer = tokenizer
        self.track_tree = track_tree
        self.managed = None
    
    async def __aenter__(self):
        from atroposlib.envs.server_handling.managed_server import ManagedServer
        
        self.managed = ManagedServer(
            self.server,
            tokenizer=self.tokenizer,
            track_tree=self.track_tree,
        )
        return self.managed
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.managed:
            self.managed.reset()
        return False
