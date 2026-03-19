"""
Cosmos Hugging Face Integration.

"Open-source AI for the swarm - thousands of models at your fingertips!"

Hugging Face provides:
- Inference API: Hosted models for text, image, audio generation
- Transformers: Local model execution with GPU support
- Hub: Access to 500k+ models
- Text Generation: Llama, Mistral, Mixtral, Qwen, Phi, and more
- Image Generation: Stable Diffusion, FLUX, etc.
- Embeddings: Sentence transformers, BGE, etc.
- Specialized: Code, math, vision-language models

API: https://huggingface.co/docs/api-inference
"""

from typing import Dict, Any, List, Optional, Union
from loguru import logger
import aiohttp
import asyncio
import os
import json
import base64
from pathlib import Path
from dataclasses import dataclass

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus


def _get_dynamic_max_tokens(model_id: str = "huggingface", task_type: str = "chat") -> int:
    """
    AGI v1.8: Get dynamic max_tokens from centralized limits.
    """
    try:
        from Cosmos.core.dynamic_limits import get_max_tokens
        return get_max_tokens(model_id, task_type)
    except Exception:
        defaults = {"chat": 1000, "thinking": 2000, "quick": 300, "code": 2000}
        return defaults.get(task_type, 1000)


@dataclass
class HFModelInfo:
    """Information about a Hugging Face model."""
    id: str
    task: str
    description: str
    context_length: int = 4096
    is_chat: bool = True


class HuggingFaceProvider(ExternalProvider):
    """
    Hugging Face integration for text generation, embeddings, and more.

    Supports both:
    - Inference API (hosted, requires HF_API_KEY)
    - Local transformers (requires transformers + torch)
    """

    def __init__(self, api_key: str = None):
        super().__init__(IntegrationConfig(name="huggingface"))
        self.api_key = api_key or os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
        self.inference_url = "https://api-inference.huggingface.co/models"
        self.chat_url = "https://api-inference.huggingface.co/models"

        # Default models for different tasks
        self.default_chat_model = "mistralai/Mistral-7B-Instruct-v0.3"
        self.default_code_model = "codellama/CodeLlama-7b-Instruct-hf"
        self.default_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.default_image_model = "black-forest-labs/FLUX.1-schnell"

        # Popular model catalog
        self.models = {
            # Chat/Instruction models
            "mistral-7b": HFModelInfo(
                id="mistralai/Mistral-7B-Instruct-v0.3",
                task="text-generation",
                description="Fast, efficient 7B instruction model",
                context_length=32768
            ),
            "mixtral-8x7b": HFModelInfo(
                id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                task="text-generation",
                description="MoE model with 8 experts, very capable",
                context_length=32768
            ),
            "llama-3-8b": HFModelInfo(
                id="meta-llama/Meta-Llama-3-8B-Instruct",
                task="text-generation",
                description="Meta's Llama 3 8B instruction model",
                context_length=8192
            ),
            "llama-3-70b": HFModelInfo(
                id="meta-llama/Meta-Llama-3-70B-Instruct",
                task="text-generation",
                description="Meta's Llama 3 70B - highly capable",
                context_length=8192
            ),
            "qwen-2.5-72b": HFModelInfo(
                id="Qwen/Qwen2.5-72B-Instruct",
                task="text-generation",
                description="Alibaba's Qwen 2.5 72B - excellent performance",
                context_length=32768
            ),
            "phi-3-mini": HFModelInfo(
                id="microsoft/Phi-3-mini-4k-instruct",
                task="text-generation",
                description="Microsoft's small but capable model",
                context_length=4096
            ),
            "zephyr-7b": HFModelInfo(
                id="HuggingFaceH4/zephyr-7b-beta",
                task="text-generation",
                description="HF's aligned chat model",
                context_length=8192
            ),

            # Code models
            "codellama-7b": HFModelInfo(
                id="codellama/CodeLlama-7b-Instruct-hf",
                task="text-generation",
                description="Meta's code-specialized Llama",
                context_length=16384
            ),
            "starcoder2-15b": HFModelInfo(
                id="bigcode/starcoder2-15b-instruct-v0.1",
                task="text-generation",
                description="BigCode's code generation model",
                context_length=16384
            ),
            "deepseek-coder": HFModelInfo(
                id="deepseek-ai/deepseek-coder-6.7b-instruct",
                task="text-generation",
                description="DeepSeek's code-focused model",
                context_length=16384
            ),

            # Embedding models
            "bge-large": HFModelInfo(
                id="BAAI/bge-large-en-v1.5",
                task="feature-extraction",
                description="BGE large embeddings (1024 dim)",
                is_chat=False
            ),
            "e5-large": HFModelInfo(
                id="intfloat/e5-large-v2",
                task="feature-extraction",
                description="E5 embeddings (1024 dim)",
                is_chat=False
            ),
            "minilm": HFModelInfo(
                id="sentence-transformers/all-MiniLM-L6-v2",
                task="feature-extraction",
                description="Fast, small embeddings (384 dim)",
                is_chat=False
            ),

            # Image models
            "flux-schnell": HFModelInfo(
                id="black-forest-labs/FLUX.1-schnell",
                task="text-to-image",
                description="Fast FLUX image generation",
                is_chat=False
            ),
            "sdxl": HFModelInfo(
                id="stabilityai/stable-diffusion-xl-base-1.0",
                task="text-to-image",
                description="Stable Diffusion XL",
                is_chat=False
            ),
        }

        # Local transformers support
        self._local_model = None
        self._local_tokenizer = None
        self._local_pipeline = None
        self._use_local = False
        self._transformers_available = self._check_transformers()

    def _check_transformers(self) -> bool:
        """Check if transformers library is available for local inference."""
        try:
            import transformers
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
            logger.info(f"Transformers available, CUDA: {self._cuda_available}")
            return True
        except ImportError:
            self._torch_available = False
            self._cuda_available = False
            return False

    def enable_local(self, model_id: str = None):
        """Enable local transformer inference (downloads model if needed)."""
        if not self._transformers_available:
            raise RuntimeError("transformers library not installed. Run: pip install transformers torch")

        model_id = model_id or "microsoft/Phi-3-mini-4k-instruct"

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            logger.info(f"Loading local model: {model_id}...")

            self._local_tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Load model with appropriate device
            if self._cuda_available:
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self._local_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )

            self._local_pipeline = pipeline(
                "text-generation",
                model=self._local_model,
                tokenizer=self._local_tokenizer
            )

            self._use_local = True
            self._local_model_id = model_id
            logger.info(f"Local model ready: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            return False

    async def local_chat(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 500,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Chat using locally loaded transformer model (no API needed)."""
        if not self._use_local or self._local_pipeline is None:
            return {"error": "Local model not loaded. Call enable_local() first.", "content": None}

        import asyncio

        # Build prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        formatted = self._format_chat_prompt(messages, self._local_model_id)

        try:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._local_pipeline(
                    formatted,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    return_full_text=False
                )
            )

            content = result[0]["generated_text"] if result else ""
            content = self._clean_response(content, self._local_model_id)

            return {
                "content": content,
                "model": self._local_model_id,
                "provider": "huggingface_local",
                "tokens": len(content.split())
            }

        except Exception as e:
            logger.error(f"Local inference error: {e}")
            return {"error": str(e), "content": None}

    async def connect(self) -> bool:
        """Test connection to Hugging Face API or local transformers."""
        # If we have local transformers, we're "connected" even without API
        if self._transformers_available:
            self.status = ConnectionStatus.CONNECTED
            logger.info("HuggingFace: Local transformers available")
            return True

        if not self.api_key:
            logger.warning("HuggingFace: No API key and no local transformers")
            self.status = ConnectionStatus.ERROR
            return False

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with session.get(
                    "https://huggingface.co/api/whoami",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.status = ConnectionStatus.CONNECTED
                        logger.info(f"HuggingFace: Connected as {data.get('name', 'unknown')}")
                        return True
                    else:
                        self.status = ConnectionStatus.ERROR
                        logger.error(f"HuggingFace: API error {response.status}")
                        return False
        except Exception as e:
            self.status = ConnectionStatus.ERROR
            logger.error(f"HuggingFace: Connection error - {e}")
            return False

    async def sync(self):
        """Sync with HuggingFace - no-op for this provider."""
        # HuggingFace doesn't need polling - it's request/response
        pass

    async def execute_action(self, action: str, params: Dict[str, Any] = None) -> Any:
        """Execute a HuggingFace action (implements abstract method)."""
        return await self.execute(action, params)

    async def execute(self, action: str, params: Dict[str, Any] = None) -> Any:
        """Execute a Hugging Face action."""
        params = params or {}

        if action == "chat":
            return await self.chat(
                prompt=params.get("prompt"),
                system=params.get("system"),
                model=params.get("model", self.default_chat_model),
                max_tokens=params.get("max_tokens", 1000),
                temperature=params.get("temperature", 0.7)
            )
        elif action == "embed":
            return await self.embed(
                text=params.get("text"),
                model=params.get("model", self.default_embedding_model)
            )
        elif action == "generate_image":
            return await self.generate_image(
                prompt=params.get("prompt"),
                model=params.get("model", self.default_image_model),
                negative_prompt=params.get("negative_prompt")
            )
        elif action == "code":
            return await self.generate_code(
                prompt=params.get("prompt"),
                language=params.get("language", "python"),
                model=params.get("model", self.default_code_model)
            )
        else:
            raise ValueError(f"Unknown HuggingFace action: {action}")

    async def chat(
        self,
        prompt: str,
        system: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = None,  # AGI v1.8: None = dynamic default
        context: str = None,
        prefer_local: bool = True
    ) -> Dict[str, Any]:
        """
        Chat with a Hugging Face model - LOCAL FIRST, then API fallback.

        Args:
            prompt: User message
            system: System prompt
            model: Model ID or alias
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (None = dynamic default)
            context: Additional context
            prefer_local: Use local transformers if available (default True)

        Returns:
            Dict with 'content', 'model', 'tokens' keys
        """
        # AGI v1.8: Resolve dynamic max_tokens default
        if max_tokens is None:
            max_tokens = _get_dynamic_max_tokens("huggingface", "chat")

        # Try local first if enabled/available
        if prefer_local and self._use_local:
            result = await self.local_chat(prompt, system, max_tokens, temperature)
            if result.get("content"):
                return result

        # Try local if transformers available but not yet loaded
        if prefer_local and self._transformers_available and not self._use_local:
            try:
                # Auto-load a local model
                self.enable_local()
                if self._use_local:
                    result = await self.local_chat(prompt, system, max_tokens, temperature)
                    if result.get("content"):
                        return result
            except Exception as e:
                logger.debug(f"Local model load failed, falling back to API: {e}")

        # Fall back to API
        if not self.api_key:
            return {"error": "No HF_API_KEY and local model unavailable", "content": None}

        # Resolve model alias
        model_id = model or self.default_chat_model
        if model_id in self.models:
            model_id = self.models[model_id].id

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})
            messages.append({"role": "assistant", "content": "I understand the context. How can I help?"})
        messages.append({"role": "user", "content": prompt})

        # Format as chat template
        formatted_prompt = self._format_chat_prompt(messages, model_id)

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "inputs": formatted_prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "do_sample": temperature > 0,
                        "return_full_text": False
                    }
                }

                url = f"{self.inference_url}/{model_id}"

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()

                        # Handle different response formats
                        if isinstance(result, list) and len(result) > 0:
                            content = result[0].get("generated_text", "")
                        elif isinstance(result, dict):
                            content = result.get("generated_text", "")
                        else:
                            content = str(result)

                        # Clean up response
                        content = self._clean_response(content, model_id)

                        return {
                            "content": content,
                            "model": model_id,
                            "tokens": len(content.split()),
                            "provider": "huggingface"
                        }

                    elif response.status == 503:
                        # Model loading
                        data = await response.json()
                        wait_time = data.get("estimated_time", 30)
                        logger.info(f"HF model loading, waiting {wait_time}s...")
                        await asyncio.sleep(min(wait_time, 60))
                        return await self.chat(prompt, system, model, temperature, max_tokens)

                    else:
                        error_text = await response.text()
                        logger.error(f"HF API error {response.status}: {error_text}")
                        return {"error": error_text, "content": None}

        except Exception as e:
            logger.error(f"HuggingFace chat error: {e}")
            return {"error": str(e), "content": None}

    def _format_chat_prompt(self, messages: List[Dict], model_id: str) -> str:
        """Format messages using appropriate chat template."""
        # Detect model family and use appropriate template
        model_lower = model_id.lower()

        if "mistral" in model_lower or "mixtral" in model_lower:
            # Mistral format: [INST] user [/INST] assistant
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"[INST] {msg['content']} [/INST] Understood.")
                elif msg["role"] == "user":
                    prompt_parts.append(f"[INST] {msg['content']} [/INST]")
                elif msg["role"] == "assistant":
                    prompt_parts.append(msg["content"])
            return " ".join(prompt_parts)

        elif "llama" in model_lower:
            # Llama 2/3 format
            prompt_parts = []
            system_msg = ""
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    if system_msg:
                        prompt_parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|>")
                        system_msg = ""
                    prompt_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}<|eot_id|>")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}<|eot_id|>")
            prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>")
            return "".join(prompt_parts)

        elif "qwen" in model_lower:
            # Qwen format
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"<|im_start|>system\n{msg['content']}<|im_end|>")
                elif msg["role"] == "user":
                    prompt_parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"<|im_start|>assistant\n{msg['content']}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            return "\n".join(prompt_parts)

        elif "zephyr" in model_lower:
            # Zephyr format
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"<|system|>\n{msg['content']}</s>")
                elif msg["role"] == "user":
                    prompt_parts.append(f"<|user|>\n{msg['content']}</s>")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"<|assistant|>\n{msg['content']}</s>")
            prompt_parts.append("<|assistant|>")
            return "\n".join(prompt_parts)

        else:
            # Generic format
            prompt_parts = []
            for msg in messages:
                role = msg["role"].upper()
                prompt_parts.append(f"{role}: {msg['content']}")
            prompt_parts.append("ASSISTANT:")
            return "\n".join(prompt_parts)

    def _clean_response(self, content: str, model_id: str) -> str:
        """Clean up model response."""
        # Remove common artifacts
        content = content.strip()

        # Remove repeated prompt echoes
        if content.startswith("ASSISTANT:"):
            content = content[10:].strip()

        # Remove end tokens
        for token in ["</s>", "<|im_end|>", "<|eot_id|>", "[/INST]"]:
            if content.endswith(token):
                content = content[:-len(token)].strip()

        return content

    async def _local_embed(
        self,
        text: Union[str, List[str]],
        model: str = None
    ) -> Dict[str, Any]:
        """Generate embeddings using local sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import asyncio
        except ImportError:
            return {"error": "sentence-transformers not installed", "embeddings": None}

        model_id = model or "all-MiniLM-L6-v2"

        # Cache the model
        if not hasattr(self, '_st_models'):
            self._st_models = {}

        if model_id not in self._st_models:
            logger.info(f"Loading local embedding model: {model_id}")
            self._st_models[model_id] = SentenceTransformer(model_id)

        st_model = self._st_models[model_id]
        texts = [text] if isinstance(text, str) else text

        # Run in executor
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: st_model.encode(texts, normalize_embeddings=True)
        )

        return {
            "embeddings": embeddings[0].tolist() if len(texts) == 1 else [e.tolist() for e in embeddings],
            "model": model_id,
            "dimensions": len(embeddings[0]),
            "provider": "huggingface_local",
            "count": len(texts)
        }

    async def embed(
        self,
        text: Union[str, List[str]],
        model: str = None,
        prefer_local: bool = True
    ) -> Dict[str, Any]:
        """
        Generate embeddings using Hugging Face model - LOCAL FIRST.

        Args:
            text: Text or list of texts to embed
            model: Model ID (default: all-MiniLM-L6-v2)
            prefer_local: Use local sentence-transformers if available

        Returns:
            Dict with 'embeddings', 'model', 'dimensions' keys
        """
        # Try local sentence-transformers first
        if prefer_local:
            try:
                result = await self._local_embed(text, model)
                if result.get("embeddings") is not None:
                    return result
            except Exception as e:
                logger.debug(f"Local embedding failed, trying API: {e}")

        if not self.api_key:
            return {"error": "No HF_API_KEY and local embeddings unavailable", "embeddings": None}

        model_id = model or self.default_embedding_model
        if model_id in self.models:
            model_id = self.models[model_id].id

        texts = [text] if isinstance(text, str) else text

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {"inputs": texts}
                url = f"{self.inference_url}/{model_id}"

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        embeddings = await response.json()

                        # Get dimensions from first embedding
                        dims = len(embeddings[0]) if embeddings else 0

                        return {
                            "embeddings": embeddings if len(texts) > 1 else embeddings[0],
                            "model": model_id,
                            "dimensions": dims,
                            "count": len(texts)
                        }

                    elif response.status == 503:
                        # Model loading
                        await asyncio.sleep(20)
                        return await self.embed(text, model)

                    else:
                        error_text = await response.text()
                        return {"error": error_text, "embeddings": None}

        except Exception as e:
            logger.error(f"HuggingFace embed error: {e}")
            return {"error": str(e), "embeddings": None}

    async def generate_image(
        self,
        prompt: str,
        model: str = None,
        negative_prompt: str = None,
        width: int = 1024,
        height: int = 1024
    ) -> Dict[str, Any]:
        """
        Generate an image using Hugging Face model.

        Args:
            prompt: Image description
            model: Model ID (default: FLUX.1-schnell)
            negative_prompt: What to avoid
            width: Image width
            height: Image height

        Returns:
            Dict with 'image_bytes', 'model' keys
        """
        if not self.api_key:
            return {"error": "No HF_API_KEY configured", "image_bytes": None}

        model_id = model or self.default_image_model
        if model_id in self.models:
            model_id = self.models[model_id].id

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "negative_prompt": negative_prompt or "",
                        "width": width,
                        "height": height
                    }
                }

                url = f"{self.inference_url}/{model_id}"

                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        image_bytes = await response.read()
                        return {
                            "image_bytes": image_bytes,
                            "model": model_id,
                            "format": "png"
                        }

                    elif response.status == 503:
                        # Model loading
                        data = await response.json()
                        wait_time = data.get("estimated_time", 60)
                        logger.info(f"HF image model loading, waiting {wait_time}s...")
                        await asyncio.sleep(min(wait_time, 120))
                        return await self.generate_image(prompt, model, negative_prompt, width, height)

                    else:
                        error_text = await response.text()
                        return {"error": error_text, "image_bytes": None}

        except Exception as e:
            logger.error(f"HuggingFace image error: {e}")
            return {"error": str(e), "image_bytes": None}

    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        model: str = None,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate code using a code-specialized model.

        Args:
            prompt: Code task description
            language: Programming language
            model: Model ID (default: CodeLlama)

        Returns:
            Dict with 'code', 'model' keys
        """
        model_id = model or self.default_code_model

        system = f"You are an expert {language} programmer. Generate clean, efficient, well-documented code."
        full_prompt = f"Write {language} code for: {prompt}\n\nCode:"

        result = await self.chat(
            prompt=full_prompt,
            system=system,
            model=model_id,
            max_tokens=max_tokens,
            temperature=0.3  # Lower temp for code
        )

        if result.get("content"):
            # Extract code block if present
            content = result["content"]
            if "```" in content:
                # Extract from code block
                import re
                code_match = re.search(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
                if code_match:
                    content = code_match.group(1).strip()

            return {
                "code": content,
                "language": language,
                "model": result.get("model")
            }

        return {"error": result.get("error", "Failed to generate code"), "code": None}

    async def swarm_respond(
        self,
        other_bots: List[str],
        last_speaker: str,
        last_content: str,
        chat_history: List[Dict] = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Generate a swarm chat response.

        For multi-bot conversations where HuggingFace adds to discussion.
        """
        system = """You are HuggingFace, an open-source AI assistant in a multi-bot swarm.
You represent the power of open-source AI and community-driven development.
Be helpful, knowledgeable about AI/ML, and collaborative with other bots.
Keep responses concise and add unique value to the conversation."""

        context = f"Other bots in swarm: {', '.join(other_bots)}\n"
        if chat_history:
            context += "Recent conversation:\n"
            for msg in chat_history[-5:]:
                context += f"- {msg.get('bot', 'User')}: {msg.get('content', '')[:100]}...\n"

        prompt = f"{last_speaker} said: {last_content}\n\nProvide a helpful response that adds to this discussion."

        return await self.chat(
            prompt=prompt,
            system=system,
            context=context,
            model=model or "mistral-7b",
            max_tokens=300
        )

    def get_available_models(self, task: str = None) -> List[Dict]:
        """Get list of available models, optionally filtered by task."""
        models = []
        for alias, info in self.models.items():
            if task is None or info.task == task:
                models.append({
                    "alias": alias,
                    "id": info.id,
                    "task": info.task,
                    "description": info.description
                })
        return models

    def get_status(self) -> Dict[str, Any]:
        """Get provider status."""
        return {
            "configured": bool(self.api_key),
            "default_chat_model": self.default_chat_model,
            "default_code_model": self.default_code_model,
            "default_embedding_model": self.default_embedding_model,
            "available_models": len(self.models),
            "use_local": self._use_local
        }


# Global instance
_hf_provider: Optional[HuggingFaceProvider] = None


def get_huggingface_provider() -> Optional[HuggingFaceProvider]:
    """Get or create the global HuggingFace provider."""
    global _hf_provider
    if _hf_provider is None:
        api_key = os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY")
        # Create provider even without API key - local transformers will work
        _hf_provider = HuggingFaceProvider(api_key)
        if api_key:
            logger.info("HuggingFace provider initialized with API key")
        elif _hf_provider._transformers_available:
            logger.info("HuggingFace provider initialized (local transformers only)")
        else:
            logger.warning("HuggingFace provider initialized but no API key or local transformers")
    return _hf_provider


async def hf_chat(prompt: str, system: str = None, model: str = None) -> str:
    """Convenience function for HuggingFace chat."""
    provider = get_huggingface_provider()
    if provider is None:
        return ""

    result = await provider.chat(prompt=prompt, system=system, model=model)
    return result.get("content", "")


async def hf_embed(text: str, model: str = None) -> List[float]:
    """Convenience function for HuggingFace embeddings."""
    provider = get_huggingface_provider()
    if provider is None:
        return []

    result = await provider.embed(text=text, model=model)
    return result.get("embeddings", [])
