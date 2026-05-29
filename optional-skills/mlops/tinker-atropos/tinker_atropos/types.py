from typing import List, Dict, Any
from pydantic import BaseModel


# Request format for /v1/completions endpoint.
class CompletionRequest(BaseModel):
    prompt: str | List[str]
    max_tokens: int = 100
    temperature: float = 1.0
    stop: List[str] | None = None
    n: int = 1


# Response format for /v1/completions endpoint.
class CompletionResponse(BaseModel):
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


# Request format for /v1/chat/completions endpoint.
class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 1.0
    stop: List[str] | None = None
    n: int = 1


# Response format for /v1/chat/completions endpoint.
class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str


# Request format for /generate endpoint (SGLang compatible).
class GenerateRequest(BaseModel):
    text: str | List[str] | None = None  # Input text or list of texts
    input_ids: List[int] | List[List[int]] | None = None  # Input token IDs (alternative to text)
    sampling_params: Dict[
        str, Any
    ] | None = None  # Sampling parameters: temperature, top_p, max_new_tokens, n, etc.
    return_logprob: bool = True  # Whether to return log probabilities
    top_logprobs_num: int = 0  # Number of top logprobs to return per token
    return_text_in_logprobs: bool = False  # Whether to return text in logprobs
    logprob_start_len: int | None = None  # Start position for logprobs (for prompt)
    stream: bool = False  # Whether to stream responses


# Response format for /generate endpoint (SGLang compatible).
# For single completion (n=1): returns one GenerateResponse
# For multiple completions (n>1): returns List[GenerateResponse]
class GenerateResponse(BaseModel):
    text: str | List[str]  # Generated text(s)
    meta_info: Dict[
        str, Any
    ]  # Contains: output_token_logprobs, finish_reason, prompt_tokens, completion_tokens
