# LLM Client

Hermes Agent uses the OpenAI Python SDK with OpenRouter as the backend, providing access to many models through a single API.

## Configuration

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)
```

## Supported Models

Any model available on [OpenRouter](https://openrouter.ai/models):

```python
# Anthropic
model = "anthropic/claude-sonnet-4"
model = "anthropic/claude-opus-4"

# OpenAI
model = "openai/gpt-4o"
model = "openai/o1"

# Google
model = "google/gemini-2.0-flash"

# Open models
model = "meta-llama/llama-3.3-70b-instruct"
model = "deepseek/deepseek-chat-v3"
model = "moonshotai/kimi-k2.5"
```

## Tool Calling

Standard OpenAI function calling format:

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        }
    ],
)

# Check for tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        # Execute tool...
```

## Reasoning Models

Some models return reasoning/thinking content:

```python
# Access reasoning if available
message = response.choices[0].message
if hasattr(message, 'reasoning_content') and message.reasoning_content:
    reasoning = message.reasoning_content
    # Store for trajectory export
```

## Provider Selection

OpenRouter allows selecting specific providers:

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={
        "provider": {
            "order": ["Anthropic", "Google"],  # Preferred providers
            "ignore": ["Novita"],              # Providers to skip
        }
    }
)
```

## Error Handling

Common errors and handling:

```python
try:
    response = client.chat.completions.create(...)
except openai.RateLimitError:
    # Back off and retry
except openai.APIError as e:
    # Check e.code for specific errors
    # 400 = bad request (often provider-specific)
    # 502 = bad gateway (retry with different provider)
```

## Cost Tracking

OpenRouter returns usage info:

```python
usage = response.usage
print(f"Tokens: {usage.prompt_tokens} + {usage.completion_tokens}")
print(f"Cost: ${usage.cost:.6f}")  # If available
```
