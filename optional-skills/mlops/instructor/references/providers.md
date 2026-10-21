# Provider Configuration

Guide to using Instructor with different LLM providers.

## Anthropic Claude

```python
import instructor
from anthropic import Anthropic

# Basic setup
client = instructor.from_anthropic(Anthropic())

# With API key
client = instructor.from_anthropic(
    Anthropic(api_key="your-api-key")
)

# Recommended mode
client = instructor.from_anthropic(
    Anthropic(),
    mode=instructor.Mode.ANTHROPIC_TOOLS
)

# Usage
result = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}],
    response_model=YourModel
)
```

## OpenAI

```python
from openai import OpenAI

client = instructor.from_openai(OpenAI())

result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=YourModel,
    messages=[{"role": "user", "content": "..."}]
)
```

## Local Models (Ollama)

```python
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ),
    mode=instructor.Mode.JSON
)

result = client.chat.completions.create(
    model="llama3.1",
    response_model=YourModel,
    messages=[...]
)
```

## Modes

- `Mode.ANTHROPIC_TOOLS`: Recommended for Claude
- `Mode.TOOLS`: OpenAI function calling
- `Mode.JSON`: Fallback for unsupported providers
