---
name: outlines
description: "Outlines: structured JSON/regex/Pydantic LLM generation."
version: 1.0.1
author: Orchestra Research
license: MIT
dependencies: [outlines, transformers, vllm, pydantic]
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Prompt Engineering, Outlines, Structured Generation, JSON Schema, Pydantic, Local Models, Grammar-Based Generation, vLLM, Transformers, Type Safety]

---

# Outlines: Structured Text Generation

## When to Use This Skill

Use Outlines when you need to:
- **Guarantee valid JSON/XML/code** structure during generation
- **Use Pydantic models** for type-safe outputs
- **Support local models** (Transformers, llama.cpp, vLLM)
- **Maximize inference speed** with zero-overhead structured generation
- **Generate against JSON schemas** automatically
- **Control token sampling** at the grammar level

**GitHub Stars**: 12,000+ | **From**: dottxt.ai (formerly .txt)

> **API note (Outlines 1.x):** This skill targets the current v1 API.
> The pre-1.0 helpers (`outlines.models.transformers(...)`,
> `outlines.generate.json/choice/regex/...`) have been **removed**. In v1 you
> create a model with `outlines.from_transformers(...)` (or `from_vllm`,
> `from_llamacpp`, `from_openai`) and then **call the model directly** with an
> output type: `model(prompt, output_type)`. JSON/Pydantic outputs are returned
> as a **JSON string** — validate with `YourModel.model_validate_json(result)`.

## Installation

```bash
# Base installation
pip install outlines

# With specific backends
pip install outlines transformers  # Hugging Face models
pip install outlines llama-cpp-python  # llama.cpp
pip install outlines vllm  # vLLM for high-throughput
```

## Quick Start

### Basic Example: Classification

```python
import outlines
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# v1: wrap a Transformers model + tokenizer
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# Call the model directly with an output type
prompt = "Sentiment of 'This product is amazing!': "
sentiment = model(prompt, Literal["positive", "negative", "neutral"])

print(sentiment)  # "positive" (guaranteed one of these)
```

### With Pydantic Models

```python
from pydantic import BaseModel
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

class User(BaseModel):
    name: str
    age: int
    email: str

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# Generate structured output (returns a JSON string)
prompt = "Extract user: John Doe, 30 years old, john@example.com"
result = model(prompt, User, max_new_tokens=200)

user = User.model_validate_json(result)  # parse into the Pydantic model
print(user.name)   # "John Doe"
print(user.age)    # 30
print(user.email)  # "john@example.com"
```

## Core Concepts

### 1. Constrained Token Sampling

Outlines constrains token generation at the logit level using a compiled
automaton derived from your output type.

**How it works:**
1. Convert the output type (JSON/Pydantic/regex/`Literal`) to a schema/grammar
2. Compile the grammar into a token-level automaton
3. Filter invalid tokens at each step during generation
4. Fast-forward when only one valid token exists

**Benefits:**
- **Zero overhead**: Filtering happens at token level
- **Speed improvement**: Fast-forward through deterministic paths
- **Guaranteed validity**: Invalid outputs impossible

```python
import outlines
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class Person(BaseModel):
    name: str
    age: int

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

result = model("Generate person: Alice, 25", Person)
person = Person.model_validate_json(result)
```

### 2. Output Types

In v1 you pass the desired **output type** directly as the second argument.

#### Multiple choice (`Literal`)

```python
from typing import Literal

sentiment = model("Review: This is great!", Literal["positive", "negative", "neutral"])
# Result: one of the three choices
```

#### JSON via Pydantic

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

result = model("Extract: iPhone 15, $999, available", Product)
product = Product.model_validate_json(result)  # valid Product instance
```

#### Regex (pass a regex string)

```python
# Generate text matching a regex pattern
phone = model("Generate phone number:", r"[0-9]{3}-[0-9]{3}-[0-9]{4}")
# Result: "555-123-4567" (guaranteed to match the pattern)
```

#### Numeric types

```python
# Pass the Python type directly
age = model("Person's age:", int)      # guaranteed integer
price = model("Product price:", float)  # guaranteed float
```

### 3. Model Backends

Outlines supports multiple local and API-based backends via `from_*` factories.

#### Transformers (Hugging Face)

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

result = model(prompt, YourModel)
```

#### llama.cpp

```python
import outlines
from llama_cpp import Llama

llm = Llama("./models/llama-3.1-8b-instruct.Q4_K_M.gguf", n_gpu_layers=35, n_ctx=4096)
model = outlines.from_llamacpp(llm)

result = model(prompt, YourModel)
```

#### vLLM (High Throughput)

```python
import outlines
from vllm import LLM

llm = LLM("meta-llama/Llama-3.1-8B-Instruct", tensor_parallel_size=2)
model = outlines.from_vllm(llm)

result = model(prompt, YourModel)
```

#### OpenAI (server-side constrained JSON)

```python
import outlines
from openai import OpenAI

client = OpenAI()
model = outlines.from_openai(client, "gpt-4o-mini")

# API backends support JSON-schema style structured output
result = model(prompt, YourModel)
```

### 4. Pydantic Integration

Outlines has first-class Pydantic support with automatic schema translation.
Generation returns a JSON string; call `model_validate_json` to get an instance.

#### Basic Models

```python
from pydantic import BaseModel, Field

class Article(BaseModel):
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    word_count: int = Field(description="Number of words", gt=0)
    tags: list[str] = Field(description="List of tags")

result = model("Generate article about AI", Article, max_new_tokens=300)
article = Article.model_validate_json(result)
print(article.title)
print(article.word_count)  # Guaranteed > 0
```

#### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    address: Address  # Nested model

result = model("Generate person in New York", Person)
person = Person.model_validate_json(result)
print(person.address.city)  # "New York"
```

#### Enums and Literals

```python
from enum import Enum
from typing import Literal

class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class Application(BaseModel):
    applicant: str
    status: Status  # Must be one of enum values
    priority: Literal["low", "medium", "high"]  # Must be one of literals

result = model("Generate application", Application)
app = Application.model_validate_json(result)
print(app.status)  # Status.PENDING (or APPROVED/REJECTED)
```

## Common Patterns

### Pattern 1: Data Extraction

```python
from pydantic import BaseModel
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

class CompanyInfo(BaseModel):
    name: str
    founded_year: int
    industry: str
    employees: int

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

text = """
Apple Inc. was founded in 1976 in the technology industry.
The company employs approximately 164,000 people worldwide.
"""

prompt = f"Extract company information:\n{text}\n\nCompany:"
company = CompanyInfo.model_validate_json(model(prompt, CompanyInfo, max_new_tokens=200))

print(f"Name: {company.name}")
print(f"Founded: {company.founded_year}")
print(f"Industry: {company.industry}")
print(f"Employees: {company.employees}")
```

### Pattern 2: Classification

```python
from typing import Literal
from pydantic import BaseModel

# Binary classification
result = model("Email: Buy now! 50% off!", Literal["spam", "not_spam"])

# Multi-class classification
category = model(
    "Article: Apple announces new iPhone...",
    Literal["technology", "business", "sports", "entertainment"],
)

# With confidence
class Classification(BaseModel):
    label: Literal["positive", "negative", "neutral"]
    confidence: float

out = model("Review: This product is okay, nothing special", Classification)
result = Classification.model_validate_json(out)
```

### Pattern 3: Structured Forms

```python
class UserProfile(BaseModel):
    full_name: str
    age: int
    email: str
    phone: str
    country: str
    interests: list[str]

prompt = """
Extract user profile from:
Name: Alice Johnson
Age: 28
Email: alice@example.com
Phone: 555-0123
Country: USA
Interests: hiking, photography, cooking
"""

profile = UserProfile.model_validate_json(model(prompt, UserProfile, max_new_tokens=250))
print(profile.full_name)
print(profile.interests)  # ["hiking", "photography", "cooking"]
```

### Pattern 4: Multi-Entity Extraction

```python
from typing import Literal

class Entity(BaseModel):
    name: str
    type: Literal["PERSON", "ORGANIZATION", "LOCATION"]

class DocumentEntities(BaseModel):
    entities: list[Entity]

text = "Tim Cook met with Satya Nadella at Microsoft headquarters in Redmond."
prompt = f"Extract entities from: {text}"

result = DocumentEntities.model_validate_json(model(prompt, DocumentEntities, max_new_tokens=300))
for entity in result.entities:
    print(f"{entity.name} ({entity.type})")
```

### Pattern 5: Code Generation

```python
class PythonFunction(BaseModel):
    function_name: str
    parameters: list[str]
    docstring: str
    body: str

prompt = "Generate a Python function to calculate factorial"
func = PythonFunction.model_validate_json(model(prompt, PythonFunction, max_new_tokens=300))

print(f"def {func.function_name}({', '.join(func.parameters)}):")
print(f'    """{func.docstring}"""')
print(f"    {func.body}")
```

### Pattern 6: Batch Processing

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map="auto"),
    AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct"),
)

texts = [
    "John is 30 years old",
    "Alice is 25 years old",
    "Bob is 40 years old",
]

# v1 accepts a list of prompts for batched generation
prompts = [f"Extract from: {t}" for t in texts]
outputs = model(prompts, Person, max_new_tokens=100)
people = [Person.model_validate_json(o) for o in outputs]
for person in people:
    print(f"{person.name}: {person.age}")
```

## Backend Configuration

### Transformers

```python
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Basic usage
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# GPU + dtype configuration is set on the HF model itself
import torch
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda", torch_dtype=torch.float16),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)

# Popular models
for name in [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]:
    model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(name, device_map="auto"),
        AutoTokenizer.from_pretrained(name),
    )
```

### llama.cpp

```python
import outlines
from llama_cpp import Llama

# Load GGUF model
llm = Llama(
    "./models/llama-3.1-8b.Q4_K_M.gguf",
    n_ctx=4096,       # Context window
    n_gpu_layers=35,  # GPU layers
    n_threads=8,      # CPU threads
)
model = outlines.from_llamacpp(llm)

# Full GPU offload: set n_gpu_layers=-1 on the Llama object
```

### vLLM (Production)

```python
import outlines
from vllm import LLM

# Single GPU
model = outlines.from_vllm(LLM("meta-llama/Llama-3.1-8B-Instruct"))

# Multi-GPU
model = outlines.from_vllm(LLM("meta-llama/Llama-3.1-70B-Instruct", tensor_parallel_size=4))

# With quantization
model = outlines.from_vllm(LLM("meta-llama/Llama-3.1-8B-Instruct", quantization="awq"))
```

## Best Practices

### 1. Use Specific Types

```python
# ✅ Good: Specific types
class Product(BaseModel):
    name: str
    price: float  # Not str
    quantity: int  # Not str
    in_stock: bool  # Not str

# ❌ Bad: Everything as string
class Product(BaseModel):
    name: str
    price: str  # Should be float
    quantity: str  # Should be int
```

### 2. Add Constraints

```python
from pydantic import Field

# ✅ Good: With constraints
class User(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=120)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

# ❌ Bad: No constraints
class User(BaseModel):
    name: str
    age: int
    email: str
```

### 3. Use Enums for Categories

```python
# ✅ Good: Enum for fixed set
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Task(BaseModel):
    title: str
    priority: Priority

# ❌ Bad: Free-form string
class Task(BaseModel):
    title: str
    priority: str  # Can be anything
```

### 4. Provide Context in Prompts

```python
# ✅ Good: Clear context
prompt = """
Extract product information from the following text.
Text: iPhone 15 Pro costs $999 and is currently in stock.
Product:
"""

# ❌ Bad: Minimal context
prompt = "iPhone 15 Pro costs $999 and is currently in stock."
```

### 5. Handle Optional Fields

```python
from typing import Optional

# ✅ Good: Optional fields for incomplete data
class Article(BaseModel):
    title: str  # Required
    author: Optional[str] = None  # Optional
    date: Optional[str] = None  # Optional
    tags: list[str] = []  # Default empty list

# Can succeed even if author/date missing
```

### 6. Always Validate JSON Output

```python
# v1 returns a JSON string for Pydantic/JSON output types.
result = model(prompt, Article)          # str
article = Article.model_validate_json(result)  # Article instance
```

## Comparison to Alternatives

| Feature | Outlines | Instructor | Guidance | LMQL |
|---------|----------|------------|----------|------|
| Pydantic Support | ✅ Native | ✅ Native | ✅ Yes | ❌ No |
| JSON Schema | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Regex Constraints | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| Local Models | ✅ Full | ⚠️ Limited | ✅ Full | ✅ Full |
| API Models | ✅ Yes | ✅ Full | ✅ Yes | ✅ Full |
| Zero Overhead | ✅ Yes | ❌ No | ⚠️ Partial | ✅ Yes |
| Automatic Retrying | ❌ No | ✅ Yes | ❌ No | ❌ No |
| Learning Curve | Low | Low | Low | High |

**When to choose Outlines:**
- Using local models (Transformers, llama.cpp, vLLM)
- Need maximum inference speed
- Want Pydantic model support
- Require zero-overhead structured generation
- Control token sampling process

**When to choose alternatives:**
- Instructor: Need API models with automatic retrying
- Guidance: Need token healing and complex workflows
- LMQL: Prefer declarative query syntax

## Performance Characteristics

**Speed:**
- **Zero overhead**: Structured generation as fast as unconstrained
- **Fast-forward optimization**: Skips deterministic tokens
- **1.2-2x faster** than post-generation validation approaches

**Memory:**
- Automaton compiled once per output type (cached)
- Minimal runtime overhead
- Efficient with vLLM for high throughput

**Accuracy:**
- **100% valid outputs** (guaranteed by the constrained automaton)
- No retry loops needed
- Deterministic token filtering

## Resources

- **Documentation**: https://dottxt-ai.github.io/outlines/
- **GitHub**: https://github.com/dottxt-ai/outlines (12k+ stars)
- **Discord**: https://discord.gg/R9DSu34mGd
- **Blog**: https://blog.dottxt.co

## See Also

- `references/json_generation.md` - Comprehensive JSON and Pydantic patterns
- `references/backends.md` - Backend-specific configuration
- `references/examples.md` - Production-ready examples
