---
name: langfuse-prompt-migration
description: Migrate hardcoded prompts to Langfuse for version control and deployment-free iteration. Use when user wants to externalize prompts, move prompts to Langfuse, or set up prompt management.
---

# Langfuse Prompt Migration

Migrate hardcoded prompts to Langfuse for version control, A/B testing, and deployment-free iteration.

## Prerequisites

Verify credentials before starting:

```bash
echo $LANGFUSE_PUBLIC_KEY   # pk-...
echo $LANGFUSE_SECRET_KEY   # sk-...
echo $LANGFUSE_HOST         # https://cloud.langfuse.com or self-hosted
```

If not set, ask user to configure them first.

## Migration Flow

```
1. Scan codebase for prompts
2. Analyze templating compatibility
3. Propose structure (names, subprompts, variables)
4. User approves
5. Create prompts in Langfuse
6. Refactor code to use get_prompt()
7. Link prompts to traces (if tracing enabled)
8. Verify application works
```

## Step 1: Find Prompts

Search for these patterns:

| Framework | Look for |
|-----------|----------|
| OpenAI | `messages=[{"role": "system", "content": "..."}]` |
| Anthropic | `system="..."` |
| LangChain | `ChatPromptTemplate`, `SystemMessage` |
| Vercel AI | `system: "..."`, `prompt: "..."` |
| Raw | Multi-line strings near LLM calls |

## Step 2: Check Templating Compatibility

**CRITICAL:** Langfuse only supports simple `{{variable}}` substitution. No conditionals, loops, or filters.

| Template Feature | Langfuse Native | Action |
|------------------|-----------------|--------|
| `{{variable}}` | ✅ | Direct migration |
| `{var}` / `${var}` | ⚠️ | Convert to `{{var}}` |
| `{% if %}` / `{% for %}` | ❌ | Move logic to code |
| `{{ var \| filter }}` | ❌ | Apply filter in code |

### Decision Tree

```
Contains {% if %}, {% for %}, or filters?
├─ No → Direct migration
└─ Yes → Choose:
    ├─ Option A (RECOMMENDED): Move logic to code, pass pre-computed values
    └─ Option B: Store raw template, compile client-side with Jinja2
        └─ ⚠️ Loses: Playground preview, UI experiments
```

### Simplifying Complex Templates

**Conditionals** → Pre-compute in code:
```python
# Instead of {% if user.is_premium %}...{% endif %} in prompt
# Use {{tier_message}} and compute value in code before compile()
```

**Loops** → Pre-format in code:
```python
# Instead of {% for tool in tools %}...{% endfor %} in prompt
# Use {{tools_list}} and format the list in code before compile()
```

For external templating details, fetch: https://langfuse.com/faq/all/using-external-templating-libraries

## Step 3: Propose Structure

### Naming Conventions

| Rule | Example | Bad |
|------|---------|-----|
| Lowercase, hyphenated | `chat-assistant` | `ChatAssistant_v2` |
| Feature-based | `document-summarizer` | `prompt1` |
| Hierarchical for related | `support/triage` | `supportTriage` |
| Prefix subprompts with `_` | `_base-personality` | `shared-personality` |

### Identify Subprompts

Extract when:
- Same text in 2+ prompts
- Represents distinct component (personality, safety rules, format)
- Would need to change together

### Variable Extraction

| Make Variable | Keep Hardcoded |
|---------------|----------------|
| User-specific (`{{user_name}}`) | Output format instructions |
| Dynamic content (`{{context}}`) | Safety guardrails |
| Per-request (`{{query}}`) | Persona/personality |
| Environment-specific (`{{company_name}}`) | Static examples |

## Step 4: Present Plan to User

Format:
```
Found N prompts across M files:

src/chat.py:
  - System prompt (47 lines) → 'chat-assistant'

src/support/triage.py:
  - Triage prompt (34 lines) → 'support/triage'
    ⚠️ Contains {% if %} - will simplify

Subprompts to extract:
  - '_base-personality' - used by: chat-assistant, support/triage

Variables to add:
  - {{user_name}} - hardcoded in 2 prompts

Proceed?
```

## Step 5: Create Prompts in Langfuse

Use `langfuse.create_prompt()` with:
- `name`: Your chosen name
- `prompt`: Template text (or message array for chat type)
- `type`: `"text"` or `"chat"`
- `labels`: `["production"]` (they're already live)
- `config`: Optional model settings

**Labeling strategy:**
- `production` → All migrated prompts
- `staging` → Add later for testing
- `latest` → Auto-applied by Langfuse

For full API: fetch https://langfuse.com/docs/prompts/get-started

## Step 6: Refactor Code

Replace hardcoded prompts with:

```python
prompt = langfuse.get_prompt("name", label="production")
messages = prompt.compile(var1=value1, var2=value2)
```

**Key points:**
- Always use `label="production"` (not `latest`) for stability
- Call `.compile()` to substitute variables
- For chat prompts, result is message array ready for API

For SDK examples (Python/JS/TS): fetch https://langfuse.com/docs/prompts/get-started

## Step 7: Link Prompts to Traces

If codebase uses Langfuse tracing, link prompts so you can see which version produced each response.

### Detect Existing Tracing

Look for:
- `@observe()` decorators
- `langfuse.trace()` calls
- `from langfuse.openai import openai` (instrumented client)

### Link Methods

| Setup | How to Link |
|-------|-------------|
| `@observe()` decorator | `langfuse_context.update_current_observation(prompt=prompt)` |
| Manual tracing | `trace.generation(prompt=prompt, ...)` |
| OpenAI integration | `openai.chat.completions.create(..., langfuse_prompt=prompt)` |

### Verify in UI

1. Go to **Traces** → select a trace
2. Click on **Generation**
3. Check **Prompt** field shows name and version

For tracing details: fetch https://langfuse.com/docs/prompts/get-started#link-with-langfuse-tracing

## Step 8: Verify Migration

### Checklist

- [ ] All prompts created with `production` label
- [ ] Code fetches with `label="production"`
- [ ] Variables compile without errors
- [ ] Subprompts resolve correctly
- [ ] Application behavior unchanged
- [ ] Generations show linked prompt in UI (if tracing)

### Common Issues

| Issue | Solution |
|-------|----------|
| `PromptNotFoundError` | Check name spelling |
| Variables not replaced | Use `{{var}}` not `{var}`, call `.compile()` |
| Subprompt not resolved | Must exist with same label |
| Old prompt cached | Restart app |

## Out of Scope

- Prompt engineering (writing better prompts)
- Evaluation setup
- A/B testing workflow
- Non-LLM string templates
