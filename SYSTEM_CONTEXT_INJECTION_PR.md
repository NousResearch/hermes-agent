# PR: Wrap Injected System Context in Clear XML Tags

Motivation: Currently, system-injected context (memory, user profile, skills, platform,
tool results) is concatenated into the system prompt as plain text. The model
cannot distinguish between:
- User messages
- Injected system context
- Tool outputs

This leads to the model sometimes treating injected context as if it were user
messages, causing confusion and hallucination.

Proposal: Wrap all injected system content in clear XML tags:
- `<system-injection type="memory">`: MEMORY.md content
- `<system-injection type="user-profile">`: USER.md content
- `<system-injection type="skills">`: Available skills list
- `<system-injection type="platform">`: Platform-specific formatting hints
- `<system-injection type="meta">`: Conversation start time, model, provider
- `<system-injection type="context-files">`: AGENTS.md, .cursorrules, etc.
- `<system-injection type="environment">`: WSL, Termux hints
- `<system-injection type="user-provided-context">`: User-provided system context
- `<system-injection type="external-memory">`: External memory blocks
- `<system-injection type="model-identity">`: Model name (alibaba workaround)
- `<system-injection type="tool-result">`: Tool outputs
- `<system-injection type="tool-call">`: Tool calls

This approach is:
- Backwards-compatible (text models can parse XML tags)
- Self-documenting (each block is self-labeling)
- Minimal token overhead (~10-15 tokens per block)

Files changed:
- `run_agent.py`: Wrap all injected system context in XML tags
- `model_tools.py`: Wrap tool call/ tool result outputs

Let's do this for real.
