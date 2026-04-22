# Hermes Agent (Fork)

This is a **fork** of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).

## What is Hermes Agent?

Hermes Agent is an open-source AI agent framework built for autonomous task execution. It can:

- Execute code, browse the web, manage files, and interact with web services
- Use multiple AI models (compression + main) for efficient context management
- Run in CLI, web dashboard, or headless mode
- Support for GitHub, GitHub Actions, and various developer tools

## What's different in this fork?

This fork contains custom modifications focused on **improving CI reliability and tool-loop prevention**:

### Circuit Breaker Improvements
- **Lowered failure threshold**: From 5 to 3 consecutive failures before triggering circuit breaker
- **Compression model suggestions**: When a tool fails repeatedly, the compression model (if configured) provides a "fresh perspective" to break the loop
- **Generic fallback**: Even without a compression model, a simple hint is provided to stop retrying the same approach

### Test Fixes
- Fixed `test_minimax_provider.py` — missing `_fallback_chain` attribute in test stub
- Fixed `test_tips.py` — truncated Tip 105 to meet 150-character limit
- Fixed `test_concurrent_interrupt.py` — resolved `polling_tool` never running and signature mismatch

## Configuration

This fork inherits all configuration from the original repository. See the [original README](https://github.com/NousResearch/hermes-agent) for setup instructions.

## License

This fork is licensed under the same license as the original repository. See the [original LICENSE](https://github.com/NousResearch/hermes-agent/blob/main/LICENSE) for details.
