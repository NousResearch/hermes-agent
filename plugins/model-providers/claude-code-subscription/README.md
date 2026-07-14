# Claude Code Subscription Provider

Runs Claude through the locally installed `claude` CLI, using its logged-in Pro or Max subscription. Hermes remains the agent runtime and owns tool execution.

```yaml
model:
  provider: claude-code-subscription
  model: sonnet

claude_code_subscription:
  command: claude
  config_dir: ~/.claude
```

Configure `fallback_providers` so a Claude subscription limit automatically switches to another provider. The adapter reports subscription exhaustion as a rate limit.
