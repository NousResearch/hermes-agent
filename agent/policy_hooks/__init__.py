r"""Reference implementations for tool-call policy hooks.

Modules here are optional examples for use with the
\`ToolCallGuardrailController.register_policy_hook()\` extension API.

To register an example at runtime, point your config at it with a
"module:callable" import path:

    tool_loop_guardrails:
      policy_hooks:
        - "agent.policy_hooks.example_policy_guardrail:risk_policy_guardrail"
"""