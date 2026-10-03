---
name: self-evolving-swarm
description: Kairos Self-Evolving Multi-Agent Swarm with autonomous tool generation and validation.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [swarm, self-evolution, tools, kairos]
    related_skills: [kairos-swarm]
prerequisites:
  pip: [pydantic]
---

# Self-Evolving Multi-Agent Swarm Skill

Autonomous multi-agent orchestration loop capable of researching, validating, and generating reusable tools dynamically at runtime.

## When to Use

- High-level complex goals requiring web research and structured validation
- Autonomous tool generation and tool evolution loops
- Multi-agent collaboration with persistent profile-isolated tool registry

## Configuration (`~/.hermes/config.yaml`)

Per Hermes project rules, non-secret behavioral settings belong in `~/.hermes/config.yaml`:

```yaml
self_evolving_swarm:
  enabled: true
  max_tools_per_run: 2
  validation_mode: strict
```

## Structure

- **Orchestrator (`agents/orchestrator.py`)**: Central swarm controller and self-evolution phase coordinator.
- **WebAgent (`agents/web_agent.py`)**: Research specialist for context gathering.
- **ValidatorAgent (`agents/validator_agent.py`)**: Code quality heuristics and validation scoring.
- **ToolRegistry (`agents/tool_registry.py`)**: Persistent tool store and Hermes runtime tool registration.

## Procedure

1. Install the skill: `hermes skills install optional/autonomous-ai-agents/self-evolving-swarm`
2. Run self-evolving goal via Python API or Hermes CLI:
   ```python
   from agents.orchestrator import run_swarm

   result = run_swarm("Build a weather lookup tool with caching")
   print(result.output)
   ```
