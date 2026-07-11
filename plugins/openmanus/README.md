# OpenManus Hermes plugin

This plugin pins [FoundationAgents/OpenManus](https://github.com/FoundationAgents/OpenManus) at the repository submodule `vendor/openmanus` and exposes it to Hermes agents through the opt-in `openmanus` toolset. The Hermes process never imports OpenManus directly. A live task runs in a child process using a copied, per-run checkout and a temporary OpenManus configuration.

## Hermes and MoA usage

Enable the plugin with `hermes plugins enable openmanus`, configure an authorised workspace, and add the `openmanus` toolset to the active agent or MoA tool allow-list when one is configured. Hermes agents can then call `openmanus_run` for one bounded delegated task. `openmanus_wide_research` starts independent workers with a configured parallelism cap and can ask the active Hermes host LLM to synthesise their receipts.

The default is a dry-run. A live call must set both `allow_side_effects` and `acknowledge_side_effects`. Each live worker is confined to the configured workspace root, rejects symlink escapes, uses a bounded step and timeout budget, and stores a redacted receipt under the active Hermes home. Network and browser tools are disabled unless both the request and plugin configuration enable them. MCP servers are not attached automatically.

Example `config.yaml` entry:

```yaml
plugins:
  enabled:
    - openmanus
  entries:
    openmanus:
      workspace_root: "C:/Users/downl/Documents/openmanus-workspace"
      max_parallel: 4
      allow_network: false
      llm:
        model: "your-model"
        base_url: "https://your-openai-compatible-endpoint/v1"
        api_type: "openai"
        api_key_env: "OPENMANUS_API_KEY"
```

Keep the key in the profile secret file, not in YAML:

```text
OPENMANUS_API_KEY=replace-me
```

Plan a task first:

```text
hermes openmanus run --prompt "Inspect the project and report the failing tests" --workspace .
```

Execute only after reviewing the plan:

```text
hermes openmanus run --execute --acknowledge-side-effects --prompt "Apply the approved local fix" --workspace .
```

## Research basis and scope

The pinned OpenManus snapshot contains the general Manus agent, a DataAnalysis agent, browser tooling, MCP support, sandbox support, and an A2A example. This plugin uses the existing agent modes and adds the Hermes boundary around them; it does not silently claim to implement proprietary Manus or Genspark services.

The official Manus documentation describes Wide Research as independent parallel workers, Browser Operator as an explicitly authorised local-browser session, Cloud Browser as an isolated browser with human take-over for verification steps, and task continuity through persistent agent work. Genspark documentation describes Claw as a persistent agent with memory and schedules, and Custom Agent as an invocation and collaboration surface. Hermes already owns memory, cron, profiles, approvals, and MoA routing, so this integration maps those concerns to Hermes rather than duplicating them inside OpenManus.

Relevant primary sources: [Manus Wide Research](https://manus.im/docs/features/wide-research), [Manus Browser Operator](https://manus.im/docs/features/browser-operator), [Manus Cloud Browser](https://manus.im/docs/features/cloud-browser), [Genspark Claw](https://www.genspark.ai/helpcenter/genspark-claw), and [Genspark Custom Agent](https://www.genspark.ai/helpcenter/custom-super-agent).

The name ManusDenspark does not identify an official product in the reviewed primary documentation. The implementation therefore targets the verifiable overlap between OpenManus, Manus, and Genspark: delegated execution, bounded parallel research, persistent Hermes state, explicit human control, and auditable results.
