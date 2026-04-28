### Configure Memory with hermes honcho

Source: https://context7.com/nousresearch/hermes-agent/llms.txt

Set up Honcho AI integration for dialectic user modeling and cross-session memory. Supports hybrid, honcho, and local memory modes.

```bash
# Setup Honcho integration
hermes honcho setup

# Check connection status
hermes honcho status

# Map current directory to session name
hermes honcho map my-project

# Configure memory mode
hermes honcho mode hybrid  # hybrid|honcho|local
```

--------------------------------

### Manage Honcho memory integration with hermes honcho

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Use `hermes honcho` to manage cross-session memory when `memory.provider` is set to `honcho`. The `--target-profile` flag allows managing another profile's config without switching.

```bash
hermes honcho [--target-profile NAME] <subcommand>
```

--------------------------------

### CLI Command: hermes honcho

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

Manage Honcho cross-session memory integration.

```APIDOC
## CLI Command: hermes honcho

### Description
Manage Honcho cross-session memory integration.

### Method
CLI

### Command Syntax
`hermes honcho`

### Example Usage
```bash
hermes honcho status
```
```

--------------------------------

### Setup Honcho Memory Provider

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/user-guide/features/honcho.md

Initialize Honcho as the memory provider using the interactive setup command or manual YAML configuration.

```bash
hermes memory setup    # select "honcho" from the provider list
```

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo "HONCHO_API_KEY=*** >> ~/.hermes/.env
```

### CLI Commands Reference > Top-level commands > Memory Management

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/reference/cli-commands.md

The `hermes honcho` command manages Honcho cross-session memory integration, and `hermes memory` configures external memory providers for persistent context across sessions.