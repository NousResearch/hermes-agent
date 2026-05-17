# Hermes Agent Home Assistant Add-on

Hermes Agent is a self-improving AI agent by Nous Research that can control your smart home, execute commands, and manage your digital life.

## Features

- **Smart Home Control**: List entities, get states, and call services directly from the agent.
- **Hermes Dashboard**: A built-in web interface for chatting with the agent and monitoring its activity.
- **OpenAI Compatible API**: Use any OpenAI-compatible frontend to connect to your agent.
- **Extensible Skills**: The agent learns and improves its capabilities over time.

## Installation

1. Add this repository to your Home Assistant Add-on Store.
2. Install the **Hermes Agent** add-on.
3. Configure your API keys (OpenRouter, OpenAI, or Anthropic) in the add-on configuration.
4. Start the add-on.

## Configuration

### API Keys

- `openrouter_api_key`: (Recommended) Your OpenRouter API key.
- `openai_api_key`: Your OpenAI API key.
- `anthropic_api_key`: Your Anthropic API key.

### Dashboard

The Hermes Dashboard is enabled by default and can be accessed via the sidebar in Home Assistant (using Ingress).

## Smart Home Integration

To allow the agent to control your Home Assistant devices:

1. Ensure `homeassistant_api: true` is set in the add-on configuration (default).
2. The agent will automatically connect to the internal Home Assistant API.
3. You can ask the agent to "List my lights" or "Turn off the kitchen switch".

### Available Tools

- `ha_list_entities`: List and filter Home Assistant entities.
- `ha_get_state`: Get detailed state of an entity.
- `ha_list_services`: Discover available actions for your devices.
- `ha_call_service`: Control devices by calling services.

## Support

For more information, visit the [Hermes Agent Documentation](https://github.com/NousResearch/hermes-agent).
