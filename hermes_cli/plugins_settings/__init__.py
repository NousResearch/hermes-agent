"""Plugin-declared settings (#46600, #87934): a manifest ``config_schema`` turned into fields and edits.

Modules:

- ``fields`` — the schema as renderable fields (type, current value, choices) and the validated
  writer; the Desktop/TUI Plugins hub and the CLI editor both read and write through it.
- ``cli`` — ``hermes plugins config <name>``: the interactive terminal editor over those fields.
"""

from hermes_cli.plugins_settings.fields import plugin_settings_fields, save_plugin_settings, secret_env_name

__all__ = ["plugin_settings_fields", "save_plugin_settings", "secret_env_name"]
