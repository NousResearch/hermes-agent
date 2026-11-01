"""Resolve application retry budgets from the shared configuration defaults."""
from hermes_cli.config_defaults import DEFAULT_CONFIG


def apply_retry_settings(agent, settings):
    """Keep explicit settings authoritative and malformed values on the schema default."""
    for key, attribute, floor in (
        ("api_max_retries", "_api_max_retries", 1),
        ("auto_recovery_cycles", "_auto_recovery_cycles", 0),
    ):
        default = DEFAULT_CONFIG["agent"][key]
        try:
            value = max(int(settings.get(key, default)), floor)
        except (TypeError, ValueError):
            value = default
        setattr(agent, attribute, value)
