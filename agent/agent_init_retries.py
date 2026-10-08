"""Configuration for independently bounded API retry policies."""


def init_retry_settings(agent, section):
    # Generic attempts (not extra retries): 1 means a single attempt.
    try:
        agent._api_max_retries = max(int(section.get("api_max_retries", 3)), 1)
    except (TypeError, ValueError):
        agent._api_max_retries = 3
    # Copilot 403 uses extra attempts. Malformed values must not opt in.
    copilot_retries = section.get("copilot_403_max_retries", 0)
    agent._copilot_403_max_retries = max(copilot_retries, 0) if type(copilot_retries) is int else 0
    # Post-exhaustion outage recovery; never extends the Copilot 403 budget.
    try:
        agent._auto_recovery_cycles = max(int(section.get("auto_recovery_cycles", 5)), 0)
    except (TypeError, ValueError):
        agent._auto_recovery_cycles = 5
