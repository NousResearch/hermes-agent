"""Dotenv file maintenance; mounted secret streams are read-only."""

from hermes_cli.env_file import is_fifo


def sanitize_env_file() -> int:
    """Normalize a regular profile dotenv without materializing a secret mount."""
    from hermes_cli import config

    env_path = config.get_env_path()
    if not env_path.exists() or is_fifo(env_path):
        return 0
    with config._env_write_lock(env_path):
        if is_fifo(env_path):
            return 0
        with open(env_path, encoding="utf-8-sig", errors="replace") as f:
            original_lines = f.readlines()
        sanitized = config._sanitize_env_lines(original_lines)
        if sanitized == original_lines:
            return 0
        fixes = abs(len(sanitized) - len(original_lines)) or sum(
            1 for a, b in zip(original_lines, sanitized) if a != b)
        config._write_env_lines(env_path, sanitized, preserve_mode=False)
    config.invalidate_env_cache()
    return fixes
