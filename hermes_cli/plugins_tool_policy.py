"""Read dispatch whitelist policy without invoking plugin hooks.

Policy storage and thread-lifetime semantics remain owned by plugins.py.
"""

from typing import Optional


def tool_whitelist_denial(tool_name: str) -> Optional[str]:
    """Return the active whitelist denial, or None for allowed/unrestricted calls."""
    from hermes_cli import plugins

    allowed = getattr(plugins._thread_tool_whitelist, "allowed", None)
    if allowed is None or tool_name in allowed:
        return None
    fmt = getattr(plugins._thread_tool_whitelist, "fmt", "Tool '{tool_name}' denied")
    return fmt.format(tool_name=tool_name)
