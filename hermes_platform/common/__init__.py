"""
Common Platform Adapter Interfaces.
"""

from hermes_platform.common.adapter import (
    PlatformAdapter,
    FileSystemAdapter,
    TerminalAdapter,
    ProcessAdapter,
    BrowserAdapter,
    NotificationAdapter,
    ClipboardAdapter,
    NetworkAdapter,
    SchedulerAdapter,
    PlatformType,
)

__all__ = [
    "PlatformAdapter",
    "FileSystemAdapter",
    "TerminalAdapter",
    "ProcessAdapter",
    "BrowserAdapter",
    "NotificationAdapter",
    "ClipboardAdapter",
    "NetworkAdapter",
    "SchedulerAdapter",
    "PlatformType",
]
