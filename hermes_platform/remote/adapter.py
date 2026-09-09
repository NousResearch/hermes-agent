"""
Remote Platform Adapter Implementation.
"""

from typing import Any, Dict, List, Optional
from hermes_platform.common.adapter import (
    PlatformAdapter,
    PlatformType,
    FileSystemAdapter,
    TerminalAdapter,
    ProcessAdapter,
    BrowserAdapter,
    NotificationAdapter,
    ClipboardAdapter,
    NetworkAdapter,
    SchedulerAdapter,
)
from hermes_platform.linux.adapter import LinuxFileSystemAdapter, LinuxTerminalAdapter, LinuxProcessAdapter, LinuxBrowserAdapter, LinuxNetworkAdapter


class RemoteAdapter(PlatformAdapter):
    def __init__(self, remote_host: str = "localhost") -> None:
        super().__init__(PlatformType.REMOTE)
        self.remote_host = remote_host
        self._fs = LinuxFileSystemAdapter()
        self._term = LinuxTerminalAdapter()
        self._proc = LinuxProcessAdapter()
        self._browser = LinuxBrowserAdapter()
        self._notif = LinuxNotificationAdapter()
        self._clip = LinuxClipboardAdapter()
        self._net = LinuxNetworkAdapter()
        self._sched = LinuxSchedulerAdapter()

    @property
    def filesystem(self) -> FileSystemAdapter:
        return self._fs

    @property
    def terminal(self) -> TerminalAdapter:
        return self._term

    @property
    def process(self) -> ProcessAdapter:
        return self._proc

    @property
    def browser(self) -> BrowserAdapter:
        return self._browser

    @property
    def notification(self) -> NotificationAdapter:
        return self._notif

    @property
    def clipboard(self) -> ClipboardAdapter:
        return self._clip

    @property
    def network(self) -> NetworkAdapter:
        return self._net

    @property
    def scheduler(self) -> SchedulerAdapter:
        return self._sched
