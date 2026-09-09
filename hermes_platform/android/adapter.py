"""
Android Platform Adapter Implementation.
"""

from typing import Any, Dict, List, Optional
import os
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


class AndroidNotificationAdapter(NotificationAdapter):
    def send_notification(self, title: str, message: str) -> bool:
        return True


class AndroidClipboardAdapter(ClipboardAdapter):
    def get_text(self) -> str:
        return ""

    def set_text(self, text: str) -> bool:
        return True


class AndroidSchedulerAdapter(SchedulerAdapter):
    def schedule_task(self, name: str, cron_or_spec: str, command: str) -> bool:
        return True


class AndroidAdapter(PlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformType.ANDROID)
        self._fs = LinuxFileSystemAdapter()
        self._term = LinuxTerminalAdapter()
        self._proc = LinuxProcessAdapter()
        self._browser = LinuxBrowserAdapter()
        self._notif = AndroidNotificationAdapter()
        self._clip = AndroidClipboardAdapter()
        self._net = LinuxNetworkAdapter()
        self._sched = AndroidSchedulerAdapter()

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
