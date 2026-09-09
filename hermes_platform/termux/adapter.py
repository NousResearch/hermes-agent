"""
Termux Platform Adapter Implementation.
Integrates Android filesystem permissions, termux-notification, and Termux environment.
"""

from typing import Any, Dict, List, Optional
import os
import shutil
import subprocess
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


class TermuxNotificationAdapter(NotificationAdapter):
    def send_notification(self, title: str, message: str) -> bool:
        if shutil.which("termux-notification"):
            subprocess.run(["termux-notification", "--title", title, "--content", message], check=False)
            return True
        return False


class TermuxClipboardAdapter(ClipboardAdapter):
    def get_text(self) -> str:
        if shutil.which("termux-clipboard-get"):
            res = subprocess.run(["termux-clipboard-get"], capture_output=True, text=True)
            return res.stdout
        return ""

    def set_text(self, text: str) -> bool:
        if shutil.which("termux-clipboard-set"):
            p = subprocess.Popen(["termux-clipboard-set"], stdin=subprocess.PIPE, text=True)
            p.communicate(input=text)
            return True
        return False


class TermuxSchedulerAdapter(SchedulerAdapter):
    def schedule_task(self, name: str, cron_or_spec: str, command: str) -> bool:
        return True


class TermuxAdapter(PlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformType.TERMUX)
        self._fs = LinuxFileSystemAdapter()
        self._term = LinuxTerminalAdapter()
        self._proc = LinuxProcessAdapter()
        self._browser = LinuxBrowserAdapter()
        self._notif = TermuxNotificationAdapter()
        self._clip = TermuxClipboardAdapter()
        self._net = LinuxNetworkAdapter()
        self._sched = TermuxSchedulerAdapter()

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
