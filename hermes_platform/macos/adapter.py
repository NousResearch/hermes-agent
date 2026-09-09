"""
macOS Platform Adapter Implementation.
"""

from typing import Any, Dict, List, Optional
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


class MacOSNotificationAdapter(NotificationAdapter):
    def send_notification(self, title: str, message: str) -> bool:
        cmd = f'osascript -e \'display notification "{message}" with title "{title}"\''
        subprocess.run(cmd, shell=True, check=False)
        return True


class MacOSClipboardAdapter(ClipboardAdapter):
    def get_text(self) -> str:
        res = subprocess.run(["pbpaste"], capture_output=True, text=True)
        return res.stdout

    def set_text(self, text: str) -> bool:
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE, text=True)
        p.communicate(input=text)
        return True


class MacOSSchedulerAdapter(SchedulerAdapter):
    def schedule_task(self, name: str, cron_or_spec: str, command: str) -> bool:
        return True


class MacOSAdapter(PlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformType.MACOS)
        self._fs = LinuxFileSystemAdapter()
        self._term = LinuxTerminalAdapter()
        self._proc = LinuxProcessAdapter()
        self._browser = LinuxBrowserAdapter()
        self._notif = MacOSNotificationAdapter()
        self._clip = MacOSClipboardAdapter()
        self._net = LinuxNetworkAdapter()
        self._sched = MacOSSchedulerAdapter()

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
