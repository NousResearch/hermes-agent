"""
Windows Platform Adapter Implementation.
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
from hermes_platform.linux.adapter import LinuxFileSystemAdapter, LinuxProcessAdapter, LinuxBrowserAdapter, LinuxNetworkAdapter


class WindowsTerminalAdapter(TerminalAdapter):
    def execute(self, command: str, cwd: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        ps_cmd = f"powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \"{command}\""
        try:
            res = subprocess.run(
                ps_cmd,
                shell=True,
                cwd=cwd,
                timeout=timeout,
                capture_output=True,
                text=True,
            )
            return {
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}


class WindowsNotificationAdapter(NotificationAdapter):
    def send_notification(self, title: str, message: str) -> bool:
        return True


class WindowsClipboardAdapter(ClipboardAdapter):
    def get_text(self) -> str:
        return ""

    def set_text(self, text: str) -> bool:
        return True


class WindowsSchedulerAdapter(SchedulerAdapter):
    def schedule_task(self, name: str, cron_or_spec: str, command: str) -> bool:
        return True


class WindowsAdapter(PlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformType.WINDOWS)
        self._fs = LinuxFileSystemAdapter()
        self._term = WindowsTerminalAdapter()
        self._proc = LinuxProcessAdapter()
        self._browser = LinuxBrowserAdapter()
        self._notif = WindowsNotificationAdapter()
        self._clip = WindowsClipboardAdapter()
        self._net = LinuxNetworkAdapter()
        self._sched = WindowsSchedulerAdapter()

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
