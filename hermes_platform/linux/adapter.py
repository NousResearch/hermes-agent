"""
Linux Platform Adapter Implementation.
"""

from typing import Any, Dict, List, Optional
import os
import shutil
import subprocess
import urllib.request
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


class LinuxFileSystemAdapter(FileSystemAdapter):
    def list(self, path: str) -> List[str]:
        return os.listdir(path)

    def read(self, path: str, encoding: str = "utf-8") -> str:
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    def write(self, path: str, content: str, encoding: str = "utf-8") -> bool:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return True

    def copy(self, src: str, dst: str) -> bool:
        shutil.copy2(src, dst)
        return True

    def move(self, src: str, dst: str) -> bool:
        shutil.move(src, dst)
        return True

    def delete(self, path: str) -> bool:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        return True

    def mkdir(self, path: str, parents: bool = True) -> bool:
        os.makedirs(path, exist_ok=parents)
        return True


class LinuxTerminalAdapter(TerminalAdapter):
    def execute(self, command: str, cwd: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        try:
            res = subprocess.run(
                command,
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
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "stdout": "", "stderr": "Command timed out"}
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}


class LinuxProcessAdapter(ProcessAdapter):
    def list_processes(self) -> List[Dict[str, Any]]:
        import psutil
        procs = []
        for p in psutil.process_iter(["pid", "name", "username"]):
            try:
                procs.append(p.info)
            except Exception:
                pass
        return procs

    def kill_process(self, pid: int) -> bool:
        import psutil
        try:
            p = psutil.Process(pid)
            p.kill()
            return True
        except Exception:
            return False


class LinuxBrowserAdapter(BrowserAdapter):
    def open_url(self, url: str) -> bool:
        import webbrowser
        return webbrowser.open(url)


class LinuxNotificationAdapter(NotificationAdapter):
    def send_notification(self, title: str, message: str) -> bool:
        if shutil.which("notify-send"):
            subprocess.run(["notify-send", title, message], check=False)
            return True
        return False


class LinuxClipboardAdapter(ClipboardAdapter):
    def get_text(self) -> str:
        if shutil.which("xclip"):
            res = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True)
            return res.stdout
        return ""

    def set_text(self, text: str) -> bool:
        if shutil.which("xclip"):
            p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE, text=True)
            p.communicate(input=text)
            return True
        return False


class LinuxNetworkAdapter(NetworkAdapter):
    def check_connectivity(self) -> bool:
        try:
            urllib.request.urlopen("https://1.1.1.1", timeout=3)
            return True
        except Exception:
            return False


class LinuxSchedulerAdapter(SchedulerAdapter):
    def schedule_task(self, name: str, cron_or_spec: str, command: str) -> bool:
        return True


class LinuxAdapter(PlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformType.LINUX)
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
