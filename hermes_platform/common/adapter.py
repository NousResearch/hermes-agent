"""
Universal Platform Adapter Interfaces.
Encapsulates operating system differences behind abstract interfaces.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
import os
import subprocess
import shutil
import platform


class PlatformType(str, Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ANDROID = "android"
    TERMUX = "termux"
    WSL = "wsl"
    REMOTE = "remote"


class FileSystemAdapter(ABC):
    @abstractmethod
    def list(self, path: str) -> List[str]:
        pass

    @abstractmethod
    def read(self, path: str, encoding: str = "utf-8") -> str:
        pass

    @abstractmethod
    def write(self, path: str, content: str, encoding: str = "utf-8") -> bool:
        pass

    @abstractmethod
    def copy(self, src: str, dst: str) -> bool:
        pass

    @abstractmethod
    def move(self, src: str, dst: str) -> bool:
        pass

    @abstractmethod
    def delete(self, path: str) -> bool:
        pass

    @abstractmethod
    def mkdir(self, path: str, parents: bool = True) -> bool:
        pass


class TerminalAdapter(ABC):
    @abstractmethod
    def execute(self, command: str, cwd: Optional[str] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        pass


class ProcessAdapter(ABC):
    @abstractmethod
    def list_processes(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def kill_process(self, pid: int) -> bool:
        pass


class BrowserAdapter(ABC):
    @abstractmethod
    def open_url(self, url: str) -> bool:
        pass


class NotificationAdapter(ABC):
    @abstractmethod
    def send_notification(self, title: str, message: str) -> bool:
        pass


class ClipboardAdapter(ABC):
    @abstractmethod
    def get_text(self) -> str:
        pass

    @abstractmethod
    def set_text(self, text: str) -> bool:
        pass


class NetworkAdapter(ABC):
    @abstractmethod
    def check_connectivity(self) -> bool:
        pass


class SchedulerAdapter(ABC):
    @abstractmethod
    def schedule_task(self, name: str, cron_or_spec: str, command: str) -> bool:
        pass


class PlatformAdapter(ABC):
    """Abstract Master Platform Adapter."""

    def __init__(self, platform_type: PlatformType) -> None:
        self.platform_type = platform_type

    @property
    @abstractmethod
    def filesystem(self) -> FileSystemAdapter:
        pass

    @property
    @abstractmethod
    def terminal(self) -> TerminalAdapter:
        pass

    @property
    @abstractmethod
    def process(self) -> ProcessAdapter:
        pass

    @property
    @abstractmethod
    def browser(self) -> BrowserAdapter:
        pass

    @property
    @abstractmethod
    def notification(self) -> NotificationAdapter:
        pass

    @property
    @abstractmethod
    def clipboard(self) -> ClipboardAdapter:
        pass

    @property
    @abstractmethod
    def network(self) -> NetworkAdapter:
        pass

    @property
    @abstractmethod
    def scheduler(self) -> SchedulerAdapter:
        pass
