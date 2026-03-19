"""
cosmos WSL Bridge

"Good news, everyone! I've bridged the gap between Windows and Linux!"

Seamless integration with Windows Subsystem for Linux (WSL).
"""

import asyncio
import subprocess
import platform
import os
import shutil
from typing import   Optional, Tuple
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath, PurePosixPath
from enum import Enum
from loguru import logger


class WSLDistro(Enum):
    """Known WSL distributions."""
    UBUNTU = "Ubuntu"
    DEBIAN = "Debian"
    KALI = "kali-linux"
    OPENSUSE = "openSUSE-Leap"
    ALPINE = "Alpine"
    ARCH = "Arch"
    DEFAULT = "default"


@dataclass
class WSLInfo:
    """WSL environment information."""
    is_available: bool = False
    version: int = 0  # WSL 1 or 2
    default_distro: str = ""
    installed_distros: list[str] = None
    is_wsl_environment: bool = False  # Running inside WSL

    def __post_init__(self):
        if self.installed_distros is None:
            self.installed_distros = []


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    returncode: int
    stdout: str
    stderr: str
    command: str


class WSLBridge:
    """
    Bridge for Windows <-> WSL integration.

    Features:
    - Detect WSL availability and version
    - Run commands in WSL from Windows
    - Path translation (Windows <-> Linux)
    - File operations across boundaries
    - Environment variable handling
    """

    def __init__(self):
        """Initialize WSL bridge."""
        self.is_windows = platform.system() == "Windows"
        self._info: Optional[WSLInfo] = None
        self._default_distro: Optional[str] = None

    @property
    def info(self) -> WSLInfo:
        """Get WSL information (cached)."""
        if self._info is None:
            self._info = self._detect_wsl()
        return self._info

    def _detect_wsl(self) -> WSLInfo:
        """Detect WSL installation and configuration."""
        info = WSLInfo()

        # Check if we're inside WSL
        if os.path.exists("/proc/version"):
            try:
                with open("/proc/version") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        info.is_wsl_environment = True
                        info.is_available = True
                        return info
            except Exception:
                pass

        # Windows detection
        if not self.is_windows:
            return info

        # Check for wsl.exe
        wsl_path = shutil.which("wsl")
        if not wsl_path:
            return info

        # Get WSL status
        try:
            result = subprocess.run(
                ["wsl", "--status"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                info.is_available = True

                # Check version
                if "WSL 2" in result.stdout or "WSL version: 2" in result.stdout:
                    info.version = 2
                else:
                    info.version = 1

        except Exception as e:
            logger.debug(f"WSL status check failed: {e}")

        # Get installed distributions
        try:
            result = subprocess.run(
                ["wsl", "--list", "--verbose"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                info.is_available = True
                info.installed_distros = self._parse_distro_list(result.stdout)

                # Find default
                for line in result.stdout.split("\n"):
                    if "*" in line:
                        parts = line.replace("*", "").strip().split()
                        if parts:
                            info.default_distro = parts[0]
                            break

        except Exception as e:
            logger.debug(f"WSL list failed: {e}")

        return info

    def _parse_distro_list(self, output: str) -> list[str]:
        """Parse WSL distribution list output."""
        distros = []

        for line in output.split("\n"):
            line = line.strip()
            # Skip header and empty lines
            if not line or "NAME" in line or "STATE" in line:
                continue

            # Remove default marker
            line = line.replace("*", "").strip()

            # Get distro name (first word)
            parts = line.split()
            if parts:
                name = parts[0]
                # Filter out noise
                if name and not name.startswith("-"):
                    distros.append(name)

        return distros

    def refresh(self):
        """Refresh WSL information."""
        self._info = None
        _ = self.info  # Trigger detection

    # ========== Path Translation ==========

    def windows_to_wsl_path(self, windows_path: str) -> str:
        """
        Convert Windows path to WSL path.

        Example: C:\\Users\\name -> /mnt/c/Users/name
        """
        path = PureWindowsPath(windows_path)

        # Handle drive letter
        if path.drive:
            drive = path.drive.rstrip(":").lower()
            rest = str(path.relative_to(path.drive + "\\"))
            return f"/mnt/{drive}/{rest}".replace("\\", "/")

        # Already a Unix-style path
        return str(path).replace("\\", "/")

    def wsl_to_windows_path(self, wsl_path: str) -> str:
        """
        Convert WSL path to Windows path.

        Example: /mnt/c/Users/name -> C:\\Users\\name
        """
        if wsl_path.startswith("/mnt/"):
            parts = wsl_path[5:].split("/", 1)
            if len(parts) == 2:
                drive = parts[0].upper()
                rest = parts[1].replace("/", "\\")
                return f"{drive}:\\{rest}"
            elif len(parts) == 1:
                return f"{parts[0].upper()}:\\"

        # Path not under /mnt, return as-is or construct UNC path
        return f"\\\\wsl$\\{self.info.default_distro or 'Ubuntu'}{wsl_path}"

    def normalize_path(self, path: str, target: str = "wsl") -> str:
        """
        Normalize path for target environment.

        Args:
            path: Input path
            target: "wsl" or "windows"
        """
        if target == "wsl":
            if "\\" in path or (len(path) > 1 and path[1] == ":"):
                return self.windows_to_wsl_path(path)
            return path
        else:
            if path.startswith("/mnt/") or path.startswith("/"):
                return self.wsl_to_windows_path(path)
            return path

    # ========== Command Execution ==========

    async def run_command(
        self,
        command: str,
        distro: Optional[str] = None,
        working_dir: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout: int = 60,
        shell: bool = True,
    ) -> CommandResult:
        """
        Run a command in WSL.

        Args:
            command: The command to run
            distro: Specific distribution (default: system default)
            working_dir: Working directory (Windows path will be converted)
            env: Environment variables
            timeout: Command timeout in seconds
            shell: Run through shell

        Returns:
            CommandResult with output
        """
        if not self.info.is_available:
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="WSL is not available",
                command=command,
            )

        # Build WSL command
        wsl_cmd = ["wsl"]

        if distro:
            wsl_cmd.extend(["-d", distro])
        elif self._default_distro:
            wsl_cmd.extend(["-d", self._default_distro])

        # Add working directory
        if working_dir:
            wsl_path = self.windows_to_wsl_path(working_dir)
            wsl_cmd.extend(["--cd", wsl_path])

        # Add the command
        if shell:
            wsl_cmd.extend(["bash", "-c", command])
        else:
            wsl_cmd.append(command)

        # Prepare environment
        process_env = os.environ.copy()
        if env:
            # Convert paths in env vars
            for key, value in env.items():
                if "\\" in value:
                    value = self.windows_to_wsl_path(value)
                process_env[key] = value

        try:
            proc = await asyncio.create_subprocess_exec(
                *wsl_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=process_env,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            return CommandResult(
                success=proc.returncode == 0,
                returncode=proc.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                command=command,
            )

        except asyncio.TimeoutError:
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                command=command,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=str(e),
                command=command,
            )

    def run_command_sync(
        self,
        command: str,
        distro: Optional[str] = None,
        timeout: int = 60,
    ) -> CommandResult:
        """Synchronous command execution."""
        if not self.info.is_available:
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr="WSL is not available",
                command=command,
            )

        wsl_cmd = ["wsl"]
        if distro:
            wsl_cmd.extend(["-d", distro])

        wsl_cmd.extend(["bash", "-c", command])

        try:
            result = subprocess.run(
                wsl_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return CommandResult(
                success=result.returncode == 0,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                command=command,
            )

        except subprocess.TimeoutExpired:
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                command=command,
            )
        except Exception as e:
            return CommandResult(
                success=False,
                returncode=-1,
                stdout="",
                stderr=str(e),
                command=command,
            )

    # ========== Distribution Management ==========

    def get_default_distro(self) -> Optional[str]:
        """Get the default WSL distribution."""
        return self.info.default_distro

    def set_default_distro(self, distro: str) -> bool:
        """Set the default WSL distribution for this session."""
        if distro in self.info.installed_distros:
            self._default_distro = distro
            return True
        return False

    async def install_package(
        self,
        package: str,
        distro: Optional[str] = None,
    ) -> CommandResult:
        """
        Install a package in WSL.

        Detects package manager automatically.
        """
        # Detect package manager
        result = await self.run_command("which apt-get", distro=distro)
        if result.success:
            return await self.run_command(
                f"sudo apt-get update && sudo apt-get install -y {package}",
                distro=distro,
                timeout=300,
            )

        result = await self.run_command("which yum", distro=distro)
        if result.success:
            return await self.run_command(
                f"sudo yum install -y {package}",
                distro=distro,
                timeout=300,
            )

        result = await self.run_command("which pacman", distro=distro)
        if result.success:
            return await self.run_command(
                f"sudo pacman -S --noconfirm {package}",
                distro=distro,
                timeout=300,
            )

        return CommandResult(
            success=False,
            returncode=-1,
            stdout="",
            stderr="No supported package manager found",
            command=f"install {package}",
        )

    # ========== File Operations ==========

    async def copy_to_wsl(
        self,
        windows_path: str,
        wsl_path: str,
        distro: Optional[str] = None,
    ) -> bool:
        """Copy a file from Windows to WSL."""
        src = self.windows_to_wsl_path(windows_path)

        result = await self.run_command(
            f"cp '{src}' '{wsl_path}'",
            distro=distro,
        )

        return result.success

    async def copy_from_wsl(
        self,
        wsl_path: str,
        windows_path: str,
        distro: Optional[str] = None,
    ) -> bool:
        """Copy a file from WSL to Windows."""
        dest = self.windows_to_wsl_path(windows_path)

        result = await self.run_command(
            f"cp '{wsl_path}' '{dest}'",
            distro=distro,
        )

        return result.success

    async def file_exists_in_wsl(
        self,
        wsl_path: str,
        distro: Optional[str] = None,
    ) -> bool:
        """Check if a file exists in WSL."""
        result = await self.run_command(
            f"test -e '{wsl_path}' && echo 'exists'",
            distro=distro,
        )

        return "exists" in result.stdout

    # ========== Utility Functions ==========

    async def get_linux_info(self, distro: Optional[str] = None) -> dict:
        """Get information about the Linux environment."""
        info = {}

        # Get OS release info
        result = await self.run_command("cat /etc/os-release", distro=distro)
        if result.success:
            for line in result.stdout.split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    info[key.lower()] = value.strip('"')

        # Get kernel version
        result = await self.run_command("uname -r", distro=distro)
        if result.success:
            info["kernel"] = result.stdout.strip()

        # Get architecture
        result = await self.run_command("uname -m", distro=distro)
        if result.success:
            info["arch"] = result.stdout.strip()

        return info

    async def check_command_available(
        self,
        command: str,
        distro: Optional[str] = None,
    ) -> bool:
        """Check if a command is available in WSL."""
        result = await self.run_command(f"which {command}", distro=distro)
        return result.success


# Global instance
wsl_bridge = WSLBridge()


# Convenience functions
def is_wsl_available() -> bool:
    """Check if WSL is available."""
    return wsl_bridge.info.is_available


def get_wsl_distros() -> list[str]:
    """Get installed WSL distributions."""
    return wsl_bridge.info.installed_distros


async def wsl_run(command: str, **kwargs) -> CommandResult:
    """Run a command in WSL."""
    return await wsl_bridge.run_command(command, **kwargs)


def to_wsl_path(windows_path: str) -> str:
    """Convert Windows path to WSL path."""
    return wsl_bridge.windows_to_wsl_path(windows_path)


def to_windows_path(wsl_path: str) -> str:
    """Convert WSL path to Windows path."""
    return wsl_bridge.wsl_to_windows_path(wsl_path)
