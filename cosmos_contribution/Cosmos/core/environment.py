"""
cosmos Environment Awareness

"Good news, everyone! I now know exactly where and what I'm running on!"

Comprehensive environment detection, terminal awareness, and context tracking.
"""

import os
import sys
import shutil
import platform
import subprocess
import json
from typing import   Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from loguru import logger


class TerminalType(Enum):
    """Types of terminals/shells."""
    POWERSHELL = "powershell"
    CMD = "cmd"
    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"
    WSL = "wsl"
    WINDOWS_TERMINAL = "windows_terminal"
    ITERM = "iterm"
    TERMINAL_APP = "terminal_app"  # macOS
    GNOME_TERMINAL = "gnome_terminal"
    KONSOLE = "konsole"
    XTERM = "xterm"
    SSH = "ssh"
    VS_CODE_TERMINAL = "vscode"
    JETBRAINS_TERMINAL = "jetbrains"
    UNKNOWN = "unknown"


class OSType(Enum):
    """Operating system types."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    WSL = "wsl"
    UNKNOWN = "unknown"


class ContainerType(Enum):
    """Container environment types."""
    NONE = "none"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    PODMAN = "podman"
    LXC = "lxc"


class CloudProvider(Enum):
    """Cloud provider types."""
    NONE = "none"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITALOCEAN = "digitalocean"
    LINODE = "linode"


@dataclass
class SystemInfo:
    """System information."""
    hostname: str = ""
    os_type: OSType = OSType.UNKNOWN
    os_version: str = ""
    os_release: str = ""
    architecture: str = ""
    cpu_count: int = 0
    memory_gb: float = 0
    python_version: str = ""
    user: str = ""
    home_dir: str = ""
    shell: str = ""


@dataclass
class TerminalInfo:
    """Terminal/shell information."""
    terminal_type: TerminalType = TerminalType.UNKNOWN
    shell: str = ""
    shell_version: str = ""
    term_program: str = ""
    term_program_version: str = ""
    color_support: bool = True
    unicode_support: bool = True
    width: int = 80
    height: int = 24
    is_interactive: bool = True
    is_tty: bool = True


@dataclass
class ContainerInfo:
    """Container environment information."""
    type: ContainerType = ContainerType.NONE
    container_id: str = ""
    image_name: str = ""
    orchestrator: str = ""


@dataclass
class CloudInfo:
    """Cloud environment information."""
    provider: CloudProvider = CloudProvider.NONE
    region: str = ""
    instance_id: str = ""
    instance_type: str = ""
    availability_zone: str = ""


@dataclass
class NetworkInfo:
    """Network environment information."""
    hostname: str = ""
    local_ip: str = ""
    public_ip: str = ""
    domain: str = ""
    dns_servers: list[str] = field(default_factory=list)


@dataclass
class EnvironmentContext:
    """Complete environment context."""
    system: SystemInfo = field(default_factory=SystemInfo)
    terminal: TerminalInfo = field(default_factory=TerminalInfo)
    container: ContainerInfo = field(default_factory=ContainerInfo)
    cloud: CloudInfo = field(default_factory=CloudInfo)
    network: NetworkInfo = field(default_factory=NetworkInfo)
    detected_at: datetime = field(default_factory=datetime.now)
    capabilities: dict[str, bool] = field(default_factory=dict)
    environment_vars: dict[str, str] = field(default_factory=dict)

    def to_any(self) -> dict:
        return {
            "system": {
                "hostname": self.system.hostname,
                "os_type": self.system.os_type.value,
                "os_version": self.system.os_version,
                "architecture": self.system.architecture,
                "cpu_count": self.system.cpu_count,
                "memory_gb": self.system.memory_gb,
                "python_version": self.system.python_version,
                "user": self.system.user,
            },
            "terminal": {
                "type": self.terminal.terminal_type.value,
                "shell": self.terminal.shell,
                "width": self.terminal.width,
                "height": self.terminal.height,
                "color_support": self.terminal.color_support,
                "unicode_support": self.terminal.unicode_support,
                "is_interactive": self.terminal.is_interactive,
            },
            "container": {
                "type": self.container.type.value,
                "container_id": self.container.container_id,
            },
            "cloud": {
                "provider": self.cloud.provider.value,
                "region": self.cloud.region,
                "instance_type": self.cloud.instance_type,
            },
            "capabilities": self.capabilities,
            "detected_at": self.detected_at.isoformat(),
        }


class EnvironmentDetector:
    """
    Comprehensive environment detection.

    Detects:
    - Operating system and version
    - Terminal type and capabilities
    - Container environment
    - Cloud provider
    - Network configuration
    - Available capabilities
    """

    def __init__(self, data_dir: str = "./data/environment"):
        """Initialize environment detector."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._context: Optional[EnvironmentContext] = None
        self._detection_cache_file = self.data_dir / "context_cache.json"

    @property
    def context(self) -> EnvironmentContext:
        """Get or detect environment context."""
        if self._context is None:
            self._context = self.detect()
        return self._context

    def detect(self, force_refresh: bool = False) -> EnvironmentContext:
        """
        Detect complete environment context.

        Args:
            force_refresh: Force fresh detection even if cached

        Returns:
            EnvironmentContext with all detected information
        """
        ctx = EnvironmentContext()

        # Detect system info
        ctx.system = self._detect_system()

        # Detect terminal
        ctx.terminal = self._detect_terminal()

        # Detect container
        ctx.container = self._detect_container()

        # Detect cloud
        ctx.cloud = self._detect_cloud()

        # Detect network
        ctx.network = self._detect_network()

        # Detect capabilities
        ctx.capabilities = self._detect_capabilities()

        # Store relevant environment variables
        ctx.environment_vars = self._get_relevant_env_vars()

        # Cache the context
        self._cache_context(ctx)

        logger.info(
            f"Environment detected: {ctx.system.os_type.value} / "
            f"{ctx.terminal.terminal_type.value} / "
            f"{ctx.container.type.value}"
        )

        return ctx

    def _detect_system(self) -> SystemInfo:
        """Detect system information."""
        info = SystemInfo()

        info.hostname = platform.node()
        info.os_version = platform.version()
        info.os_release = platform.release()
        info.architecture = platform.machine()
        info.python_version = platform.python_version()
        info.user = os.getenv("USER") or os.getenv("USERNAME", "")
        info.home_dir = str(Path.home())
        info.shell = os.getenv("SHELL", "")

        # Detect OS type
        system = platform.system().lower()
        if system == "windows":
            info.os_type = OSType.WINDOWS
        elif system == "linux":
            # Check for WSL
            if self._is_wsl():
                info.os_type = OSType.WSL
            else:
                info.os_type = OSType.LINUX
        elif system == "darwin":
            info.os_type = OSType.MACOS
        else:
            info.os_type = OSType.UNKNOWN

        # Get CPU count
        try:
            info.cpu_count = os.cpu_count() or 0
        except Exception:
            pass

        # Get memory
        try:
            import psutil
            info.memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            pass

        return info

    def _is_wsl(self) -> bool:
        """Check if running in WSL."""
        if os.path.exists("/proc/version"):
            try:
                with open("/proc/version") as f:
                    version = f.read().lower()
                    return "microsoft" in version or "wsl" in version
            except Exception:
                pass
        return False

    def _detect_terminal(self) -> TerminalInfo:
        """Detect terminal information."""
        info = TerminalInfo()

        # Check TTY
        info.is_tty = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
        info.is_interactive = info.is_tty

        # Get terminal size
        try:
            size = shutil.get_terminal_size()
            info.width = size.columns
            info.height = size.lines
        except Exception:
            pass

        # Get shell
        info.shell = os.getenv("SHELL", os.getenv("COMSPEC", ""))

        # Detect terminal program
        term_program = os.getenv("TERM_PROGRAM", "")
        info.term_program = term_program
        info.term_program_version = os.getenv("TERM_PROGRAM_VERSION", "")

        # Detect terminal type
        info.terminal_type = self._identify_terminal(term_program)

        # Check color support
        info.color_support = self._check_color_support()

        # Check unicode support
        info.unicode_support = self._check_unicode_support()

        # Get shell version
        info.shell_version = self._get_shell_version(info.shell)

        return info

    def _identify_terminal(self, term_program: str) -> TerminalType:
        """Identify terminal type."""
        term_program_lower = term_program.lower()

        # Check specific terminal programs
        if "vscode" in term_program_lower or os.getenv("TERM_PROGRAM") == "vscode":
            return TerminalType.VS_CODE_TERMINAL

        if "iterm" in term_program_lower:
            return TerminalType.ITERM

        if term_program == "Apple_Terminal":
            return TerminalType.TERMINAL_APP

        if os.getenv("WT_SESSION"):
            return TerminalType.WINDOWS_TERMINAL

        if os.getenv("JETBRAINS_TERMINAL"):
            return TerminalType.JETBRAINS_TERMINAL

        # Check by shell/environment
        if os.getenv("SSH_CLIENT") or os.getenv("SSH_CONNECTION"):
            return TerminalType.SSH

        shell = os.getenv("SHELL", "").lower()
        comspec = os.getenv("COMSPEC", "").lower()

        if "powershell" in comspec or os.getenv("PSModulePath"):
            return TerminalType.POWERSHELL

        if "cmd" in comspec:
            return TerminalType.CMD

        if "bash" in shell:
            return TerminalType.BASH

        if "zsh" in shell:
            return TerminalType.ZSH

        if "fish" in shell:
            return TerminalType.FISH

        # Check TERM environment variable
        term = os.getenv("TERM", "").lower()

        if "xterm" in term:
            return TerminalType.XTERM

        if "gnome" in term_program_lower:
            return TerminalType.GNOME_TERMINAL

        if "konsole" in term_program_lower:
            return TerminalType.KONSOLE

        return TerminalType.UNKNOWN

    def _check_color_support(self) -> bool:
        """Check if terminal supports colors."""
        # Check TERM
        term = os.getenv("TERM", "")
        if term in ["dumb", ""]:
            return False

        # Check Windows
        if platform.system() == "Windows":
            # Windows 10+ supports colors
            return True

        # Check COLORTERM
        if os.getenv("COLORTERM"):
            return True

        # Check common color-supporting terminals
        color_terms = ["xterm", "linux", "screen", "vt100", "ansi", "color"]
        return any(ct in color_terms)

    def _check_unicode_support(self) -> bool:
        """Check if terminal supports unicode."""
        # Check encoding
        try:
            encoding = sys.stdout.encoding or ""
            return "utf" in encoding.lower()
        except Exception:
            pass

        # Check locale
        lang = os.getenv("LANG", "").lower()
        return "utf" in lang

    def _get_shell_version(self, shell: str) -> str:
        """Get shell version."""
        try:
            if "bash" in shell.lower():
                result = subprocess.run(
                    ["bash", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return result.stdout.split("\n")[0]

            elif "zsh" in shell.lower():
                result = subprocess.run(
                    ["zsh", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return result.stdout.strip()

            elif "powershell" in shell.lower():
                result = subprocess.run(
                    ["powershell", "-Command", "$PSVersionTable.PSVersion.ToString()"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return f"PowerShell {result.stdout.strip()}"

        except Exception:
            pass

        return ""

    def _detect_container(self) -> ContainerInfo:
        """Detect container environment."""
        info = ContainerInfo()

        # Check Docker
        if os.path.exists("/.dockerenv"):
            info.type = ContainerType.DOCKER

            # Try to get container ID
            try:
                with open("/proc/self/cgroup") as f:
                    for line in f:
                        if "docker" in line:
                            parts = line.strip().split("/")
                            if len(parts) > 2:
                                info.container_id = parts[-1][:12]
                            break
            except Exception:
                pass

        # Check Kubernetes
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            info.type = ContainerType.KUBERNETES
            info.orchestrator = "kubernetes"

        # Check Podman
        elif os.path.exists("/run/.containerenv"):
            info.type = ContainerType.PODMAN

        return info

    def _detect_cloud(self) -> CloudInfo:
        """Detect cloud provider."""
        info = CloudInfo()

        # Check AWS
        if os.path.exists("/sys/hypervisor/uuid"):
            try:
                with open("/sys/hypervisor/uuid") as f:
                    uuid = f.read().strip().lower()
                    if uuid.startswith("ec2"):
                        info.provider = CloudProvider.AWS
            except Exception:
                pass

        # Check AWS metadata service
        if info.provider == CloudProvider.NONE:
            try:
                import httpx
                response = httpx.get(
                    "http://169.254.169.254/latest/meta-data/instance-id",
                    timeout=1.0,
                )
                if response.status_code == 200:
                    info.provider = CloudProvider.AWS
                    info.instance_id = response.text
            except Exception:
                pass

        # Check Azure
        if info.provider == CloudProvider.NONE:
            try:
                import httpx
                response = httpx.get(
                    "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                    headers={"Metadata": "true"},
                    timeout=1.0,
                )
                if response.status_code == 200:
                    info.provider = CloudProvider.AZURE
                    data = response.json()
                    info.region = data.get("compute", {}).get("location", "")
                    info.instance_type = data.get("compute", {}).get("vmSize", "")
            except Exception:
                pass

        # Check GCP
        if info.provider == CloudProvider.NONE:
            try:
                import httpx
                response = httpx.get(
                    "http://metadata.google.internal/computeMetadata/v1/instance/id",
                    headers={"Metadata-Flavor": "Google"},
                    timeout=1.0,
                )
                if response.status_code == 200:
                    info.provider = CloudProvider.GCP
                    info.instance_id = response.text
            except Exception:
                pass

        return info

    def _detect_network(self) -> NetworkInfo:
        """Detect network information."""
        info = NetworkInfo()

        info.hostname = platform.node()

        # Get local IP
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            info.local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            info.local_ip = "127.0.0.1"

        # Get domain
        try:
            import socket
            fqdn = socket.getfqdn()
            if "." in fqdn:
                info.domain = ".".join(fqdn.split(".")[1:])
        except Exception:
            pass

        return info

    def _detect_capabilities(self) -> dict[str, bool]:
        """Detect available capabilities."""
        caps = {}

        # Check for common tools
        tools = [
            "git", "docker", "kubectl", "python", "node", "npm",
            "pip", "curl", "wget", "ssh", "scp", "rsync",
            "aws", "az", "gcloud", "terraform", "ansible",
        ]

        for tool in tools:
            caps[f"has_{tool}"] = shutil.which(tool) is not None

        # Check Python packages
        packages = ["psutil", "httpx", "asyncio", "rich"]
        for pkg in packages:
            try:
                __import__(pkg)
                caps[f"has_pkg_{pkg}"] = True
            except ImportError:
                caps[f"has_pkg_{pkg}"] = False

        # Check permissions
        caps["is_admin"] = self._check_admin()

        # Check network access
        caps["has_internet"] = self._check_internet()

        return caps

    def _check_admin(self) -> bool:
        """Check if running with admin/root privileges."""
        try:
            if platform.system() == "Windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            else:
                return os.geteuid() == 0
        except Exception:
            return False

    def _check_internet(self) -> bool:
        """Check for internet connectivity."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except Exception:
            return False

    def _get_relevant_env_vars(self) -> dict[str, str]:
        """Get relevant environment variables."""
        relevant_prefixes = [
            "cosmos_", "PATH", "HOME", "USER", "SHELL",
            "TERM", "LANG", "LC_", "DISPLAY", "SSH_",
            "DOCKER_", "KUBERNETES_", "AWS_", "AZURE_", "GOOGLE_",
            "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        ]

        env_vars = {}
        for key, value in os.environ.items():
            for prefix in relevant_prefixes:
                if key.startswith(prefix) or key == prefix:
                    # Don't include secrets
                    if "SECRET" not in key and "KEY" not in key and "TOKEN" not in key and "PASSWORD" not in key:
                        env_vars[key] = value[:200]  # Truncate long values
                    break

        return env_vars

    def _cache_context(self, ctx: EnvironmentContext):
        """Cache context to disk."""
        try:
            self._detection_cache_file.write_text(
                json.dumps(ctx.to_any(), indent=2)
            )
        except Exception as e:
            logger.debug(f"Failed to cache context: {e}")

    def get_summary(self) -> str:
        """Get human-readable environment summary."""
        ctx = self.context

        lines = [
            "=" * 50,
            "cosmos ENVIRONMENT SUMMARY",
            "=" * 50,
            "",
            "SYSTEM",
            "-" * 30,
            f"  OS: {ctx.system.os_type.value} ({ctx.system.os_version})",
            f"  Host: {ctx.system.hostname}",
            f"  Arch: {ctx.system.architecture}",
            f"  CPUs: {ctx.system.cpu_count}",
            f"  Memory: {ctx.system.memory_gb:.1f} GB",
            f"  Python: {ctx.system.python_version}",
            f"  User: {ctx.system.user}",
            "",
            "TERMINAL",
            "-" * 30,
            f"  Type: {ctx.terminal.terminal_type.value}",
            f"  Shell: {ctx.terminal.shell}",
            f"  Size: {ctx.terminal.width}x{ctx.terminal.height}",
            f"  Color: {'Yes' if ctx.terminal.color_support else 'No'}",
            f"  Unicode: {'Yes' if ctx.terminal.unicode_support else 'No'}",
            f"  Interactive: {'Yes' if ctx.terminal.is_interactive else 'No'}",
            "",
        ]

        if ctx.container.type != ContainerType.NONE:
            lines.extend([
                "CONTAINER",
                "-" * 30,
                f"  Type: {ctx.container.type.value}",
                f"  ID: {ctx.container.container_id or 'N/A'}",
                "",
            ])

        if ctx.cloud.provider != CloudProvider.NONE:
            lines.extend([
                "CLOUD",
                "-" * 30,
                f"  Provider: {ctx.cloud.provider.value}",
                f"  Region: {ctx.cloud.region or 'N/A'}",
                f"  Instance: {ctx.cloud.instance_type or 'N/A'}",
                "",
            ])

        lines.extend([
            "NETWORK",
            "-" * 30,
            f"  Local IP: {ctx.network.local_ip}",
            f"  Domain: {ctx.network.domain or 'N/A'}",
            "",
            "CAPABILITIES",
            "-" * 30,
        ])

        # Group capabilities
        has_tools = [k.replace("has_", "") for k, v in ctx.capabilities.items() if v and k.startswith("has_") and not k.startswith("has_pkg_")]
        has_pkgs = [k.replace("has_pkg_", "") for k, v in ctx.capabilities.items() if v and k.startswith("has_pkg_")]

        lines.append(f"  Tools: {', '.join(has_tools[:10])}")
        lines.append(f"  Python Pkgs: {', '.join(has_pkgs)}")
        lines.append(f"  Admin: {'Yes' if ctx.capabilities.get('is_admin') else 'No'}")
        lines.append(f"  Internet: {'Yes' if ctx.capabilities.get('has_internet') else 'No'}")

        lines.extend([
            "",
            "=" * 50,
        ])

        return "\n".join(lines)

    def refresh(self) -> EnvironmentContext:
        """Force refresh environment detection."""
        self._context = None
        return self.detect(force_refresh=True)

    def adapt_output(self, text: str) -> str:
        """
        Adapt output based on terminal capabilities.

        Removes colors/unicode if not supported.
        """
        ctx = self.context

        if not ctx.terminal.color_support:
            # Remove ANSI color codes
            import re
            text = re.sub(r'\x1b\[[0-9;]*m', '', text)

        if not ctx.terminal.unicode_support:
            # Replace common unicode with ASCII
            replacements = {
                '→': '->',
                '←': '<-',
                '↑': '^',
                '↓': 'v',
                '✓': '[OK]',
                '✗': '[X]',
                '●': '*',
                '○': 'o',
                '▶': '>',
                '◀': '<',
                '━': '-',
                '│': '|',
                '┌': '+',
                '┐': '+',
                '└': '+',
                '┘': '+',
            }
            for uni, ascii_char in replacements.items():
                text = text.replace(uni, ascii_char)

        return text


# Global instance
environment_detector = EnvironmentDetector()


def get_environment() -> EnvironmentContext:
    """Get current environment context."""
    return environment_detector.context


def get_environment_summary() -> str:
    """Get environment summary string."""
    return environment_detector.get_summary()


def adapt_for_terminal(text: str) -> str:
    """Adapt text for current terminal capabilities."""
    return environment_detector.adapt_output(text)
