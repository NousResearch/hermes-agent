"""
cosmos Forensics Toolkit

"Good news, everyone! I can analyze digital evidence like a 31st century detective!"

Digital forensics capabilities for security investigation.
"""

import os
import hashlib
import stat
import subprocess
import platform
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from loguru import logger


class ArtifactType(Enum):
    """Types of forensic artifacts."""
    FILE = "file"
    PROCESS = "process"
    NETWORK = "network"
    REGISTRY = "registry"
    LOG = "log"
    MEMORY = "memory"
    BROWSER = "browser"
    PERSISTENCE = "persistence"


@dataclass
class FileArtifact:
    """File forensic artifact."""
    path: str
    name: str
    size_bytes: int = 0
    md5: str = ""
    sha256: str = ""
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    accessed: Optional[datetime] = None
    owner: str = ""
    permissions: str = ""
    file_type: str = ""
    is_executable: bool = False
    is_hidden: bool = False
    entropy: float = 0.0
    suspicious_indicators: List[str] = field(default_factory=list)


@dataclass
class ProcessArtifact:
    """Process forensic artifact."""
    pid: int
    name: str
    path: str = ""
    cmdline: str = ""
    user: str = ""
    parent_pid: int = 0
    parent_name: str = ""
    start_time: Optional[datetime] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    open_files: List[str] = field(default_factory=list)
    network_connections: List[Dict] = field(default_factory=list)
    suspicious_indicators: List[str] = field(default_factory=list)


@dataclass
class ForensicTimeline:
    """Timeline of forensic events."""
    events: List[Dict[str, Any]] = field(default_factory=list)


class ForensicsToolkit:
    """
    Digital forensics toolkit.

    Capabilities:
    - File system analysis
    - Process inspection
    - Artifact collection
    - Timeline generation
    - Evidence hashing
    """

    # Suspicious file locations
    SUSPICIOUS_LOCATIONS = [
        # Windows
        r"C:\Windows\Temp",
        r"C:\Users\*\AppData\Local\Temp",
        r"C:\ProgramData",
        r"C:\Users\*\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup",
        # Linux
        "/tmp",
        "/var/tmp",
        "/dev/shm",
        "/home/*/.config/autostart",
        "/etc/cron.d",
        "/etc/init.d",
    ]

    # Suspicious file extensions
    SUSPICIOUS_EXTENSIONS = {
        ".exe", ".dll", ".scr", ".bat", ".cmd", ".ps1", ".vbs", ".js",
        ".hta", ".msi", ".jar", ".com", ".pif", ".wsf",
    }

    # Known malware file names
    KNOWN_MALWARE_NAMES = {
        "mimikatz", "lazagne", "psexec", "procdump", "nc.exe", "netcat",
        "cobalt", "beacon", "meterpreter", "powersploit",
    }

    def __init__(self, evidence_dir: str = "./evidence"):
        """Initialize forensics toolkit."""
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.platform = platform.system().lower()

    # ========== File Analysis ==========

    def analyze_file(self, file_path: str) -> FileArtifact:
        """
        Perform forensic analysis on a file.

        Args:
            file_path: Path to file

        Returns:
            FileArtifact with analysis results
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        artifact = FileArtifact(
            path=str(path.absolute()),
            name=path.name,
        )

        try:
            # Get file stats
            stat_info = path.stat()
            artifact.size_bytes = stat_info.st_size
            artifact.created = datetime.fromtimestamp(stat_info.st_ctime)
            artifact.modified = datetime.fromtimestamp(stat_info.st_mtime)
            artifact.accessed = datetime.fromtimestamp(stat_info.st_atime)
            artifact.permissions = oct(stat_info.st_mode)[-3:]

            # Check if executable
            artifact.is_executable = bool(stat_info.st_mode & stat.S_IXUSR)

            # Check if hidden
            artifact.is_hidden = path.name.startswith(".") or (
                self.platform == "windows" and self._is_hidden_windows(str(path))
            )

            # Calculate hashes
            artifact.md5, artifact.sha256 = self._calculate_hashes(path)

            # Calculate entropy (indicator of encryption/packing)
            artifact.entropy = self._calculate_entropy(path)

            # Get file type
            artifact.file_type = self._get_file_type(path)

            # Get owner
            artifact.owner = self._get_file_owner(path)

            # Check for suspicious indicators
            artifact.suspicious_indicators = self._check_file_indicators(artifact)

        except Exception as e:
            logger.error(f"Error analyzing file: {e}")

        return artifact

    def _calculate_hashes(self, path: Path) -> tuple:
        """Calculate MD5 and SHA256 hashes."""
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()

        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    md5.update(chunk)
                    sha256.update(chunk)
        except Exception as e:
            logger.error(f"Hash calculation error: {e}")
            return "", ""

        return md5.hexdigest(), sha256.hexdigest()

    def _calculate_entropy(self, path: Path, sample_size: int = 10240) -> float:
        """Calculate file entropy (randomness indicator)."""
        import math

        try:
            with open(path, "rb") as f:
                data = f.read(sample_size)

            if not data:
                return 0.0

            # Count byte frequencies
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1

            # Calculate entropy
            entropy = 0.0
            length = len(data)
            for count in byte_counts:
                if count > 0:
                    freq = count / length
                    entropy -= freq * math.log2(freq)

            return entropy

        except Exception:
            return 0.0

    def _get_file_type(self, path: Path) -> str:
        """Determine file type."""
        # Check magic bytes
        try:
            with open(path, "rb") as f:
                magic = f.read(8)

            signatures = {
                b"MZ": "PE Executable",
                b"\x7fELF": "ELF Executable",
                b"PK\x03\x04": "ZIP Archive",
                b"\x1f\x8b": "GZIP Archive",
                b"Rar!": "RAR Archive",
                b"%PDF": "PDF Document",
                b"\xff\xd8\xff": "JPEG Image",
                b"\x89PNG": "PNG Image",
            }

            for sig, file_type in signatures.items():
                if magic.startswith(sig):
                    return file_type

        except Exception:
            pass

        # Fall back to extension
        return path.suffix.lower() or "Unknown"

    def _get_file_owner(self, path: Path) -> str:
        """Get file owner."""
        try:
            if self.platform == "windows":
                # Windows - use PowerShell
                result = subprocess.run(
                    ["powershell", "-Command",
                     f"(Get-Acl '{path}').Owner"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout.strip()
            else:
                # Unix
                import pwd
                stat_info = path.stat()
                return pwd.getpwuid(stat_info.st_uid).pw_name
        except Exception:
            return "Unknown"

    def _is_hidden_windows(self, path: str) -> bool:
        """Check if file is hidden on Windows."""
        try:
            import ctypes
            attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
            return bool(attrs & 0x02)  # FILE_ATTRIBUTE_HIDDEN
        except Exception:
            return False

    def _check_file_indicators(self, artifact: FileArtifact) -> List[str]:
        """Check for suspicious file indicators."""
        indicators = []

        # Check extension
        ext = Path(artifact.name).suffix.lower()
        if ext in self.SUSPICIOUS_EXTENSIONS:
            indicators.append(f"Suspicious extension: {ext}")

        # Check name
        name_lower = artifact.name.lower()
        for malware_name in self.KNOWN_MALWARE_NAMES:
            if malware_name in name_lower:
                indicators.append(f"Known malware name pattern: {malware_name}")

        # Check entropy (high entropy = likely packed/encrypted)
        if artifact.entropy > 7.5:
            indicators.append(f"High entropy ({artifact.entropy:.2f}) - possibly packed/encrypted")

        # Check hidden executable
        if artifact.is_hidden and artifact.is_executable:
            indicators.append("Hidden executable file")

        # Check location
        for suspicious_loc in self.SUSPICIOUS_LOCATIONS:
            if suspicious_loc.replace("*", "") in artifact.path:
                indicators.append(f"Located in suspicious directory")
                break

        # Check for double extension
        if artifact.name.count(".") > 1:
            inner_ext = artifact.name.rsplit(".", 2)[-2] if artifact.name.count(".") >= 2 else ""
            if inner_ext in ["exe", "dll", "scr"]:
                indicators.append("Double extension detected")

        return indicators

    # ========== Process Analysis ==========

    def analyze_process(self, pid: int) -> ProcessArtifact:
        """
        Analyze a running process.

        Args:
            pid: Process ID

        Returns:
            ProcessArtifact with analysis
        """
        artifact = ProcessArtifact(pid=pid, name="Unknown")

        try:
            import psutil

            proc = psutil.Process(pid)
            artifact.name = proc.name()
            artifact.path = proc.exe() if proc.exe() else ""
            artifact.cmdline = " ".join(proc.cmdline()) if proc.cmdline() else ""
            artifact.user = proc.username()
            artifact.parent_pid = proc.ppid()
            artifact.cpu_percent = proc.cpu_percent()
            artifact.memory_mb = proc.memory_info().rss / (1024 * 1024)

            # Get parent info
            try:
                parent = psutil.Process(proc.ppid())
                artifact.parent_name = parent.name()
            except Exception:
                pass

            # Get start time
            try:
                artifact.start_time = datetime.fromtimestamp(proc.create_time())
            except Exception:
                pass

            # Get open files
            try:
                artifact.open_files = [f.path for f in proc.open_files()]
            except Exception:
                pass

            # Get network connections
            try:
                for conn in proc.connections():
                    artifact.network_connections.append({
                        "local_addr": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                        "remote_addr": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                        "status": conn.status,
                    })
            except Exception:
                pass

            # Check for suspicious indicators
            artifact.suspicious_indicators = self._check_process_indicators(artifact)

        except ImportError:
            logger.warning("psutil not installed")
        except Exception as e:
            logger.error(f"Process analysis error: {e}")

        return artifact

    def _check_process_indicators(self, artifact: ProcessArtifact) -> List[str]:
        """Check for suspicious process indicators."""
        indicators = []

        # Check for suspicious names
        name_lower = artifact.name.lower()
        for malware_name in self.KNOWN_MALWARE_NAMES:
            if malware_name in name_lower:
                indicators.append(f"Known tool name: {malware_name}")

        # Check for process running from temp
        if artifact.path:
            path_lower = artifact.path.lower()
            if "temp" in path_lower or "tmp" in path_lower:
                indicators.append("Running from temp directory")

        # Check for high resource usage
        if artifact.cpu_percent > 80:
            indicators.append(f"High CPU usage: {artifact.cpu_percent}%")
        if artifact.memory_mb > 1024:
            indicators.append(f"High memory usage: {artifact.memory_mb:.0f}MB")

        # Check for suspicious parent-child relationships
        suspicious_parents = {"cmd.exe", "powershell.exe", "wscript.exe", "cscript.exe"}
        if artifact.parent_name.lower() in suspicious_parents:
            indicators.append(f"Spawned by scripting host: {artifact.parent_name}")

        # Check for encoded commands in cmdline
        if artifact.cmdline:
            if "-enc" in artifact.cmdline.lower() or "-encodedcommand" in artifact.cmdline.lower():
                indicators.append("Encoded PowerShell command detected")
            if "bypass" in artifact.cmdline.lower():
                indicators.append("Execution policy bypass detected")

        # Check for suspicious network connections
        for conn in artifact.network_connections:
            if conn.get("remote_addr"):
                # External connections from unexpected processes
                if ":" in conn["remote_addr"]:
                    port = conn["remote_addr"].split(":")[-1]
                    if port in ["4444", "5555", "6666", "1337"]:
                        indicators.append(f"Connection to suspicious port: {port}")

        return indicators

    def list_processes(self) -> List[ProcessArtifact]:
        """List all running processes with basic info."""
        processes = []

        try:
            import psutil

            for proc in psutil.process_iter(["pid", "name", "username"]):
                try:
                    processes.append(ProcessArtifact(
                        pid=proc.info["pid"],
                        name=proc.info["name"] or "Unknown",
                        user=proc.info["username"] or "",
                    ))
                except Exception:
                    continue

        except ImportError:
            logger.warning("psutil not installed")

        return processes

    # ========== Timeline Generation ==========

    def generate_timeline(
        self,
        paths: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> ForensicTimeline:
        """
        Generate forensic timeline from file system.

        Args:
            paths: Paths to include
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            ForensicTimeline with events
        """
        timeline = ForensicTimeline()

        for path_str in paths:
            path = Path(path_str)

            if path.is_file():
                self._add_file_to_timeline(path, timeline, start_time, end_time)
            elif path.is_dir():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        self._add_file_to_timeline(file_path, timeline, start_time, end_time)

        # Sort by timestamp
        timeline.events.sort(key=lambda x: x.get("timestamp", datetime.min))

        return timeline

    def _add_file_to_timeline(
        self,
        path: Path,
        timeline: ForensicTimeline,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ):
        """Add file events to timeline."""
        try:
            stat_info = path.stat()

            events = [
                ("created", datetime.fromtimestamp(stat_info.st_ctime)),
                ("modified", datetime.fromtimestamp(stat_info.st_mtime)),
                ("accessed", datetime.fromtimestamp(stat_info.st_atime)),
            ]

            for event_type, timestamp in events:
                # Apply time filters
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                timeline.events.append({
                    "timestamp": timestamp,
                    "type": event_type,
                    "artifact_type": ArtifactType.FILE.value,
                    "path": str(path),
                    "name": path.name,
                })

        except Exception as e:
            logger.debug(f"Timeline error for {path}: {e}")

    # ========== Evidence Collection ==========

    def collect_evidence(
        self,
        paths: List[str],
        case_name: str,
    ) -> str:
        """
        Collect and hash evidence files.

        Args:
            paths: Paths to collect
            case_name: Case identifier

        Returns:
            Path to evidence collection
        """
        import shutil

        case_dir = self.evidence_dir / case_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        case_dir.mkdir(parents=True, exist_ok=True)

        manifest = []

        for path_str in paths:
            path = Path(path_str)

            if not path.exists():
                continue

            try:
                # Copy file
                dest = case_dir / path.name
                shutil.copy2(path, dest)

                # Calculate hash
                _, sha256 = self._calculate_hashes(dest)

                manifest.append({
                    "original_path": str(path.absolute()),
                    "collected_path": str(dest),
                    "sha256": sha256,
                    "collected_at": datetime.now().isoformat(),
                })

            except Exception as e:
                logger.error(f"Evidence collection error: {e}")

        # Write manifest
        manifest_path = case_dir / "manifest.json"
        import json
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info(f"Evidence collected: {len(manifest)} files in {case_dir}")

        return str(case_dir)

    def scan_for_artifacts(
        self,
        path: str,
        artifact_type: ArtifactType = ArtifactType.FILE,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Scan for forensic artifacts.

        Args:
            path: Path to scan
            artifact_type: Type of artifacts to find

        Yields:
            Found artifacts
        """
        scan_path = Path(path)

        if artifact_type == ArtifactType.FILE:
            for file_path in scan_path.rglob("*"):
                if file_path.is_file():
                    try:
                        artifact = self.analyze_file(str(file_path))
                        if artifact.suspicious_indicators:
                            yield {
                                "type": "file",
                                "path": artifact.path,
                                "indicators": artifact.suspicious_indicators,
                                "md5": artifact.md5,
                                "sha256": artifact.sha256,
                            }
                    except Exception:
                        continue

        elif artifact_type == ArtifactType.PERSISTENCE:
            # Check common persistence locations
            persistence_locations = self._get_persistence_locations()
            for loc in persistence_locations:
                loc_path = Path(loc)
                if loc_path.exists():
                    if loc_path.is_file():
                        yield {
                            "type": "persistence",
                            "location": str(loc_path),
                            "category": "startup",
                        }
                    elif loc_path.is_dir():
                        for item in loc_path.iterdir():
                            yield {
                                "type": "persistence",
                                "location": str(item),
                                "category": "startup",
                            }

    def _get_persistence_locations(self) -> List[str]:
        """Get OS-specific persistence locations."""
        if self.platform == "windows":
            user_home = os.environ.get("USERPROFILE", "")
            return [
                rf"{user_home}\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup",
                r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\StartUp",
                r"C:\Windows\System32\Tasks",
            ]
        else:
            user_home = os.environ.get("HOME", "")
            return [
                f"{user_home}/.config/autostart",
                "/etc/init.d",
                "/etc/systemd/system",
                "/etc/cron.d",
                "/var/spool/cron",
            ]


# Global instance
forensics_toolkit = ForensicsToolkit()
