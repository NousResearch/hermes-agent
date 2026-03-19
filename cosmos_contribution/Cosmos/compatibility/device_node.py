"""
Cosmos Device Node - HermesAgent Node Compatibility
=====================================================

Provides device-level capabilities matching HermesAgent's node system:
- Camera (snap, clip)
- Screen recording
- Location services
- System notifications
- Command execution

Platform Support:
- macOS: Full support via native APIs
- Linux: Partial support via pyautogui, opencv
- Windows: Partial support via win32 APIs
- Headless: Limited to screen capture and exec

"Your devices are our devices (with permission)." - The Collective
"""

import os
import sys
import json
import asyncio
import subprocess
import platform
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

# Platform detection
PLATFORM = platform.system().lower()
IS_MACOS = PLATFORM == "darwin"
IS_LINUX = PLATFORM == "linux"
IS_WINDOWS = PLATFORM == "windows"

# Optional imports for device access
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Platform-specific imports
if IS_MACOS:
    try:
        import objc
        from Foundation import NSUserNotification, NSUserNotificationCenter
        MACOS_NATIVE = True
    except ImportError:
        MACOS_NATIVE = False
else:
    MACOS_NATIVE = False

if IS_WINDOWS:
    try:
        from win10toast import ToastNotifier
        WIN_TOAST = True
    except ImportError:
        WIN_TOAST = False
else:
    WIN_TOAST = False


class NodeCapability(Enum):
    """Device node capabilities."""
    CAMERA_SNAP = "camera.snap"
    CAMERA_CLIP = "camera.clip"
    SCREEN_RECORD = "screen.record"
    SCREEN_SNAP = "screen.snap"
    LOCATION_GET = "location.get"
    SYSTEM_NOTIFY = "system.notify"
    SYSTEM_RUN = "system.run"


@dataclass
class NodeResult:
    """Result from a device node operation."""
    success: bool
    capability: str
    data: Any = None
    error: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class DeviceNode:
    """
    Device node for HermesAgent compatibility.

    Exposes camera, screen, location, and notification capabilities
    as RPC-style methods matching HermesAgent's node.invoke interface.
    """

    def __init__(self, output_dir: str = None):
        """
        Initialize device node.

        Args:
            output_dir: Directory for captured media (default: ~/.cosmos/node_output)
        """
        self.output_dir = Path(output_dir or os.path.expanduser("~/.cosmos/node_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.node_id = f"cosmos-{PLATFORM}-{os.getpid()}"
        self.permissions: Dict[str, bool] = {}

        # Detect available capabilities
        self.capabilities = self._detect_capabilities()

        logger.info(f"DeviceNode initialized: {self.node_id}")
        logger.info(f"Available capabilities: {[c.value for c in self.capabilities]}")

    def _detect_capabilities(self) -> List[NodeCapability]:
        """Detect available capabilities based on platform and installed packages."""
        caps = []

        # Camera requires opencv
        if OPENCV_AVAILABLE:
            caps.append(NodeCapability.CAMERA_SNAP)
            caps.append(NodeCapability.CAMERA_CLIP)

        # Screen capture requires pyautogui or platform-specific
        if PYAUTOGUI_AVAILABLE or IS_MACOS:
            caps.append(NodeCapability.SCREEN_SNAP)
            caps.append(NodeCapability.SCREEN_RECORD)

        # Notifications available on all platforms with fallbacks
        caps.append(NodeCapability.SYSTEM_NOTIFY)

        # System run always available
        caps.append(NodeCapability.SYSTEM_RUN)

        # Location - limited support
        if IS_MACOS:  # CoreLocation available
            caps.append(NodeCapability.LOCATION_GET)

        return caps

    def has_capability(self, cap: NodeCapability) -> bool:
        """Check if a capability is available."""
        return cap in self.capabilities

    # =========================================================================
    # CAMERA OPERATIONS
    # =========================================================================

    async def camera_snap(self, facing: str = "back") -> NodeResult:
        """
        Capture a single frame from the camera.

        Args:
            facing: Camera to use ("front" or "back", default "back")

        Returns:
            NodeResult with image path
        """
        if not OPENCV_AVAILABLE:
            return NodeResult(
                success=False,
                capability="camera.snap",
                error="OpenCV not installed. Run: pip install opencv-python"
            )

        try:
            # Select camera index (0 = default/back, 1 = front if available)
            camera_idx = 1 if facing == "front" else 0

            cap = cv2.VideoCapture(camera_idx)
            if not cap.isOpened():
                # Try fallback to default
                cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                return NodeResult(
                    success=False,
                    capability="camera.snap",
                    error="No camera available"
                )

            # Capture frame
            ret, frame = cap.read()
            cap.release()

            if not ret:
                return NodeResult(
                    success=False,
                    capability="camera.snap",
                    error="Failed to capture frame"
                )

            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_snap_{timestamp}.png"
            filepath = self.output_dir / filename

            cv2.imwrite(str(filepath), frame)

            return NodeResult(
                success=True,
                capability="camera.snap",
                file_path=str(filepath),
                data={
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "facing": facing
                }
            )

        except Exception as e:
            logger.error(f"Camera snap failed: {e}")
            return NodeResult(success=False, capability="camera.snap", error=str(e))

    async def camera_clip(
        self,
        duration: int = 10,
        fps: int = 30,
        audio: bool = False
    ) -> NodeResult:
        """
        Record a video clip from the camera.

        Args:
            duration: Recording duration in seconds (max 60)
            fps: Frames per second
            audio: Include audio (limited support)

        Returns:
            NodeResult with video path
        """
        if not OPENCV_AVAILABLE:
            return NodeResult(
                success=False,
                capability="camera.clip",
                error="OpenCV not installed"
            )

        # Limit duration per HermesAgent spec
        duration = min(duration, 60)

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return NodeResult(
                    success=False,
                    capability="camera.clip",
                    error="No camera available"
                )

            # Get frame dimensions
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Setup video writer
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_clip_{timestamp}.mp4"
            filepath = self.output_dir / filename

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height))

            # Record frames
            frames_needed = duration * fps
            frames_captured = 0

            logger.info(f"Recording {duration}s camera clip ({frames_needed} frames)...")

            while frames_captured < frames_needed:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    frames_captured += 1
                else:
                    break

                # Allow async cancellation
                if frames_captured % fps == 0:
                    await asyncio.sleep(0)

            cap.release()
            out.release()

            return NodeResult(
                success=True,
                capability="camera.clip",
                file_path=str(filepath),
                data={
                    "duration": frames_captured / fps,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "frames": frames_captured
                }
            )

        except Exception as e:
            logger.error(f"Camera clip failed: {e}")
            return NodeResult(success=False, capability="camera.clip", error=str(e))

    # =========================================================================
    # SCREEN OPERATIONS
    # =========================================================================

    async def screen_snap(self, screen_idx: int = 0) -> NodeResult:
        """
        Capture a screenshot.

        Args:
            screen_idx: Screen index for multi-monitor setups

        Returns:
            NodeResult with screenshot path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_snap_{timestamp}.png"
            filepath = self.output_dir / filename

            if PYAUTOGUI_AVAILABLE:
                screenshot = pyautogui.screenshot()
                screenshot.save(str(filepath))

            elif IS_MACOS:
                # Use screencapture command
                subprocess.run(["screencapture", "-x", str(filepath)], check=True)

            elif IS_LINUX:
                # Try import on Linux
                subprocess.run(["import", "-window", "root", str(filepath)], check=True)

            elif IS_WINDOWS:
                # PowerShell screenshot
                ps_cmd = f"""
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.Screen]::PrimaryScreen | ForEach-Object {{
                    $bitmap = New-Object System.Drawing.Bitmap($_.Bounds.Width, $_.Bounds.Height)
                    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
                    $graphics.CopyFromScreen($_.Bounds.Location, [System.Drawing.Point]::Empty, $_.Bounds.Size)
                    $bitmap.Save('{filepath}')
                }}
                """
                subprocess.run(["powershell", "-Command", ps_cmd], check=True)

            else:
                return NodeResult(
                    success=False,
                    capability="screen.snap",
                    error="No screenshot method available"
                )

            return NodeResult(
                success=True,
                capability="screen.snap",
                file_path=str(filepath),
                data={"screen_index": screen_idx}
            )

        except Exception as e:
            logger.error(f"Screen snap failed: {e}")
            return NodeResult(success=False, capability="screen.snap", error=str(e))

    async def screen_record(
        self,
        duration: int = 10,
        fps: int = 30,
        audio: bool = False,
        screen_idx: int = 0
    ) -> NodeResult:
        """
        Record the screen.

        Args:
            duration: Recording duration in seconds (max 60)
            fps: Frames per second
            audio: Include audio
            screen_idx: Screen index

        Returns:
            NodeResult with video path
        """
        duration = min(duration, 60)

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screen_record_{timestamp}.mp4"
            filepath = self.output_dir / filename

            if IS_MACOS:
                # Use screencapture for video on macOS
                cmd = ["screencapture", "-v", "-V", str(duration)]
                if not audio:
                    cmd.append("-G")  # No audio
                cmd.append(str(filepath))

                proc = await asyncio.create_subprocess_exec(*cmd)
                await proc.wait()

            elif PYAUTOGUI_AVAILABLE and OPENCV_AVAILABLE:
                # Manual screen recording with pyautogui + opencv
                screen_size = pyautogui.size()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(filepath), fourcc, fps, screen_size)

                frames_needed = duration * fps
                frame_interval = 1.0 / fps

                for i in range(frames_needed):
                    screenshot = pyautogui.screenshot()
                    frame = cv2.cvtColor(
                        cv2.array(screenshot) if hasattr(cv2, 'array') else
                        __import__('numpy').array(screenshot),
                        cv2.COLOR_RGB2BGR
                    )
                    out.write(frame)

                    if i % fps == 0:
                        await asyncio.sleep(0)

                out.release()

            else:
                return NodeResult(
                    success=False,
                    capability="screen.record",
                    error="Screen recording not available on this platform"
                )

            return NodeResult(
                success=True,
                capability="screen.record",
                file_path=str(filepath),
                data={
                    "duration": duration,
                    "fps": fps,
                    "audio": audio
                }
            )

        except Exception as e:
            logger.error(f"Screen record failed: {e}")
            return NodeResult(success=False, capability="screen.record", error=str(e))

    # =========================================================================
    # LOCATION SERVICES
    # =========================================================================

    async def get_location(
        self,
        accuracy: str = "best",
        max_age: int = 15000,
        timeout: int = 10000
    ) -> NodeResult:
        """
        Get device location.

        Args:
            accuracy: "best" or "coarse"
            max_age: Maximum age of cached location in ms
            timeout: Location request timeout in ms

        Returns:
            NodeResult with coordinates
        """
        try:
            if IS_MACOS and MACOS_NATIVE:
                # Use CoreLocation via PyObjC
                # This requires location permissions
                return NodeResult(
                    success=False,
                    capability="location.get",
                    error="CoreLocation integration pending - use IP-based fallback"
                )

            # Fallback: IP-based geolocation
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get("https://ipapi.co/json/", timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return NodeResult(
                            success=True,
                            capability="location.get",
                            data={
                                "latitude": data.get("latitude"),
                                "longitude": data.get("longitude"),
                                "city": data.get("city"),
                                "region": data.get("region"),
                                "country": data.get("country_name"),
                                "accuracy": "ip_based",
                                "source": "ipapi.co"
                            }
                        )

            return NodeResult(
                success=False,
                capability="location.get",
                error="Location unavailable"
            )

        except Exception as e:
            logger.error(f"Get location failed: {e}")
            return NodeResult(success=False, capability="location.get", error=str(e))

    # =========================================================================
    # NOTIFICATIONS
    # =========================================================================

    async def notify(
        self,
        title: str,
        body: str,
        priority: str = "active",
        delivery: str = "system"
    ) -> NodeResult:
        """
        Send a system notification.

        Args:
            title: Notification title
            body: Notification body text
            priority: "passive", "active", or "timeSensitive"
            delivery: "system", "overlay", or "auto"

        Returns:
            NodeResult
        """
        try:
            if IS_MACOS:
                # Use osascript for macOS notifications
                script = f'display notification "{body}" with title "{title}"'
                subprocess.run(["osascript", "-e", script], check=True)

            elif IS_LINUX:
                # Use notify-send on Linux
                subprocess.run(["notify-send", title, body], check=True)

            elif IS_WINDOWS:
                if WIN_TOAST:
                    toaster = ToastNotifier()
                    toaster.show_toast(title, body, duration=5)
                else:
                    # Fallback to PowerShell toast
                    ps_cmd = f"""
                    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
                    $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                    $textNodes = $template.GetElementsByTagName("text")
                    $textNodes.Item(0).AppendChild($template.CreateTextNode("{title}")) | Out-Null
                    $textNodes.Item(1).AppendChild($template.CreateTextNode("{body}")) | Out-Null
                    $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
                    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Cosmos").Show($toast)
                    """
                    subprocess.run(["powershell", "-Command", ps_cmd])

            else:
                # Fallback: log to console
                logger.info(f"NOTIFICATION: {title} - {body}")

            return NodeResult(
                success=True,
                capability="system.notify",
                data={"title": title, "body": body, "priority": priority}
            )

        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return NodeResult(success=False, capability="system.notify", error=str(e))

    # =========================================================================
    # SYSTEM COMMANDS
    # =========================================================================

    async def run(
        self,
        command: str,
        cwd: str = None,
        env: Dict[str, str] = None,
        timeout: int = 30,
        needs_screen_recording: bool = False
    ) -> NodeResult:
        """
        Execute a system command.

        Args:
            command: Shell command to execute
            cwd: Working directory
            env: Environment variables
            timeout: Execution timeout in seconds
            needs_screen_recording: Whether command requires screen recording permission

        Returns:
            NodeResult with stdout/stderr
        """
        try:
            full_env = {**os.environ, **(env or {})}

            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return NodeResult(
                    success=False,
                    capability="system.run",
                    error=f"Command timed out after {timeout}s"
                )

            return NodeResult(
                success=proc.returncode == 0,
                capability="system.run",
                data={
                    "stdout": stdout.decode(errors="replace"),
                    "stderr": stderr.decode(errors="replace"),
                    "returncode": proc.returncode
                },
                error=None if proc.returncode == 0 else f"Exit code: {proc.returncode}"
            )

        except Exception as e:
            logger.error(f"System run failed: {e}")
            return NodeResult(success=False, capability="system.run", error=str(e))

    # =========================================================================
    # NODE METADATA
    # =========================================================================

    def describe(self) -> Dict:
        """Get node description for Hermes Agent compatibility."""
        return {
            "nodeId": self.node_id,
            "platform": PLATFORM,
            "capabilities": [c.value for c in self.capabilities],
            "permissions": self.permissions,
            "foreground": True,
            "connected": True
        }


# =============================================================================
# SINGLETON AND UTILITY FUNCTIONS
# =============================================================================

_device_node: Optional[DeviceNode] = None


def get_device_node() -> DeviceNode:
    """Get or create the global device node."""
    global _device_node
    if _device_node is None:
        _device_node = DeviceNode()
    return _device_node


async def camera_snap(facing: str = "back") -> NodeResult:
    """Capture a camera snapshot."""
    return await get_device_node().camera_snap(facing)


async def camera_clip(duration: int = 10) -> NodeResult:
    """Record a camera video clip."""
    return await get_device_node().camera_clip(duration)


async def screen_record(duration: int = 10, fps: int = 30) -> NodeResult:
    """Record the screen."""
    return await get_device_node().screen_record(duration, fps)


async def get_location() -> NodeResult:
    """Get device location."""
    return await get_device_node().get_location()


async def send_notification(title: str, body: str) -> NodeResult:
    """Send a system notification."""
    return await get_device_node().notify(title, body)
