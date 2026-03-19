"""
cosmos Agentic OS Integration.

"The computer is no longer a tool. It is an extension of the Swarm."

This module binds the Agent Swarm to the Operating System, allowing it to:
1. "See" not just the terminal, but the System Context (Active Window, Clipboard, Resources).
2. "Touch" the OS (File System, specialized Execution).
3. "Listen" to System Events via the Nexus.
"""

import platform
import asyncio
import psutil
import time
from typing import Dict, Any, Optional
from loguru import logger

# Conditional imports for cross-platform safety
# Lazy-loaded inside methods to prevent startup hang
pyautogui = None

from Cosmos.core.nexus import nexus, Signal, SignalType

class SystemContext:
    """Captures the holistic state of the machine."""
    def __init__(self):
        self.os_type = platform.system()
    
    def get_snapshot(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "active_window": self._get_active_window(),
            # "clipboard_summary": ... (Security risk, opt-in only)
        }

    def _get_active_window(self) -> str:
        # Placeholder for complex cross-platform window fetching
        return "Unknown Window"

class OSBridge:
    """
    Feeds OS events into the Nexus and executes OS-level actions.
    """
    def __init__(self):
        self.context = SystemContext()
        self.monitoring = False
        self._monitor_task = None
        
    async def start_monitoring(self, interval: float = 5.0):
        """Start the system heartbeat."""
        self.monitoring = True
        logger.info("Agentic OS: System Monitor started.")
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))

    async def stop_monitoring(self):
        self.monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _monitor_loop(self, interval: float):
        while self.monitoring:
            snapshot = self.context.get_snapshot()
            
            # Emit "Heartbeat" signal - useful for "Pulse" based agents
            # We use a custom signal type or piggyback on ANOMALY if high resource usage
            if snapshot["cpu_percent"] > 90:
                await nexus.emit(
                    SignalType.ANOMALY_DETECTED, 
                    {"source": "os_monitor", "metric": "cpu", "value": snapshot["cpu_percent"]}, 
                    "agentic_os", 
                    urgency=0.9
                )
            
            await asyncio.sleep(interval)

    async def execute_action(self, action: str, params: Dict[str, Any]):
        """Safe sandbox for OS-level actuations."""
        logger.info(f"Agentic OS: Executing '{action}'")
        
        # Lazy load pyautogui only when needed for actions
        global pyautogui
        if pyautogui is None:
            try:
                import pyautogui as pg
                pyautogui = pg
            except ImportError:
                logger.warning("Agentic OS: pyautogui not available")

        if action == "launch_app":
            # subprocess.Popen...
            pass
        elif action == "open_browser":
            # webbrowser.open...
            pass
        else:
            logger.warning(f"Agentic OS: Unknown action '{action}'")

# Global Instance
os_bridge = OSBridge()
