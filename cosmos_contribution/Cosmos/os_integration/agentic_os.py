"""
cosmos Agentic OS Bridge - Advanced System Control.

"I am the master of my process, the captain of my kernel."

This module enables deep OS-level monitoring and control (Process management, 
System stats, and Shell automation with safety).
"""

import os
import psutil
import platform
import subprocess
from typing import List, Dict, Any, Optional
from loguru import logger

class AgenticOSBridge:
    def __init__(self):
        self.os_type = platform.system()
        
    def get_system_load(self) -> Dict[str, Any]:
        """Get CPU, Memory and Disk usage."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "boot_time": psutil.boot_time()
        }

    def list_processes(self, filter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List running processes with basic stats."""
        procs = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_info']):
            try:
                pinfo = proc.info
                if not filter_name or filter_name.lower() in pinfo['name'].lower():
                    procs.append(pinfo)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return procs[:50] # Limit to top 50

    async def execute_shell_command(self, cmd: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a shell command with strict timeout and logging."""
        logger.warning(f"OS Bridge: Executing shell command: {cmd}")
        # In a real environment, this would have a 'Safety Middleware' check
        try:
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode()
            }
        except asyncio.TimeoutError:
            return {"error": "Command timed out."}
        except Exception as e:
            return {"error": str(e)}

    def get_network_stats(self) -> Dict[str, Any]:
        """Get current network I/O and active connections."""
        return {
            "io": psutil.net_io_counters()._asdict(),
            "connections": len(psutil.net_connections())
        }

# Global Instance
agentic_os = AgenticOSBridge()
