"""
Nomad API Client for atropos-agent.

Provides a simple async client for interacting with the Nomad HTTP API:
- Submit/stop jobs
- Query allocations
- Get allocation addresses
- Scale jobs up/down
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


class AllocationStatus(Enum):
    """Nomad allocation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    LOST = "lost"


@dataclass
class Allocation:
    """Information about a Nomad allocation."""
    id: str
    job_id: str
    task_group: str
    node_id: str
    status: AllocationStatus
    # Network info for reaching the allocation
    address: Optional[str] = None
    port: Optional[int] = None
    
    @property
    def http_address(self) -> Optional[str]:
        """Get full HTTP address for the allocation."""
        if self.address and self.port:
            return f"http://{self.address}:{self.port}"
        return None


@dataclass
class JobStatus:
    """Status of a Nomad job."""
    id: str
    name: str
    status: str
    allocations: List[Allocation] = field(default_factory=list)
    count: int = 0  # Number of task groups


class NomadClient:
    """
    Async client for Nomad HTTP API.
    
    Usage:
        client = NomadClient(address="http://localhost:4646")
        
        # Submit a job
        await client.submit_job(job_spec)
        
        # Get allocations
        allocs = await client.get_job_allocations("sandbox-python")
        
        # Scale job
        await client.scale_job("sandbox-python", count=5)
    """
    
    def __init__(
        self,
        address: str = "http://localhost:4646",
        token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.address = address.rstrip("/")
        self.token = token or os.environ.get("NOMAD_TOKEN")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.token:
                headers["X-Nomad-Token"] = self.token
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers,
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to Nomad API."""
        session = await self._get_session()
        url = f"{self.address}{path}"
        
        try:
            async with session.request(method, url, json=data) as response:
                if response.status == 404:
                    return {"error": "not_found", "status": 404}
                
                text = await response.text()
                if not text:
                    return {"status": response.status}
                
                try:
                    result = json.loads(text)
                except json.JSONDecodeError:
                    return {"text": text, "status": response.status}
                
                if response.status >= 400:
                    return {"error": result, "status": response.status}
                
                return result if isinstance(result, dict) else {"data": result, "status": response.status}
                
        except aiohttp.ClientError as e:
            return {"error": str(e), "status": 0}
    
    # Job Operations
    
    async def submit_job(self, job_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a job to Nomad.
        
        Args:
            job_spec: Job specification dict (HCL converted to JSON)
            
        Returns:
            Response with EvalID if successful
        """
        return await self._request("POST", "/v1/jobs", {"Job": job_spec})
    
    async def stop_job(self, job_id: str, purge: bool = False) -> Dict[str, Any]:
        """
        Stop (and optionally purge) a job.
        
        Args:
            job_id: Job identifier
            purge: If True, completely remove the job
        """
        path = f"/v1/job/{job_id}"
        if purge:
            path += "?purge=true"
        return await self._request("DELETE", path)
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details."""
        result = await self._request("GET", f"/v1/job/{job_id}")
        if "error" in result and result.get("status") == 404:
            return None
        return result
    
    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status with allocations."""
        job = await self.get_job(job_id)
        if not job:
            return None
        
        allocs = await self.get_job_allocations(job_id)
        
        # Get count from task groups
        count = 0
        task_groups = job.get("TaskGroups", [])
        for tg in task_groups:
            count += tg.get("Count", 1)
        
        return JobStatus(
            id=job_id,
            name=job.get("Name", job_id),
            status=job.get("Status", "unknown"),
            allocations=allocs,
            count=count,
        )
    
    # Allocation Operations
    
    async def get_job_allocations(self, job_id: str) -> List[Allocation]:
        """Get all allocations for a job."""
        result = await self._request("GET", f"/v1/job/{job_id}/allocations")
        
        if "error" in result:
            return []
        
        allocs_data = result.get("data", result) if isinstance(result, dict) else result
        if not isinstance(allocs_data, list):
            return []
        
        allocations = []
        for alloc_data in allocs_data:
            # Parse allocation info
            alloc_id = alloc_data.get("ID", "")
            status_str = alloc_data.get("ClientStatus", "unknown")
            
            try:
                status = AllocationStatus(status_str)
            except ValueError:
                status = AllocationStatus.PENDING
            
            # Get network info - need to fetch detailed allocation for this
            address = None
            port = None
            
            # First try the summary data
            resources = alloc_data.get("AllocatedResources") or {}
            shared = resources.get("Shared") or {}
            networks = shared.get("Networks") or []
            
            # If no networks in summary, fetch detailed allocation
            if not networks and alloc_id:
                detailed = await self.get_allocation(alloc_id)
                if detailed:
                    resources = detailed.get("AllocatedResources") or {}
                    shared = resources.get("Shared") or {}
                    networks = shared.get("Networks") or []
            
            if networks:
                network = networks[0]
                address = network.get("IP")
                # Look for dynamic ports OR reserved ports (Singularity/raw_exec uses reserved)
                dyn_ports = network.get("DynamicPorts") or []
                reserved_ports = network.get("ReservedPorts") or []
                for dp in dyn_ports + reserved_ports:
                    if dp.get("Label") == "http":
                        port = dp.get("Value")
                        break
            
            allocations.append(Allocation(
                id=alloc_id,
                job_id=job_id,
                task_group=alloc_data.get("TaskGroup", ""),
                node_id=alloc_data.get("NodeID", ""),
                status=status,
                address=address,
                port=port,
            ))
        
        return allocations
    
    async def get_allocation(self, alloc_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed allocation info."""
        result = await self._request("GET", f"/v1/allocation/{alloc_id}")
        if "error" in result and result.get("status") == 404:
            return None
        return result
    
    # Scaling Operations
    
    async def scale_job(self, job_id: str, count: int, task_group: str = "sandbox") -> Dict[str, Any]:
        """
        Scale a job's task group to specified count.
        
        Args:
            job_id: Job identifier
            count: Desired number of allocations
            task_group: Name of task group to scale
        """
        payload = {
            "Count": count,
            "Target": {
                "Group": task_group,
            },
        }
        return await self._request("POST", f"/v1/job/{job_id}/scale", payload)
    
    async def get_job_scale_status(self, job_id: str) -> Dict[str, int]:
        """
        Get current scale status for a job.
        
        Returns:
            Dict mapping task group name to count
        """
        result = await self._request("GET", f"/v1/job/{job_id}/scale")
        
        if "error" in result:
            return {}
        
        task_groups = result.get("TaskGroups", {})
        return {
            name: info.get("Running", 0)
            for name, info in task_groups.items()
        }
    
    # Health Check
    
    async def is_healthy(self) -> bool:
        """Check if Nomad is reachable and healthy."""
        try:
            result = await self._request("GET", "/v1/status/leader")
            return "error" not in result
        except Exception:
            return False
    
    async def get_leader(self) -> Optional[str]:
        """Get current Nomad leader address."""
        result = await self._request("GET", "/v1/status/leader")
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return None


def load_job_template(
    template_name: str = "sandbox",
    **kwargs,
) -> Dict[str, Any]:
    """
    Load and configure a job template.
    
    Args:
        template_name: Name of template (e.g., "sandbox")
        **kwargs: Template variables to substitute
        
    Returns:
        Job specification dict ready for Nomad API
    """
    # Default job template for sandbox container
    if template_name == "sandbox":
        return create_sandbox_job(**kwargs)
    else:
        raise ValueError(f"Unknown template: {template_name}")


def create_sandbox_job(
    job_id: str = "atropos-sandbox",
    image: str = "atropos-sandbox:local",  # Use :local tag to avoid registry pull
    count: int = 1,
    slots_per_container: int = 10,
    privileged: bool = False,
    cpu: int = 500,
    memory: int = 512,
    port: int = 8080,
    datacenter: str = "dc1",
    driver: str = "docker",  # "docker" or "singularity"
    singularity_image: str = None,  # Path to .sif file for singularity driver
) -> Dict[str, Any]:
    """
    Create a sandbox job specification.
    
    This job runs the sandbox_server.py inside a container,
    with the specified number of slots for agent workspaces.
    
    Args:
        job_id: Unique job identifier
        image: Docker image to use (for docker driver)
        count: Number of container instances
        slots_per_container: Number of slots per container
        privileged: Run container in privileged mode (recommended for bubblewrap)
        cpu: CPU allocation in MHz
        memory: Memory allocation in MB
        port: HTTP port for sandbox server
        datacenter: Nomad datacenter
        driver: Container driver - "docker" or "singularity"
        singularity_image: Path to .sif file (required if driver="singularity")
        
    Returns:
        Job specification dict
    """
    # Build task config based on driver
    if driver == "singularity":
        if not singularity_image:
            raise ValueError("singularity_image path required when driver='singularity'")
        
        # Use raw_exec driver to run apptainer via shell for variable expansion
        # The container binds the allocation directory for workspace persistence
        # For raw_exec, we use static port since Nomad's dynamic port mapping doesn't
        # work the same as Docker - the process runs directly on the host.
        shell_cmd = (
            f'apptainer run '
            f'--bind "$NOMAD_ALLOC_DIR/data:/data" '
            f'--pwd /app '
            f'--env PYTHONUNBUFFERED=1 '
            f'{singularity_image} '
            f'python sandbox_server.py '
            f'--port {port} '
            f'--slots {slots_per_container} '
            f'--data-dir /data'
        )
        task_config = {
            "command": "/bin/sh",
            "args": ["-c", shell_cmd],
        }
        task_driver = "raw_exec"
    else:
        # Docker driver (default)
        task_config = {
            "image": image,
            "force_pull": False,  # Use local image, don't try to pull
            "ports": ["http"],
            "privileged": privileged,
            "command": "python",
            "args": [
                "sandbox_server.py",
                "--port", str(port),
                "--slots", str(slots_per_container),
                "--data-dir", "/data",
            ],
            # Note: On Linux, you can mount persistent storage:
            # "volumes": ["${NOMAD_ALLOC_DIR}/data:/data"],
            # On macOS/Docker Desktop, skip volumes for PoC
            # (container /data is ephemeral but works for testing)
        }
        task_driver = "docker"
    
    # For Singularity/raw_exec, use static ports since the process runs directly on host.
    # For Docker, use dynamic ports with port mapping.
    if driver == "singularity":
        network_config = {
            "Mode": "host",
            "ReservedPorts": [
                {
                    "Label": "http",
                    "Value": port,
                }
            ],
        }
    else:
        network_config = {
            "Mode": "host",
            "DynamicPorts": [
                {
                    "Label": "http",
                    "To": port,
                }
            ],
        }
    
    return {
        "ID": job_id,
        "Name": job_id,
        "Type": "service",
        "Datacenters": [datacenter],
        "TaskGroups": [
            {
                "Name": "sandbox",
                "Count": count,
                # Speed up deployments and avoid Consul checks. Without this, Nomad may
                # keep an "active deployment" around for the default MinHealthyTime,
                # which blocks immediate scaling under load.
                "Update": {
                    "HealthCheck": "task_states",
                    "MinHealthyTime": 0,
                },
                "Networks": [network_config],
                "Tasks": [
                    {
                        "Name": "sandbox-server",
                        "Driver": task_driver,
                        "Config": task_config,
                        "Env": {
                            "PYTHONUNBUFFERED": "1",
                            "NOMAD_ALLOC_DIR": "${NOMAD_ALLOC_DIR}",
                        },
                        "Resources": {
                            "CPU": cpu,
                            "MemoryMB": memory,
                        },
                        # Note: Services with Checks require Consul, which we skip for the PoC
                    }
                ],
                "RestartPolicy": {
                    "Attempts": 3,
                    "Interval": 300_000_000_000,  # 5 minutes
                    "Delay": 10_000_000_000,     # 10 seconds
                    "Mode": "delay",
                },
                "ReschedulePolicy": {
                    "Attempts": 5,
                    "Interval": 3600_000_000_000,  # 1 hour
                    "Delay": 30_000_000_000,      # 30 seconds
                    "DelayFunction": "exponential",
                    "MaxDelay": 300_000_000_000,  # 5 minutes
                    "Unlimited": False,
                },
            }
        ],
    }
