from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, ClassVar

from ..slots.executor import ExecutionResult
from ..slots.slot import Slot, SlotState
from .base import ToolBackend

import yaml

@dataclass
class ModalSandboxConfig:
    """
    Unified configuration for Modal sandbox pools.
    
    This single class handles both profile definitions and runtime configuration.
    Use `with_app_name()` to set the full app_name for a specific deployment.
    
    Example profiles:
        - "default": Basic Python environment, CPU only
        - "pytorch-gpu": PyTorch with T4 GPU for training
        - "high-memory": 64GB RAM for large model inference
    
    Usage:
        # Single profile
        config = ModalSandboxConfig(name="default")
        config = config.with_app_name("my-training")
        backend = ModalToolBackend(config)
        
        # Multi-profile
        profiles = ModalSandboxConfig.load_profiles()
        backend = ModalToolBackend.with_profiles(profiles=profiles)
    """
    # Identity
    name: str = "default"
    app_name: Optional[str] = None  # Full app name (computed via with_app_name)
    
    # Container image
    image: str = "python:3.11"
    
    # Resource allocation
    gpu: Optional[str] = None  # None, "T4", "A10G", "A100", "H100"
    cpu: float = 1.0
    memory: int = 2048  # MB
    
    # Pool sizing (slot-based multiplexing)
    slots_per_sandbox: int = 10
    min_sandboxes: int = 1
    max_sandboxes: int = 5
    
    # Timeouts
    idle_timeout: int = 120  # Modal server-side auto-cleanup
    max_lifetime: int = 3600  # Max sandbox lifetime
    acquire_timeout_s: float = 60.0  # Timeout waiting for slot
    execution_timeout_s: float = 30.0  # Default command timeout
    
    # Credentials
    secrets: List[str] = field(default_factory=list)  # Modal Secret names
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    # Working directory
    workspace_base: str = "/data"
    
    def with_app_name(self, base_app: str) -> "ModalSandboxConfig":
        """Return a copy with computed app_name: {base_app}-{name}."""
        return replace(self, app_name=f"{base_app}-{self.name}")
    
    def get_app_name(self, fallback: str = "atropos-sandbox") -> str:
        """Get app_name, using fallback if not set."""
        return self.app_name or f"{fallback}-{self.name}"
    
    # -------------------------------------------------------------------------
    # Loading methods
    # -------------------------------------------------------------------------
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ModalSandboxConfig":
        """Create config from dictionary (e.g., from YAML)."""
        return cls(
            name=name,
            app_name=data.get("app_name"),
            image=data.get("image", "python:3.11"),
            gpu=data.get("gpu"),
            cpu=float(data.get("cpu", 1.0)),
            memory=int(data.get("memory", 2048)),
            slots_per_sandbox=int(data.get("slots_per_sandbox", 10)),
            min_sandboxes=int(data.get("min_sandboxes", 1)),
            max_sandboxes=int(data.get("max_sandboxes", 5)),
            idle_timeout=int(data.get("idle_timeout", 120)),
            max_lifetime=int(data.get("max_lifetime", 3600)),
            acquire_timeout_s=float(data.get("acquire_timeout_s", 60.0)),
            execution_timeout_s=float(data.get("execution_timeout_s", 30.0)),
            secrets=list(data.get("secrets", [])),
            env_vars=dict(data.get("env_vars", {})),
            workspace_base=data.get("workspace_base", "/data"),
        )
    
    @classmethod
    def from_env(cls, profile_name: str = "default") -> "ModalSandboxConfig":
        """Create config from environment variables."""
        prefix = f"ATROPOS_MODAL_PROFILE_{profile_name.upper().replace('-', '_')}_"
        
        def get_env(key: str, default: Any) -> str:
            return os.environ.get(f"{prefix}{key}", os.environ.get(f"ATROPOS_MODAL_{key}", str(default)))
        
        secrets = []
        secrets_str = get_env("SECRETS", "")
        if secrets_str:
            secrets = [s.strip() for s in secrets_str.split(",") if s.strip()]
        
        env_vars = {}
        env_vars_str = get_env("ENV_VARS", "")
        if env_vars_str:
            for pair in env_vars_str.split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    env_vars[k.strip()] = v.strip()
        
        return cls(
            name=profile_name,
            image=get_env("IMAGE", "python:3.11"),
            gpu=get_env("GPU", "") or None,
            cpu=float(get_env("CPU", 1.0)),
            memory=int(get_env("MEMORY", 2048)),
            slots_per_sandbox=int(get_env("SLOTS_PER_SANDBOX", 10)),
            min_sandboxes=int(get_env("MIN_SANDBOXES", 1)),
            max_sandboxes=int(get_env("MAX_SANDBOXES", 5)),
            idle_timeout=int(get_env("IDLE_TIMEOUT", 120)),
            max_lifetime=int(get_env("MAX_LIFETIME", 3600)),
            acquire_timeout_s=float(get_env("ACQUIRE_TIMEOUT", 60.0)),
            execution_timeout_s=float(get_env("EXECUTION_TIMEOUT", 30.0)),
            secrets=secrets,
            env_vars=env_vars,
            workspace_base=get_env("WORKSPACE_BASE", "/data"),
        )
    
    @classmethod
    def from_agent_env_config(cls, cfg: Any) -> "ModalSandboxConfig":
        """Create config from AgentEnv configuration object."""
        secrets = []
        secrets_str = str(getattr(cfg, "modal_secrets", ""))
        if secrets_str:
            secrets = [s.strip() for s in secrets_str.split(",") if s.strip()]
        
        env_vars = {}
        env_vars_str = str(getattr(cfg, "modal_env_vars", ""))
        if env_vars_str:
            for pair in env_vars_str.split(";"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    env_vars[k.strip()] = v.strip()
        
        return cls(
            name="default",
            app_name=str(getattr(cfg, "modal_app_name", None)) or None,
            image=str(getattr(cfg, "modal_image", "python:3.11")),
            gpu=getattr(cfg, "modal_gpu", None) or None,
            cpu=float(getattr(cfg, "modal_cpu", 1.0)),
            memory=int(getattr(cfg, "modal_memory", 2048)),
            slots_per_sandbox=int(getattr(cfg, "modal_slots_per_sandbox", 10)),
            min_sandboxes=int(getattr(cfg, "modal_min_sandboxes", 1)),
            max_sandboxes=int(getattr(cfg, "modal_max_sandboxes", 5)),
            idle_timeout=int(getattr(cfg, "modal_idle_timeout", 120)),
            max_lifetime=int(getattr(cfg, "modal_max_lifetime", 3600)),
            acquire_timeout_s=float(getattr(cfg, "modal_acquire_timeout", 60.0)),
            execution_timeout_s=float(getattr(cfg, "modal_execution_timeout", 30.0)),
            secrets=secrets,
            env_vars=env_vars,
            workspace_base=str(getattr(cfg, "modal_workspace_base", "/data")),
        )
    
    @classmethod
    def load_profiles(cls, config_file: Optional[str] = None) -> Dict[str, "ModalSandboxConfig"]:
        """
        Load profiles from YAML file or environment variables.
        
        Priority:
        1. Specified config file
        2. ATROPOS_MODAL_PROFILES_FILE env var
        3. Default profiles from env vars
        """
        profiles: Dict[str, ModalSandboxConfig] = {}
        
        # Try loading from YAML
        config_path = config_file or os.environ.get("ATROPOS_MODAL_PROFILES_FILE")
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
                
                if data and "profiles" in data:
                    for name, profile_data in data["profiles"].items():
                        profiles[name] = cls.from_dict(name, profile_data)
                    
                    print(f"[Modal] Loaded {len(profiles)} profile(s) from {config_path}")
                    return profiles
            except Exception as e:
                print(f"[Modal] Warning: Could not load profiles from {config_path}: {e}")
        
        # Load from environment variables
        profile_names_str = os.environ.get("ATROPOS_MODAL_PROFILES", "default")
        profile_names = [p.strip() for p in profile_names_str.split(",") if p.strip()]
        
        for name in profile_names:
            profiles[name] = cls.from_env(name)
        
        # Always ensure "default" profile exists
        if "default" not in profiles:
            profiles["default"] = cls.from_env("default")
        
        return profiles


class _ModalSandboxWithSlots:
    """
    A Modal sandbox hosting multiple slots (isolated workspaces).
    
    Each slot has its own workspace directory for filesystem isolation.
    Multiple trajectories can run in the same sandbox via different slots.
    """
    
    def __init__(
        self,
        sandbox: Any,  # modal.Sandbox
        sandbox_id: str,
        config: ModalSandboxConfig,
    ):
        self.sandbox = sandbox
        self.sandbox_id = sandbox_id
        self.config = config
        self.slots: Dict[str, Slot] = {}
        self._lock = asyncio.Lock()
        
        # Create slots
        for i in range(config.slots_per_sandbox):
            slot_id = f"{sandbox_id}_slot_{i}"
            workspace_dir = f"{config.workspace_base}/{slot_id}"
            self.slots[slot_id] = Slot(
                slot_id=slot_id,
                alloc_id=sandbox_id,
                container_addr=f"modal://{sandbox_id}",  # Virtual address
                workspace_dir=workspace_dir,
                state=SlotState.AVAILABLE,
            )
    
    async def initialize_workspaces(self):
        """Create workspace directories for all slots."""
        try:
            # Create all workspace directories in parallel
            commands = [f"mkdir -p {slot.workspace_dir}" for slot in self.slots.values()]
            combined_cmd = " && ".join(commands)
            
            process = self.sandbox.exec("bash", "-c", combined_cmd, timeout=30)
            process.wait()
            
        except Exception as e:
            print(f"[Modal] Warning: Could not initialize workspaces: {e}")
    
    async def acquire_slot(self, trajectory_id: Optional[str] = None) -> Optional[Slot]:
        """Acquire an available slot."""
        async with self._lock:
            for slot in self.slots.values():
                if slot.is_available:
                    slot.acquire(trajectory_id)
                    return slot
            return None
    
    async def release_slot(self, slot: Slot, reset_workspace: bool = False):
        """Release a slot back to available."""
        async with self._lock:
            if slot.slot_id in self.slots:
                if reset_workspace:
                    await self._reset_workspace(slot)
                slot.release()
    
    async def _reset_workspace(self, slot: Slot):
        """Reset a slot's workspace (delete all files)."""
        try:
            cmd = f"rm -rf {slot.workspace_dir}/* {slot.workspace_dir}/.[!.]* 2>/dev/null || true"
            process = self.sandbox.exec("bash", "-c", cmd, timeout=30)
            process.wait()
        except Exception as e:
            print(f"[Modal] Warning: Could not reset workspace {slot.slot_id}: {e}")
    
    async def execute(
        self,
        slot: Slot,
        tool_name: str,
        args: Dict[str, Any],
        timeout: float = 30.0,
    ) -> ExecutionResult:
        """Execute a tool in a slot's workspace."""
        execution_id = str(uuid.uuid4())
        
        try:
            # Mark slot as executing
            if slot.state == SlotState.ACQUIRED:
                slot.start_execution(execution_id)
            
            # Build command based on tool type
            if tool_name == "bash":
                command = args.get("command", "")
            elif tool_name == "read_file":
                path = args.get("path", "")
                full_path = f"{slot.workspace_dir}/{path}" if not path.startswith("/") else path
                command = f"cat {full_path}"
            elif tool_name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                full_path = f"{slot.workspace_dir}/{path}" if not path.startswith("/") else path
                # Escape content for shell
                escaped_content = content.replace("'", "'\\''")
                command = f"mkdir -p $(dirname {full_path}) && printf '%s' '{escaped_content}' > {full_path}"
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown tool: {tool_name}",
                    execution_id=execution_id,
                    slot_id=slot.slot_id,
                )
            
            # Execute in workspace directory
            full_command = f"cd {slot.workspace_dir} && {command}"
            
            process = self.sandbox.exec(
                "bash", "-c", full_command,
                timeout=int(timeout),
            )
            
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            process.wait()
            
            output = stdout
            if stderr:
                output = f"{stdout}\n{stderr}" if stdout else stderr
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=output,
                error="" if process.returncode == 0 else f"Exit code: {process.returncode}",
                execution_id=execution_id,
                slot_id=slot.slot_id,
                metadata={"returncode": process.returncode},
            )
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                error_msg = f"Command timed out after {timeout}s"
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_id=execution_id,
                slot_id=slot.slot_id,
            )
        finally:
            if slot.state == SlotState.EXECUTING:
                slot.end_execution()
    
    def available_slots(self) -> int:
        """Count available slots."""
        return sum(1 for slot in self.slots.values() if slot.is_available)
    
    def is_healthy(self) -> bool:
        """Check if sandbox is still running."""
        try:
            return self.sandbox.poll() is None
        except Exception:
            return False
    
    def terminate(self):
        """Terminate this sandbox."""
        try:
            self.sandbox.terminate()
        except Exception:
            pass


class _ModalSandboxPool:
    """
    Pool of Modal sandboxes with slot-based multiplexing.
    
    Manages multiple sandboxes, each hosting multiple slots.
    Provides acquire/release semantics for slots.
    Auto-scales sandboxes based on demand.
    """
    
    def __init__(self, config: ModalSandboxConfig):
        self.config = config
        self._sandboxes: Dict[str, _ModalSandboxWithSlots] = {}
        self._lock = asyncio.Lock()
        self._app = None
        self._image = None
        self._started = False
        self._next_sandbox_idx = 0
    
    async def start(self):
        """Initialize Modal app and create minimum sandboxes."""
        if self._started:
            return
        
        try:
            import modal
            
            app_name = self.config.get_app_name()
            self._app = modal.App.lookup(app_name, create_if_missing=True)
            self._image = modal.Image.from_registry(self.config.image)
            
            # Create minimum sandboxes
            for _ in range(self.config.min_sandboxes):
                await self._create_sandbox()
            
            self._started = True
            print(f"[Modal] Pool started with {len(self._sandboxes)} sandbox(es), "
                  f"{self.config.slots_per_sandbox} slots each")
            
        except ImportError:
            raise ImportError("Modal package not installed. Run: pip install modal")
    
    async def stop(self, purge: bool = False):
        """Stop all sandboxes."""
        async with self._lock:
            for sandbox_wrapper in self._sandboxes.values():
                sandbox_wrapper.terminate()
            
            if purge:
                self._sandboxes.clear()
            
            self._started = False
            print(f"[Modal] Pool stopped")
    
    async def _create_sandbox(self) -> _ModalSandboxWithSlots:
        """Create a new sandbox with slots."""
        import modal
        
        sandbox_id = f"sandbox_{self._next_sandbox_idx}"
        self._next_sandbox_idx += 1
        app_name = self.config.get_app_name()
        
        # Build secrets list
        secrets_list = []
        for secret_name in self.config.secrets:
            try:
                secrets_list.append(modal.Secret.from_name(secret_name))
            except Exception as e:
                print(f"[Modal] Warning: Could not load secret '{secret_name}': {e}")
        
        # Add env_vars as a programmatic secret
        if self.config.env_vars:
            secrets_list.append(modal.Secret.from_dict(self.config.env_vars))
        
        # Build create kwargs
        create_kwargs = {
            "app": self._app,
            "name": f"{app_name}-{sandbox_id}",
            "image": self._image,
            "timeout": self.config.max_lifetime,
            "idle_timeout": self.config.idle_timeout,
            "workdir": self.config.workspace_base,
        }
        
        if self.config.cpu != 1.0:
            create_kwargs["cpu"] = self.config.cpu
        if self.config.memory != 2048:
            create_kwargs["memory"] = self.config.memory
        if self.config.gpu:
            create_kwargs["gpu"] = self.config.gpu
        if secrets_list:
            create_kwargs["secrets"] = secrets_list
        
        # Try to recover existing sandbox or create new
        try:
            sandbox = modal.Sandbox.from_name(
                app_name, 
                f"{app_name}-{sandbox_id}"
            )
            if sandbox.poll() is None:
                print(f"[Modal] Recovered existing sandbox: {sandbox_id}")
            else:
                sandbox = modal.Sandbox.create(**create_kwargs)
                print(f"[Modal] Created new sandbox: {sandbox_id}")
        except modal.exception.NotFoundError:
            sandbox = modal.Sandbox.create(**create_kwargs)
            print(f"[Modal] Created new sandbox: {sandbox_id}")
        
        wrapper = _ModalSandboxWithSlots(sandbox, sandbox_id, self.config)
        await wrapper.initialize_workspaces()
        
        self._sandboxes[sandbox_id] = wrapper
        return wrapper
    
    async def acquire(self, trajectory_id: Optional[str] = None) -> Slot:
        """Acquire a slot for a trajectory."""
        deadline = asyncio.get_event_loop().time() + self.config.acquire_timeout_s
        
        while True:
            async with self._lock:
                # Try to find an available slot in existing sandboxes
                for sandbox_wrapper in self._sandboxes.values():
                    if sandbox_wrapper.is_healthy():
                        slot = await sandbox_wrapper.acquire_slot(trajectory_id)
                        if slot:
                            return slot
                
                # No slots available - try to scale up
                if len(self._sandboxes) < self.config.max_sandboxes:
                    try:
                        new_sandbox = await self._create_sandbox()
                        slot = await new_sandbox.acquire_slot(trajectory_id)
                        if slot:
                            return slot
                    except Exception as e:
                        print(f"[Modal] Failed to create sandbox: {e}")
            
            # Check timeout
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"No slot available within {self.config.acquire_timeout_s}s "
                    f"(sandboxes: {len(self._sandboxes)}/{self.config.max_sandboxes})"
                )
            
            await asyncio.sleep(0.5)
    
    async def release(self, slot: Slot, reset_workspace: bool = False):
        """Release a slot back to its sandbox."""
        sandbox_id = slot.alloc_id
        
        async with self._lock:
            if sandbox_id in self._sandboxes:
                await self._sandboxes[sandbox_id].release_slot(slot, reset_workspace)
    
    async def execute(
        self,
        slot: Slot,
        tool_name: str,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a tool in a slot."""
        sandbox_id = slot.alloc_id
        exec_timeout = timeout or self.config.execution_timeout_s
        
        if sandbox_id not in self._sandboxes:
            return ExecutionResult(
                success=False,
                error=f"Sandbox {sandbox_id} not found",
                slot_id=slot.slot_id,
            )
        
        return await self._sandboxes[sandbox_id].execute(
            slot, tool_name, args, exec_timeout
        )
    
    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Execute multiple tools in parallel across slots.
        
        This is the key optimization - batched execution maximizes
        container utilization while agents wait for LLM responses.
        """
        if not requests:
            return []
        
        exec_timeout = timeout or self.config.execution_timeout_s
        
        # Execute all requests in parallel
        tasks = [
            self.execute(slot, tool_name, args, exec_timeout)
            for slot, tool_name, args in requests
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ExecutionResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                slot, tool_name, _ = requests[i]
                final_results.append(ExecutionResult(
                    success=False,
                    error=str(result),
                    slot_id=slot.slot_id,
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status."""
        total_slots = 0
        available_slots = 0
        healthy_sandboxes = 0
        
        for sandbox_wrapper in self._sandboxes.values():
            if sandbox_wrapper.is_healthy():
                healthy_sandboxes += 1
                total_slots += len(sandbox_wrapper.slots)
                available_slots += sandbox_wrapper.available_slots()
        
        return {
            "sandboxes": len(self._sandboxes),
            "healthy_sandboxes": healthy_sandboxes,
            "max_sandboxes": self.config.max_sandboxes,
            "total_slots": total_slots,
            "available_slots": available_slots,
            "slots_per_sandbox": self.config.slots_per_sandbox,
        }


class _ModalMultiProfileManager:
    """
    Manages multiple sandbox pools across different profiles.
    
    This enables heterogeneous resource allocation - different trajectory
    types can request different profiles (GPU vs CPU, etc.)
    
    Architecture:
        Manager
            ├── "default" profile → _ModalSandboxPool (CPU, 2GB)
            ├── "pytorch-gpu" profile → _ModalSandboxPool (T4, 16GB)
            └── "high-memory" profile → _ModalSandboxPool (CPU, 64GB)
    """
    
    def __init__(
        self,
        app_name: str = "atropos-sandbox",
        profiles: Optional[Dict[str, ModalSandboxConfig]] = None,
        default_profile: str = "default",
    ):
        self.app_name = app_name
        self.default_profile = default_profile
        self._profiles = profiles or ModalSandboxConfig.load_profiles()
        self._pools: Dict[str, _ModalSandboxPool] = {}
        self._slot_profile_map: Dict[str, str] = {}  # slot_id -> profile_name
        self._lock = asyncio.Lock()
        self._started = False
    
    async def start(self, profiles_to_start: Optional[List[str]] = None):
        """
        Start sandbox pools for specified profiles.
        
        Args:
            profiles_to_start: Profile names to start. If None, starts default only.
        """
        if self._started:
            return
        
        profiles = profiles_to_start or [self.default_profile]
        
        for profile_name in profiles:
            if profile_name not in self._profiles:
                print(f"[Modal] Warning: Profile '{profile_name}' not found, skipping")
                continue
            
            await self._ensure_pool(profile_name)
        
        self._started = True
        print(f"[Modal] Multi-profile manager started with {len(self._pools)} pool(s)")
    
    async def stop(self, purge: bool = False):
        """Stop all pools."""
        async with self._lock:
            for pool in self._pools.values():
                await pool.stop(purge=purge)
            
            if purge:
                self._pools.clear()
                self._slot_profile_map.clear()
            
            self._started = False
            print(f"[Modal] Multi-profile manager stopped")
    
    async def _ensure_pool(self, profile_name: str) -> _ModalSandboxPool:
        """Ensure a pool exists for the given profile, create if needed."""
        if profile_name not in self._pools:
            if profile_name not in self._profiles:
                raise ValueError(f"Unknown profile: {profile_name}")
            
            config = self._profiles[profile_name].with_app_name(self.app_name)
            
            pool = _ModalSandboxPool(config)
            await pool.start()
            
            self._pools[profile_name] = pool
            print(f"[Modal] Started pool for profile '{profile_name}'")
        
        return self._pools[profile_name]
    
    async def acquire(
        self,
        trajectory_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Slot:
        """
        Acquire a slot from the specified profile's pool.
        
        Args:
            trajectory_id: ID of the trajectory requesting the slot
            profile: Profile name to use. If None, uses default profile.
            
        Returns:
            Acquired Slot
        """
        profile_name = profile or self.default_profile
        
        async with self._lock:
            pool = await self._ensure_pool(profile_name)
        
        slot = await pool.acquire(trajectory_id)
        
        # Track which profile this slot belongs to
        self._slot_profile_map[slot.slot_id] = profile_name
        
        return slot
    
    async def release(self, slot: Slot, reset_workspace: bool = False):
        """Release a slot back to its profile's pool."""
        profile_name = self._slot_profile_map.get(slot.slot_id, self.default_profile)
        
        if profile_name in self._pools:
            await self._pools[profile_name].release(slot, reset_workspace)
        
        # Clean up mapping
        self._slot_profile_map.pop(slot.slot_id, None)
    
    async def execute(
        self,
        slot: Slot,
        tool_name: str,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a tool in a slot."""
        profile_name = self._slot_profile_map.get(slot.slot_id, self.default_profile)
        
        if profile_name not in self._pools:
            return ExecutionResult(
                success=False,
                error=f"Pool for profile '{profile_name}' not found",
                slot_id=slot.slot_id,
            )
        
        return await self._pools[profile_name].execute(slot, tool_name, args, timeout)
    
    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """Execute batch across potentially different profile pools."""
        if not requests:
            return []
        
        # Group requests by profile
        by_profile: Dict[str, List[Tuple[int, Slot, str, Dict[str, Any]]]] = {}
        
        for idx, (slot, tool_name, args) in enumerate(requests):
            profile_name = self._slot_profile_map.get(slot.slot_id, self.default_profile)
            if profile_name not in by_profile:
                by_profile[profile_name] = []
            by_profile[profile_name].append((idx, slot, tool_name, args))
        
        # Execute each profile's batch in parallel
        async def execute_profile_batch(
            profile_name: str,
            profile_requests: List[Tuple[int, Slot, str, Dict[str, Any]]]
        ) -> List[Tuple[int, ExecutionResult]]:
            if profile_name not in self._pools:
                return [
                    (idx, ExecutionResult(
                        success=False,
                        error=f"Pool for profile '{profile_name}' not found",
                        slot_id=slot.slot_id,
                    ))
                    for idx, slot, _, _ in profile_requests
                ]
            
            pool = self._pools[profile_name]
            batch_requests = [(slot, tool_name, args) for _, slot, tool_name, args in profile_requests]
            results = await pool.execute_batch(batch_requests, timeout=timeout)
            
            return [(profile_requests[i][0], result) for i, result in enumerate(results)]
        
        # Run all profile batches in parallel
        tasks = [
            execute_profile_batch(profile_name, profile_requests)
            for profile_name, profile_requests in by_profile.items()
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results in original order
        results: List[Optional[ExecutionResult]] = [None] * len(requests)
        
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                continue
            for idx, result in batch_result:
                results[idx] = result
        
        # Fill in any missing results
        for idx, result in enumerate(results):
            if result is None:
                slot, _, _ = requests[idx]
                results[idx] = ExecutionResult(
                    success=False,
                    error="Batch execution failed",
                    slot_id=slot.slot_id,
                )
        
        return results  # type: ignore
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all pools."""
        status = {
            "profiles": list(self._profiles.keys()),
            "active_pools": list(self._pools.keys()),
            "default_profile": self.default_profile,
            "pools": {},
        }
        
        for profile_name, pool in self._pools.items():
            status["pools"][profile_name] = pool.get_status()
        
        return status
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """List all available profiles and their configurations."""
        return {
            name: {
                "image": profile.image,
                "gpu": profile.gpu,
                "cpu": profile.cpu,
                "memory": profile.memory,
                "slots_per_sandbox": profile.slots_per_sandbox,
                "max_sandboxes": profile.max_sandboxes,
                "active": name in self._pools,
            }
            for name, profile in self._profiles.items()
        }


class ModalToolBackend(ToolBackend):
    """
    Modal-based tool backend with slot-based multiplexing and multi-profile support.
    
    This backend provides scalable execution for RL training:
    - Multiple trajectories share Modal sandboxes via slots
    - Batched parallel execution across slots
    - Auto-scaling sandbox pool per profile
    - Named sandbox recovery after restart
    - Multi-profile support for heterogeneous resources
    
    Usage (single profile):
        config = ModalSandboxConfig(
            name="default",
            slots_per_sandbox=10,
            max_sandboxes=5,
        ).with_app_name("my-training")
        
        backend = ModalToolBackend(config)
        await backend.start()
        slot = await backend.acquire("trajectory_1")
        ...
        await backend.stop()
    
    Usage (multi-profile):
        backend = ModalToolBackend.with_profiles(
            app_name="my-training",
            profiles={
                "default": ModalSandboxConfig(name="default"),
                "pytorch-gpu": ModalSandboxConfig(
                    name="pytorch-gpu",
                    image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                    gpu="T4",
                    memory=16384,
                ),
            }
        )
        
        await backend.start(profiles_to_start=["default", "pytorch-gpu"])
        
        # CPU task
        slot1 = await backend.acquire("traj_1", profile="default")
        
        # GPU task  
        slot2 = await backend.acquire("traj_2", profile="pytorch-gpu")
        
        # Execute in parallel across different profiles
        results = await backend.execute_batch([
            (slot1, "bash", {"command": "python preprocess.py"}),
            (slot2, "bash", {"command": "python train.py"}),
        ])
        
        await backend.release(slot1)
        await backend.release(slot2)
        await backend.stop()
    """
    
    def __init__(
        self,
        config: Optional[ModalSandboxConfig] = None,
        *,
        multi_profile_manager: Optional[_ModalMultiProfileManager] = None,
    ):
        """
        Initialize backend with either single config or multi-profile manager.
        
        Args:
            config: Sandbox configuration (ModalSandboxConfig)
            multi_profile_manager: Manager for multiple profiles
        """
        self._multi_profile = multi_profile_manager is not None
        
        if self._multi_profile:
            self._manager = multi_profile_manager
            self.config = None
            self.pool = None
        else:
            self.config = config or ModalSandboxConfig()
            self.pool = _ModalSandboxPool(self.config)
            self._manager = None
    
    @classmethod
    def with_profiles(
        cls,
        app_name: str = "atropos-sandbox",
        profiles: Optional[Dict[str, ModalSandboxConfig]] = None,
        default_profile: str = "default",
        profiles_file: Optional[str] = None,
    ) -> "ModalToolBackend":
        """
        Create backend with multi-profile support.
        
        Args:
            app_name: Modal app name prefix
            profiles: Dict of profile name -> ModalSandboxConfig. If None, loads from file/env.
            default_profile: Default profile name
            profiles_file: Path to YAML profiles file
            
        Returns:
            ModalToolBackend with multi-profile manager
        """
        if profiles is None:
            profiles = ModalSandboxConfig.load_profiles(profiles_file)
        
        manager = _ModalMultiProfileManager(
            app_name=app_name,
            profiles=profiles,
            default_profile=default_profile,
        )
        
        return cls(multi_profile_manager=manager)
    
    @property
    def default_timeout_s(self) -> Optional[float]:
        if self._multi_profile:
            # Return default profile's timeout
            return 30.0  # Default fallback
        return self.config.execution_timeout_s
    
    async def start(self, profiles_to_start: Optional[List[str]] = None) -> None:
        """
        Start the Modal pool(s).
        
        Args:
            profiles_to_start: For multi-profile, which profiles to start.
                              If None, starts default profile only.
        """
        if self._multi_profile:
            await self._manager.start(profiles_to_start)
        else:
            await self.pool.start()
    
    async def stop(self, *, purge: bool = False) -> None:
        """Stop the Modal pool(s)."""
        if self._multi_profile:
            await self._manager.stop(purge=purge)
        else:
            await self.pool.stop(purge=purge)
    
    async def acquire(
        self,
        trajectory_id: Optional[str] = None,
        profile: Optional[str] = None,
    ) -> Slot:
        """
        Acquire a slot for a trajectory.
        
        Args:
            trajectory_id: ID of the trajectory
            profile: Profile name (multi-profile mode only). If None, uses default.
            
        Returns:
            Acquired Slot
        """
        if self._multi_profile:
            return await self._manager.acquire(trajectory_id, profile)
        else:
            return await self.pool.acquire(trajectory_id)
    
    async def release(self, slot: Slot, *, reset_workspace: bool = False) -> None:
        """Release a slot back to the pool."""
        if self._multi_profile:
            await self._manager.release(slot, reset_workspace)
        else:
            await self.pool.release(slot, reset_workspace=reset_workspace)
    
    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        *,
        timeout_s: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """Execute a batch of tools in parallel across slots."""
        if self._multi_profile:
            return await self._manager.execute_batch(requests, timeout=timeout_s)
        else:
            return await self.pool.execute_batch(requests, timeout=timeout_s)
    
    # -------------------------------------------------------------------------
    # Artifact helpers
    # -------------------------------------------------------------------------
    
    async def _execute_in_slot(
        self,
        slot: Slot,
        command: str,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Helper to execute a command in a slot."""
        if self._multi_profile:
            return await self._manager.execute(slot, "bash", {"command": command}, timeout)
        return await self.pool.execute(slot, "bash", {"command": command}, timeout)
    
    async def read_artifact(
        self,
        slot: Slot,
        path: str,
        *,
        encoding: str = "text",
        max_bytes: Optional[int] = None,
        include_sha256: bool = False,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Read a file from a slot's workspace."""
        full_path = f"{slot.workspace_dir}/{path}" if not path.startswith("/") else path
        
        # Build command based on options
        if encoding == "base64":
            cmd = f"base64 {full_path}"
        else:
            cmd = f"cat {full_path}"
        
        if max_bytes:
            cmd = f"head -c {max_bytes} {full_path}"
            if encoding == "base64":
                cmd = f"head -c {max_bytes} {full_path} | base64"
        
        result = await self._execute_in_slot(slot, cmd, timeout_s)
        
        response: Dict[str, Any] = {
            "success": result.success,
            "content": result.output if result.success else "",
            "error": result.error,
        }
        
        if include_sha256 and result.success:
            sha_result = await self._execute_in_slot(
                slot, f"sha256sum {full_path} | cut -d' ' -f1", timeout_s
            )
            if sha_result.success:
                response["sha256"] = sha_result.output.strip()
        
        return response
    
    async def list_artifacts(
        self,
        slot: Slot,
        path: str = ".",
        *,
        recursive: bool = False,
        max_entries: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """List files in a slot's workspace."""
        full_path = f"{slot.workspace_dir}/{path}" if not path.startswith("/") else path
        
        if recursive:
            cmd = f"find {full_path} -type f"
        else:
            cmd = f"ls -1 {full_path}"
        
        if max_entries:
            cmd = f"{cmd} | head -n {max_entries}"
        
        result = await self._execute_in_slot(slot, cmd, timeout_s)
        
        entries = []
        if result.success and result.output.strip():
            entries = result.output.strip().split("\n")
        
        return {
            "success": result.success,
            "entries": entries,
            "error": result.error,
        }
    
    async def archive_artifacts(
        self,
        slot: Slot,
        path: str = ".",
        *,
        archive_format: str = "tar.gz",
        max_bytes: Optional[int] = None,
        max_entries: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create an archive of files in a slot's workspace."""
        full_path = f"{slot.workspace_dir}/{path}" if not path.startswith("/") else path
        
        if archive_format == "tar.gz":
            cmd = f"tar -czf - -C {full_path} . | base64"
        elif archive_format == "tar":
            cmd = f"tar -cf - -C {full_path} . | base64"
        elif archive_format == "zip":
            cmd = f"cd {full_path} && zip -r - . | base64"
        else:
            return {"success": False, "error": f"Unknown format: {archive_format}"}
        
        result = await self._execute_in_slot(slot, cmd, timeout_s)
        
        return {
            "success": result.success,
            "archive_base64": result.output if result.success else "",
            "format": archive_format,
            "error": result.error,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get backend status."""
        if self._multi_profile:
            return self._manager.get_status()
        return self.pool.get_status()
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        List available profiles (multi-profile mode only).
        
        Returns:
            Dict mapping profile names to their configs
        """
        if self._multi_profile:
            return self._manager.list_profiles()
        return {
            "default": {
                "image": self.config.image,
                "gpu": self.config.gpu,
                "cpu": self.config.cpu,
                "memory": self.config.memory,
                "slots_per_sandbox": self.config.slots_per_sandbox,
                "max_sandboxes": self.config.max_sandboxes,
                "active": True,
            }
        }
