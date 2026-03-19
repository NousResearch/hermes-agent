"""
cosmos Docker Manager

"Containers within containers! It's like Russian nesting dolls, but for code!"

Comprehensive Docker container and image management.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from loguru import logger

try:
    import docker
    from docker.models.containers import Container as DockerContainer
    from docker.models.images import Image as DockerImageModel
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False
    DockerContainer = Any
    DockerImageModel = Any


class ContainerStatus(Enum):
    """Container status."""
    RUNNING = "running"
    EXITED = "exited"
    CREATED = "created"
    PAUSED = "paused"
    RESTARTING = "restarting"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class Container:
    """Container information."""
    id: str
    name: str
    image: str
    status: ContainerStatus
    created_at: datetime
    ports: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    mounts: List[Dict[str, Any]] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    command: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "image": self.image,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "ports": self.ports,
            "labels": self.labels,
            "networks": self.networks,
            "command": self.command,
        }


@dataclass
class DockerImage:
    """Docker image information."""
    id: str
    tags: List[str]
    size: int
    created_at: datetime
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tags": self.tags,
            "size": self.size,
            "size_mb": round(self.size / 1024 / 1024, 2),
            "created_at": self.created_at.isoformat(),
            "labels": self.labels,
        }


@dataclass
class ContainerStats:
    """Container resource statistics."""
    container_id: str
    cpu_percent: float
    memory_usage: int
    memory_limit: int
    memory_percent: float
    network_rx: int
    network_tx: int
    block_read: int
    block_write: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DockerManager:
    """
    Comprehensive Docker management for cosmos.

    Features:
    - Container lifecycle management
    - Image management
    - Network management
    - Volume management
    - Docker Compose support
    - Resource monitoring
    """

    def __init__(self, docker_host: str = None):
        self.docker_host = docker_host
        self._client: Optional[docker.DockerClient] = None

        if HAS_DOCKER:
            try:
                self._client = docker.from_env() if not docker_host else docker.DockerClient(base_url=docker_host)
                logger.info(f"Connected to Docker: {self._client.version()['Version']}")
            except Exception as e:
                logger.warning(f"Docker not available: {e}")

    def _ensure_client(self):
        """Ensure Docker client is available."""
        if not self._client:
            raise RuntimeError("Docker client not available")

    # =========================================================================
    # CONTAINER MANAGEMENT
    # =========================================================================

    def list_containers(
        self,
        all_containers: bool = True,
        filters: Dict[str, Any] = None,
    ) -> List[Container]:
        """List all containers."""
        self._ensure_client()

        containers = self._client.containers.list(all=all_containers, filters=filters or {})
        return [self._container_to_model(c) for c in containers]

    def get_container(self, container_id: str) -> Optional[Container]:
        """Get a container by ID or name."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            return self._container_to_model(container)
        except docker.errors.NotFound:
            return None

    def _container_to_model(self, container: DockerContainer) -> Container:
        """Convert Docker container to our model."""
        attrs = container.attrs

        status_map = {
            "running": ContainerStatus.RUNNING,
            "exited": ContainerStatus.EXITED,
            "created": ContainerStatus.CREATED,
            "paused": ContainerStatus.PAUSED,
            "restarting": ContainerStatus.RESTARTING,
            "dead": ContainerStatus.DEAD,
        }

        return Container(
            id=container.id[:12],
            name=container.name,
            image=attrs.get("Config", {}).get("Image", ""),
            status=status_map.get(container.status, ContainerStatus.UNKNOWN),
            created_at=datetime.fromisoformat(attrs.get("Created", "").replace("Z", "+00:00")),
            ports=attrs.get("NetworkSettings", {}).get("Ports", {}),
            labels=attrs.get("Config", {}).get("Labels", {}),
            networks=list(attrs.get("NetworkSettings", {}).get("Networks", {}).keys()),
            command=attrs.get("Config", {}).get("Cmd", []),
        )

    def run_container(
        self,
        image: str,
        name: str = None,
        command: str = None,
        environment: Dict[str, str] = None,
        ports: Dict[str, int] = None,
        volumes: Dict[str, Dict] = None,
        network: str = None,
        detach: bool = True,
        auto_remove: bool = False,
        labels: Dict[str, str] = None,
        restart_policy: Dict[str, Any] = None,
        **kwargs,
    ) -> Container:
        """Run a new container."""
        self._ensure_client()

        container = self._client.containers.run(
            image,
            name=name,
            command=command,
            environment=environment or {},
            ports=ports or {},
            volumes=volumes or {},
            network=network,
            detach=detach,
            auto_remove=auto_remove,
            labels=labels or {},
            restart_policy=restart_policy,
            **kwargs,
        )

        logger.info(f"Started container: {container.name} ({container.id[:12]})")
        return self._container_to_model(container)

    def start_container(self, container_id: str) -> bool:
        """Start a stopped container."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            container.start()
            logger.info(f"Started container: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return False

    def stop_container(self, container_id: str, timeout: int = 10) -> bool:
        """Stop a running container."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            container.stop(timeout=timeout)
            logger.info(f"Stopped container: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    def restart_container(self, container_id: str, timeout: int = 10) -> bool:
        """Restart a container."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            container.restart(timeout=timeout)
            logger.info(f"Restarted container: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return False

    def remove_container(
        self,
        container_id: str,
        force: bool = False,
        volumes: bool = False,
    ) -> bool:
        """Remove a container."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            container.remove(force=force, v=volumes)
            logger.info(f"Removed container: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")
            return False

    def get_container_logs(
        self,
        container_id: str,
        tail: int = 100,
        since: datetime = None,
        until: datetime = None,
        timestamps: bool = True,
    ) -> str:
        """Get container logs."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            logs = container.logs(
                tail=tail,
                since=since,
                until=until,
                timestamps=timestamps,
            )
            return logs.decode() if isinstance(logs, bytes) else logs
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return ""

    def exec_in_container(
        self,
        container_id: str,
        command: str,
        user: str = None,
        workdir: str = None,
    ) -> Dict[str, Any]:
        """Execute a command in a running container."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            exit_code, output = container.exec_run(
                command,
                user=user,
                workdir=workdir,
            )
            return {
                "exit_code": exit_code,
                "output": output.decode() if isinstance(output, bytes) else output,
            }
        except Exception as e:
            logger.error(f"Exec failed: {e}")
            return {"exit_code": -1, "output": str(e)}

    def get_container_stats(self, container_id: str) -> Optional[ContainerStats]:
        """Get container resource statistics."""
        self._ensure_client()

        try:
            container = self._client.containers.get(container_id)
            stats = container.stats(stream=False)

            # Calculate CPU percent
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0

            # Memory
            memory_usage = stats["memory_stats"].get("usage", 0)
            memory_limit = stats["memory_stats"].get("limit", 1)
            memory_percent = (memory_usage / memory_limit) * 100

            # Network
            networks = stats.get("networks", {})
            network_rx = sum(n.get("rx_bytes", 0) for n in networks.values())
            network_tx = sum(n.get("tx_bytes", 0) for n in networks.values())

            return ContainerStats(
                container_id=container_id,
                cpu_percent=round(cpu_percent, 2),
                memory_usage=memory_usage,
                memory_limit=memory_limit,
                memory_percent=round(memory_percent, 2),
                network_rx=network_rx,
                network_tx=network_tx,
                block_read=0,
                block_write=0,
            )
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return None

    # =========================================================================
    # IMAGE MANAGEMENT
    # =========================================================================

    def list_images(self, filters: Dict[str, Any] = None) -> List[DockerImage]:
        """List all images."""
        self._ensure_client()

        images = self._client.images.list(filters=filters or {})
        return [self._image_to_model(img) for img in images]

    def get_image(self, image_id: str) -> Optional[DockerImage]:
        """Get an image by ID or tag."""
        self._ensure_client()

        try:
            image = self._client.images.get(image_id)
            return self._image_to_model(image)
        except docker.errors.ImageNotFound:
            return None

    def _image_to_model(self, image: DockerImageModel) -> DockerImage:
        """Convert Docker image to our model."""
        attrs = image.attrs

        created = attrs.get("Created", "")
        if created:
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        else:
            created = datetime.utcnow()

        return DockerImage(
            id=image.id[:12] if image.id else "",
            tags=image.tags or [],
            size=attrs.get("Size", 0),
            created_at=created,
            labels=attrs.get("Config", {}).get("Labels", {}) or {},
        )

    def pull_image(self, repository: str, tag: str = "latest") -> Optional[DockerImage]:
        """Pull an image from registry."""
        self._ensure_client()

        try:
            logger.info(f"Pulling image: {repository}:{tag}")
            image = self._client.images.pull(repository, tag=tag)
            logger.info(f"Pulled image: {repository}:{tag}")
            return self._image_to_model(image)
        except Exception as e:
            logger.error(f"Failed to pull image: {e}")
            return None

    def build_image(
        self,
        path: str,
        tag: str,
        dockerfile: str = "Dockerfile",
        build_args: Dict[str, str] = None,
        no_cache: bool = False,
    ) -> Optional[DockerImage]:
        """Build an image from Dockerfile."""
        self._ensure_client()

        try:
            logger.info(f"Building image: {tag}")
            image, logs = self._client.images.build(
                path=path,
                tag=tag,
                dockerfile=dockerfile,
                buildargs=build_args or {},
                nocache=no_cache,
            )
            for log in logs:
                if "stream" in log:
                    logger.debug(log["stream"].strip())
            logger.info(f"Built image: {tag}")
            return self._image_to_model(image)
        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            return None

    def push_image(
        self,
        repository: str,
        tag: str = "latest",
        auth_config: Dict[str, str] = None,
    ) -> bool:
        """Push an image to registry."""
        self._ensure_client()

        try:
            logger.info(f"Pushing image: {repository}:{tag}")
            result = self._client.images.push(
                repository,
                tag=tag,
                auth_config=auth_config,
            )
            logger.info(f"Pushed image: {repository}:{tag}")
            return True
        except Exception as e:
            logger.error(f"Failed to push image: {e}")
            return False

    def remove_image(
        self,
        image_id: str,
        force: bool = False,
        no_prune: bool = False,
    ) -> bool:
        """Remove an image."""
        self._ensure_client()

        try:
            self._client.images.remove(image_id, force=force, noprune=no_prune)
            logger.info(f"Removed image: {image_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove image: {e}")
            return False

    def tag_image(
        self,
        image_id: str,
        repository: str,
        tag: str = "latest",
    ) -> bool:
        """Tag an image."""
        self._ensure_client()

        try:
            image = self._client.images.get(image_id)
            image.tag(repository, tag=tag)
            logger.info(f"Tagged image: {image_id} as {repository}:{tag}")
            return True
        except Exception as e:
            logger.error(f"Failed to tag image: {e}")
            return False

    def prune_images(self, all_unused: bool = False) -> Dict[str, Any]:
        """Remove unused images."""
        self._ensure_client()

        try:
            filters = {"dangling": True} if not all_unused else {}
            result = self._client.images.prune(filters=filters)
            logger.info(f"Pruned images: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to prune images: {e}")
            return {}

    # =========================================================================
    # NETWORK MANAGEMENT
    # =========================================================================

    def list_networks(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List all networks."""
        self._ensure_client()

        networks = self._client.networks.list(filters=filters or {})
        return [{
            "id": n.id[:12],
            "name": n.name,
            "driver": n.attrs.get("Driver"),
            "scope": n.attrs.get("Scope"),
        } for n in networks]

    def create_network(
        self,
        name: str,
        driver: str = "bridge",
        internal: bool = False,
        labels: Dict[str, str] = None,
    ) -> Optional[str]:
        """Create a network."""
        self._ensure_client()

        try:
            network = self._client.networks.create(
                name=name,
                driver=driver,
                internal=internal,
                labels=labels or {},
            )
            logger.info(f"Created network: {name}")
            return network.id
        except Exception as e:
            logger.error(f"Failed to create network: {e}")
            return None

    def remove_network(self, network_id: str) -> bool:
        """Remove a network."""
        self._ensure_client()

        try:
            network = self._client.networks.get(network_id)
            network.remove()
            logger.info(f"Removed network: {network_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove network: {e}")
            return False

    # =========================================================================
    # VOLUME MANAGEMENT
    # =========================================================================

    def list_volumes(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List all volumes."""
        self._ensure_client()

        volumes = self._client.volumes.list(filters=filters or {})
        return [{
            "name": v.name,
            "driver": v.attrs.get("Driver"),
            "mountpoint": v.attrs.get("Mountpoint"),
            "labels": v.attrs.get("Labels", {}),
        } for v in volumes]

    def create_volume(
        self,
        name: str,
        driver: str = "local",
        labels: Dict[str, str] = None,
    ) -> Optional[str]:
        """Create a volume."""
        self._ensure_client()

        try:
            volume = self._client.volumes.create(
                name=name,
                driver=driver,
                labels=labels or {},
            )
            logger.info(f"Created volume: {name}")
            return volume.name
        except Exception as e:
            logger.error(f"Failed to create volume: {e}")
            return None

    def remove_volume(self, volume_name: str, force: bool = False) -> bool:
        """Remove a volume."""
        self._ensure_client()

        try:
            volume = self._client.volumes.get(volume_name)
            volume.remove(force=force)
            logger.info(f"Removed volume: {volume_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove volume: {e}")
            return False

    # =========================================================================
    # DOCKER COMPOSE
    # =========================================================================

    async def compose_up(
        self,
        compose_file: Path,
        project_name: str = None,
        detach: bool = True,
        build: bool = False,
    ) -> Dict[str, Any]:
        """Run docker-compose up."""
        command = ["docker-compose", "-f", str(compose_file)]

        if project_name:
            command.extend(["-p", project_name])

        command.append("up")

        if detach:
            command.append("-d")
        if build:
            command.append("--build")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }

    async def compose_down(
        self,
        compose_file: Path,
        project_name: str = None,
        volumes: bool = False,
        remove_orphans: bool = False,
    ) -> Dict[str, Any]:
        """Run docker-compose down."""
        command = ["docker-compose", "-f", str(compose_file)]

        if project_name:
            command.extend(["-p", project_name])

        command.append("down")

        if volumes:
            command.append("-v")
        if remove_orphans:
            command.append("--remove-orphans")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        return {
            "success": process.returncode == 0,
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
        }

    async def compose_ps(
        self,
        compose_file: Path,
        project_name: str = None,
    ) -> List[Dict[str, Any]]:
        """List compose services."""
        command = ["docker-compose", "-f", str(compose_file)]

        if project_name:
            command.extend(["-p", project_name])

        command.extend(["ps", "--format", "json"])

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            try:
                return json.loads(stdout.decode())
            except json.JSONDecodeError:
                return []
        return []


# Singleton instance
docker_manager = DockerManager()
