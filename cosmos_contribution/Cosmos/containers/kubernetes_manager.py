"""
cosmos Kubernetes Manager

"I've orchestrated a symphony of containers! Kubernetes is my conductor's baton!"

Comprehensive Kubernetes cluster and workload management.
"""

import asyncio
import yaml
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from loguru import logger

try:
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException
    HAS_K8S = True
except ImportError:
    HAS_K8S = False


class PodPhase(Enum):
    """Kubernetes pod phases."""
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    UNKNOWN = "Unknown"


class DeploymentStatus(Enum):
    """Deployment status."""
    AVAILABLE = "available"
    PROGRESSING = "progressing"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class K8sPod:
    """Kubernetes pod information."""
    name: str
    namespace: str
    phase: PodPhase
    node: str
    ip: str
    containers: List[Dict[str, Any]]
    created_at: datetime
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "phase": self.phase.value,
            "node": self.node,
            "ip": self.ip,
            "containers": self.containers,
            "created_at": self.created_at.isoformat(),
            "labels": self.labels,
        }


@dataclass
class K8sDeployment:
    """Kubernetes deployment information."""
    name: str
    namespace: str
    replicas: int
    available_replicas: int
    ready_replicas: int
    image: str
    status: DeploymentStatus
    created_at: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    selector: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "replicas": self.replicas,
            "available_replicas": self.available_replicas,
            "ready_replicas": self.ready_replicas,
            "image": self.image,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "labels": self.labels,
        }


@dataclass
class K8sService:
    """Kubernetes service information."""
    name: str
    namespace: str
    type: str
    cluster_ip: str
    external_ip: Optional[str]
    ports: List[Dict[str, Any]]
    selector: Dict[str, str]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "type": self.type,
            "cluster_ip": self.cluster_ip,
            "external_ip": self.external_ip,
            "ports": self.ports,
            "selector": self.selector,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class K8sNamespace:
    """Kubernetes namespace information."""
    name: str
    status: str
    labels: Dict[str, str]
    created_at: datetime


class KubernetesManager:
    """
    Comprehensive Kubernetes management for cosmos.

    Features:
    - Multi-cluster support
    - Workload management (Deployments, StatefulSets, DaemonSets)
    - Service management
    - ConfigMap and Secret management
    - Namespace management
    - Resource monitoring
    - Helm integration
    """

    def __init__(
        self,
        kubeconfig_path: str = None,
        context: str = None,
    ):
        self.kubeconfig_path = kubeconfig_path
        self.context = context
        self._core_v1: Optional[client.CoreV1Api] = None
        self._apps_v1: Optional[client.AppsV1Api] = None
        self._batch_v1: Optional[client.BatchV1Api] = None
        self._networking_v1: Optional[client.NetworkingV1Api] = None

        if HAS_K8S:
            self._load_config()

    def _load_config(self):
        """Load Kubernetes configuration."""
        try:
            if self.kubeconfig_path:
                config.load_kube_config(
                    config_file=self.kubeconfig_path,
                    context=self.context,
                )
            else:
                try:
                    config.load_incluster_config()
                    logger.info("Loaded in-cluster Kubernetes config")
                except config.ConfigException:
                    config.load_kube_config(context=self.context)
                    logger.info("Loaded Kubernetes config from kubeconfig")

            self._core_v1 = client.CoreV1Api()
            self._apps_v1 = client.AppsV1Api()
            self._batch_v1 = client.BatchV1Api()
            self._networking_v1 = client.NetworkingV1Api()

        except Exception as e:
            logger.warning(f"Kubernetes not available: {e}")

    def _ensure_client(self):
        """Ensure Kubernetes client is available."""
        if not self._core_v1:
            raise RuntimeError("Kubernetes client not available")

    # =========================================================================
    # NAMESPACE MANAGEMENT
    # =========================================================================

    def list_namespaces(self) -> List[K8sNamespace]:
        """List all namespaces."""
        self._ensure_client()

        namespaces = self._core_v1.list_namespace()
        return [
            K8sNamespace(
                name=ns.metadata.name,
                status=ns.status.phase,
                labels=ns.metadata.labels or {},
                created_at=ns.metadata.creation_timestamp,
            )
            for ns in namespaces.items
        ]

    def create_namespace(
        self,
        name: str,
        labels: Dict[str, str] = None,
    ) -> bool:
        """Create a namespace."""
        self._ensure_client()

        try:
            namespace = client.V1Namespace(
                metadata=client.V1ObjectMeta(
                    name=name,
                    labels=labels or {},
                )
            )
            self._core_v1.create_namespace(namespace)
            logger.info(f"Created namespace: {name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to create namespace: {e}")
            return False

    def delete_namespace(self, name: str) -> bool:
        """Delete a namespace."""
        self._ensure_client()

        try:
            self._core_v1.delete_namespace(name)
            logger.info(f"Deleted namespace: {name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to delete namespace: {e}")
            return False

    # =========================================================================
    # POD MANAGEMENT
    # =========================================================================

    def list_pods(
        self,
        namespace: str = "default",
        label_selector: str = None,
    ) -> List[K8sPod]:
        """List pods in a namespace."""
        self._ensure_client()

        pods = self._core_v1.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector,
        )

        return [self._pod_to_model(pod) for pod in pods.items]

    def _pod_to_model(self, pod) -> K8sPod:
        """Convert Kubernetes pod to our model."""
        containers = []
        for c in pod.spec.containers:
            containers.append({
                "name": c.name,
                "image": c.image,
                "ports": [{"port": p.container_port, "protocol": p.protocol}
                         for p in (c.ports or [])],
            })

        return K8sPod(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
            phase=PodPhase(pod.status.phase) if pod.status.phase else PodPhase.UNKNOWN,
            node=pod.spec.node_name or "",
            ip=pod.status.pod_ip or "",
            containers=containers,
            created_at=pod.metadata.creation_timestamp,
            labels=pod.metadata.labels or {},
        )

    def get_pod(self, name: str, namespace: str = "default") -> Optional[K8sPod]:
        """Get a specific pod."""
        self._ensure_client()

        try:
            pod = self._core_v1.read_namespaced_pod(name, namespace)
            return self._pod_to_model(pod)
        except ApiException:
            return None

    def delete_pod(
        self,
        name: str,
        namespace: str = "default",
        grace_period: int = 30,
    ) -> bool:
        """Delete a pod."""
        self._ensure_client()

        try:
            self._core_v1.delete_namespaced_pod(
                name,
                namespace,
                grace_period_seconds=grace_period,
            )
            logger.info(f"Deleted pod: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to delete pod: {e}")
            return False

    def get_pod_logs(
        self,
        name: str,
        namespace: str = "default",
        container: str = None,
        tail_lines: int = 100,
        since_seconds: int = None,
    ) -> str:
        """Get pod logs."""
        self._ensure_client()

        try:
            return self._core_v1.read_namespaced_pod_log(
                name,
                namespace,
                container=container,
                tail_lines=tail_lines,
                since_seconds=since_seconds,
            )
        except ApiException as e:
            logger.error(f"Failed to get pod logs: {e}")
            return ""

    def exec_in_pod(
        self,
        name: str,
        namespace: str,
        command: List[str],
        container: str = None,
    ) -> Dict[str, Any]:
        """Execute a command in a pod."""
        self._ensure_client()

        from kubernetes.stream import stream

        try:
            resp = stream(
                self._core_v1.connect_get_namespaced_pod_exec,
                name,
                namespace,
                command=command,
                container=container,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
            )
            return {"output": resp, "success": True}
        except ApiException as e:
            return {"output": str(e), "success": False}

    # =========================================================================
    # DEPLOYMENT MANAGEMENT
    # =========================================================================

    def list_deployments(
        self,
        namespace: str = "default",
        label_selector: str = None,
    ) -> List[K8sDeployment]:
        """List deployments in a namespace."""
        self._ensure_client()

        deployments = self._apps_v1.list_namespaced_deployment(
            namespace=namespace,
            label_selector=label_selector,
        )

        return [self._deployment_to_model(d) for d in deployments.items]

    def _deployment_to_model(self, deployment) -> K8sDeployment:
        """Convert Kubernetes deployment to our model."""
        status = DeploymentStatus.UNKNOWN
        if deployment.status.available_replicas == deployment.spec.replicas:
            status = DeploymentStatus.AVAILABLE
        elif deployment.status.unavailable_replicas:
            status = DeploymentStatus.PROGRESSING

        # Get image from first container
        image = ""
        if deployment.spec.template.spec.containers:
            image = deployment.spec.template.spec.containers[0].image

        return K8sDeployment(
            name=deployment.metadata.name,
            namespace=deployment.metadata.namespace,
            replicas=deployment.spec.replicas or 0,
            available_replicas=deployment.status.available_replicas or 0,
            ready_replicas=deployment.status.ready_replicas or 0,
            image=image,
            status=status,
            created_at=deployment.metadata.creation_timestamp,
            labels=deployment.metadata.labels or {},
            selector=deployment.spec.selector.match_labels or {},
        )

    def get_deployment(
        self,
        name: str,
        namespace: str = "default",
    ) -> Optional[K8sDeployment]:
        """Get a specific deployment."""
        self._ensure_client()

        try:
            deployment = self._apps_v1.read_namespaced_deployment(name, namespace)
            return self._deployment_to_model(deployment)
        except ApiException:
            return None

    def create_deployment(
        self,
        name: str,
        namespace: str,
        image: str,
        replicas: int = 1,
        ports: List[int] = None,
        env: Dict[str, str] = None,
        labels: Dict[str, str] = None,
        resources: Dict[str, Any] = None,
    ) -> bool:
        """Create a deployment."""
        self._ensure_client()

        container_ports = [
            client.V1ContainerPort(container_port=p)
            for p in (ports or [])
        ]

        env_vars = [
            client.V1EnvVar(name=k, value=v)
            for k, v in (env or {}).items()
        ]

        container = client.V1Container(
            name=name,
            image=image,
            ports=container_ports or None,
            env=env_vars or None,
            resources=client.V1ResourceRequirements(
                requests=resources.get("requests") if resources else None,
                limits=resources.get("limits") if resources else None,
            ) if resources else None,
        )

        template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels or {"app": name}),
            spec=client.V1PodSpec(containers=[container]),
        )

        spec = client.V1DeploymentSpec(
            replicas=replicas,
            template=template,
            selector=client.V1LabelSelector(match_labels=labels or {"app": name}),
        )

        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=name, labels=labels or {"app": name}),
            spec=spec,
        )

        try:
            self._apps_v1.create_namespaced_deployment(namespace, deployment)
            logger.info(f"Created deployment: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to create deployment: {e}")
            return False

    def update_deployment_image(
        self,
        name: str,
        namespace: str,
        image: str,
        container_name: str = None,
    ) -> bool:
        """Update deployment image."""
        self._ensure_client()

        try:
            deployment = self._apps_v1.read_namespaced_deployment(name, namespace)

            for container in deployment.spec.template.spec.containers:
                if container_name is None or container.name == container_name:
                    container.image = image
                    break

            self._apps_v1.patch_namespaced_deployment(name, namespace, deployment)
            logger.info(f"Updated deployment image: {namespace}/{name} -> {image}")
            return True
        except ApiException as e:
            logger.error(f"Failed to update deployment: {e}")
            return False

    def scale_deployment(
        self,
        name: str,
        namespace: str,
        replicas: int,
    ) -> bool:
        """Scale a deployment."""
        self._ensure_client()

        try:
            self._apps_v1.patch_namespaced_deployment_scale(
                name,
                namespace,
                {"spec": {"replicas": replicas}},
            )
            logger.info(f"Scaled deployment: {namespace}/{name} to {replicas} replicas")
            return True
        except ApiException as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False

    def restart_deployment(self, name: str, namespace: str) -> bool:
        """Restart a deployment (rolling restart)."""
        self._ensure_client()

        try:
            patch = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": datetime.utcnow().isoformat()
                            }
                        }
                    }
                }
            }
            self._apps_v1.patch_namespaced_deployment(name, namespace, patch)
            logger.info(f"Restarted deployment: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to restart deployment: {e}")
            return False

    def delete_deployment(
        self,
        name: str,
        namespace: str,
    ) -> bool:
        """Delete a deployment."""
        self._ensure_client()

        try:
            self._apps_v1.delete_namespaced_deployment(name, namespace)
            logger.info(f"Deleted deployment: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to delete deployment: {e}")
            return False

    # =========================================================================
    # SERVICE MANAGEMENT
    # =========================================================================

    def list_services(
        self,
        namespace: str = "default",
        label_selector: str = None,
    ) -> List[K8sService]:
        """List services in a namespace."""
        self._ensure_client()

        services = self._core_v1.list_namespaced_service(
            namespace=namespace,
            label_selector=label_selector,
        )

        return [self._service_to_model(s) for s in services.items]

    def _service_to_model(self, service) -> K8sService:
        """Convert Kubernetes service to our model."""
        ports = []
        for p in (service.spec.ports or []):
            ports.append({
                "name": p.name,
                "port": p.port,
                "target_port": p.target_port,
                "protocol": p.protocol,
                "node_port": p.node_port,
            })

        external_ip = None
        if service.status.load_balancer.ingress:
            external_ip = service.status.load_balancer.ingress[0].ip or \
                         service.status.load_balancer.ingress[0].hostname

        return K8sService(
            name=service.metadata.name,
            namespace=service.metadata.namespace,
            type=service.spec.type,
            cluster_ip=service.spec.cluster_ip or "",
            external_ip=external_ip,
            ports=ports,
            selector=service.spec.selector or {},
            created_at=service.metadata.creation_timestamp,
        )

    def create_service(
        self,
        name: str,
        namespace: str,
        selector: Dict[str, str],
        ports: List[Dict[str, Any]],
        service_type: str = "ClusterIP",
    ) -> bool:
        """Create a service."""
        self._ensure_client()

        service_ports = [
            client.V1ServicePort(
                name=p.get("name"),
                port=p["port"],
                target_port=p.get("target_port", p["port"]),
                protocol=p.get("protocol", "TCP"),
            )
            for p in ports
        ]

        service = client.V1Service(
            metadata=client.V1ObjectMeta(name=name),
            spec=client.V1ServiceSpec(
                selector=selector,
                ports=service_ports,
                type=service_type,
            ),
        )

        try:
            self._core_v1.create_namespaced_service(namespace, service)
            logger.info(f"Created service: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to create service: {e}")
            return False

    def delete_service(self, name: str, namespace: str) -> bool:
        """Delete a service."""
        self._ensure_client()

        try:
            self._core_v1.delete_namespaced_service(name, namespace)
            logger.info(f"Deleted service: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to delete service: {e}")
            return False

    # =========================================================================
    # CONFIGMAP AND SECRET MANAGEMENT
    # =========================================================================

    def create_configmap(
        self,
        name: str,
        namespace: str,
        data: Dict[str, str],
    ) -> bool:
        """Create a ConfigMap."""
        self._ensure_client()

        configmap = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(name=name),
            data=data,
        )

        try:
            self._core_v1.create_namespaced_config_map(namespace, configmap)
            logger.info(f"Created ConfigMap: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to create ConfigMap: {e}")
            return False

    def create_secret(
        self,
        name: str,
        namespace: str,
        data: Dict[str, str],
        secret_type: str = "Opaque",
    ) -> bool:
        """Create a Secret."""
        self._ensure_client()

        import base64
        encoded_data = {k: base64.b64encode(v.encode()).decode() for k, v in data.items()}

        secret = client.V1Secret(
            metadata=client.V1ObjectMeta(name=name),
            type=secret_type,
            data=encoded_data,
        )

        try:
            self._core_v1.create_namespaced_secret(namespace, secret)
            logger.info(f"Created Secret: {namespace}/{name}")
            return True
        except ApiException as e:
            logger.error(f"Failed to create Secret: {e}")
            return False

    # =========================================================================
    # HELM INTEGRATION
    # =========================================================================

    async def helm_install(
        self,
        release_name: str,
        chart: str,
        namespace: str = "default",
        values: Dict[str, Any] = None,
        values_file: Path = None,
        wait: bool = True,
    ) -> Dict[str, Any]:
        """Install a Helm chart."""
        command = [
            "helm", "install", release_name, chart,
            "--namespace", namespace,
            "--create-namespace",
        ]

        if values:
            # Write values to temp file
            values_path = Path(f"/tmp/helm-values-{release_name}.yaml")
            with open(values_path, "w") as f:
                yaml.dump(values, f)
            command.extend(["-f", str(values_path)])

        if values_file:
            command.extend(["-f", str(values_file)])

        if wait:
            command.append("--wait")

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

    async def helm_upgrade(
        self,
        release_name: str,
        chart: str,
        namespace: str = "default",
        values: Dict[str, Any] = None,
        install: bool = True,
    ) -> Dict[str, Any]:
        """Upgrade a Helm release."""
        command = [
            "helm", "upgrade", release_name, chart,
            "--namespace", namespace,
        ]

        if install:
            command.append("--install")

        if values:
            values_path = Path(f"/tmp/helm-values-{release_name}.yaml")
            with open(values_path, "w") as f:
                yaml.dump(values, f)
            command.extend(["-f", str(values_path)])

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

    async def helm_uninstall(
        self,
        release_name: str,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """Uninstall a Helm release."""
        command = ["helm", "uninstall", release_name, "--namespace", namespace]

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

    async def helm_list(
        self,
        namespace: str = None,
        all_namespaces: bool = False,
    ) -> List[Dict[str, Any]]:
        """List Helm releases."""
        command = ["helm", "list", "--output", "json"]

        if all_namespaces:
            command.append("--all-namespaces")
        elif namespace:
            command.extend(["--namespace", namespace])

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
kubernetes_manager = KubernetesManager()
