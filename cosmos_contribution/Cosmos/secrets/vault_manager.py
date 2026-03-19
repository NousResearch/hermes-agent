"""
cosmos Secrets and Credentials Vault Manager

"I've hidden my secrets so well, even I can't find them!"

Unified secrets management with rotation, versioning, and audit logging.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import json
import hashlib
import base64
from loguru import logger

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class SecretType(Enum):
    """Types of secrets."""
    GENERIC = "generic"
    PASSWORD = "password"
    API_KEY = "api_key"
    SSH_KEY = "ssh_key"
    CERTIFICATE = "certificate"
    DATABASE = "database"
    AWS_CREDENTIALS = "aws_credentials"
    OAUTH_TOKEN = "oauth_token"
    ENCRYPTION_KEY = "encryption_key"
    CONNECTION_STRING = "connection_string"


class SecretStatus(Enum):
    """Secret status."""
    ACTIVE = "active"
    ROTATING = "rotating"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class SecretVersion:
    """A version of a secret."""
    version: int
    value: str
    created_at: datetime
    created_by: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_current: bool = True


@dataclass
class Secret:
    """A secret with versioning."""
    id: str
    name: str
    secret_type: SecretType
    path: str  # Hierarchical path like "prod/database/mysql"
    current_value: str = ""
    versions: List[SecretVersion] = field(default_factory=list)
    status: SecretStatus = SecretStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Rotation settings
    rotation_enabled: bool = False
    rotation_days: int = 90
    last_rotated: Optional[datetime] = None
    next_rotation: Optional[datetime] = None
    rotation_lambda: Optional[str] = None  # Function to call for rotation

    # Access control
    allowed_services: List[str] = field(default_factory=list)
    allowed_roles: List[str] = field(default_factory=list)

    # Audit
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.secret_type.value,
            "path": self.path,
            "status": self.status.value,
            "metadata": self.metadata,
            "tags": self.tags,
            "rotation_enabled": self.rotation_enabled,
            "rotation_days": self.rotation_days,
            "last_rotated": self.last_rotated.isoformat() if self.last_rotated else None,
            "next_rotation": self.next_rotation.isoformat() if self.next_rotation else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version_count": len(self.versions),
            "access_count": self.access_count,
        }
        if include_value:
            result["value"] = self.current_value
        return result


@dataclass
class SecretAccessLog:
    """Audit log entry for secret access."""
    secret_id: str
    secret_path: str
    action: str  # read, write, delete, rotate
    actor: str
    service: str
    timestamp: datetime
    ip_address: str = ""
    success: bool = True
    error_message: str = ""


class SecretsProvider(ABC):
    """Abstract base class for secrets providers."""

    @abstractmethod
    async def get_secret(self, path: str, version: int = None) -> Optional[Secret]:
        """Get a secret by path."""
        pass

    @abstractmethod
    async def set_secret(self, secret: Secret) -> bool:
        """Create or update a secret."""
        pass

    @abstractmethod
    async def delete_secret(self, path: str) -> bool:
        """Delete a secret."""
        pass

    @abstractmethod
    async def list_secrets(self, path_prefix: str = "") -> List[str]:
        """List secret paths."""
        pass

    @abstractmethod
    async def rotate_secret(self, path: str) -> bool:
        """Rotate a secret."""
        pass


class LocalVaultProvider(SecretsProvider):
    """
    Local file-based secrets vault with encryption.
    For development and testing purposes.
    """

    def __init__(
        self,
        storage_path: Path,
        master_password: str,
    ):
        if not HAS_CRYPTOGRAPHY:
            raise ImportError("cryptography is required for local vault")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.secrets_file = self.storage_path / "secrets.enc"
        self.fernet = self._derive_key(master_password)
        self.secrets: Dict[str, Secret] = {}

        self._load_secrets()

    def _derive_key(self, password: str) -> Fernet:
        """Derive encryption key from password."""
        salt = b"cosmos_vault_salt"  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return Fernet(key)

    def _load_secrets(self):
        """Load secrets from encrypted file."""
        if not self.secrets_file.exists():
            return

        try:
            with open(self.secrets_file, "rb") as f:
                encrypted = f.read()

            decrypted = self.fernet.decrypt(encrypted)
            data = json.loads(decrypted.decode())

            for path, secret_data in data.items():
                self.secrets[path] = Secret(
                    id=secret_data["id"],
                    name=secret_data["name"],
                    secret_type=SecretType(secret_data["type"]),
                    path=path,
                    current_value=secret_data["value"],
                    status=SecretStatus(secret_data.get("status", "active")),
                    metadata=secret_data.get("metadata", {}),
                    tags=secret_data.get("tags", []),
                    rotation_enabled=secret_data.get("rotation_enabled", False),
                    rotation_days=secret_data.get("rotation_days", 90),
                    created_at=datetime.fromisoformat(secret_data["created_at"]),
                    updated_at=datetime.fromisoformat(secret_data["updated_at"]),
                )

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")

    def _save_secrets(self):
        """Save secrets to encrypted file."""
        data = {}
        for path, secret in self.secrets.items():
            data[path] = {
                "id": secret.id,
                "name": secret.name,
                "type": secret.secret_type.value,
                "value": secret.current_value,
                "status": secret.status.value,
                "metadata": secret.metadata,
                "tags": secret.tags,
                "rotation_enabled": secret.rotation_enabled,
                "rotation_days": secret.rotation_days,
                "created_at": secret.created_at.isoformat(),
                "updated_at": secret.updated_at.isoformat(),
            }

        encrypted = self.fernet.encrypt(json.dumps(data).encode())
        with open(self.secrets_file, "wb") as f:
            f.write(encrypted)

    async def get_secret(self, path: str, version: int = None) -> Optional[Secret]:
        secret = self.secrets.get(path)
        if secret:
            secret.last_accessed_at = datetime.utcnow()
            secret.access_count += 1
        return secret

    async def set_secret(self, secret: Secret) -> bool:
        secret.updated_at = datetime.utcnow()
        self.secrets[secret.path] = secret
        self._save_secrets()
        return True

    async def delete_secret(self, path: str) -> bool:
        if path in self.secrets:
            del self.secrets[path]
            self._save_secrets()
            return True
        return False

    async def list_secrets(self, path_prefix: str = "") -> List[str]:
        return [p for p in self.secrets.keys() if p.startswith(path_prefix)]

    async def rotate_secret(self, path: str) -> bool:
        secret = self.secrets.get(path)
        if not secret:
            return False

        # Store old version
        version = SecretVersion(
            version=len(secret.versions) + 1,
            value=secret.current_value,
            created_at=datetime.utcnow(),
            created_by="system",
            is_current=False,
        )
        secret.versions.append(version)

        # Generate new value (in production, use actual rotation logic)
        import secrets as sec
        secret.current_value = sec.token_urlsafe(32)
        secret.last_rotated = datetime.utcnow()
        secret.next_rotation = datetime.utcnow() + timedelta(days=secret.rotation_days)
        secret.updated_at = datetime.utcnow()

        self._save_secrets()
        return True


class VaultManager:
    """
    Unified secrets management across multiple providers.

    Features:
    - Multi-provider support (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault)
    - Secret versioning
    - Automatic rotation
    - Access audit logging
    - Path-based organization
    - Encryption at rest
    """

    def __init__(
        self,
        default_provider: str = "local",
    ):
        self.providers: Dict[str, SecretsProvider] = {}
        self.default_provider = default_provider
        self.audit_logs: List[SecretAccessLog] = []
        self.rotation_callbacks: Dict[str, Callable] = {}

    def register_provider(self, name: str, provider: SecretsProvider):
        """Register a secrets provider."""
        self.providers[name] = provider
        logger.info(f"Registered secrets provider: {name}")

    def get_provider(self, name: str = None) -> Optional[SecretsProvider]:
        """Get a provider by name or the default."""
        if name:
            return self.providers.get(name)
        return self.providers.get(self.default_provider)

    # =========================================================================
    # SECRET CRUD
    # =========================================================================

    async def get_secret(
        self,
        path: str,
        provider: str = None,
        version: int = None,
        actor: str = "system",
        service: str = "cosmos",
    ) -> Optional[Secret]:
        """Get a secret by path."""
        prov = self.get_provider(provider)
        if not prov:
            logger.error(f"Provider not found: {provider or self.default_provider}")
            return None

        secret = await prov.get_secret(path, version)

        # Audit log
        self.audit_logs.append(SecretAccessLog(
            secret_id=secret.id if secret else "",
            secret_path=path,
            action="read",
            actor=actor,
            service=service,
            timestamp=datetime.utcnow(),
            success=secret is not None,
        ))

        return secret

    async def create_secret(
        self,
        path: str,
        value: str,
        secret_type: SecretType = SecretType.GENERIC,
        name: str = None,
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
        rotation_enabled: bool = False,
        rotation_days: int = 90,
        provider: str = None,
        actor: str = "system",
    ) -> bool:
        """Create a new secret."""
        prov = self.get_provider(provider)
        if not prov:
            return False

        import uuid
        secret = Secret(
            id=str(uuid.uuid4()),
            name=name or path.split("/")[-1],
            secret_type=secret_type,
            path=path,
            current_value=value,
            metadata=metadata or {},
            tags=tags or [],
            rotation_enabled=rotation_enabled,
            rotation_days=rotation_days,
            next_rotation=datetime.utcnow() + timedelta(days=rotation_days) if rotation_enabled else None,
            created_by=actor,
        )

        # Add initial version
        secret.versions.append(SecretVersion(
            version=1,
            value=value,
            created_at=datetime.utcnow(),
            created_by=actor,
        ))

        success = await prov.set_secret(secret)

        self.audit_logs.append(SecretAccessLog(
            secret_id=secret.id,
            secret_path=path,
            action="create",
            actor=actor,
            service="cosmos",
            timestamp=datetime.utcnow(),
            success=success,
        ))

        if success:
            logger.info(f"Created secret: {path}")
        return success

    async def update_secret(
        self,
        path: str,
        value: str,
        provider: str = None,
        actor: str = "system",
    ) -> bool:
        """Update an existing secret."""
        prov = self.get_provider(provider)
        if not prov:
            return False

        secret = await prov.get_secret(path)
        if not secret:
            logger.error(f"Secret not found: {path}")
            return False

        # Add old value as previous version
        secret.versions.append(SecretVersion(
            version=len(secret.versions) + 1,
            value=secret.current_value,
            created_at=secret.updated_at,
            created_by=actor,
            is_current=False,
        ))

        secret.current_value = value
        secret.updated_at = datetime.utcnow()

        success = await prov.set_secret(secret)

        self.audit_logs.append(SecretAccessLog(
            secret_id=secret.id,
            secret_path=path,
            action="update",
            actor=actor,
            service="cosmos",
            timestamp=datetime.utcnow(),
            success=success,
        ))

        return success

    async def delete_secret(
        self,
        path: str,
        provider: str = None,
        actor: str = "system",
    ) -> bool:
        """Delete a secret."""
        prov = self.get_provider(provider)
        if not prov:
            return False

        secret = await prov.get_secret(path)
        success = await prov.delete_secret(path)

        self.audit_logs.append(SecretAccessLog(
            secret_id=secret.id if secret else "",
            secret_path=path,
            action="delete",
            actor=actor,
            service="cosmos",
            timestamp=datetime.utcnow(),
            success=success,
        ))

        if success:
            logger.info(f"Deleted secret: {path}")
        return success

    async def list_secrets(
        self,
        path_prefix: str = "",
        provider: str = None,
    ) -> List[str]:
        """List secrets by path prefix."""
        prov = self.get_provider(provider)
        if not prov:
            return []

        return await prov.list_secrets(path_prefix)

    # =========================================================================
    # SECRET ROTATION
    # =========================================================================

    def register_rotation_callback(
        self,
        secret_type: SecretType,
        callback: Callable,
    ):
        """Register a callback for rotating a secret type."""
        self.rotation_callbacks[secret_type.value] = callback
        logger.info(f"Registered rotation callback for {secret_type.value}")

    async def rotate_secret(
        self,
        path: str,
        provider: str = None,
        actor: str = "system",
    ) -> bool:
        """Rotate a secret."""
        prov = self.get_provider(provider)
        if not prov:
            return False

        secret = await prov.get_secret(path)
        if not secret:
            return False

        # Check if there's a custom rotation callback
        callback = self.rotation_callbacks.get(secret.secret_type.value)
        if callback:
            try:
                new_value = await callback(secret)
                if new_value:
                    secret.current_value = new_value
            except Exception as e:
                logger.error(f"Rotation callback failed: {e}")
                return False

        success = await prov.rotate_secret(path)

        self.audit_logs.append(SecretAccessLog(
            secret_id=secret.id,
            secret_path=path,
            action="rotate",
            actor=actor,
            service="cosmos",
            timestamp=datetime.utcnow(),
            success=success,
        ))

        if success:
            logger.info(f"Rotated secret: {path}")
        return success

    async def check_rotation_needed(
        self,
        provider: str = None,
    ) -> List[Secret]:
        """Check for secrets that need rotation."""
        prov = self.get_provider(provider)
        if not prov:
            return []

        needs_rotation = []
        paths = await prov.list_secrets()

        for path in paths:
            secret = await prov.get_secret(path)
            if not secret:
                continue

            if secret.rotation_enabled and secret.next_rotation:
                if datetime.utcnow() >= secret.next_rotation:
                    needs_rotation.append(secret)

        return needs_rotation

    async def auto_rotate_all(
        self,
        provider: str = None,
    ) -> Dict[str, bool]:
        """Automatically rotate all secrets that need it."""
        results = {}
        secrets = await self.check_rotation_needed(provider)

        for secret in secrets:
            success = await self.rotate_secret(secret.path, provider)
            results[secret.path] = success

        return results

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    async def import_secrets(
        self,
        secrets: Dict[str, str],
        path_prefix: str = "",
        provider: str = None,
        actor: str = "system",
    ) -> Dict[str, bool]:
        """Import multiple secrets from a dictionary."""
        results = {}
        for name, value in secrets.items():
            path = f"{path_prefix}/{name}" if path_prefix else name
            success = await self.create_secret(
                path=path,
                value=value,
                provider=provider,
                actor=actor,
            )
            results[path] = success
        return results

    async def export_secrets(
        self,
        path_prefix: str = "",
        provider: str = None,
        include_values: bool = False,
    ) -> Dict[str, Dict]:
        """Export secrets metadata (optionally with values)."""
        prov = self.get_provider(provider)
        if not prov:
            return {}

        result = {}
        paths = await prov.list_secrets(path_prefix)

        for path in paths:
            secret = await prov.get_secret(path)
            if secret:
                result[path] = secret.to_dict(include_value=include_values)

        return result

    # =========================================================================
    # AUDIT AND REPORTING
    # =========================================================================

    def get_audit_logs(
        self,
        path: str = None,
        action: str = None,
        actor: str = None,
        since: datetime = None,
        limit: int = 100,
    ) -> List[SecretAccessLog]:
        """Get audit logs with optional filters."""
        logs = self.audit_logs

        if path:
            logs = [l for l in logs if l.secret_path == path]
        if action:
            logs = [l for l in logs if l.action == action]
        if actor:
            logs = [l for l in logs if l.actor == actor]
        if since:
            logs = [l for l in logs if l.timestamp >= since]

        return sorted(logs, key=lambda l: l.timestamp, reverse=True)[:limit]

    def generate_report(self) -> Dict[str, Any]:
        """Generate a secrets usage report."""
        total_logs = len(self.audit_logs)
        if not total_logs:
            return {"total_operations": 0}

        by_action = {}
        by_actor = {}
        failed = []

        for log in self.audit_logs:
            by_action[log.action] = by_action.get(log.action, 0) + 1
            by_actor[log.actor] = by_actor.get(log.actor, 0) + 1
            if not log.success:
                failed.append(log)

        return {
            "total_operations": total_logs,
            "by_action": by_action,
            "by_actor": by_actor,
            "failed_operations": len(failed),
            "recent_failures": [
                {
                    "path": f.secret_path,
                    "action": f.action,
                    "actor": f.actor,
                    "timestamp": f.timestamp.isoformat(),
                    "error": f.error_message,
                }
                for f in failed[-10:]
            ],
        }

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def generate_password(
        length: int = 32,
        include_special: bool = True,
    ) -> str:
        """Generate a secure random password."""
        import secrets as sec
        import string

        alphabet = string.ascii_letters + string.digits
        if include_special:
            alphabet += "!@#$%^&*"

        return "".join(sec.choice(alphabet) for _ in range(length))

    @staticmethod
    def generate_api_key(prefix: str = "farn") -> str:
        """Generate an API key with prefix."""
        import secrets as sec
        return f"{prefix}_{sec.token_urlsafe(32)}"


# Singleton instance
vault_manager = VaultManager()
