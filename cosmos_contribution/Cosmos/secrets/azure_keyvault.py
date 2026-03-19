"""
cosmos Azure Key Vault Integration

"Microsoft's vault? I hope it doesn't need rebooting!"

Azure Key Vault for secrets, keys, and certificates.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from loguru import logger

try:
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.keyvault.secrets import SecretClient
    from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False

from Cosmos.secrets.vault_manager import (
    SecretsProvider,
    Secret,
    SecretVersion,
    SecretType,
    SecretStatus,
)


class AzureKeyVaultProvider(SecretsProvider):
    """
    Azure Key Vault secrets provider.

    Features:
    - Secret CRUD operations
    - Versioning
    - Soft delete and purge protection
    - Managed identity support
    - Certificate management
    """

    def __init__(
        self,
        vault_url: str,
        tenant_id: str = None,
        client_id: str = None,
        client_secret: str = None,
    ):
        if not HAS_AZURE:
            raise ImportError("azure-keyvault-secrets is required for Azure Key Vault")

        self.vault_url = vault_url

        # Use service principal if credentials provided, otherwise default
        if tenant_id and client_id and client_secret:
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret,
            )
        else:
            credential = DefaultAzureCredential()

        self.client = SecretClient(
            vault_url=vault_url,
            credential=credential,
        )

    # =========================================================================
    # SECRET OPERATIONS
    # =========================================================================

    async def get_secret(
        self,
        path: str,
        version: int = None,
    ) -> Optional[Secret]:
        """Get a secret from Azure Key Vault."""
        try:
            # Azure uses the secret name directly (no hierarchy in path)
            name = path.replace("/", "-")  # Convert path to valid name

            if version:
                azure_secret = self.client.get_secret(name, version=str(version))
            else:
                azure_secret = self.client.get_secret(name)

            # Parse value as JSON if possible
            try:
                secret_data = json.loads(azure_secret.value)
                value = secret_data.get("value", azure_secret.value)
                metadata = {k: v for k, v in secret_data.items() if k != "value"}
            except (json.JSONDecodeError, TypeError):
                value = azure_secret.value
                metadata = {}

            # Add Azure properties to metadata
            metadata["content_type"] = azure_secret.properties.content_type

            secret = Secret(
                id=azure_secret.properties.id,
                name=azure_secret.name,
                secret_type=SecretType(metadata.get("_type", "generic")),
                path=path,
                current_value=value,
                metadata=metadata,
                tags=list(azure_secret.properties.tags.keys()) if azure_secret.properties.tags else [],
                created_at=azure_secret.properties.created_on or datetime.utcnow(),
                updated_at=azure_secret.properties.updated_on or datetime.utcnow(),
            )

            if azure_secret.properties.expires_on:
                secret.next_rotation = azure_secret.properties.expires_on
                secret.rotation_enabled = True

            return secret

        except ResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Azure Key Vault error: {e}")
            return None

    async def set_secret(self, secret: Secret) -> bool:
        """Create or update a secret in Azure Key Vault."""
        try:
            name = secret.path.replace("/", "-")

            # Prepare secret value as JSON
            secret_data = {
                "value": secret.current_value,
                "_type": secret.secret_type.value,
                **secret.metadata,
            }
            secret_string = json.dumps(secret_data)

            # Set optional properties
            kwargs = {}

            if secret.tags:
                kwargs["tags"] = {str(tag): "" for tag in secret.tags}

            if secret.rotation_enabled and secret.rotation_days:
                kwargs["expires_on"] = datetime.utcnow() + timedelta(days=secret.rotation_days)

            self.client.set_secret(
                name=name,
                value=secret_string,
                content_type="application/json",
                **kwargs,
            )

            logger.info(f"Set secret: {secret.path}")
            return True

        except Exception as e:
            logger.error(f"Failed to set secret: {e}")
            return False

    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from Azure Key Vault."""
        try:
            name = path.replace("/", "-")

            # Begin delete operation
            poller = self.client.begin_delete_secret(name)
            poller.wait()

            logger.info(f"Deleted secret: {path}")
            return True

        except ResourceNotFoundError:
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret: {e}")
            return False

    async def purge_secret(self, path: str) -> bool:
        """Permanently purge a deleted secret."""
        try:
            name = path.replace("/", "-")
            self.client.purge_deleted_secret(name)
            logger.info(f"Purged secret: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to purge secret: {e}")
            return False

    async def recover_secret(self, path: str) -> bool:
        """Recover a deleted secret."""
        try:
            name = path.replace("/", "-")
            poller = self.client.begin_recover_deleted_secret(name)
            poller.wait()
            logger.info(f"Recovered secret: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to recover secret: {e}")
            return False

    async def list_secrets(self, path_prefix: str = "") -> List[str]:
        """List all secrets in the vault."""
        try:
            secrets = []
            prefix = path_prefix.replace("/", "-") if path_prefix else ""

            for secret_properties in self.client.list_properties_of_secrets():
                name = secret_properties.name
                # Convert back to path format
                path = name.replace("-", "/")

                if not prefix or name.startswith(prefix):
                    secrets.append(path)

            return secrets

        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []

    async def list_deleted_secrets(self) -> List[str]:
        """List deleted secrets that can be recovered."""
        try:
            secrets = []
            for deleted_secret in self.client.list_deleted_secrets():
                secrets.append(deleted_secret.name.replace("-", "/"))
            return secrets

        except Exception as e:
            logger.error(f"Failed to list deleted secrets: {e}")
            return []

    async def rotate_secret(self, path: str) -> bool:
        """Rotate a secret by creating a new version."""
        try:
            secret = await self.get_secret(path)
            if not secret:
                return False

            # Generate new value
            import secrets as sec
            secret.current_value = sec.token_urlsafe(32)
            secret.last_rotated = datetime.utcnow()
            secret.next_rotation = datetime.utcnow() + timedelta(days=secret.rotation_days)

            return await self.set_secret(secret)

        except Exception as e:
            logger.error(f"Failed to rotate secret: {e}")
            return False

    # =========================================================================
    # VERSIONING
    # =========================================================================

    async def list_versions(self, path: str) -> List[Dict]:
        """List all versions of a secret."""
        try:
            name = path.replace("/", "-")
            versions = []

            for version in self.client.list_properties_of_secret_versions(name):
                versions.append({
                    "version": version.version,
                    "created_on": version.created_on,
                    "updated_on": version.updated_on,
                    "enabled": version.enabled,
                    "expires_on": version.expires_on,
                    "is_current": version.version == self.client.get_secret(name).properties.version,
                })

            return versions

        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []

    async def get_version(
        self,
        path: str,
        version: str,
    ) -> Optional[str]:
        """Get a specific version of a secret."""
        try:
            name = path.replace("/", "-")
            azure_secret = self.client.get_secret(name, version=version)
            return azure_secret.value

        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            return None

    # =========================================================================
    # PROPERTIES MANAGEMENT
    # =========================================================================

    async def update_properties(
        self,
        path: str,
        enabled: bool = None,
        expires_on: datetime = None,
        not_before: datetime = None,
        content_type: str = None,
        tags: Dict[str, str] = None,
    ) -> bool:
        """Update secret properties without changing the value."""
        try:
            name = path.replace("/", "-")

            # Get current properties
            current = self.client.get_secret(name)

            kwargs = {}
            if enabled is not None:
                kwargs["enabled"] = enabled
            if expires_on:
                kwargs["expires_on"] = expires_on
            if not_before:
                kwargs["not_before"] = not_before
            if content_type:
                kwargs["content_type"] = content_type
            if tags:
                kwargs["tags"] = tags

            self.client.update_secret_properties(
                name,
                version=current.properties.version,
                **kwargs,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update properties: {e}")
            return False

    # =========================================================================
    # BACKUP AND RESTORE
    # =========================================================================

    async def backup_secret(self, path: str) -> Optional[bytes]:
        """Backup a secret."""
        try:
            name = path.replace("/", "-")
            backup = self.client.backup_secret(name)
            return backup

        except Exception as e:
            logger.error(f"Failed to backup secret: {e}")
            return None

    async def restore_secret(self, backup: bytes) -> bool:
        """Restore a secret from backup."""
        try:
            self.client.restore_secret_backup(backup)
            return True

        except Exception as e:
            logger.error(f"Failed to restore secret: {e}")
            return False

    # =========================================================================
    # BULK OPERATIONS
    # =========================================================================

    async def export_all(self, include_values: bool = False) -> Dict[str, Dict]:
        """Export all secrets metadata."""
        result = {}

        try:
            for secret_properties in self.client.list_properties_of_secrets():
                path = secret_properties.name.replace("-", "/")

                if include_values:
                    secret = await self.get_secret(path)
                    if secret:
                        result[path] = secret.to_dict(include_value=True)
                else:
                    result[path] = {
                        "name": secret_properties.name,
                        "enabled": secret_properties.enabled,
                        "created_on": secret_properties.created_on.isoformat() if secret_properties.created_on else None,
                        "updated_on": secret_properties.updated_on.isoformat() if secret_properties.updated_on else None,
                        "expires_on": secret_properties.expires_on.isoformat() if secret_properties.expires_on else None,
                        "content_type": secret_properties.content_type,
                        "tags": dict(secret_properties.tags) if secret_properties.tags else {},
                    }

            return result

        except Exception as e:
            logger.error(f"Failed to export secrets: {e}")
            return {}
