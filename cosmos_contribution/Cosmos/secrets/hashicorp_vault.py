"""
cosmos HashiCorp Vault Integration

"A vault within a vault? How delightfully recursive!"

HashiCorp Vault secrets management.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from Cosmos.secrets.vault_manager import (
    SecretsProvider,
    Secret,
    SecretVersion,
    SecretType,
    SecretStatus,
)


class HashiCorpVaultProvider(SecretsProvider):
    """
    HashiCorp Vault secrets provider.

    Features:
    - KV secrets engine (v1 and v2)
    - Dynamic secrets
    - Transit encryption
    - Token and AppRole authentication
    - Namespace support
    """

    def __init__(
        self,
        url: str,
        token: str = None,
        role_id: str = None,
        secret_id: str = None,
        namespace: str = None,
        mount_point: str = "secret",
        kv_version: int = 2,
    ):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for HashiCorp Vault")

        self.url = url.rstrip("/")
        self.token = token
        self.namespace = namespace
        self.mount_point = mount_point
        self.kv_version = kv_version

        # If using AppRole auth, we need to authenticate first
        if role_id and secret_id and not token:
            self._role_id = role_id
            self._secret_id = secret_id
        else:
            self._role_id = None
            self._secret_id = None

    async def _get_headers(self) -> Dict[str, str]:
        """Get request headers including auth token."""
        headers = {"Content-Type": "application/json"}

        if self.token:
            headers["X-Vault-Token"] = self.token
        elif self._role_id and self._secret_id:
            # Authenticate with AppRole
            self.token = await self._approle_login()
            headers["X-Vault-Token"] = self.token

        if self.namespace:
            headers["X-Vault-Namespace"] = self.namespace

        return headers

    async def _approle_login(self) -> str:
        """Authenticate using AppRole."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.url}/v1/auth/approle/login",
                json={
                    "role_id": self._role_id,
                    "secret_id": self._secret_id,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["auth"]["client_token"]

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> Optional[Dict]:
        """Make an authenticated request to Vault."""
        headers = await self._get_headers()
        url = f"{self.url}/v1/{path}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs,
                )

                if response.status_code == 404:
                    return None

                response.raise_for_status()

                if response.status_code == 204:
                    return {}
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"Vault API error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"Vault request failed: {e}")
                return None

    def _get_kv_path(self, secret_path: str, data_path: bool = True) -> str:
        """Get the correct KV path based on version."""
        if self.kv_version == 2:
            if data_path:
                return f"{self.mount_point}/data/{secret_path}"
            return f"{self.mount_point}/metadata/{secret_path}"
        return f"{self.mount_point}/{secret_path}"

    # =========================================================================
    # SECRET OPERATIONS
    # =========================================================================

    async def get_secret(
        self,
        path: str,
        version: int = None,
    ) -> Optional[Secret]:
        """Get a secret from Vault."""
        kv_path = self._get_kv_path(path)

        params = {}
        if version and self.kv_version == 2:
            params["version"] = version

        result = await self._request("GET", kv_path, params=params)

        if not result:
            return None

        # Parse based on KV version
        if self.kv_version == 2:
            data = result.get("data", {})
            secret_data = data.get("data", {})
            metadata = data.get("metadata", {})

            secret = Secret(
                id=path,
                name=path.split("/")[-1],
                secret_type=SecretType(secret_data.get("_type", "generic")),
                path=path,
                current_value=secret_data.get("value", ""),
                metadata={k: v for k, v in secret_data.items() if not k.startswith("_")},
                created_at=datetime.fromisoformat(metadata.get("created_time", "").replace("Z", "+00:00")) if metadata.get("created_time") else datetime.utcnow(),
                updated_at=datetime.fromisoformat(metadata.get("created_time", "").replace("Z", "+00:00")) if metadata.get("created_time") else datetime.utcnow(),
            )
        else:
            secret_data = result.get("data", {})
            secret = Secret(
                id=path,
                name=path.split("/")[-1],
                secret_type=SecretType(secret_data.get("_type", "generic")),
                path=path,
                current_value=secret_data.get("value", ""),
                metadata={k: v for k, v in secret_data.items() if not k.startswith("_")},
            )

        return secret

    async def set_secret(self, secret: Secret) -> bool:
        """Create or update a secret in Vault."""
        kv_path = self._get_kv_path(secret.path)

        # Prepare data
        data = {
            "value": secret.current_value,
            "_type": secret.secret_type.value,
            **secret.metadata,
        }

        if self.kv_version == 2:
            payload = {"data": data}
        else:
            payload = data

        result = await self._request("POST", kv_path, json=payload)
        return result is not None

    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from Vault."""
        if self.kv_version == 2:
            # Delete all versions
            kv_path = self._get_kv_path(path, data_path=False)
            result = await self._request("DELETE", kv_path)
        else:
            kv_path = self._get_kv_path(path)
            result = await self._request("DELETE", kv_path)

        return result is not None

    async def list_secrets(self, path_prefix: str = "") -> List[str]:
        """List secrets at a path."""
        if self.kv_version == 2:
            list_path = f"{self.mount_point}/metadata/{path_prefix}"
        else:
            list_path = f"{self.mount_point}/{path_prefix}"

        result = await self._request("LIST", list_path)

        if not result:
            return []

        keys = result.get("data", {}).get("keys", [])

        # Recursively list subdirectories
        all_paths = []
        for key in keys:
            full_path = f"{path_prefix}/{key}" if path_prefix else key
            if key.endswith("/"):
                # It's a directory, recurse
                subpaths = await self.list_secrets(full_path.rstrip("/"))
                all_paths.extend(subpaths)
            else:
                all_paths.append(full_path)

        return all_paths

    async def rotate_secret(self, path: str) -> bool:
        """Rotate a secret (create new version with new value)."""
        secret = await self.get_secret(path)
        if not secret:
            return False

        # Generate new value
        import secrets as sec
        secret.current_value = sec.token_urlsafe(32)
        secret.last_rotated = datetime.utcnow()
        secret.next_rotation = datetime.utcnow() + timedelta(days=secret.rotation_days)

        return await self.set_secret(secret)

    # =========================================================================
    # VERSION MANAGEMENT (KV v2 only)
    # =========================================================================

    async def list_versions(self, path: str) -> List[Dict]:
        """List all versions of a secret."""
        if self.kv_version != 2:
            logger.warning("Version listing only available for KV v2")
            return []

        metadata_path = self._get_kv_path(path, data_path=False)
        result = await self._request("GET", metadata_path)

        if not result:
            return []

        versions = result.get("data", {}).get("versions", {})
        return [
            {
                "version": int(v),
                "created_time": meta.get("created_time"),
                "deleted": meta.get("deletion_time") is not None,
                "destroyed": meta.get("destroyed", False),
            }
            for v, meta in versions.items()
        ]

    async def destroy_version(self, path: str, versions: List[int]) -> bool:
        """Permanently destroy specific versions of a secret."""
        if self.kv_version != 2:
            return False

        destroy_path = f"{self.mount_point}/destroy/{path}"
        result = await self._request(
            "POST",
            destroy_path,
            json={"versions": versions},
        )
        return result is not None

    async def undelete_version(self, path: str, versions: List[int]) -> bool:
        """Undelete specific versions of a secret."""
        if self.kv_version != 2:
            return False

        undelete_path = f"{self.mount_point}/undelete/{path}"
        result = await self._request(
            "POST",
            undelete_path,
            json={"versions": versions},
        )
        return result is not None

    # =========================================================================
    # TRANSIT ENGINE (Encryption as a Service)
    # =========================================================================

    async def encrypt(
        self,
        key_name: str,
        plaintext: str,
        mount_point: str = "transit",
    ) -> Optional[str]:
        """Encrypt data using Transit engine."""
        import base64

        # Base64 encode the plaintext
        encoded = base64.b64encode(plaintext.encode()).decode()

        result = await self._request(
            "POST",
            f"{mount_point}/encrypt/{key_name}",
            json={"plaintext": encoded},
        )

        if result:
            return result.get("data", {}).get("ciphertext")
        return None

    async def decrypt(
        self,
        key_name: str,
        ciphertext: str,
        mount_point: str = "transit",
    ) -> Optional[str]:
        """Decrypt data using Transit engine."""
        import base64

        result = await self._request(
            "POST",
            f"{mount_point}/decrypt/{key_name}",
            json={"ciphertext": ciphertext},
        )

        if result:
            plaintext_b64 = result.get("data", {}).get("plaintext")
            if plaintext_b64:
                return base64.b64decode(plaintext_b64).decode()
        return None

    # =========================================================================
    # DYNAMIC SECRETS
    # =========================================================================

    async def get_database_creds(
        self,
        role: str,
        mount_point: str = "database",
    ) -> Optional[Dict]:
        """Get dynamic database credentials."""
        result = await self._request(
            "GET",
            f"{mount_point}/creds/{role}",
        )

        if result:
            data = result.get("data", {})
            lease = result.get("lease_id")
            return {
                "username": data.get("username"),
                "password": data.get("password"),
                "lease_id": lease,
                "lease_duration": result.get("lease_duration"),
            }
        return None

    async def get_aws_creds(
        self,
        role: str,
        mount_point: str = "aws",
    ) -> Optional[Dict]:
        """Get dynamic AWS credentials."""
        result = await self._request(
            "GET",
            f"{mount_point}/creds/{role}",
        )

        if result:
            data = result.get("data", {})
            return {
                "access_key": data.get("access_key"),
                "secret_key": data.get("secret_key"),
                "security_token": data.get("security_token"),
                "lease_id": result.get("lease_id"),
                "lease_duration": result.get("lease_duration"),
            }
        return None

    async def revoke_lease(self, lease_id: str) -> bool:
        """Revoke a dynamic secret lease."""
        result = await self._request(
            "POST",
            "sys/leases/revoke",
            json={"lease_id": lease_id},
        )
        return result is not None

    # =========================================================================
    # TOKEN MANAGEMENT
    # =========================================================================

    async def lookup_self(self) -> Optional[Dict]:
        """Get information about the current token."""
        result = await self._request("GET", "auth/token/lookup-self")
        if result:
            return result.get("data")
        return None

    async def renew_self(self, increment: int = None) -> bool:
        """Renew the current token."""
        payload = {}
        if increment:
            payload["increment"] = increment

        result = await self._request("POST", "auth/token/renew-self", json=payload)
        return result is not None

    async def create_token(
        self,
        policies: List[str] = None,
        ttl: str = "1h",
        renewable: bool = True,
    ) -> Optional[Dict]:
        """Create a new token."""
        payload = {
            "ttl": ttl,
            "renewable": renewable,
        }
        if policies:
            payload["policies"] = policies

        result = await self._request("POST", "auth/token/create", json=payload)
        if result:
            auth = result.get("auth", {})
            return {
                "token": auth.get("client_token"),
                "accessor": auth.get("accessor"),
                "policies": auth.get("policies"),
                "lease_duration": auth.get("lease_duration"),
            }
        return None

    # =========================================================================
    # HEALTH AND STATUS
    # =========================================================================

    async def health(self) -> Optional[Dict]:
        """Get Vault health status."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.url}/v1/sys/health")
                return response.json()
            except Exception as e:
                logger.error(f"Failed to get Vault health: {e}")
                return None

    async def seal_status(self) -> Optional[Dict]:
        """Get Vault seal status."""
        result = await self._request("GET", "sys/seal-status")
        return result
