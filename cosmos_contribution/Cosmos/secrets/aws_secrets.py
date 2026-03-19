"""
cosmos AWS Secrets Manager Integration

"Amazon has secrets? I thought they just had packages!"

AWS Secrets Manager for cloud-native secrets management.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from loguru import logger

try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

from Cosmos.secrets.vault_manager import (
    SecretsProvider,
    Secret,
    SecretVersion,
    SecretType,
    SecretStatus,
)


class AWSSecretsProvider(SecretsProvider):
    """
    AWS Secrets Manager provider.

    Features:
    - Secret CRUD operations
    - Automatic rotation with Lambda
    - Versioning
    - Cross-region replication
    - Resource-based policies
    """

    def __init__(
        self,
        region: str = "us-east-1",
        access_key_id: str = None,
        secret_access_key: str = None,
        session_token: str = None,
        profile_name: str = None,
    ):
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for AWS Secrets Manager")

        session_kwargs = {"region_name": region}

        if access_key_id and secret_access_key:
            session_kwargs["aws_access_key_id"] = access_key_id
            session_kwargs["aws_secret_access_key"] = secret_access_key
            if session_token:
                session_kwargs["aws_session_token"] = session_token
        elif profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)
        self.client = session.client("secretsmanager")
        self.region = region

    # =========================================================================
    # SECRET OPERATIONS
    # =========================================================================

    async def get_secret(
        self,
        path: str,
        version: int = None,
    ) -> Optional[Secret]:
        """Get a secret from AWS Secrets Manager."""
        try:
            kwargs = {"SecretId": path}
            if version:
                kwargs["VersionId"] = str(version)

            response = self.client.get_secret_value(**kwargs)

            # Parse the secret value
            if "SecretString" in response:
                secret_value = response["SecretString"]
                try:
                    secret_data = json.loads(secret_value)
                    value = secret_data.get("value", secret_value)
                    metadata = {k: v for k, v in secret_data.items() if k != "value"}
                except json.JSONDecodeError:
                    value = secret_value
                    metadata = {}
            else:
                # Binary secret
                value = response["SecretBinary"].decode()
                metadata = {}

            # Get metadata
            describe = self.client.describe_secret(SecretId=path)

            secret = Secret(
                id=describe["ARN"],
                name=describe["Name"],
                secret_type=SecretType(metadata.get("_type", "generic")),
                path=path,
                current_value=value,
                metadata=metadata,
                tags=[{"Key": t["Key"], "Value": t["Value"]} for t in describe.get("Tags", [])],
                rotation_enabled=describe.get("RotationEnabled", False),
                last_rotated=describe.get("LastRotatedDate"),
                created_at=describe.get("CreatedDate", datetime.utcnow()),
                updated_at=describe.get("LastChangedDate", datetime.utcnow()),
            )

            if describe.get("RotationRules"):
                secret.rotation_days = describe["RotationRules"].get("AutomaticallyAfterDays", 90)

            return secret

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return None
            logger.error(f"AWS Secrets Manager error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get secret: {e}")
            return None

    async def set_secret(self, secret: Secret) -> bool:
        """Create or update a secret in AWS Secrets Manager."""
        try:
            # Prepare secret value as JSON
            secret_data = {
                "value": secret.current_value,
                "_type": secret.secret_type.value,
                **secret.metadata,
            }
            secret_string = json.dumps(secret_data)

            # Try to update existing secret
            try:
                self.client.put_secret_value(
                    SecretId=secret.path,
                    SecretString=secret_string,
                )
                logger.info(f"Updated secret: {secret.path}")
                return True

            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceNotFoundException":
                    raise

            # Create new secret
            create_kwargs = {
                "Name": secret.path,
                "SecretString": secret_string,
            }

            if secret.tags:
                create_kwargs["Tags"] = [
                    {"Key": str(k), "Value": str(v)}
                    for tag in secret.tags
                    for k, v in (tag.items() if isinstance(tag, dict) else [(tag, "")])
                ]

            self.client.create_secret(**create_kwargs)
            logger.info(f"Created secret: {secret.path}")

            # Configure rotation if enabled
            if secret.rotation_enabled and secret.rotation_lambda:
                self.client.rotate_secret(
                    SecretId=secret.path,
                    RotationLambdaARN=secret.rotation_lambda,
                    RotationRules={
                        "AutomaticallyAfterDays": secret.rotation_days,
                    },
                )

            return True

        except Exception as e:
            logger.error(f"Failed to set secret: {e}")
            return False

    async def delete_secret(self, path: str) -> bool:
        """Delete a secret from AWS Secrets Manager."""
        try:
            # Schedule deletion (can be recovered within 7-30 days)
            self.client.delete_secret(
                SecretId=path,
                RecoveryWindowInDays=7,
            )
            logger.info(f"Scheduled deletion for secret: {path}")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return True
            logger.error(f"Failed to delete secret: {e}")
            return False

    async def force_delete_secret(self, path: str) -> bool:
        """Immediately delete a secret (no recovery)."""
        try:
            self.client.delete_secret(
                SecretId=path,
                ForceDeleteWithoutRecovery=True,
            )
            logger.info(f"Force deleted secret: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to force delete secret: {e}")
            return False

    async def restore_secret(self, path: str) -> bool:
        """Restore a deleted secret that's in recovery window."""
        try:
            self.client.restore_secret(SecretId=path)
            logger.info(f"Restored secret: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore secret: {e}")
            return False

    async def list_secrets(self, path_prefix: str = "") -> List[str]:
        """List all secrets, optionally filtered by prefix."""
        try:
            secrets = []
            paginator = self.client.get_paginator("list_secrets")

            filters = []
            if path_prefix:
                filters.append({
                    "Key": "name",
                    "Values": [path_prefix],
                })

            for page in paginator.paginate(Filters=filters if filters else []):
                for secret in page.get("SecretList", []):
                    name = secret["Name"]
                    if not path_prefix or name.startswith(path_prefix):
                        secrets.append(name)

            return secrets

        except Exception as e:
            logger.error(f"Failed to list secrets: {e}")
            return []

    async def rotate_secret(self, path: str) -> bool:
        """Trigger rotation for a secret."""
        try:
            self.client.rotate_secret(SecretId=path)
            logger.info(f"Triggered rotation for: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate secret: {e}")
            return False

    # =========================================================================
    # VERSIONING
    # =========================================================================

    async def list_versions(self, path: str) -> List[Dict]:
        """List all versions of a secret."""
        try:
            response = self.client.list_secret_version_ids(SecretId=path)

            versions = []
            for version in response.get("Versions", []):
                versions.append({
                    "version_id": version["VersionId"],
                    "stages": version.get("VersionStages", []),
                    "created_date": version.get("CreatedDate"),
                    "is_current": "AWSCURRENT" in version.get("VersionStages", []),
                })

            return versions

        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            return []

    async def get_version(
        self,
        path: str,
        version_id: str = None,
        version_stage: str = None,
    ) -> Optional[str]:
        """Get a specific version of a secret."""
        try:
            kwargs = {"SecretId": path}
            if version_id:
                kwargs["VersionId"] = version_id
            if version_stage:
                kwargs["VersionStage"] = version_stage

            response = self.client.get_secret_value(**kwargs)

            if "SecretString" in response:
                return response["SecretString"]
            return response.get("SecretBinary", b"").decode()

        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            return None

    # =========================================================================
    # ROTATION CONFIGURATION
    # =========================================================================

    async def configure_rotation(
        self,
        path: str,
        lambda_arn: str,
        rotation_days: int = 30,
    ) -> bool:
        """Configure automatic rotation for a secret."""
        try:
            self.client.rotate_secret(
                SecretId=path,
                RotationLambdaARN=lambda_arn,
                RotationRules={
                    "AutomaticallyAfterDays": rotation_days,
                },
            )
            logger.info(f"Configured rotation for: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to configure rotation: {e}")
            return False

    async def cancel_rotation(self, path: str) -> bool:
        """Cancel pending rotation for a secret."""
        try:
            self.client.cancel_rotate_secret(SecretId=path)
            logger.info(f"Cancelled rotation for: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel rotation: {e}")
            return False

    # =========================================================================
    # TAGS AND POLICIES
    # =========================================================================

    async def add_tags(
        self,
        path: str,
        tags: Dict[str, str],
    ) -> bool:
        """Add tags to a secret."""
        try:
            self.client.tag_resource(
                SecretId=path,
                Tags=[{"Key": k, "Value": v} for k, v in tags.items()],
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add tags: {e}")
            return False

    async def remove_tags(
        self,
        path: str,
        tag_keys: List[str],
    ) -> bool:
        """Remove tags from a secret."""
        try:
            self.client.untag_resource(
                SecretId=path,
                TagKeys=tag_keys,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to remove tags: {e}")
            return False

    async def put_resource_policy(
        self,
        path: str,
        policy: Dict,
    ) -> bool:
        """Attach a resource-based policy to a secret."""
        try:
            self.client.put_resource_policy(
                SecretId=path,
                ResourcePolicy=json.dumps(policy),
            )
            return True

        except Exception as e:
            logger.error(f"Failed to put resource policy: {e}")
            return False

    async def get_resource_policy(self, path: str) -> Optional[Dict]:
        """Get the resource-based policy for a secret."""
        try:
            response = self.client.get_resource_policy(SecretId=path)
            policy_str = response.get("ResourcePolicy")
            if policy_str:
                return json.loads(policy_str)
            return None

        except Exception as e:
            logger.error(f"Failed to get resource policy: {e}")
            return None

    # =========================================================================
    # REPLICATION
    # =========================================================================

    async def replicate_secret(
        self,
        path: str,
        regions: List[str],
    ) -> bool:
        """Replicate a secret to other regions."""
        try:
            replica_regions = [{"Region": r} for r in regions]

            self.client.replicate_secret_to_regions(
                SecretId=path,
                AddReplicaRegions=replica_regions,
            )
            logger.info(f"Replicated {path} to {regions}")
            return True

        except Exception as e:
            logger.error(f"Failed to replicate secret: {e}")
            return False

    async def remove_replica(
        self,
        path: str,
        regions: List[str],
    ) -> bool:
        """Remove secret replicas from regions."""
        try:
            self.client.remove_regions_from_replication(
                SecretId=path,
                RemoveReplicaRegions=regions,
            )
            return True

        except Exception as e:
            logger.error(f"Failed to remove replica: {e}")
            return False
