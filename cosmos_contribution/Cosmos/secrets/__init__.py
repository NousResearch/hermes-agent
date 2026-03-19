"""
cosmos Secrets and Credentials Vault

"I've hidden my most dangerous inventions in a vault...
 but I can never remember the combination!"

Multi-provider secrets management with rotation and audit logging.
"""

from Cosmos.secrets.vault_manager import (
    VaultManager,
    Secret,
    SecretVersion,
    SecretType,
)
from Cosmos.secrets.hashicorp_vault import HashiCorpVaultProvider
from Cosmos.secrets.aws_secrets import AWSSecretsProvider
from Cosmos.secrets.azure_keyvault import AzureKeyVaultProvider

__all__ = [
    "VaultManager",
    "Secret",
    "SecretVersion",
    "SecretType",
    "HashiCorpVaultProvider",
    "AWSSecretsProvider",
    "AzureKeyVaultProvider",
]
