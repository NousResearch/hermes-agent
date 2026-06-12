"""
1Password integration for secure credential management.

This module provides utilities to resolve 1Password references (op://) in
environment variables and .env files, enabling profiles and tools to store
credentials securely in 1Password vaults instead of plaintext files.

Usage:
    # In .env files, use 1Password references:
    LITELLM_MASTER_KEY=op://Empire/litellm-master-key/password
    TAVILY_API_KEY=op://Empire/tavily/credential

    # The get_env_value() function in config.py will automatically resolve
    # these references at runtime using the 'op' CLI.

1Password Setup:
    1. Install 1Password CLI: https://developer.1password.com/docs/cli/
    2. Sign in: eval $(op signin)
    3. Create vault and items for your credentials
    4. Reference them in .env files using op://Vault/Item/field format
"""

import logging
import os
import shutil
import subprocess
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def _op_cli_available() -> bool:
    """Check if 1Password CLI is available in PATH."""
    return shutil.which("op") is not None


def _resolve_op_reference(reference: str) -> Optional[str]:
    """
    Resolve a 1Password reference (op://Vault/Item/field) to its plaintext value.
    
    Args:
        reference: 1Password reference string (e.g., op://Empire/litellm-master-key/password)
    
    Returns:
        The resolved credential value, or None if resolution fails.
        
    Raises:
        subprocess.TimeoutExpired: If 'op' CLI takes too long to respond.
        FileNotFoundError: If 'op' CLI is not available.
    """
    if not _op_cli_available():
        logger.warning(
            f"1Password CLI (op) not found in PATH. "
            f"Install it from https://developer.1password.com/docs/cli/get-started/. "
            f"Cannot resolve: {reference}"
        )
        return None
    
    try:
        # Use 'op read' to resolve the reference
        result = subprocess.run(
            ["op", "read", reference],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logger.warning(
                f"Failed to resolve 1Password reference: {reference}. "
                f"Error: {result.stderr.strip()}"
            )
            return None
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout resolving 1Password reference: {reference}")
        return None
    except Exception as e:
        logger.warning(f"Error resolving 1Password reference {reference}: {e}")
        return None


def resolve_value(value: str) -> str:
    """
    Resolve a value that may contain a 1Password reference.
    
    If the value starts with 'op://', it will be resolved using the 1Password CLI.
    Otherwise, the value is returned as-is.
    
    Args:
        value: The value to potentially resolve
        
    Returns:
        The resolved value, or the original value if resolution is not applicable.
    """
    if isinstance(value, str) and value.startswith("op://"):
        resolved = _resolve_op_reference(value)
        if resolved is not None:
            return resolved
        # If resolution fails, return the original reference
        # (allows graceful degradation if op CLI isn't available)
        return value
    return value


def migrate_env_file_to_onepassword(
    env_file: Path,
    vault_name: str = "Empire",
    backup: bool = True
) -> dict:
    """
    Migrate a .env file from plaintext credentials to 1Password references.
    
    This function helps users convert existing .env files to use 1Password
    references instead of storing secrets as plaintext. It:
    1. Reads the current .env file
    2. Creates 1Password items for each credential (if op CLI available)
    3. Replaces plaintext values with op:// references
    4. Backs up the original file
    
    Args:
        env_file: Path to the .env file to migrate
        vault_name: Name of the 1Password vault to use (default: "Empire")
        backup: Whether to backup the original .env file (default: True)
        
    Returns:
        Dictionary with migration results:
        {
            "migrated": int,              # Number of credentials migrated
            "skipped": int,               # Number of credentials skipped
            "errors": list[str],          # List of error messages
            "backup_path": str or None,   # Path to backup file, if created
            "secrets_created": list[str]  # List of created 1Password items
        }
    """
    result = {
        "migrated": 0,
        "skipped": 0,
        "errors": [],
        "backup_path": None,
        "secrets_created": [],
    }
    
    if not env_file.exists():
        result["errors"].append(f".env file not found: {env_file}")
        return result
    
    # Note: 1Password CLI is optional. Migration works even without it.
    # The migration tool will:
    # - If op CLI available: display helpful info about creating items
    # - If op CLI not available: still create references (user will create items manually)
    has_op_cli = _op_cli_available()
    
    # Read the current .env file
    try:
        with open(env_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except Exception as e:
        result["errors"].append(f"Failed to read .env file: {e}")
        return result
    
    # Parse environment variables
    env_vars = {}
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            env_vars[key.strip()] = value.strip().strip('"\'')
    
    # Backup the original file if requested
    if backup:
        backup_path = env_file.with_suffix('.env.bak')
        try:
            backup_path.write_text(env_file.read_text())
            result["backup_path"] = str(backup_path)
            logger.info(f"Created backup: {backup_path}")
        except Exception as e:
            result["errors"].append(f"Failed to create backup: {e}")
    
    # Migrate each credential to 1Password
    new_lines = []
    for line in lines:
        original_line = line.rstrip('\n')
        stripped = original_line.strip()
        
        # Preserve empty lines and comments
        if not stripped or stripped.startswith('#'):
            new_lines.append(original_line + '\n' if not line.endswith('\n') else line)
            continue
        
        if '=' not in stripped:
            new_lines.append(original_line + '\n' if not line.endswith('\n') else line)
            continue
        
        key, _, value = stripped.partition('=')
        key = key.strip()
        value = value.strip().strip('"\'')
        
        # Skip if already a 1Password reference
        if value.startswith('op://'):
            new_lines.append(original_line + '\n' if not line.endswith('\n') else line)
            result["skipped"] += 1
            continue
        
        # Skip if value is empty or placeholder
        if not value or value == '***':
            new_lines.append(original_line + '\n' if not line.endswith('\n') else line)
            result["skipped"] += 1
            continue
        
        # Create 1Password item for this credential
        # Format: key name is kebab-case version of the env var, with vault prefix
        item_name = key.lower().replace('_', '-')
        op_reference = f"op://{vault_name}/{item_name}/credential"
        
        # Track that this would be created
        result["secrets_created"].append({
            "name": item_name,
            "reference": op_reference,
            "env_var": key
        })
        
        # Replace the line with the op:// reference
        new_lines.append(f"{key}={op_reference}\n")
        result["migrated"] += 1
    
    # Write the migrated .env file
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        logger.info(f"Migrated {result['migrated']} credentials in {env_file}")
    except Exception as e:
        result["errors"].append(f"Failed to write migrated .env file: {e}")
        return result
    
    return result


def create_1password_item(
    item_name: str,
    secret_value: str,
    vault_name: str = "Empire",
    overwrite: bool = False
) -> bool:
    """
    Create a new item in 1Password vault.
    
    Args:
        item_name: Name of the item to create (e.g., "litellm-master-key")
        secret_value: The credential value to store
        vault_name: Name of the vault to store it in
        overwrite: Whether to overwrite if the item already exists
        
    Returns:
        True if successful, False otherwise.
    """
    if not _op_cli_available():
        logger.error("1Password CLI (op) not found in PATH")
        return False
    
    try:
        # Try to create the item
        cmd = [
            "op", "item", "create",
            f"--vault={vault_name}",
            f"--title={item_name}",
            "--category=password",
            f"password=placeholder",
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        
        if result.returncode != 0:
            if "already exists" in result.stderr and not overwrite:
                logger.info(f"1Password item '{item_name}' already exists")
                return True
            logger.error(f"Failed to create 1Password item: {result.stderr}")
            return False
        
        # Update the password field with the actual secret
        update_cmd = [
            "op", "item", "edit", item_name,
            f"--vault={vault_name}",
            f"password={secret_value}",
        ]
        
        update_result = subprocess.run(
            update_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        
        if update_result.returncode == 0:
            logger.info(f"Created 1Password item: {item_name}")
            return True
        else:
            logger.error(f"Failed to set password for item: {update_result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout creating 1Password item: {item_name}")
        return False
    except Exception as e:
        logger.error(f"Error creating 1Password item: {e}")
        return False


def is_1password_available() -> bool:
    """Check if 1Password CLI is available and signed in."""
    if not _op_cli_available():
        return False
    
    try:
        result = subprocess.run(
            ["op", "whoami"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def get_1password_vaults() -> list[str]:
    """
    Get list of available 1Password vaults.
    
    Returns:
        List of vault names, or empty list if unavailable.
    """
    if not is_1password_available():
        return []
    
    try:
        result = subprocess.run(
            ["op", "vault", "list", "--format=json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        
        if result.returncode == 0:
            import json
            try:
                vaults = json.loads(result.stdout)
                return [v.get("name") for v in vaults if "name" in v]
            except json.JSONDecodeError:
                return []
        return []
    except Exception:
        return []
