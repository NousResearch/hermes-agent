"""
wallet_manager.py - Burner Wallet Manager
===========================================
Functions:
- Create burner wallets
- Import encrypted wallets
- List wallets
- Balance checker per wallet
- Encrypt sensitive data (NEVER expose private keys)
- Future OWS (Open Wallet Standard) compatibility placeholder

SECURITY:
- Private keys are NEVER printed or logged
- All keys stored encrypted with Fernet
- Wallet files stored in .wallets/ directory

Usage:
    python -m custom_tools.wallet_manager create --label "burner1"
    python -m custom_tools.wallet_manager list
    python -m custom_tools.wallet_manager balance --label "burner1" --chain ethereum
"""

import os
import sys
import json
import hashlib
import getpass
from pathlib import Path
from datetime import datetime

from eth_account import Account
from web3 import Web3
from cryptography.fernet import Fernet

from custom_tools.check_wallet import get_web3



# Wallet storage directory
WALLETS_DIR = Path(os.getenv("WALLETS_DIR", ".wallets"))
WALLETS_DIR.mkdir(exist_ok=True)


def _get_encryption_key() -> bytes:
    """Get or create encryption key for wallet storage."""
    key_file = WALLETS_DIR / ".key"
    if key_file.exists():
        return key_file.read_bytes()
    else:
        key = Fernet.generate_key()
        key_file.write_bytes(key)
        os.chmod(str(key_file), 0o600)
        return key


def _encrypt(data: str) -> str:
    """Encrypt sensitive data."""
    key = _get_encryption_key()
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()


def _decrypt(encrypted: str) -> str:
    """Decrypt sensitive data."""
    key = _get_encryption_key()
    f = Fernet(key)
    return f.decrypt(encrypted.encode()).decode()



def create_burner_wallet(label: str) -> dict:
    """
    Create a new burner wallet with encrypted key storage.
    
    Args:
        label: Human-readable label for the wallet
    
    Returns:
        dict with address and label (NEVER returns private key)
    """
    account = Account.create()
    address = account.address
    
    # Encrypt private key - NEVER store plaintext
    encrypted_key = _encrypt(account.key.hex())
    
    wallet_data = {
        "label": label,
        "address": address,
        "encrypted_key": encrypted_key,
        "created_at": datetime.utcnow().isoformat(),
        "type": "burner",
    }
    
    # Save wallet file
    wallet_file = WALLETS_DIR / f"{label}.json"
    with open(wallet_file, "w") as f:
        json.dump(wallet_data, f, indent=2)
    os.chmod(str(wallet_file), 0o600)
    
    # Return safe data only - NO PRIVATE KEY
    return {
        "label": label,
        "address": address,
        "created_at": wallet_data["created_at"],
        "status": "created",
        "warning": "Private key stored encrypted. NEVER share or expose.",
    }



def import_wallet(label: str, private_key: str) -> dict:
    """
    Import an existing wallet with encrypted storage.
    
    Args:
        label: Label for the wallet
        private_key: Private key (will be encrypted immediately)
    
    Returns:
        dict with address and label (NEVER returns private key)
    """
    # Validate key by deriving address
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key
    
    account = Account.from_key(private_key)
    address = account.address
    
    # Encrypt immediately
    encrypted_key = _encrypt(private_key)
    
    wallet_data = {
        "label": label,
        "address": address,
        "encrypted_key": encrypted_key,
        "created_at": datetime.utcnow().isoformat(),
        "type": "imported",
    }
    
    wallet_file = WALLETS_DIR / f"{label}.json"
    with open(wallet_file, "w") as f:
        json.dump(wallet_data, f, indent=2)
    os.chmod(str(wallet_file), 0o600)
    
    # Clear private key from memory reference
    private_key = None
    
    return {
        "label": label,
        "address": address,
        "status": "imported",
        "warning": "Private key stored encrypted. Original cleared.",
    }



def list_wallets() -> list:
    """List all stored wallets (addresses only, no keys)."""
    wallets = []
    for wallet_file in WALLETS_DIR.glob("*.json"):
        try:
            with open(wallet_file) as f:
                data = json.load(f)
            wallets.append({
                "label": data.get("label", wallet_file.stem),
                "address": data.get("address", "unknown"),
                "type": data.get("type", "unknown"),
                "created_at": data.get("created_at", "unknown"),
            })
        except Exception:
            continue
    return wallets


def get_wallet_key(label: str) -> str:
    """
    Get decrypted private key for transaction signing.
    INTERNAL USE ONLY - never expose to logs or output.
    
    Args:
        label: Wallet label
    
    Returns:
        Decrypted private key string
    """
    wallet_file = WALLETS_DIR / f"{label}.json"
    if not wallet_file.exists():
        raise FileNotFoundError(f"Wallet '{label}' not found")
    
    with open(wallet_file) as f:
        data = json.load(f)
    
    return _decrypt(data["encrypted_key"])



def check_wallet_balance(label: str, chain: str = "ethereum") -> dict:
    """Check balance of a stored wallet."""
    wallet_file = WALLETS_DIR / f"{label}.json"
    if not wallet_file.exists():
        raise FileNotFoundError(f"Wallet '{label}' not found")
    
    with open(wallet_file) as f:
        data = json.load(f)
    
    address = data["address"]
    w3 = get_web3(chain)
    balance_wei = w3.eth.get_balance(Web3.to_checksum_address(address))
    
    return {
        "label": label,
        "address": address,
        "chain": chain,
        "balance_wei": str(balance_wei),
        "balance_eth": str(Web3.from_wei(balance_wei, "ether")),
    }


def import_wallets_from_csv(csv_path: str) -> list:
    """
    Import multiple wallets from CSV file.
    CSV format: label,private_key
    """
    import csv
    results = []
    
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                label, key = row[0].strip(), row[1].strip()
                if label and key and not label.startswith("#"):
                    try:
                        result = import_wallet(label, key)
                        results.append(result)
                    except Exception as e:
                        results.append({"label": label, "error": str(e)})
    
    return results



# === OWS (Open Wallet Standard) Placeholder ===
class OWSAdapter:
    """
    Future: Open Wallet Standard adapter.
    Placeholder for OWS protocol integration.
    """
    
    def __init__(self):
        self.supported = False
        self.version = "placeholder"
    
    def connect(self):
        raise NotImplementedError("OWS support coming in future version")
    
    def sign_transaction(self, tx):
        raise NotImplementedError("OWS support coming in future version")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Burner Wallet Manager")
    sub = parser.add_subparsers(dest="command")
    
    # Create command
    create_p = sub.add_parser("create", help="Create new burner wallet")
    create_p.add_argument("--label", required=True, help="Wallet label")
    
    # List command
    sub.add_parser("list", help="List all wallets")
    
    # Balance command
    bal_p = sub.add_parser("balance", help="Check wallet balance")
    bal_p.add_argument("--label", required=True, help="Wallet label")
    bal_p.add_argument("--chain", default="ethereum", help="Chain")
    
    # Import CSV command
    imp_p = sub.add_parser("import-csv", help="Import wallets from CSV")
    imp_p.add_argument("--file", required=True, help="CSV file path")
    
    args = parser.parse_args()
    
    if args.command == "create":
        result = create_burner_wallet(args.label)
        print(json.dumps(result, indent=2))
    elif args.command == "list":
        wallets = list_wallets()
        print(json.dumps(wallets, indent=2))
    elif args.command == "balance":
        result = check_wallet_balance(args.label, args.chain)
        print(json.dumps(result, indent=2))
    elif args.command == "import-csv":
        results = import_wallets_from_csv(args.file)
        print(json.dumps(results, indent=2))
    else:
        parser.print_help()
