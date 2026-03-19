"""DropClaw SDK — Permanent Encrypted On-Chain Storage for AI Agents."""

from dropclaw.client import VaultClient
from dropclaw.crypto import encrypt, decrypt

__all__ = ["VaultClient", "encrypt", "decrypt"]
__version__ = "1.0.0"
