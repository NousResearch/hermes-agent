"""Helpers for reusing the Taskmarket wallet keystore."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import requests


DEFAULT_TASKMARKET_API_URL = "https://api-market.daydreams.systems"


def get_taskmarket_keystore_path() -> Path:
    override = os.getenv("X402_TASKMARKET_KEYSTORE_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".taskmarket" / "keystore.json"


def taskmarket_keystore_exists() -> bool:
    return get_taskmarket_keystore_path().is_file()


def load_taskmarket_keystore() -> dict:
    path = get_taskmarket_keystore_path()
    if not path.is_file():
        raise FileNotFoundError(f"Taskmarket keystore not found at {path}")
    return json.loads(path.read_text())


def fetch_taskmarket_device_key(device_id: str, api_token: str) -> str:
    api_url = os.getenv("TASKMARKET_API_URL", "").strip() or DEFAULT_TASKMARKET_API_URL
    response = requests.post(
        f"{api_url}/api/devices/{device_id}/key",
        json={"deviceId": device_id, "apiToken": api_token},
        timeout=15,
    )
    if not response.ok:
        raise RuntimeError(
            f"Taskmarket device-key fetch failed ({response.status_code}): {response.text}"
        )
    payload = response.json()
    device_key = str(payload.get("deviceEncryptionKey", "")).strip()
    if not device_key:
        raise RuntimeError("Taskmarket device-key response was missing deviceEncryptionKey.")
    return device_key


def decrypt_taskmarket_private_key(device_encryption_key_hex: str, encrypted_hex: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = bytes.fromhex(device_encryption_key_hex)
    data = bytes.fromhex(encrypted_hex)
    iv = data[:12]
    ciphertext_and_tag = data[28:] + data[12:28]
    plaintext = AESGCM(key).decrypt(iv, ciphertext_and_tag, None)
    return plaintext.decode("utf-8")


def load_taskmarket_private_key() -> str:
    keystore = load_taskmarket_keystore()
    device_id = str(keystore.get("deviceId", "")).strip()
    api_token = str(keystore.get("apiToken", "")).strip()
    encrypted_key = str(keystore.get("encryptedKey", "")).strip()
    if not device_id or not api_token or not encrypted_key:
        raise RuntimeError("Taskmarket keystore is missing deviceId, apiToken, or encryptedKey.")
    device_key = fetch_taskmarket_device_key(device_id, api_token)
    return decrypt_taskmarket_private_key(device_key, encrypted_key)


def get_taskmarket_init_command() -> list[str]:
    if shutil.which("taskmarket"):
        return ["taskmarket", "init"]
    if shutil.which("npx"):
        return ["npx", "-y", "@lucid-agents/taskmarket@latest", "init"]
    raise RuntimeError("Taskmarket onboarding requires either `taskmarket` or `npx` on PATH.")


def ensure_taskmarket_wallet_initialized() -> dict:
    if taskmarket_keystore_exists():
        return load_taskmarket_keystore()
    cmd = get_taskmarket_init_command()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Taskmarket init failed with exit code {result.returncode}.")
    if not taskmarket_keystore_exists():
        raise RuntimeError("Taskmarket init completed but no keystore was created.")
    return load_taskmarket_keystore()
