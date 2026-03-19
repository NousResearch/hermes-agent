import hashlib
import time
from dataclasses import dataclass, field

import httpx

from dropclaw.crypto import encrypt, decrypt
from dropclaw.compression import compress, decompress
from dropclaw.skill_file import validate


@dataclass
class StoreResult:
    skill_file: dict
    key: str
    file_id: str
    total_chunks: int = 0
    total_size: int = 0


@dataclass
class PaymentRequired:
    needs_payment: bool = True
    payment_options: dict = field(default_factory=dict)


class VaultClient:
    """Client for storing and retrieving encrypted files via the Agent Vault gateway."""

    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url.rstrip("/")

    def store(self, file_bytes: bytes, payment_header: str | None = None) -> StoreResult | PaymentRequired:
        """Store a file on-chain via Agent Vault.

        Flow:
        1. Compute SHA-256 hash of original
        2. Compress with zlib
        3. Encrypt with AES-256-GCM (caller keeps key)
        4. POST encrypted blob to gateway
        5. Handle 402 (payment required)
        6. Handle async job polling
        7. Return StoreResult with skill file + encryption key
        """
        # 1. Hash original
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        # 2. Compress
        compressed = compress(file_bytes)

        # 3. Encrypt
        encrypted, key = encrypt(compressed)

        # 4. Build multipart form
        files = {"file": ("vault.bin", encrypted, "application/octet-stream")}
        data = {
            "originalSize": str(len(file_bytes)),
            "contentHash": content_hash,
        }

        headers = {"X-File-Size": str(len(encrypted))}
        if payment_header:
            headers["X-PAYMENT"] = payment_header

        # 5. POST to gateway
        with httpx.Client(timeout=60.0) as client:
            res = client.post(
                f"{self.gateway_url}/vault/store",
                files=files,
                data=data,
                headers=headers,
            )

        body = res.json()

        # Handle 402
        if res.status_code == 402:
            return PaymentRequired(needs_payment=True, payment_options=body)

        if res.status_code >= 400:
            raise RuntimeError(f"Store failed ({res.status_code}): {body.get('error', 'Unknown error')}")

        # Handle async job response
        job_id = body.get("jobId")
        if job_id:
            body = self.wait_for_completion(job_id)

        return StoreResult(
            skill_file=body.get("skillFile", {}),
            key=key,
            file_id=body.get("fileId", ""),
            total_chunks=body.get("totalChunks", 0),
            total_size=body.get("totalSize", 0),
        )

    def retrieve(self, skill_file: dict, key_hex: str) -> bytes:
        """Retrieve and decrypt a file from on-chain storage."""
        # Validate skill file
        result = validate(skill_file)
        if not result.valid:
            raise ValueError(f"Invalid skill file: {', '.join(result.errors)}")

        file_id = skill_file["file"]["fileId"]

        # Fetch encrypted blob from gateway
        with httpx.Client(timeout=60.0) as client:
            res = client.post(
                f"{self.gateway_url}/vault/retrieve/{file_id}",
                json={"skillFile": skill_file},
            )

        if res.status_code >= 400:
            try:
                err = res.json()
            except Exception:
                err = {"error": "Unknown error"}
            raise RuntimeError(f"Retrieve failed ({res.status_code}): {err.get('error', 'Unknown error')}")

        encrypted_blob = res.content

        # Decrypt
        compressed = decrypt(encrypted_blob, key_hex)

        # Decompress
        original = decompress(compressed)

        # Verify hash
        computed_hash = hashlib.sha256(original).hexdigest()
        expected_hash = skill_file.get("file", {}).get("contentHash", "").replace("sha256:", "")
        if expected_hash and computed_hash != expected_hash:
            raise ValueError(f"Content hash mismatch: got {computed_hash}, expected {expected_hash}")

        return original

    def wait_for_completion(self, job_id: str, poll_interval: float = 2.0, timeout: float = 300.0) -> dict:
        """Poll the gateway for job completion."""
        start = time.monotonic()
        with httpx.Client(timeout=30.0) as client:
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

                res = client.get(f"{self.gateway_url}/vault/status/{job_id}")
                if res.status_code >= 400:
                    raise RuntimeError(f"Status check failed ({res.status_code})")

                data = res.json()
                status = data.get("status")

                if status == "completed":
                    return data
                if status == "failed":
                    raise RuntimeError(f"Job {job_id} failed: {data.get('error', 'Unknown error')}")

                time.sleep(poll_interval)

    def estimate_cost(self, file_size: int) -> dict:
        """Estimate storage cost for a file."""
        with httpx.Client(timeout=30.0) as client:
            res = client.get(f"{self.gateway_url}/vault/pricing", params={"size": file_size})

        if res.status_code >= 400:
            raise RuntimeError(f"Pricing request failed: {res.status_code}")

        return res.json()
