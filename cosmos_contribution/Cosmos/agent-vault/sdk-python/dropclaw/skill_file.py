from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    errors: list


def validate(skill_file: dict) -> ValidationResult:
    """Validate a skill file has all required fields."""
    errors = []
    if not skill_file:
        return ValidationResult(valid=False, errors=["Skill file is None"])
    if skill_file.get("protocol") != "agent-vault":
        errors.append(f"Invalid protocol: {skill_file.get('protocol')} (expected agent-vault)")
    chain = skill_file.get("chain", {})
    if not chain.get("rpcUrl"):
        errors.append("Missing chain.rpcUrl")
    file_info = skill_file.get("file", {})
    if not file_info.get("fileId"):
        errors.append("Missing file.fileId")
    storage = skill_file.get("storage", {})
    tx_hashes = storage.get("txHashes", [])
    if not tx_hashes:
        errors.append("Missing or empty storage.txHashes")
    if storage.get("encryption") != "aes-256-gcm":
        errors.append(f"Unsupported encryption: {storage.get('encryption')}")
    if storage.get("compression") != "zlib":
        errors.append(f"Unsupported compression: {storage.get('compression')}")
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def get_reconstruction_info(skill_file: dict) -> dict:
    """Extract reconstruction info from a skill file."""
    storage = skill_file.get("storage", {})
    file_info = skill_file.get("file", {})
    chain = skill_file.get("chain", {})
    return {
        "tx_hashes": storage.get("txHashes", []),
        "chunk_size": storage.get("chunkSize"),
        "header_bytes": storage.get("headerBytes"),
        "file_id": file_info.get("fileId"),
        "total_chunks": storage.get("totalChunks"),
        "rpc_url": chain.get("rpcUrl"),
    }
