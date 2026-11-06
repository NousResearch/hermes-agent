"""Windows interop PATH entries for generated WSL service definitions."""

from __future__ import annotations


def deduplicate_wsl_paths(path_entries: list[str], candidates: list[str]) -> list[str]:
    """Return additional interop directories without changing the caller's PATH."""
    def path_key(entry: str) -> str:
        normalized = entry.rstrip("/") or entry
        return normalized.lower() if normalized.startswith("/mnt/") else normalized

    result: list[str] = []
    seen = {path_key(entry) for entry in path_entries}
    for entry in candidates:
        normalized = entry.rstrip("/") or entry
        key = path_key(normalized)
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


def build_wsl_interop_paths(path_entries: list[str]) -> list[str]:
    """Persist only discovered Windows tool dirs and the System32 defaults.

    Scraping arbitrary shell PATH entries can exhaust WSL's 9p server
    connection limit (#73163); interop tools do not need those directories.
    """
    from hermes_cli import gateway as gw

    if not gw.is_wsl():
        return []

    candidates: list[str] = []
    for executable in ("powershell.exe", "cmd.exe", "explorer.exe", "wsl.exe"):
        resolved = gw.shutil.which(executable)
        if resolved:
            candidates.append(str(gw.Path(resolved).parent))
    candidates += [
        entry
        for entry in (
            "/mnt/c/WINDOWS/system32",
            "/mnt/c/WINDOWS",
            "/mnt/c/WINDOWS/System32/Wbem",
            "/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0/",
            "/mnt/c/WINDOWS/System32/OpenSSH/",
        )
        if gw.Path(entry).exists()
    ]
    return deduplicate_wsl_paths(path_entries, candidates)
