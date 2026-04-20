from __future__ import annotations


def format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{int(value)} B"


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    if original_bytes <= 0:
        return 0.0
    return round((compressed_bytes / original_bytes) * 100, 1)
