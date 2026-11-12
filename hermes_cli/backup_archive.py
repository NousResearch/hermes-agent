"""Write backup entries using SQLite-safe files and immutable profile UI snapshots."""
from pathlib import Path
import zipfile


def write_zip_entries(
    zf: zipfile.ZipFile, files_to_add: list[tuple[Path, Path]], out_path: Path,
    *, on_db_failure, on_error, on_progress, track_bytes: bool, ui_snapshots=None,
) -> int:
    """Archive entries; callbacks preserve the full/automatic backup failure policies."""
    # These are the facade's existing patch seams for SQLite and streamed file writes.
    from hermes_cli.backup import _write_zip_file, _zip_sqlite_snapshot

    total_bytes = 0
    for i, (abs_path, rel_path) in enumerate(files_to_add, 1):
        try:
            if ui_snapshots is not None and rel_path in ui_snapshots:
                zf.writestr(rel_path.as_posix(), ui_snapshots[rel_path])
                if track_bytes:
                    total_bytes += len(ui_snapshots[rel_path])
            elif abs_path.suffix == ".db":
                size = _zip_sqlite_snapshot(zf, abs_path, rel_path, out_path)
                if size is None:
                    on_db_failure(rel_path)
                    continue
                total_bytes += size
            else:
                _write_zip_file(zf, abs_path, str(rel_path))
                if track_bytes:
                    total_bytes += abs_path.stat().st_size
        except (PermissionError, OSError, ValueError) as exc:
            on_error(rel_path, exc)
            continue
        if i % 500 == 0:
            on_progress(i)
    return total_bytes
