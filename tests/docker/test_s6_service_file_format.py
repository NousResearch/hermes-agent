from __future__ import annotations

from pathlib import Path


def test_docker_runtime_files_are_utf8_without_bom_and_lf_only() -> None:
    root = Path(__file__).resolve().parents[2]
    docker_dir = root / "docker"
    failures: list[str] = []
    checked_paths: list[Path] = []

    checked_paths.extend(sorted((docker_dir / "s6-rc.d").rglob("*")))
    checked_paths.extend(sorted((docker_dir / "cont-init.d").rglob("*")))
    checked_paths.extend(sorted(docker_dir.glob("*.sh")))

    for path in checked_paths:
        if not path.is_file():
            continue
        data = path.read_bytes()
        rel = path.relative_to(root).as_posix()
        if data.startswith(b"\xef\xbb\xbf"):
            failures.append(f"{rel}: has UTF-8 BOM")
        if b"\r\n" in data:
            failures.append(f"{rel}: has CRLF line endings")

    assert not failures, "\n".join(failures)