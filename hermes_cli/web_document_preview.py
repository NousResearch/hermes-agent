"""Bounded, tokenized document preview helpers for the authenticated web API."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time

from tools.environments.local import hermes_subprocess_env

PDF_INITIAL_BYTES = 64 * 1024
PDF_MAX_RANGE_BYTES = 1024 * 1024
PDF_MAX_OPEN_DOCUMENTS = 32
PDF_IDLE_SECONDS = 15 * 60
TEX_MAX_LOG_BYTES = 1024 * 1024
TEX_TIMEOUT_SECONDS = 60
_ALLOWED_TEX_PROGRAMS = {"lualatex", "pdflatex", "xelatex"}


@dataclass
class _Document:
    path: Path
    size: int
    modified_at: float
    last_access: float
    temporary_root: Path | None = None


_documents: dict[str, _Document] = {}
_documents_lock = threading.Lock()
_active_compiles: dict[str, subprocess.Popen[bytes]] = {}
_compile_lock = threading.Lock()
_compile_slots = threading.BoundedSemaphore(2)


def _revision(size: int, modified_at: float) -> str:
    return f"{size}:{modified_at}"


def _prune_documents(now: float) -> None:
    expired = [key for key, value in _documents.items() if now - value.last_access > PDF_IDLE_SECONDS]
    while len(_documents) - len(expired) >= PDF_MAX_OPEN_DOCUMENTS:
        candidates = [(value.last_access, key) for key, value in _documents.items() if key not in expired]
        if not candidates:
            break
        expired.append(min(candidates)[1])
    for key in expired:
        document = _documents.pop(key, None)
        if document and document.temporary_root:
            shutil.rmtree(document.temporary_root, ignore_errors=True)


def open_pdf(path: Path, *, temporary_root: Path | None = None) -> dict:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    if not resolved.is_file() or resolved.suffix.lower() != ".pdf":
        raise ValueError("PDF preview requires a regular .pdf file")
    now = time.monotonic()
    token = secrets.token_urlsafe(32)
    document = _Document(resolved, stat.st_size, stat.st_mtime, now, temporary_root)
    with _documents_lock:
        _prune_documents(now)
        _documents[token] = document
    with resolved.open("rb") as handle:
        initial = handle.read(PDF_INITIAL_BYTES)
    return {
        "byteLength": stat.st_size,
        "id": token,
        "initialData": base64.b64encode(initial).decode("ascii"),
        "modifiedAt": stat.st_mtime * 1000,
        "revision": _revision(stat.st_size, stat.st_mtime),
    }


def read_pdf_range(token: str, begin: int, end: int, revision: str | None = None) -> bytes:
    if begin < 0 or end <= begin or end - begin > PDF_MAX_RANGE_BYTES:
        raise ValueError("Invalid PDF byte range")
    with _documents_lock:
        document = _documents.get(token)
        if document:
            document.last_access = time.monotonic()
    if not document:
        raise FileNotFoundError("PDF document is closed or unavailable")
    stat = document.path.stat()
    current_revision = _revision(stat.st_size, stat.st_mtime)
    if stat.st_size != document.size or stat.st_mtime != document.modified_at or (
        revision and revision != current_revision
    ):
        raise RuntimeError("PDF_CHANGED")
    if end > document.size:
        raise ValueError("Invalid PDF byte range")
    with document.path.open("rb") as handle:
        handle.seek(begin)
        return handle.read(end - begin)


def close_pdf(token: str) -> bool:
    with _documents_lock:
        document = _documents.pop(token, None)
    if document and document.temporary_root:
        shutil.rmtree(document.temporary_root, ignore_errors=True)
    return document is not None


def _tex_program(source: str) -> str | None:
    header = "\n".join(source.splitlines()[:24])
    program_match = re.search(r"^\s*%\s*!TeX\s+program\s*=\s*([\w-]+)\s*$", header, re.I | re.M)
    program = program_match.group(1).lower() if program_match else None
    return program if program in _ALLOWED_TEX_PROGRAMS else None


def tex_directives(source: str, source_path: Path, workspace_root: Path | None = None) -> tuple[Path, str | None]:
    header = "\n".join(source.splitlines()[:24])
    root_match = re.search(r"^\s*%\s*!TeX\s+root\s*=\s*(.+?)\s*$", header, re.I | re.M)
    program = _tex_program(source)
    root = (source_path.parent / root_match.group(1).strip()).resolve() if root_match else source_path
    project_root: Path | None = None
    if workspace_root:
        try:
            candidate = workspace_root.resolve(strict=True)
            source_path.relative_to(candidate)
            project_root = candidate
        except (FileNotFoundError, OSError, ValueError):
            project_root = None
    if project_root is None:
        project_root = source_path.parent.resolve()
        cursor = project_root
        while True:
            if (cursor / ".git").exists():
                project_root = cursor
                break
            if cursor == cursor.parent:
                break
            cursor = cursor.parent
    try:
        root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("TeX root must remain inside the source project") from exc
    return root, program


def _diagnostics(log: str) -> list[dict]:
    result: list[dict] = []
    for line in log.splitlines():
        located = re.match(r"^(.+?):(\d+):\s*(.+)$", line)
        if located:
            result.append({"file": located.group(1), "line": int(located.group(2)), "message": located.group(3)})
        elif line.startswith("! "):
            result.append({"message": line[2:].strip()})
        if len(result) >= 50:
            break
    return result


def _missing_engine() -> str:
    return "No supported TeX engine was found. Install latexmk with XeLaTeX/LuaLaTeX/pdfLaTeX, or Tectonic, then restart Hermes."


def cancel_tex(request_id: str) -> bool:
    with _compile_lock:
        process = _active_compiles.pop(request_id, None)
    if not process:
        return False
    try:
        process.kill()
    except OSError:
        pass
    return True


def compile_tex(source_path: Path, request_id: str, workspace_root: Path | None = None) -> dict:
    started = time.monotonic()
    source_path = source_path.resolve(strict=True)
    if source_path.suffix.lower() != ".tex" or not source_path.is_file():
        raise ValueError("TeX preview requires a regular .tex file")
    source = source_path.read_text(encoding="utf-8", errors="replace")
    root_path, requested = tex_directives(source, source_path, workspace_root)
    if not root_path.is_file() or root_path.suffix.lower() != ".tex":
        raise ValueError("TeX root must be a regular .tex file")
    if requested is None and root_path != source_path:
        requested = _tex_program(root_path.read_text(encoding="utf-8", errors="replace"))
    output_root = Path(tempfile.mkdtemp(prefix="hermes-tex-preview-"))
    latexmk = shutil.which("latexmk")
    programs = [requested] if requested else ["xelatex", "lualatex", "pdflatex"]
    program = next((name for name in programs if name and shutil.which(name)), None)
    tectonic = None if requested else shutil.which("tectonic")
    command: list[str] | None = None
    engine: str | None = None
    direct_program = False
    if latexmk and program:
        mode = "-xelatex" if program == "xelatex" else "-lualatex" if program == "lualatex" else "-pdf"
        engine = f"latexmk/{program}"
        command = [latexmk, "-norc", mode, "-recorder", f"-outdir={output_root}",
                   "-latexoption=-no-shell-escape", "-latexoption=-interaction=nonstopmode",
                   "-latexoption=-halt-on-error", "-latexoption=-file-line-error", str(root_path)]
    elif tectonic:
        engine = "tectonic"
        command = [tectonic, "--keep-logs", "--outdir", str(output_root), "--untrusted", str(root_path)]
    elif program:
        engine = program
        direct_program = True
        command = [shutil.which(program) or program, "-no-shell-escape", "-interaction=nonstopmode",
                   "-halt-on-error", "-file-line-error", f"-output-directory={output_root}", str(root_path)]
    if not command:
        shutil.rmtree(output_root, ignore_errors=True)
        return {"diagnostics": [], "durationMs": int((time.monotonic() - started) * 1000),
                "log": _missing_engine(), "rootPath": str(root_path), "stale": False, "status": "missing-engine"}
    if not _compile_slots.acquire(timeout=1):
        shutil.rmtree(output_root, ignore_errors=True)
        raise RuntimeError("Too many TeX previews are compiling")
    try:
        environment = hermes_subprocess_env(inherit_credentials=False)
        environment.update({"max_print_line": "1000", "openin_any": "p", "openout_any": "p"})
        process = subprocess.Popen(command, cwd=root_path.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
        with _compile_lock:
            previous = _active_compiles.pop(request_id, None)
            if previous:
                previous.kill()
            _active_compiles[request_id] = process
        try:
            try:
                output, _ = process.communicate(timeout=TEX_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
                output += b"\nCompilation timed out after 60 seconds.\n"

            if direct_program and process.returncode == 0:
                second = subprocess.Popen(
                    command,
                    cwd=root_path.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
                with _compile_lock:
                    if _active_compiles.get(request_id) is process:
                        _active_compiles[request_id] = second
                try:
                    second_output, _ = second.communicate(timeout=TEX_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    second.kill()
                    second_output, _ = second.communicate()
                    second_output += b"\nCompilation timed out after 60 seconds.\n"
                output = (output + second_output)[:TEX_MAX_LOG_BYTES]
                process = second
        finally:
            with _compile_lock:
                if _active_compiles.get(request_id) is process:
                    _active_compiles.pop(request_id, None)
        log = output[:TEX_MAX_LOG_BYTES].decode("utf-8", errors="replace")
        pdf_path = output_root / f"{root_path.stem}.pdf"
        success = process.returncode == 0 and pdf_path.is_file()
        result = {"diagnostics": _diagnostics(log), "durationMs": int((time.monotonic() - started) * 1000),
                  "engine": engine, "log": log, "rootPath": str(root_path), "stale": False,
                  "status": "success" if success else "error"}
        if success:
            result["pdfDocument"] = open_pdf(pdf_path, temporary_root=output_root)
        else:
            shutil.rmtree(output_root, ignore_errors=True)
        return result
    finally:
        _compile_slots.release()
