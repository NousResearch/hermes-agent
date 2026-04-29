"""OpenCLI-backed image generation for Hermes-owned Image2 jobs."""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _extract_json(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", raw)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _normalise_opencli_file(value: Any) -> str:
    text = str(value or "").strip()
    # OpenCLI table/json rows may preserve the display emoji prefix.
    text = text.replace("📁", "").strip()
    if text in {"", "-"}:
        return ""
    return str(Path(text).expanduser())


def _saved_files_from_rows(rows: Any) -> list[str]:
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list):
        return []
    files: list[str] = []
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("status") or "").lower()
        path_text = _normalise_opencli_file(item.get("file"))
        if "saved" not in status and not path_text:
            continue
        if not path_text:
            continue
        path = Path(path_text).expanduser()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files.append(str(path))
    return list(dict.fromkeys(files))


def _first_link(rows: Any) -> str:
    if isinstance(rows, Mapping):
        rows = [rows]
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, Mapping):
                link = str(item.get("link") or item.get("page_url") or "").replace("🔗", "").strip()
                if link and link != "-":
                    return link
    return ""


def _source_image_path(source_files: Sequence[Any] | None) -> str:
    for item in source_files or []:
        if isinstance(item, (str, os.PathLike)):
            path = Path(str(item)).expanduser()
        elif isinstance(item, Mapping):
            path = Path(str(item.get("path") or item.get("local_path") or item.get("file_path") or "")).expanduser()
        else:
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return str(path)
    return ""


def _base_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Build a minimized environment for OpenCLI image subprocesses.

    The Image2 worker may receive delivery credentials and unrelated profile
    secrets. OpenCLI generation only needs browser/profile/proxy plumbing, so
    pass an allowlisted environment instead of inheriting all of os.environ.
    """
    allowed_exact = {
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "PATH",
        "LANG",
        "LC_ALL",
        "SSH_AUTH_SOCK",
        "HERMES_HOME",
    }
    allowed_prefixes = (
        "OPENCLI_",
        "CHATGPT_",
        "CHROME_",
        "PLAYWRIGHT_",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    )
    merged: dict[str, str] = {}
    for source in (os.environ, environ):
        for key, value in dict(source).items():
            key_s = str(key)
            if key_s in allowed_exact or key_s.startswith(allowed_prefixes):
                merged[key_s] = str(value)
    home_local = str(Path.home() / ".local/bin")
    merged["PATH"] = f"{home_local}:{merged.get('PATH', os.environ.get('PATH', ''))}"
    merged.setdefault("HOME", str(Path.home()))
    merged.setdefault("OPENCLI_CHROME_CDP_GUIDANCE", "0")
    return merged




def _clip_title(value: str, limit: int = 48) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" -_｜|：:")
    return cleaned[:limit].rstrip() if len(cleaned) > limit else cleaned


def build_opencli_window_title(*, job_dir: Path, prompt_text: str, environ: Mapping[str, str]) -> str:
    task_id = Path(job_dir).name or "img2"
    subject = ""
    for pattern in (r"主视觉对象[:：]\s*([^。；;\n]+)", r"主标题[「:\s：]+([^」；;\n]+)"):
        match = re.search(pattern, prompt_text or "")
        if match:
            subject = match.group(1).strip("「」 ：:")
            break
    explicit = str(environ.get("IMAGE2_OPENCLI_TITLE") or "").strip()
    parts = ["Image2", task_id]
    if subject:
        parts.append(subject)
    if explicit and explicit not in {"海报设计", "poster design", "Image2"} and explicit not in parts:
        parts.append(explicit)
    if len(parts) <= 2:
        parts.append("Image2任务")
    return _clip_title(" ".join(parts), 64)


def probe_opencli_browser_state(*, job_dir: Path, environ: Mapping[str, str], timeout: int = 60) -> dict[str, Any]:
    """Probe ChatGPT Images page via OpenCLI without sending a generation prompt."""
    root = Path(job_dir)
    root.mkdir(parents=True, exist_ok=True)
    cmd = ["opencli", "chatgpt", "image-capabilities", "-f", "json"]
    started_at = _utc_now()
    try:
        proc = subprocess.run(cmd, cwd=str(root), env=_base_env(environ), text=True, capture_output=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - fail closed into browser_state
        result = {
            "cdp_reachable": False,
            "active_url": "",
            "title": "",
            "probe_error": f"{exc.__class__.__name__}: {exc}",
            "started_at": started_at,
            "completed_at": _utc_now(),
        }
        (root / "browser_state.json").write_text(_safe_json(result), encoding="utf-8")
        return result
    parsed = _extract_json(proc.stdout)
    url = ""
    title = ""
    if isinstance(parsed, list):
        for row in parsed:
            if not isinstance(row, Mapping):
                continue
            if row.get("Category") == "page" and row.get("Name") == "url":
                url = str(row.get("Value") or "")
            if row.get("Category") == "page" and row.get("Name") == "title":
                title = str(row.get("Value") or "")
    result = {
        "cdp_reachable": proc.returncode == 0 and bool(url),
        "active_url": url,
        "title": title,
        "opencli_returncode": proc.returncode,
        "started_at": started_at,
        "completed_at": _utc_now(),
    }
    if proc.returncode != 0:
        result["probe_error"] = (proc.stderr or proc.stdout or "opencli image-capabilities failed")[-1000:]
    (root / "browser_state.json").write_text(_safe_json(result), encoding="utf-8")
    return result


def run_opencli_generation(
    *,
    job_dir: Path,
    prompt_text: str,
    environ: Mapping[str, str],
    source_files: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Generate a candidate through OpenCLI ChatGPT Images and persist logs.

    This function treats OpenCLI JSON status as data: exit code 0 with
    `blocked`/`failed` and no file is not success. If an aspect/size option blocks,
    it retries once without the aspect because the live ChatGPT UI is conditional.
    """
    root = Path(job_dir)
    candidates = root / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    env = _base_env(environ)
    timeout = int(env.get("IMAGE2_SUBPROCESS_TIMEOUT") or env.get("IMAGE2_OPENCLI_TIMEOUT") or 600)
    opencli_timeout = str(env.get("IMAGE2_OPENCLI_TIMEOUT") or 420)
    title = build_opencli_window_title(job_dir=root, prompt_text=prompt_text, environ=env)
    source_path = _source_image_path(source_files)
    requested_aspect = str(env.get("IMAGE2_OPENCLI_ASPECT") or env.get("IMAGE2_OPENCLI_SIZE") or "").strip()
    attempts: list[dict[str, Any]] = []
    existing_candidates = {str(p) for p in candidates.iterdir() if p.is_file()}

    aspect_values = [requested_aspect] if requested_aspect else [""]
    if requested_aspect:
        aspect_values.append("")

    for index, aspect in enumerate(aspect_values, start=1):
        cmd = ["opencli", "chatgpt", "image", "--title", title, "--op", str(candidates), "--timeout", opencli_timeout, "-f", "json"]
        if aspect:
            cmd.extend(["--aspect", aspect])
        if source_path:
            cmd.extend(["--file", source_path])
        cmd.append(str(prompt_text))
        started_at = _utc_now()
        stdout_path = candidates / f"opencli.attempt{index}.stdout.json"
        stderr_path = candidates / f"opencli.attempt{index}.stderr.log"
        try:
            proc = subprocess.run(cmd, cwd=str(root), env=env, text=True, capture_output=True, timeout=timeout)
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
            parsed = _extract_json(proc.stdout)
            saved = _saved_files_from_rows(parsed)
            discovered = [str(p) for p in sorted(candidates.iterdir(), key=lambda item: item.stat().st_mtime) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and str(p) not in existing_candidates]
            saved = list(dict.fromkeys(saved + discovered))
            attempt = {
                "attempt": index,
                "aspect": aspect,
                "returncode": proc.returncode,
                "rows": parsed,
                "files": saved,
                "link": _first_link(parsed),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "started_at": started_at,
                "completed_at": _utc_now(),
            }
        except Exception as exc:  # noqa: BLE001 - preserve evidence and continue/fail
            stderr_path.write_text(f"{exc.__class__.__name__}: {exc}", encoding="utf-8")
            attempt = {
                "attempt": index,
                "aspect": aspect,
                "returncode": None,
                "rows": None,
                "files": [],
                "link": "",
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "error": f"{exc.__class__.__name__}: {exc}",
                "started_at": started_at,
                "completed_at": _utc_now(),
            }
        attempts.append(attempt)
        if attempt["files"]:
            result_status = "saved" if attempt.get("returncode") == 0 else "saved_with_nonzero_exit"
            result = {"status": result_status, "files": attempt["files"], "link": attempt.get("link") or "", "attempts": attempts, "source_file": source_path, "title": title}
            (root / "generation_result.json").write_text(_safe_json(result), encoding="utf-8")
            return result
        # Only retry aspect-specific failures once without aspect.
        if not aspect:
            break

    result = {"status": "no_file", "files": [], "link": _first_link(attempts[-1].get("rows") if attempts else None), "attempts": attempts, "source_file": source_path, "title": title}
    (root / "generation_result.json").write_text(_safe_json(result), encoding="utf-8")
    return result
