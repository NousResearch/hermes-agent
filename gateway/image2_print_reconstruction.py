"""ChatGPT reconstruction step for Image2 print-final jobs.

This is the bridge between a chat-approved preview and print packaging. It asks
ChatGPT Images/OpenCLI to rebuild the approved visual before the deterministic
DPI/size packager runs. The packager may still resize pixels to the requested
physical size, but the source entering that packager is no longer just the small
preview unless reconstruction fails closed.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway.image2_generation import (
    IMAGE_SUFFIXES,
    _base_env,
    _extract_json,
    _first_link,
    _saved_files_from_rows,
    _utc_now,
)

CommandRunner = Callable[..., Any]


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: Any, *, default: bool = True) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on"}


def _history_url(print_request: Mapping[str, Any]) -> str:
    for key in ("chatgpt_url", "conversation_url", "generation_url", "history_url"):
        value = str(print_request.get(key) or "").strip()
        if value:
            return value
    preview = print_request.get("approved_preview")
    if isinstance(preview, Mapping):
        for key in ("chatgpt_url", "conversation_url", "generation_url", "history_url"):
            value = str(preview.get(key) or "").strip()
            if value:
                return value
    return ""


def build_print_reconstruction_prompt(*, print_request: Mapping[str, Any], prompt_text: str) -> str:
    spec = print_request.get("spec") if isinstance(print_request.get("spec"), Mapping) else {}
    width = spec.get("width_mm") or spec.get("width_cm") or ""
    height = spec.get("height_mm") or spec.get("height_cm") or ""
    dpi = spec.get("dpi") or ""
    target_width = spec.get("target_width_px") or ""
    target_height = spec.get("target_height_px") or ""
    size_line = ""
    if width and height:
        size_line = f"目标印刷规格：{width}×{height}mm"
        if dpi:
            size_line += f"，{dpi}DPI。"
    elif dpi:
        size_line = f"目标印刷规格：{dpi}DPI。"
    pixel_line = ""
    if target_width and target_height:
        pixel_line = f"目标像素尺寸：{target_width}×{target_height}px。请尽量直接输出接近或达到该像素尺寸的完整竖版成图；如果平台无法直接输出该像素尺寸，也要生成可作为后续精确缩放到该尺寸的最高质量源图。"
    original = str(prompt_text or "").strip()
    excerpt = original[:700]
    return "\n".join(
        line
        for line in [
            "请基于已确认的海报预览做高分辨率重建/AI超分，不要重新创作新方案。",
            size_line,
            pixel_line,
            "必须严格保持原构图、主体数量、产品位置、标题层级、价格/文字含义、色调和留白；只提升清晰度、边缘质感、材质细节与印刷观感。",
            "不得添加或改写 Logo、二维码、品牌字样、价格、菜名/饮品名；看不清的小字宁可保持原样，不要编造。",
            "输出一张完整竖版成图，作为后续 PDF/PNG/PSD 印刷打包的源图。",
            "原始任务提示摘要：" + excerpt if excerpt else "",
        ]
        if line
    )


def _candidate_files(candidates: Path, existing: set[str]) -> list[str]:
    if not candidates.is_dir():
        return []
    files = [p for p in candidates.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and str(p) not in existing]
    return [str(p) for p in sorted(files, key=lambda item: (item.stat().st_mtime, item.name), reverse=True)]


def _attempt_command(
    *,
    candidates: Path,
    opencli_timeout: str,
    history: str,
    title: str,
    approved_image_path: Path,
    prompt: str,
    include_file: bool,
) -> list[str]:
    cmd = ["opencli", "chatgpt", "image"]
    if history:
        cmd.extend(["--history", history])
    else:
        cmd.extend(["--title", title])
    cmd.extend(["--op", str(candidates), "--timeout", opencli_timeout, "-f", "json"])
    if include_file:
        cmd.extend(["--file", str(approved_image_path)])
    cmd.append(prompt)
    return cmd


def reconstruct_print_source_with_chatgpt(
    *,
    job_dir: Path,
    approved_image_path: Path,
    approved_sha256: str,
    print_request: Mapping[str, Any],
    prompt_text: str,
    environ: Mapping[str, str],
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Return a fresh ChatGPT-reconstructed image for print packaging.

    The first attempt is grounded with ``--file <approved_preview>``. If that
    route is blocked or produces no usable image, the second attempt retries in
    the same ChatGPT history without file upload, matching the field-proven
    workflow used manually on the marketing Mac.
    """
    root = Path(job_dir)
    approved = Path(approved_image_path).expanduser()
    reports = root / "print" / "reports"
    candidates = root / "print" / "reconstruction" / "candidates"
    reports.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(parents=True, exist_ok=True)
    result_path = reports / "reconstruction_result.json"

    if not _as_bool(environ.get("IMAGE2_PRINT_RECONSTRUCT_ENABLED"), default=True):
        result = {"status": "disabled", "reason": "IMAGE2_PRINT_RECONSTRUCT_ENABLED is false", "mode": "disabled"}
        result_path.write_text(_safe_json(result), encoding="utf-8")
        return result
    if not approved.is_file() or not str(approved_sha256 or "").strip():
        result = {"status": "failed", "reason": "approved_source_missing", "mode": "chatgpt_reconstruction"}
        result_path.write_text(_safe_json(result), encoding="utf-8")
        return result
    actual_approved_sha = _sha256_file(approved)
    if actual_approved_sha != str(approved_sha256).lower():
        result = {"status": "failed", "reason": "approved_source_sha_mismatch", "approved_sha256_actual": actual_approved_sha, "mode": "chatgpt_reconstruction"}
        result_path.write_text(_safe_json(result), encoding="utf-8")
        return result

    env = _base_env(environ)
    runner = command_runner or subprocess.run
    history = _history_url(print_request)
    title = str(environ.get("IMAGE2_PRINT_RECONSTRUCT_TITLE") or print_request.get("generation_title") or f"Image2 Print {root.name} 高清重建")[:80]
    base_prompt = build_print_reconstruction_prompt(print_request=print_request, prompt_text=prompt_text)
    fallback_prompt = base_prompt + "\n如果上一轮文件上传被阻挡，请直接基于当前 ChatGPT Images 会话里已确认的海报继续高清重建。"
    opencli_timeout = str(environ.get("IMAGE2_PRINT_RECONSTRUCT_OPENCLI_TIMEOUT") or environ.get("IMAGE2_OPENCLI_TIMEOUT") or 420)
    timeout = int(environ.get("IMAGE2_PRINT_RECONSTRUCT_SUBPROCESS_TIMEOUT") or environ.get("IMAGE2_SUBPROCESS_TIMEOUT") or 600)
    existing = {str(p) for p in candidates.iterdir() if p.is_file()}
    attempts: list[dict[str, Any]] = []
    source_echo_seen = False

    for index, include_file in enumerate((True, False), start=1):
        prompt = base_prompt if include_file else fallback_prompt
        cmd = _attempt_command(candidates=candidates, opencli_timeout=opencli_timeout, history=history, title=title, approved_image_path=approved, prompt=prompt, include_file=include_file)
        stdout_path = candidates / f"opencli.print_reconstruct.attempt{index}.stdout.json"
        stderr_path = candidates / f"opencli.print_reconstruct.attempt{index}.stderr.log"
        started_at = _utc_now()
        try:
            proc = runner(cmd, cwd=str(root), env=env, text=True, capture_output=True, timeout=timeout)
            stdout_path.write_text(str(getattr(proc, "stdout", "") or ""), encoding="utf-8")
            stderr_path.write_text(str(getattr(proc, "stderr", "") or ""), encoding="utf-8")
            parsed = _extract_json(getattr(proc, "stdout", "") or "")
            saved = _saved_files_from_rows(parsed)
            saved = list(dict.fromkeys(saved + _candidate_files(candidates, existing)))
            attempt = {
                "attempt": index,
                "file_upload": include_file,
                "history": bool(history),
                "returncode": getattr(proc, "returncode", None),
                "rows": parsed,
                "files": saved,
                "link": _first_link(parsed),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "started_at": started_at,
                "completed_at": _utc_now(),
            }
        except Exception as exc:  # noqa: BLE001 - preserve evidence and try fallback
            stderr_path.write_text(f"{exc.__class__.__name__}: {exc}", encoding="utf-8")
            attempt = {
                "attempt": index,
                "file_upload": include_file,
                "history": bool(history),
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
        for file_text in attempt.get("files", []):
            path = Path(str(file_text)).expanduser()
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = _sha256_file(path)
            if digest == actual_approved_sha:
                source_echo_seen = True
                continue
            result = {
                "status": "pass",
                "mode": "chatgpt_reconstruction",
                "image_path": str(path),
                "image_sha256": digest,
                "approved_image_path": str(approved),
                "approved_sha256": actual_approved_sha,
                "link": str(attempt.get("link") or history or ""),
                "history": history,
                "attempts": attempts,
            }
            result_path.write_text(_safe_json(result), encoding="utf-8")
            return result
        # If grounded upload failed/no-file, retry without file. If no-file fallback also failed, break naturally.

    reason = "source_sha_match" if source_echo_seen else "chatgpt_reconstruction_no_candidate"
    result = {"status": "rejected" if source_echo_seen else "failed", "reason": reason, "mode": "chatgpt_reconstruction", "approved_image_path": str(approved), "approved_sha256": actual_approved_sha, "history": history, "attempts": attempts}
    result_path.write_text(_safe_json(result), encoding="utf-8")
    return result
