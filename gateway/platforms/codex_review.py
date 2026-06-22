"""codex_review — Codex 执行 + 审查子模块

由 feishu_codex.handle_codex_message 调用。
职责：执行 Codex CLI → 提取 git diff → 调用 codex review 评分。
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

CODEX_TIMEOUT = 300
CODEX_WORKDIR = os.path.expanduser("~/hermes-source")
REVIEW_SCRIPT = os.path.expanduser("~/.hermes/scripts/codex_review.py")
LANDING_SCRIPT = os.path.expanduser("~/.hermes/scripts/codex_landing.py")


async def run_review(
    project_name: str, description: str
) -> tuple[Path, Path, str, str, float | None]:
    """执行 Codex → 提取 diff → 评分

    Returns:
        (md_path, patch_path, status, result_msg, score)
    """
    prompt = (
        description[len("codex"):].strip()
        if description.startswith("codex")
        else description
    )

    # 1. 执行 Codex
    try:
        result = await asyncio.to_thread(
            _run_codex, prompt, CODEX_WORKDIR, CODEX_TIMEOUT
        )
    except asyncio.TimeoutError:
        return Path(), Path(), "failure", "❌ Codex 执行超时", None
    except Exception:
        logger.exception("Codex execution failed")
        return Path(), Path(), "failure", "❌ Codex 执行出错", None

    if result["exit_code"] != 0:
        return (
            Path(), Path(), "failure",
            f"❌ Codex 执行失败 (exit={result['exit_code']})\n\n"
            f"输出:\n```\n{result['stdout']}\n```\n"
            f"错误:\n```\n{result['stderr']}\n```", None,
        )

    # 2. 提取 diff（git diff + 新文件检测）
    md_path, patch_path, new_files = _extract_diff(CODEX_WORKDIR, project_name)
    if md_path is None:
        # 无 git 变更但有新文件 → 列出新文件
        if new_files:
            files_list = "\n".join(f"- `{f}`" for f in new_files[:20])
            return (
                Path(), Path(), "success",
                f"✅ Codex 执行完成\n\n"
                f"新增/修改文件 ({len(new_files)} 个):\n{files_list}\n\n"
                f"Codex 输出:\n```\n{result['stdout'][:500]}\n```", None,
            )
        return (
            Path(), Path(), "success",
            f"✅ Codex 执行完成\n\n"
            f"输出:\n```\n{result['stdout'][:500]}\n```", None,
        )

    # 3. 评分（传用户原始指令，用于相关性检查）
    score = _run_codex_review(md_path, prompt)
    if score is None:
        return (
            md_path, patch_path, "success",
            f"📝 变更已提取，自动评分失败，请手动审核\n审查文件: {md_path}",
            None,
        )

    # 4. 落地（≥9 自动 commit+push）
    if score >= 9:
        verdict = "pass"
    elif score >= 7:
        verdict = "warn"
    else:
        verdict = "fail"

    land_output = _run_landing(CODEX_WORKDIR, score, verdict)

    if verdict == "pass":
        result_msg = (
            f"✅ 审核通过 ({score}/10)，已自动提交并推送\n{land_output}"
        )
    elif verdict == "warn":
        result_msg = (
            f"⚠️ 审核通过 ({score}/10)，已自动提交\n"
            f"建议人工复查\n{land_output}"
        )
    else:
        result_msg = (
            f"❌ 审核不通过 ({score}/10)，不落地\n"
            f"审查报告: {md_path}\n完整 diff: {patch_path}"
        )

    return md_path, patch_path, "success", result_msg, score


def _run_codex(prompt: str, workdir: str, timeout: int) -> dict:
    # 强制 Codex 用 heredoc 创建新文件，绕过 apply_patch bug
    full_prompt = (
        "文件操作规则：\n"
        "1. 创建新文件：用 cat > file << 'ENDOFFILE' 方式\n"
        "2. 编辑已有文件：用 sed -i '' 命令\n"
        "3. 禁止使用 echo 创建文件（会丢引号）\n\n"
        + prompt
    )
    proc = subprocess.run(
        ["codex", "exec", "-s", "danger-full-access", full_prompt],
        capture_output=True, text=True, timeout=timeout,
        cwd=workdir,
    )
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-500:],
    }


def _extract_diff(
    project_dir: str, project_name: str
) -> tuple[Path | None, Path | None, list[str]]:
    # 1. git diff
    result = subprocess.run(
        ["git", "-C", project_dir, "diff", "--stat"],
        capture_output=True, text=True, timeout=10,
    )
    
    # 2. 未跟踪的新文件
    untracked = subprocess.run(
        ["git", "-C", project_dir, "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, timeout=10,
    )
    new_files = [f.strip() for f in untracked.stdout.strip().split("\n") if f.strip()]

    if not result.stdout.strip() and not new_files:
        return None, None, []

    # 清理旧的审查报告，避免用上次的 diff 评分
    safe_name = "".join(c for c in project_name if c.isalnum() or c in "_-")
    md_path = Path(f"/tmp/codex_review_{safe_name}.md")
    patch_path = Path(f"/tmp/codex_full_diff_{safe_name}.patch")
    try:
        md_path.unlink(missing_ok=True)
    except OSError:
        pass

    subprocess.run(
        ["python3", REVIEW_SCRIPT, project_dir],
        capture_output=True, timeout=15,
    )

    diff_result = subprocess.run(
        ["git", "-C", project_dir, "diff"],
        capture_output=True, text=True, timeout=10,
    )
    try:
        patch_path.write_text(diff_result.stdout)
    except OSError:
        logger.warning("Failed to write full diff patch: %s", patch_path)

    return (md_path if md_path.exists() else None), patch_path, new_files


HERMES_BIN = os.path.expanduser("~/hermes-source/venv/bin/hermes")


def _run_codex_review(review_file: Path, user_prompt: str = "") -> float | None:
    try:
        content = review_file.read_text()
    except OSError:
        return None

    # 用 Hermes fallback 链审核，不硬绑模型，禁工具防止文件修改
    review_prompt = (
        f"用户指令：{user_prompt}\n\n"
        f"代码变更：\n{content}\n\n"
        f"评分标准 (0-10分)：\n"
        f"- 产出是否与用户指令直接相关（无关=0分，完全匹配=高分）\n"
        f"- 代码质量和正确性\n"
        f"- 是否有不必要的文件或垃圾代码\n"
        f"只输出一个数字分数，不要其他内容。"
    )
    proc = subprocess.run(
        [HERMES_BIN, "-z", review_prompt, "-t", "", "--yolo"],
        capture_output=True, text=True, timeout=180,
        cwd=CODEX_WORKDIR,
    )

    match = re.search(r"(\d+(?:\.\d+)?)", proc.stdout)
    if match:
        return float(match.group(1))
    return None


def _run_landing(project_dir: str, score: float, verdict: str) -> str:
    proc = subprocess.run(
        ["python3", LANDING_SCRIPT, project_dir, str(score), verdict],
        capture_output=True, text=True, timeout=30,
        cwd=project_dir,
    )
    return proc.stdout.strip() or proc.stderr.strip()