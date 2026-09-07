"""Shared SKILL.md preprocessing helpers: ``${HERMES_*}`` template tokens and
inline ``!`cmd``` shell expansion."""

import logging
import re
import subprocess
from pathlib import Path

from hermes_cli._subprocess_compat import IS_WINDOWS, windows_hide_flags

logger = logging.getLogger(__name__)

# ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID} tokens. Unresolvable ones (e.g. no
# session) are left as-is so the author can spot them.
_SKILL_TEMPLATE_RE = re.compile(r"\$\{(HERMES_SKILL_DIR|HERMES_SESSION_ID)\}")
# Inline shell snippets like !`date +%Y-%m-%d` — single-line only.
_INLINE_SHELL_RE = re.compile(r"!`([^`\n]+)`")
# Cap inline-shell output so a runaway command can't blow out the context.
_INLINE_SHELL_MAX_OUTPUT = 4000


def load_skills_config() -> dict:
    """Load the ``skills`` section of config.yaml (best-effort)."""
    try:
        from hermes_cli.config import load_config_readonly
        skills_cfg = (load_config_readonly() or {}).get("skills")
        if isinstance(skills_cfg, dict):
            return skills_cfg
    except Exception:
        logger.debug("Could not read skills config", exc_info=True)
    return {}


def substitute_template_vars(content: str, skill_dir: Path | None, session_id: str | None) -> str:
    """Replace ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID}; tokens without a value stay in place."""
    if not content:
        return content
    values = {
        "HERMES_SKILL_DIR": str(skill_dir) if skill_dir else None,
        "HERMES_SESSION_ID": str(session_id) if session_id else None,
    }
    return _SKILL_TEMPLATE_RE.sub(lambda m: values[m.group(1)] or m.group(0), content)


def run_inline_shell(command: str, cwd: Path | None, timeout: int) -> str:
    """Run one inline-shell snippet and return its stdout (trimmed; stderr when
    stdout is empty). Failures return an ``[inline-shell ...]`` marker instead
    of raising, so one bad snippet can't wreck the whole skill message."""
    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    try:
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=max(1, int(timeout)),
            check=False,
            stdin=subprocess.DEVNULL,
            **_popen_kwargs,
        )
    except subprocess.TimeoutExpired:
        return f"[inline-shell timeout after {timeout}s: {command}]"
    except FileNotFoundError:
        return "[inline-shell error: bash not found]"
    except Exception as exc:
        # tests/conftest.py's live-system guard may block the os.kill that
        # subprocess.run uses to clean up a timed-out shell; report the timeout.
        if isinstance(exc, RuntimeError) and "live-system guard: blocked os.kill" in str(exc):
            return f"[inline-shell timeout after {timeout}s: {command}]"
        return f"[inline-shell error: {exc}]"
    output = (completed.stdout or "").rstrip("\n") or (completed.stderr or "").rstrip("\n")
    if len(output) > _INLINE_SHELL_MAX_OUTPUT:
        output = output[:_INLINE_SHELL_MAX_OUTPUT] + "...[truncated]"
    return output


def expand_inline_shell(content: str, skill_dir: Path | None, timeout: int) -> str:
    """Replace every !`cmd` snippet with its stdout, run with the skill dir as CWD."""
    if "!`" not in content:
        return content
    def _replace(match: re.Match) -> str:
        cmd = match.group(1).strip()
        return run_inline_shell(cmd, skill_dir, timeout) if cmd else ""
    return _INLINE_SHELL_RE.sub(_replace, content)


def preprocess_skill_content(
    content: str,
    skill_dir: Path | None,
    session_id: str | None = None,
    skills_cfg: dict | None = None,
) -> str:
    """Apply configured SKILL.md template and inline-shell preprocessing."""
    if not content:
        return content
    cfg = skills_cfg if isinstance(skills_cfg, dict) else load_skills_config()
    if cfg.get("template_vars", True):
        content = substitute_template_vars(content, skill_dir, session_id)
    if cfg.get("inline_shell", False):
        content = expand_inline_shell(content, skill_dir, int(cfg.get("inline_shell_timeout", 10) or 10))
    return content
