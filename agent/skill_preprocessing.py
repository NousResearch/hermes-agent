"""Shared SKILL.md preprocessing helpers."""

import logging
import os
import re
import subprocess
from pathlib import Path

from hermes_cli._subprocess_compat import IS_WINDOWS, windows_hide_flags

logger = logging.getLogger(__name__)

# Matches ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID} tokens in SKILL.md.
# Tokens that don't resolve (e.g. ${HERMES_SESSION_ID} with no session) are
# left as-is so the user can debug them.
_SKILL_TEMPLATE_RE = re.compile(r"\$\{(HERMES_SKILL_DIR|HERMES_SESSION_ID)\}")

# Matches inline shell snippets like:  !`date +%Y-%m-%d`
# Non-greedy, single-line only -- no newlines inside the backticks.
_INLINE_SHELL_RE = re.compile(r"!`([^`\n]+)`")

# Cap inline-shell output so a runaway command can't blow out the context.
_INLINE_SHELL_MAX_OUTPUT = 4000


def load_skills_config() -> dict:
    """Load the ``skills`` section of config.yaml (best-effort)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        skills_cfg = cfg.get("skills")
        if isinstance(skills_cfg, dict):
            return skills_cfg
    except Exception:
        logger.debug("Could not read skills config", exc_info=True)
    return {}


def substitute_template_vars(
    content: str,
    skill_dir: Path | None,
    session_id: str | None,
) -> str:
    """Replace ${HERMES_SKILL_DIR} / ${HERMES_SESSION_ID} in skill content.

    Only substitutes tokens for which a concrete value is available --
    unresolved tokens are left in place so the author can spot them.
    """
    if not content:
        return content

    skill_dir_str = str(skill_dir) if skill_dir else None

    def _replace(match: re.Match) -> str:
        token = match.group(1)
        if token == "HERMES_SKILL_DIR" and skill_dir_str:
            return skill_dir_str
        if token == "HERMES_SESSION_ID" and session_id:
            return str(session_id)
        return match.group(0)

    return _SKILL_TEMPLATE_RE.sub(_replace, content)


def _shell_candidates() -> list[str]:
    """Return shell executables to try for inline snippets.

    Windows may expose a broken WSL ``bash.exe`` ahead of Git Bash in clean
    subprocess environments. Trying ``sh`` as a fallback keeps simple POSIX
    snippets working when ``bash`` only prints the WSL relay error.
    """
    if os.name == "nt":
        return ["bash", "sh"]
    return ["bash"]


def _normalize_inline_shell_output(output: str, cwd: Path | None) -> str:
    """Normalize shell cwd echoes back to the native skill directory path."""
    if not output or not cwd:
        return output
    native_cwd = str(cwd)
    aliases = {native_cwd, native_cwd.replace("\\", "/")}
    try:
        resolved = str(cwd.resolve())
        aliases.add(resolved)
        aliases.add(resolved.replace("\\", "/"))
    except Exception:
        pass
    try:
        _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
        cyg = subprocess.run(
            ["sh", "-c", "pwd"],
            cwd=native_cwd,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            stdin=subprocess.DEVNULL,
            **_popen_kwargs,
        )
        if cyg.stdout:
            aliases.add(cyg.stdout.strip())
    except Exception:
        pass
    for alias in sorted((a for a in aliases if a), key=len, reverse=True):
        output = output.replace(alias, native_cwd)
    return output


def run_inline_shell(command: str, cwd: Path | None, timeout: int) -> str:
    """Execute a single inline-shell snippet and return its stdout (trimmed).

    Failures return a short ``[inline-shell error: ...]`` marker instead of
    raising, so one bad snippet can't wreck the whole skill message.
    """
    last_error = "bash not found"
    _popen_kwargs = {"creationflags": windows_hide_flags()} if IS_WINDOWS else {}
    for shell in _shell_candidates():
        try:
            completed = subprocess.run(
                [shell, "-c", command],
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout)),
                check=False,
                stdin=subprocess.DEVNULL,
                **_popen_kwargs,
            )
        except subprocess.TimeoutExpired:
            return f"[inline-shell timeout after {timeout}s: {command}]"
        except FileNotFoundError:
            last_error = f"{shell} not found"
            continue
        except RuntimeError as exc:
            # tests/conftest.py installs a live-system guard that blocks real
            # os.kill on out-of-tree PIDs. subprocess.run(timeout=...) may trip
            # that guard while trying to clean up the timed-out shell; treat it
            # as the same timeout outcome instead of surfacing the guard error.
            if "live-system guard: blocked os.kill" in str(exc):
                return f"[inline-shell timeout after {timeout}s: {command}]"
            return f"[inline-shell error: {exc}]"
        except Exception as exc:
            return f"[inline-shell error: {exc}]"

        output = (completed.stdout or "").rstrip("\n")
        stderr = (completed.stderr or "").rstrip("\n")
        combined = "\n".join(part for part in (output, stderr) if part)
        if completed.returncode != 0 and "execvpe(/bin/bash) failed" in combined:
            last_error = combined
            continue
        if not output and stderr:
            output = stderr
        output = _normalize_inline_shell_output(output, cwd)
        if len(output) > _INLINE_SHELL_MAX_OUTPUT:
            output = output[:_INLINE_SHELL_MAX_OUTPUT] + "...[truncated]"
        return output
    return f"[inline-shell error: {last_error}]"


def expand_inline_shell(
    content: str,
    skill_dir: Path | None,
    timeout: int,
) -> str:
    """Replace every !`cmd` snippet in ``content`` with its stdout.

    Runs each snippet with the skill directory as CWD so relative paths in
    the snippet work the way the author expects.
    """
    if "!`" not in content:
        return content

    def _replace(match: re.Match) -> str:
        cmd = match.group(1).strip()
        if not cmd:
            return ""
        return run_inline_shell(cmd, skill_dir, timeout)

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
        timeout = int(cfg.get("inline_shell_timeout", 10) or 10)
        content = expand_inline_shell(content, skill_dir, timeout)
    return content
