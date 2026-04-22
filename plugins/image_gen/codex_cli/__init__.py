"""Codex CLI image generation backend.

Uses Codex CLI's built-in image generation capability under the user's ChatGPT /
Codex login rather than the public OpenAI Images API.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "codex-cli-default"
_MODELS = [
    {
        "id": DEFAULT_MODEL,
        "display": "Codex CLI Built-in Image Generation",
        "speed": "varies",
        "strengths": "Uses ChatGPT/Codex built-in image tool",
        "price": "included with Codex auth",
    }
]
_ASPECT_GUIDANCE = {
    "landscape": "Use a landscape orientation.",
    "square": "Use a square composition.",
    "portrait": "Use a portrait orientation.",
}


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _plugin_config() -> Dict[str, Any]:
    cfg = _load_config()
    for key in ("codex_cli", "codex-cli"):
        value = cfg.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _codex_home() -> Path:
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env).expanduser()
    cfg_home = _plugin_config().get("codex_home")
    if isinstance(cfg_home, str) and cfg_home.strip():
        return Path(cfg_home).expanduser()
    raw = Path.home() / ".codex"
    return raw


def _generated_images_dir() -> Path:
    return _codex_home() / "generated_images"


def _subprocess_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["CODEX_HOME"] = str(_codex_home())
    return env


def _codex_binary() -> Optional[str]:
    configured = _plugin_config().get("command")
    if isinstance(configured, str) and configured.strip():
        binary = configured.strip().split()[0]
        found = shutil.which(binary)
        return found or configured.strip()
    return shutil.which("codex")


def _run_login_status(codex_bin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [codex_bin, "login", "status"],
        capture_output=True,
        text=True,
        timeout=30,
        env=_subprocess_env(),
    )


def _snapshot_generated_images() -> set[Path]:
    root = _generated_images_dir()
    if not root.exists():
        return set()
    return {p for p in root.rglob("*") if p.is_file()}


def _parse_json_lines(stdout: str) -> Iterable[dict]:
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _extract_thread_id(stdout: str) -> Optional[str]:
    for obj in _parse_json_lines(stdout):
        if obj.get("type") == "thread.started":
            thread_id = obj.get("thread_id")
            if isinstance(thread_id, str) and thread_id.strip():
                return thread_id.strip()
    return None


def _newest_file(paths: Iterable[Path]) -> Optional[Path]:
    files = [p for p in paths if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _find_generated_image(thread_id: Optional[str], before: set[Path]) -> Optional[Path]:
    root = _generated_images_dir()
    if thread_id:
        thread_dir = root / thread_id
        if thread_dir.exists():
            found = _newest_file(thread_dir.rglob("*"))
            if found is not None:
                return found
    after = _snapshot_generated_images()
    new_files = sorted((after - before), key=lambda p: p.stat().st_mtime, reverse=True)
    return new_files[0] if new_files else None


def _build_prompt(prompt: str, aspect_ratio: str) -> str:
    guidance = _ASPECT_GUIDANCE.get(aspect_ratio, _ASPECT_GUIDANCE[DEFAULT_ASPECT_RATIO])
    return (
        f"Generate an image for this request: {prompt}\n"
        f"{guidance}\n"
        "Use Codex built-in image generation if available."
    )


def _run_codex_image_generation(codex_bin: str, prompt: str, aspect_ratio: str) -> Tuple[subprocess.CompletedProcess, Optional[str], Optional[Path]]:
    before = _snapshot_generated_images()
    built_prompt = _build_prompt(prompt, aspect_ratio)
    with tempfile.TemporaryDirectory(prefix="codex-imagegen-") as tmp:
        tmp_path = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, capture_output=True, text=True, timeout=30, env=_subprocess_env())
        cmd = [
            codex_bin,
            "exec",
            "--json",
            "--full-auto",
            "--enable",
            "image_generation",
            built_prompt,
        ]
        result = subprocess.run(
            cmd,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
            env=_subprocess_env(),
        )
    thread_id = _extract_thread_id(result.stdout)
    image_path = _find_generated_image(thread_id, before)
    return result, thread_id, image_path


class CodexCLIImageGenProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "codex-cli"

    @property
    def display_name(self) -> str:
        return "Codex CLI"

    def is_available(self) -> bool:
        codex_bin = _codex_binary()
        if not codex_bin:
            return False
        try:
            result = _run_login_status(codex_bin)
        except Exception:
            return False
        return result.returncode == 0 and "logged in" in (result.stdout or "").lower()

    def list_models(self) -> List[Dict[str, Any]]:
        return list(_MODELS)

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Codex CLI",
            "badge": "chatgpt",
            "tag": "Built-in image generation via local Codex CLI login",
            "env_vars": [],
        }

    def generate(self, prompt: str, aspect_ratio: str = DEFAULT_ASPECT_RATIO, **kwargs: Any) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider=self.name,
                aspect_ratio=aspect,
            )

        codex_bin = _codex_binary()
        if not codex_bin:
            return error_response(
                error="codex CLI not installed or not on PATH",
                error_type="missing_dependency",
                provider=self.name,
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result, thread_id, image_path = _run_codex_image_generation(codex_bin, prompt, aspect)
        except subprocess.TimeoutExpired:
            return error_response(
                error="Codex CLI image generation timed out",
                error_type="timeout",
                provider=self.name,
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except Exception as exc:
            logger.debug("Codex CLI image generation failed", exc_info=True)
            return error_response(
                error=f"Codex CLI image generation failed: {exc}",
                error_type="provider_error",
                provider=self.name,
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            return error_response(
                error=f"Codex CLI image generation failed: {detail}",
                error_type="api_error",
                provider=self.name,
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        if image_path is None:
            return error_response(
                error="Codex CLI completed but no generated image file was found",
                error_type="empty_response",
                provider=self.name,
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {}
        if thread_id:
            extra["thread_id"] = thread_id

        return success_response(
            image=str(image_path),
            model=DEFAULT_MODEL,
            prompt=prompt,
            aspect_ratio=aspect,
            provider=self.name,
            extra=extra,
        )


def register(ctx) -> None:
    ctx.register_image_gen_provider(CodexCLIImageGenProvider())
