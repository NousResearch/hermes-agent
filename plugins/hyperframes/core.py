"""Hermes bridge for https://github.com/heygen-com/hyperframes."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

PLUGIN_ID = "hyperframes"
CONFIG_ALIASES = (PLUGIN_ID, "hyper_frames")
TOOLSET = "hyperframes"
SKILL_NAME = "hyperframes"
UPSTREAM_REPO = "https://github.com/heygen-com/hyperframes.git"
DEFAULT_REF = "main"
MIN_NODE_MAJOR = 22
MIN_HYPERFRAMES_VERSION = "0.4.2"
PREVIEW_STATE_NAME = "hyperframes_preview_state.json"
DEFAULT_PREVIEW_PORT = 3002

# HyperFrames itself does not use Google OAuth. Optional post-render review only.
OPTIONAL_ENV_KEYS = {
    "GEMINI_API_KEY": "Optional: native video_analyze QA after render (Gemini via OpenRouter/auxiliary).",
    "GOOGLE_API_KEY": "Alias for GEMINI_API_KEY on some provider paths.",
    "PRODUCER_FORCE_SCREENSHOT": "Escape hatch when Chromium BeginFrame is unavailable (set to true).",
}


def plugin_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return plugin_dir().parents[1]


def bundled_skill_dir() -> Path:
    return repo_root() / "optional-skills" / "creative" / "hyperframes"


def vendor_root() -> Path:
    return plugin_dir() / "vendor" / "hyperframes"


def user_skill_path() -> Path:
    return get_hermes_home() / "skills" / SKILL_NAME


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _path_text(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _load_config_readonly() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _plugin_config() -> dict[str, Any]:
    plugins = _load_config_readonly().get("plugins", {})
    if not isinstance(plugins, dict):
        return {}
    entries = plugins.get("entries", {})
    if not isinstance(entries, dict):
        return {}
    for key in CONFIG_ALIASES:
        value = entries.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def projects_root(value: Any = None) -> Path:
    text = _path_text(value or _plugin_config().get("projects_dir"))
    if text:
        path = Path(text).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    default = get_hermes_home() / "hyperframes" / "projects"
    default.mkdir(parents=True, exist_ok=True)
    return default


def _preview_state_file() -> Path:
    return get_hermes_home() / PREVIEW_STATE_NAME


def _read_preview_state() -> dict[str, Any]:
    path = _preview_state_file()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_preview_state(payload: dict[str, Any]) -> None:
    path = _preview_state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _clear_preview_state() -> None:
    try:
        _preview_state_file().unlink()
    except FileNotFoundError:
        pass


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        if os.name == "nt":
            return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok - POSIX-only fallback when psutil is unavailable.
        return True
    except OSError:
        return False


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=merged,
        check=False,
    )


def _run_git(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    git = _git_exe()
    if not git:
        raise FileNotFoundError("git")
    return _run([git, *args], cwd=cwd)


def _which(name: str) -> str | None:
    return shutil.which(name)


def _npm_exe() -> str | None:
    return os.environ.get("HYPERFRAMES_NPM") or _which("npm")


def _npx_exe() -> str | None:
    return os.environ.get("HYPERFRAMES_NPX") or _which("npx")


def _node_exe() -> str | None:
    return os.environ.get("HYPERFRAMES_NODE") or _which("node")


def _git_exe() -> str | None:
    return os.environ.get("HYPERFRAMES_GIT") or _which("git")


def _node_major_version() -> int | None:
    node = _node_exe()
    if not node:
        return None
    try:
        proc = _run([node, "--version"])
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    match = re.match(r"v?(\d+)", (proc.stdout or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _hyperframes_version() -> str | None:
    candidates: list[list[str]] = []
    hf = _which("hyperframes")
    if hf:
        candidates.append([hf, "--version"])
    npx = _npx_exe()
    if npx:
        candidates.append([npx, "--yes", "hyperframes", "--version"])
    for cmd in candidates:
        try:
            proc = _run(cmd, timeout=120.0)
        except (FileNotFoundError, OSError):
            continue
        if proc.returncode != 0:
            continue
        lines = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        if not lines:
            continue
        version = lines[-1].lstrip("v")
        if version:
            return version
    return None


def _version_ge(left: str, right: str) -> bool:
    def _parts(value: str) -> list[int]:
        chunks: list[int] = []
        for piece in re.split(r"[.\-+]", value):
            if piece.isdigit():
                chunks.append(int(piece))
        return chunks

    a = _parts(left)
    b = _parts(right)
    length = max(len(a), len(b))
    a.extend([0] * (length - len(a)))
    b.extend([0] * (length - len(b)))
    return a >= b


def hyperframes_cmd() -> list[str]:
    hf = _which("hyperframes")
    if hf:
        return [hf]
    npx = _npx_exe()
    if npx:
        return [npx, "--yes", "hyperframes"]
    return ["npx", "--yes", "hyperframes"]


def check_available() -> bool:
    return _node_major_version() is not None and bool(_which("ffmpeg")) and bool(_npm_exe() or _npx_exe())


def _materialize_skill_source(src: Path, dst: Path) -> dict[str, Any]:
    if not (src / "SKILL.md").is_file():
        return {"ok": False, "error": f"Missing SKILL.md in bundled skill: {src}"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(str(src.resolve()), str(dst))
        return {"ok": True, "action": "symlink", "source": str(src.resolve()), "destination": str(dst)}
    except (OSError, NotImplementedError) as sym_err:
        try:
            shutil.copytree(
                str(src.resolve()),
                str(dst),
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
                dirs_exist_ok=False,
            )
            return {
                "ok": True,
                "action": "copytree",
                "source": str(src.resolve()),
                "destination": str(dst),
                "note": f"Symlink failed ({sym_err}); used copy fallback.",
            }
        except Exception as copy_err:
            return {
                "ok": False,
                "action": "copy_failed",
                "error": str(copy_err),
                "symlink_error": str(sym_err),
            }


def sync_skill_link(*, force: bool = False) -> dict[str, Any]:
    src = bundled_skill_dir()
    dst = user_skill_path()
    if not (src / "SKILL.md").is_file():
        return {
            "ok": False,
            "error": "Bundled hyperframes skill missing in this Hermes checkout.",
            "expected_path": str(src),
            "hint": "Run: hermes skills install official/creative/hyperframes",
        }
    if dst.exists() and not force:
        return {
            "ok": True,
            "skipped": True,
            "action": "already_linked",
            "destination": str(dst),
            "is_symlink": dst.is_symlink(),
        }
    return _materialize_skill_source(src, dst)


def clone_upstream(*, force: bool = False, ref: str | None = None) -> dict[str, Any]:
    dest = vendor_root()
    if (dest / ".git").exists():
        if not force:
            return {"ok": True, "skipped": True, "action": "already_cloned", "path": str(dest)}
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    branch = (ref or DEFAULT_REF).strip() or DEFAULT_REF
    attempts = [
        ["clone", "--depth", "1", "--branch", branch, UPSTREAM_REPO, str(dest)],
        ["clone", "--depth", "1", UPSTREAM_REPO, str(dest)],
    ]
    last: subprocess.CompletedProcess[str] | None = None
    for args in attempts:
        last = _run_git(args)
        if last.returncode == 0 and dest.is_dir():
            return {"ok": True, "action": "cloned", "path": str(dest), "ref": branch}
    return {
        "ok": False,
        "action": "clone_failed",
        "path": str(dest),
        "ref": branch,
        "stderr": (last.stderr if last else "").strip(),
        "stdout": (last.stdout if last else "").strip(),
    }


def setup_environment(*, skip_chrome: bool = False, auto_prereqs: bool = False) -> dict[str, Any]:
    prereqs: dict[str, Any] | None = None
    if auto_prereqs:
        prereqs = ensure_prerequisites()
        if prereqs.get("uac_pending"):
            return {
                "ok": False,
                "action": "uac_pending",
                "prerequisites": prereqs,
                "note": "Approve the UAC prompt, then re-run: hermes hyperframes setup",
            }

    node_major = _node_major_version()
    if node_major is None:
        return {"ok": False, "error": "Node.js is not installed.", "min_node_major": MIN_NODE_MAJOR}
    if node_major < MIN_NODE_MAJOR:
        return {
            "ok": False,
            "error": f"Node.js {node_major} is too old; HyperFrames requires >= {MIN_NODE_MAJOR}.",
        }
    if not _npm_exe():
        return {"ok": False, "error": "npm is not installed."}
    if not _which("ffmpeg"):
        return {"ok": False, "error": "FFmpeg is not installed.", "hint_windows": "winget install Gyan.FFmpeg"}

    npm = _npm_exe()
    assert npm is not None
    npm_install = _install_hyperframes_cli(npm)
    if not npm_install.get("ok"):
        return {
            "ok": False,
            "action": "npm_install_failed",
            "prerequisites": prereqs,
            **npm_install,
        }

    version = _hyperframes_version()
    if not version or not _version_ge(version, MIN_HYPERFRAMES_VERSION):
        return {
            "ok": False,
            "action": "version_check_failed",
            "version": version,
            "required": MIN_HYPERFRAMES_VERSION,
        }

    chrome: dict[str, Any] | None = None
    if not skip_chrome:
        npx = _npx_exe()
        if npx:
            try:
                chrome_proc = _run(
                    [npx, "--yes", "puppeteer", "browsers", "install", "chrome-headless-shell"],
                    timeout=900.0,
                )
            except (FileNotFoundError, OSError) as exc:
                chrome_proc = None
                chrome = {"ok": False, "error": str(exc)}
            else:
                chrome = {
                    "ok": chrome_proc.returncode == 0,
                    "returncode": chrome_proc.returncode,
                    "stderr": chrome_proc.stderr.strip()[:500],
                    "stdout": chrome_proc.stdout.strip()[:500],
                }
        else:
            chrome = {"ok": False, "error": "npx not found"}

    doctor: dict[str, Any]
    try:
        doctor_proc = _run(hyperframes_cmd() + ["doctor"], timeout=300.0)
        doctor = {
            "returncode": doctor_proc.returncode,
            "stdout": doctor_proc.stdout.strip(),
            "stderr": doctor_proc.stderr.strip(),
        }
    except (FileNotFoundError, OSError) as exc:
        doctor = {"returncode": 1, "stderr": str(exc), "stdout": ""}
    return {
        "ok": doctor.get("returncode") == 0,
        "prerequisites": prereqs,
        "node_major": node_major,
        "hyperframes_version": version,
        "chrome_headless_shell": chrome,
        "npm_install": npm_install,
        "doctor": {
            "returncode": doctor.get("returncode"),
            "stdout": doctor.get("stdout", ""),
            "stderr": doctor.get("stderr", ""),
        },
    }


def _install_hyperframes_cli(npm: str) -> dict[str, Any]:
    try:
        proc = _run([npm, "install", "-g", "hyperframes@latest"], timeout=600.0)
    except (FileNotFoundError, OSError) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode == 0:
        return {"ok": True, "returncode": 0}
    if os.name == "nt":
        from . import windows_install

        elevated = windows_install.ensure_npm_global("hyperframes@latest")
        if elevated.get("action") == "uac_handoff":
            return {"ok": False, "uac_pending": True, **elevated}
        if elevated.get("ok"):
            return elevated
    return {
        "ok": False,
        "returncode": proc.returncode,
        "stderr": proc.stderr.strip(),
        "stdout": proc.stdout.strip(),
    }


def ensure_prerequisites() -> dict[str, Any]:
    """Install Node/FFmpeg/HyperFrames CLI when missing (Windows winget + UAC)."""
    steps: dict[str, Any] = {}
    uac_pending = False

    node_major = _node_major_version()
    if node_major is None or node_major < MIN_NODE_MAJOR:
        if os.name == "nt":
            from . import windows_install

            steps["node"] = windows_install.ensure_node(min_major=MIN_NODE_MAJOR)
            if steps["node"].get("action") == "uac_handoff":
                uac_pending = True
        else:
            steps["node"] = {
                "ok": False,
                "error": f"Install Node.js >= {MIN_NODE_MAJOR} from https://nodejs.org/",
            }

    if not _which("ffmpeg"):
        if os.name == "nt":
            from . import windows_install

            steps["ffmpeg"] = windows_install.ensure_ffmpeg()
            if steps["ffmpeg"].get("action") == "uac_handoff":
                uac_pending = True
        else:
            steps["ffmpeg"] = {"ok": False, "error": "Install FFmpeg via your package manager."}

    return {
        "ok": not uac_pending and all(step.get("ok", False) for step in steps.values()) if steps else True,
        "uac_pending": uac_pending,
        "steps": steps,
    }


def apply_video_defaults() -> dict[str, Any]:
    """Pin HyperFrames as the default HTML/video path in Hermes config."""
    try:
        from hermes_cli.config import load_config, save_config as persist_config
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    cfg = load_config()
    plugins = cfg.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        cfg["plugins"] = plugins
    entries = plugins.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins["entries"] = entries
    entry = entries.setdefault(PLUGIN_ID, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[PLUGIN_ID] = entry
    entry.setdefault("auto_install", True)
    entry.setdefault("preferred_video_renderer", True)
    entry.setdefault("projects_dir", str(projects_root()))

    tools = cfg.setdefault("tools", {})
    if isinstance(tools, dict):
        cli_tools = tools.setdefault("cli", {})
        if isinstance(cli_tools, dict):
            enabled = cli_tools.setdefault("enabled", [])
            if isinstance(enabled, list):
                for name in (TOOLSET, "terminal", "file"):
                    if name not in enabled:
                        enabled.append(name)

    persist_config(cfg)
    return {
        "ok": True,
        "config_key": f"plugins.entries.{PLUGIN_ID}",
        "preferred_video_renderer": True,
        "tools_cli_enabled": cfg.get("tools", {}).get("cli", {}).get("enabled"),
        "note": (
            "HyperFrames is the default HTML-to-video path. "
            "Use manim-video only for pure math/geometry explainers."
        ),
    }


def install(
    *,
    force: bool = False,
    ref: str | None = None,
    skip_vendor: bool = False,
    auto_prereqs: bool = True,
) -> dict[str, Any]:
    defaults = apply_video_defaults()
    skill = sync_skill_link(force=force)
    setup = setup_environment(skip_chrome=False, auto_prereqs=auto_prereqs)
    vendor: dict[str, Any] | None = None
    if not skip_vendor:
        vendor = clone_upstream(force=force, ref=ref)
    ok = bool(skill.get("ok")) and bool(setup.get("ok")) and bool(defaults.get("ok"))
    return {
        "ok": ok,
        "defaults": defaults,
        "skill_link": skill,
        "setup": setup,
        "vendor": vendor,
        "slash_command": f"/{SKILL_NAME}",
        "projects_root": str(projects_root()),
        "display_projects_root": f"{display_hermes_home()}/hyperframes/projects",
        "oauth_note": (
            "HyperFrames render/TTS/transcribe do NOT require Google OAuth. "
            "Optional GEMINI_API_KEY/GOOGLE_API_KEY in ~/.hermes/.env only for video_analyze QA."
        ),
        "next_steps": [
            "Approve any UAC prompts, then: hermes hyperframes status",
            "Start a new session or run /skills reload (skill is at ~/.hermes/skills/hyperframes).",
            "Scaffold: hermes hyperframes init my-video",
            "Workflow: lint → validate → render via hyperframes_* tools or /hyperframes",
        ],
    }


def status() -> dict[str, Any]:
    skill = user_skill_path()
    bundled = bundled_skill_dir()
    vendor = vendor_root()
    preview = _read_preview_state()
    preview_pid = preview.get("pid")
    return {
        "ok": True,
        "plugin": PLUGIN_ID,
        "upstream_repo": UPSTREAM_REPO,
        "bundled_skill": {
            "path": str(bundled),
            "present": (bundled / "SKILL.md").is_file(),
        },
        "hermes_skill": {
            "path": str(skill),
            "present": skill.exists() or skill.is_symlink(),
            "is_symlink": skill.is_symlink() if skill.exists() else False,
        },
        "vendor": {
            "path": str(vendor),
            "present": vendor.is_dir(),
        },
        "environment": {
            "node_major": _node_major_version(),
            "ffmpeg": bool(_which("ffmpeg")),
            "npm": bool(_npm_exe()),
            "npx": bool(_npx_exe()),
            "hyperframes_version": _hyperframes_version(),
            "ready": check_available() and bool(_hyperframes_version()),
        },
        "projects_root": str(projects_root()),
        "display_projects_root": f"{display_hermes_home()}/hyperframes/projects",
        "preview": {
            "running": _pid_alive(int(preview_pid)) if preview_pid else False,
            "pid": preview_pid,
            "project_dir": preview.get("project_dir"),
            "port": preview.get("port", DEFAULT_PREVIEW_PORT),
        },
        "auth_and_env": {
            "hyperframes_requires_oauth": False,
            "optional_env": OPTIONAL_ENV_KEYS,
            "heygen_note": (
                "heygen-com/hyperframes is HTML→video via npm CLI. "
                "HeyGen Video Agent is a separate product (see surfsense plugin)."
            ),
        },
        "routing": {
            "preferred_for": [
                "motion graphics",
                "captioned narration",
                "website-to-video",
                "social overlays",
                "product promos",
            ],
            "fallback_skill": "manim-video (math/geometry only)",
        },
    }


def _resolve_project_dir(project_dir: Any = None, project_name: Any = None) -> Path:
    text = _path_text(project_dir)
    if text:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    name = _path_text(project_name)
    if name:
        return (projects_root() / name).resolve()
    return Path.cwd().resolve()


def run_cli(
    subcommand: str,
    *,
    project_dir: Path | None = None,
    extra_args: list[str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    if not _node_exe():
        return {"ok": False, "error": "Node.js is not installed."}
    cmd = hyperframes_cmd()
    if cmd[0] == _npx_exe() or (len(cmd) >= 2 and cmd[0].endswith("npx")):
        if not _npx_exe():
            return {"ok": False, "error": "npx is not installed (install Node.js >= 22)."}
    cmd = list(cmd) + [subcommand]
    if extra_args:
        cmd.extend(extra_args)
    cwd = project_dir or Path.cwd()
    try:
        proc = _run(cmd, cwd=cwd, timeout=timeout)
    except (FileNotFoundError, OSError) as exc:
        return {"ok": False, "error": str(exc), "command": cmd, "cwd": str(cwd)}
    return {
        "ok": proc.returncode == 0,
        "command": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def init_project(
    *,
    project_name: str,
    project_dir: str | None = None,
    example: str | None = None,
    video: str | None = None,
    audio: str | None = None,
    non_interactive: bool = True,
) -> dict[str, Any]:
    name = _path_text(project_name) or "my-video"
    root = _resolve_project_dir(project_dir, name)
    if root.exists() and any(root.iterdir()):
        return {"ok": False, "error": f"Project directory already exists and is not empty: {root}"}
    root.parent.mkdir(parents=True, exist_ok=True)
    args = [name, "--non-interactive"] if non_interactive else [name]
    if example:
        args.extend(["--example", _path_text(example)])
    if video:
        args.extend(["--video", _path_text(video)])
    if audio:
        args.extend(["--audio", _path_text(audio)])
    result = run_cli("init", project_dir=root.parent, extra_args=args, timeout=600.0)
    result["project_dir"] = str((root.parent / name).resolve())
    return result


def validate_project(
    *,
    project_dir: str | None = None,
    lint: bool = True,
    contrast: bool = True,
    inspect_layout: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    path = _resolve_project_dir(project_dir)
    if not path.is_dir():
        return {"ok": False, "error": f"Project directory not found: {path}"}
    results: dict[str, Any] = {"project_dir": str(path), "steps": {}}
    ok = True
    if lint:
        args = ["--strict"] if strict else []
        step = run_cli("lint", project_dir=path, extra_args=args or None, timeout=300.0)
        results["steps"]["lint"] = step
        ok = ok and step.get("ok", False)
    if contrast:
        step = run_cli("validate", project_dir=path, timeout=300.0)
        results["steps"]["validate"] = step
        ok = ok and step.get("ok", False)
    if inspect_layout:
        step = run_cli("inspect", project_dir=path, extra_args=["--json"], timeout=300.0)
        results["steps"]["inspect"] = step
        ok = ok and step.get("ok", False)
    results["ok"] = ok
    return results


def render_project(
    *,
    project_dir: str | None = None,
    output: str | None = None,
    quality: str = "standard",
    fps: int | None = None,
    format: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    path = _resolve_project_dir(project_dir)
    if not path.is_dir():
        return {"ok": False, "error": f"Project directory not found: {path}"}
    out_path = Path(_path_text(output) or (path / "output" / "final.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["--output", str(out_path), "--quality", _path_text(quality) or "standard"]
    if fps:
        args.extend(["--fps", str(int(fps))])
    if format:
        args.extend(["--format", _path_text(format)])
    if strict:
        args.append("--strict")
    result = run_cli("render", project_dir=path, extra_args=args, timeout=3600.0)
    result["output"] = str(out_path)
    return result


def capture_url(
    *,
    url: str,
    output_dir: str | None = None,
    skip_assets: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    target = _path_text(url)
    if not target:
        return {"ok": False, "error": "url is required"}
    args = [target]
    if output_dir:
        args.extend(["-o", _path_text(output_dir)])
    if skip_assets:
        args.append("--skip-assets")
    if json_output:
        args.append("--json")
    cwd = projects_root()
    return run_cli("capture", project_dir=cwd, extra_args=args, timeout=900.0)


def audio_command(
    *,
    action: str,
    text: str | None = None,
    input_path: str | None = None,
    output_path: str | None = None,
    voice: str | None = None,
    lang: str | None = None,
) -> dict[str, Any]:
    verb = _path_text(action).lower()
    if verb == "tts":
        script = _path_text(text)
        if not script:
            return {"ok": False, "error": "text is required for tts"}
        out = _path_text(output_path) or str(projects_root() / "narration.wav")
        args = [script, "--output", out]
        if voice:
            args.extend(["--voice", _path_text(voice)])
        if lang:
            args.extend(["--lang", _path_text(lang)])
        result = run_cli("tts", project_dir=projects_root(), extra_args=args, timeout=900.0)
        result["output"] = out
        return result
    if verb == "transcribe":
        src = _path_text(input_path)
        if not src:
            return {"ok": False, "error": "input_path is required for transcribe"}
        args = [src]
        if output_path:
            args.extend(["--output", _path_text(output_path)])
        return run_cli("transcribe", project_dir=projects_root(), extra_args=args, timeout=900.0)
    return {"ok": False, "error": f"Unknown audio action: {action}"}


def preview_control(
    *,
    action: str,
    project_dir: str | None = None,
    port: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    verb = _path_text(action).lower()
    if verb == "stop":
        state = _read_preview_state()
        pid = state.get("pid")
        if pid and _pid_alive(int(pid)):
            try:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/T", "/F"],
                        capture_output=True,
                        check=False,
                    )
                else:
                    os.kill(int(pid), 15)
            except OSError as exc:
                return {"ok": False, "error": str(exc), "pid": pid}
        _clear_preview_state()
        return {"ok": True, "action": "stopped", "pid": pid}

    path = _resolve_project_dir(project_dir)
    if not path.is_dir():
        return {"ok": False, "error": f"Project directory not found: {path}"}
    listen_port = int(port or DEFAULT_PREVIEW_PORT)
    state = _read_preview_state()
    if state.get("pid") and _pid_alive(int(state["pid"])) and not force:
        return {
            "ok": True,
            "action": "already_running",
            "pid": state.get("pid"),
            "project_dir": state.get("project_dir"),
            "port": state.get("port", listen_port),
            "url": f"http://127.0.0.1:{state.get('port', listen_port)}",
        }
    cmd = hyperframes_cmd() + ["preview", "--port", str(listen_port)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0,
    )
    _write_preview_state(
        {
            "pid": proc.pid,
            "project_dir": str(path),
            "port": listen_port,
            "command": cmd,
        }
    )
    return {
        "ok": True,
        "action": "started",
        "pid": proc.pid,
        "project_dir": str(path),
        "port": listen_port,
        "url": f"http://127.0.0.1:{listen_port}",
    }


def _args_dict(args: dict[str, Any] | None, **kwargs: Any) -> dict[str, Any]:
    payload = dict(args or {})
    payload.update({k: v for k, v in kwargs.items() if v is not None})
    return payload


def handle_status(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    return _json(status())


def handle_setup(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(setup_environment(skip_chrome=bool(data.get("skip_chrome"))))


def handle_install(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(
        install(
            force=bool(data.get("force")),
            ref=data.get("ref"),
            skip_vendor=bool(data.get("skip_vendor")),
        )
    )


def handle_init(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(
        init_project(
            project_name=_path_text(data.get("project_name")) or "my-video",
            project_dir=data.get("project_dir"),
            example=data.get("example"),
            video=data.get("video"),
            audio=data.get("audio"),
            non_interactive=bool(data.get("non_interactive", True)),
        )
    )


def handle_validate(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(
        validate_project(
            project_dir=data.get("project_dir"),
            lint=bool(data.get("lint", True)),
            contrast=bool(data.get("contrast", True)),
            inspect_layout=bool(data.get("inspect", True)),
            strict=bool(data.get("strict")),
        )
    )


def handle_render(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(
        render_project(
            project_dir=data.get("project_dir"),
            output=data.get("output"),
            quality=_path_text(data.get("quality")) or "standard",
            fps=data.get("fps"),
            format=data.get("format"),
            strict=bool(data.get("strict")),
        )
    )


def handle_preview(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(
        preview_control(
            action=_path_text(data.get("action")) or "start",
            project_dir=data.get("project_dir"),
            port=data.get("port"),
            force=bool(data.get("force")),
        )
    )


def handle_capture(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(
        capture_url(
            url=_path_text(data.get("url")),
            output_dir=data.get("output_dir"),
            skip_assets=bool(data.get("skip_assets")),
            json_output=bool(data.get("json_output")),
        )
    )


def handle_audio(args: dict[str, Any] | None = None, **kwargs: Any) -> str:
    data = _args_dict(args, **kwargs)
    return _json(
        audio_command(
            action=_path_text(data.get("action")),
            text=data.get("text"),
            input_path=data.get("input_path"),
            output_path=data.get("output_path"),
            voice=data.get("voice"),
            lang=data.get("lang"),
        )
    )


def handle_slash(args: str) -> str:
    text = (args or "").strip()
    if not text or text.lower() in {"status", "help"}:
        payload = status()
        payload["hint"] = "Usage: /hyperframes status | install | init <name> | lint <dir> | render <dir>"
        return _json(payload)
    parts = text.split()
    cmd = parts[0].lower()
    if cmd == "install":
        return handle_install({"force": "force" in parts[1:]})
    if cmd == "init" and len(parts) >= 2:
        return handle_init({"project_name": parts[1]})
    if cmd in {"lint", "validate"} and len(parts) >= 2:
        return handle_validate({"project_dir": parts[1]})
    if cmd == "render" and len(parts) >= 2:
        return handle_render({"project_dir": parts[1]})
    if cmd == "preview" and len(parts) >= 2:
        action = "stop" if parts[1].lower() == "stop" else "start"
        project = None if action == "stop" else parts[1]
        return handle_preview({"action": action, "project_dir": project})
    return _json({"ok": False, "error": f"Unknown /hyperframes subcommand: {text}"})


STATUS_SCHEMA = {
    "name": "hyperframes_status",
    "description": "Report HyperFrames skill link, npm CLI, FFmpeg, and preview server readiness.",
    "parameters": {"type": "object", "properties": {}},
}

SETUP_SCHEMA = {
    "name": "hyperframes_setup",
    "description": "Install or upgrade the hyperframes npm CLI, chrome-headless-shell, and run doctor.",
    "parameters": {
        "type": "object",
        "properties": {
            "skip_chrome": {
                "type": "boolean",
                "description": "Skip puppeteer chrome-headless-shell pre-cache.",
            },
        },
    },
}

INSTALL_SCHEMA = {
    "name": "hyperframes_install",
    "description": "Link the bundled hyperframes skill, set up the npm CLI, and optionally clone upstream examples.",
    "parameters": {
        "type": "object",
        "properties": {
            "force": {"type": "boolean"},
            "ref": {"type": "string", "description": "Git ref for optional upstream clone."},
            "skip_vendor": {
                "type": "boolean",
                "description": "Skip cloning heygen-com/hyperframes into the plugin vendor dir.",
            },
        },
    },
}

INIT_SCHEMA = {
    "name": "hyperframes_init",
    "description": "Scaffold a new HyperFrames HTML video project with hyperframes init.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_name": {"type": "string", "description": "Directory name for the new project."},
            "project_dir": {"type": "string", "description": "Parent directory; defaults to ~/.hermes/hyperframes/projects."},
            "example": {
                "type": "string",
                "description": "Template example: blank, warm-grain, swiss-grid, kinetic-type, product-promo, etc.",
            },
            "video": {"type": "string", "description": "Optional seed video file path."},
            "audio": {"type": "string", "description": "Optional seed audio file path."},
            "non_interactive": {"type": "boolean", "description": "Skip interactive prompts (default true)."},
        },
        "required": ["project_name"],
    },
}

VALIDATE_SCHEMA = {
    "name": "hyperframes_validate",
    "description": "Run hyperframes lint, validate (contrast), and inspect (layout) on a project.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_dir": {"type": "string", "description": "HyperFrames project directory."},
            "lint": {"type": "boolean"},
            "contrast": {"type": "boolean"},
            "inspect": {"type": "boolean"},
            "strict": {"type": "boolean"},
        },
        "required": ["project_dir"],
    },
}

RENDER_SCHEMA = {
    "name": "hyperframes_render",
    "description": "Render a HyperFrames HTML composition to MP4/WebM.",
    "parameters": {
        "type": "object",
        "properties": {
            "project_dir": {"type": "string"},
            "output": {"type": "string", "description": "Output media path."},
            "quality": {
                "type": "string",
                "enum": ["draft", "standard", "high"],
                "description": "Render quality preset.",
            },
            "fps": {"type": "integer"},
            "format": {"type": "string", "enum": ["mp4", "webm"]},
            "strict": {"type": "boolean"},
        },
        "required": ["project_dir"],
    },
}

PREVIEW_SCHEMA = {
    "name": "hyperframes_preview",
    "description": "Start or stop the HyperFrames live preview server for a project.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["start", "stop"]},
            "project_dir": {"type": "string"},
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "force": {"type": "boolean", "description": "Restart preview even if one is already running."},
        },
        "required": ["action"],
    },
}

CAPTURE_SCHEMA = {
    "name": "hyperframes_capture",
    "description": "Capture a website into editable HyperFrames capture assets.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "output_dir": {"type": "string"},
            "skip_assets": {"type": "boolean"},
            "json_output": {"type": "boolean"},
        },
        "required": ["url"],
    },
}

AUDIO_SCHEMA = {
    "name": "hyperframes_audio",
    "description": "Generate narration (tts) or word-level captions (transcribe) for HyperFrames projects.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["tts", "transcribe"]},
            "text": {"type": "string", "description": "Narration script for tts."},
            "input_path": {"type": "string", "description": "Audio file for transcribe."},
            "output_path": {"type": "string"},
            "voice": {"type": "string"},
            "lang": {"type": "string"},
        },
        "required": ["action"],
    },
}
