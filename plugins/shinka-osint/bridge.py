"""Bridge to an external ShinkaEvolve-OSINT checkout."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

from . import providers

try:
    from hermes_cli.config import get_env_value, save_env_value
except Exception:  # pragma: no cover - import safety during early plugin load
    get_env_value = None  # type: ignore[assignment]
    save_env_value = None  # type: ignore[assignment]


def _desktop_dir() -> Path:
    if os.name == "nt":
        try:
            import winreg

            key_path = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                raw, _ = winreg.QueryValueEx(key, "Desktop")
            return Path(os.path.expandvars(str(raw)))
        except Exception:
            pass
    return Path.home() / "Desktop"


DEFAULT_ROOT = _desktop_dir() / "ShinkaEvolve-OSINT-main" / "ShinkaEvolve-OSINT-main"
DEFAULT_EXAMPLE = "milspec_security_jp"
ENV_ROOT = "SHINKA_OSINT_ROOT"
ENV_DEFAULT_EXAMPLE = "SHINKA_OSINT_DEFAULT_EXAMPLE"
STATE_CONFIG = "config.json"

_MCP_MODULE = None
_MCP_ROOT: Path | None = None

# Example-local imports (e.g. ``from tools.audit_logger``) collide with Hermes'
# top-level ``tools`` package when handlers run in-process.
_ISOLATED_TOOLS = frozenset(
    {
        "shinka_evaluate",
        "shinka_evaluate_all",
        "shinka_run_tests",
        "shinka_start_evolution",
        "shinka_mutate",
        "shinka_apply_patch",
    }
)

_ISOLATED_RUNNER = r"""
import importlib.util
import json
import os
import sys
from pathlib import Path

payload = json.loads(sys.stdin.read())
root = Path(os.environ["SHINKA_OSINT_ROOT"]).resolve()
example = (payload.get("arguments") or {}).get("example", "")
example_dir = root / "examples" / example if example else root

sys.path[:0] = [str(example_dir), str(root)]
os.chdir(str(example_dir if example_dir.is_dir() else root))

spec = importlib.util.spec_from_file_location("shinka_mcp_server", root / "shinka_mcp_server.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

handler = module.TOOL_HANDLERS[payload["tool"]]
result = handler(payload.get("arguments") or {})
print(json.dumps(result, ensure_ascii=False, default=str))
"""


def _state_dir() -> Path:
    path = get_hermes_home() / "shinka-osint"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_state_config() -> dict[str, Any]:
    path = _state_dir() / STATE_CONFIG
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_state_config(data: dict[str, Any]) -> None:
    path = _state_dir() / STATE_CONFIG
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_root() -> Path:
    """Return the configured ShinkaEvolve-OSINT checkout root."""
    for candidate in (
        os.environ.get(ENV_ROOT, "").strip(),
        (_read_state_config().get("root") or "").strip(),
        (get_env_value(ENV_ROOT) if get_env_value else "") or "",
    ):
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_dir():
                return path.resolve()
    if DEFAULT_ROOT.is_dir():
        return DEFAULT_ROOT.resolve()
    return DEFAULT_ROOT


def resolve_default_example() -> str:
    for candidate in (
        os.environ.get(ENV_DEFAULT_EXAMPLE, "").strip(),
        (_read_state_config().get("default_example") or "").strip(),
        (get_env_value(ENV_DEFAULT_EXAMPLE) if get_env_value else "") or "",
    ):
        if candidate:
            return candidate
    return DEFAULT_EXAMPLE


def save_root(path: str | Path, *, persist_env: bool = True) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"ShinkaEvolve-OSINT root not found: {resolved}")
    mcp = resolved / "shinka_mcp_server.py"
    if not mcp.is_file():
        raise FileNotFoundError(f"Missing shinka_mcp_server.py under {resolved}")

    cfg = _read_state_config()
    cfg["root"] = str(resolved)
    _write_state_config(cfg)
    os.environ[ENV_ROOT] = str(resolved)

    if persist_env and save_env_value:
        save_env_value(ENV_ROOT, str(resolved), secret=False)
    return resolved


def save_default_example(example: str, *, persist_env: bool = True) -> str:
    example = (example or "").strip() or DEFAULT_EXAMPLE
    cfg = _read_state_config()
    cfg["default_example"] = example
    _write_state_config(cfg)
    os.environ[ENV_DEFAULT_EXAMPLE] = example
    if persist_env and save_env_value:
        save_env_value(ENV_DEFAULT_EXAMPLE, example, secret=False)
    return example


def root_status() -> dict[str, Any]:
    root = resolve_root()
    mcp_path = root / "shinka_mcp_server.py"
    pyproject = root / "pyproject.toml"
    example = resolve_default_example()
    example_dir = root / "examples" / example
    return {
        "root": str(root),
        "root_exists": root.is_dir(),
        "mcp_server_exists": mcp_path.is_file(),
        "pyproject_exists": pyproject.is_file(),
        "default_example": example,
        "default_example_ready": example_dir.is_dir(),
        "importable": False,
        "tool_count": 0,
    }


def _load_mcp_module():
    global _MCP_MODULE, _MCP_ROOT
    root = resolve_root()
    if _MCP_MODULE is not None and _MCP_ROOT == root:
        return _MCP_MODULE

    mcp_path = root / "shinka_mcp_server.py"
    if not mcp_path.is_file():
        raise FileNotFoundError(
            f"ShinkaEvolve-OSINT not found at {root}. "
            f"Run `hermes shinka-osint setup --root <path>`."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    module_name = "shinka_mcp_server_hermes_bridge"
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, mcp_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MCP bridge from {mcp_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    _MCP_MODULE = module
    _MCP_ROOT = root
    return module


def _resolve_isolated_python() -> list[str]:
    """Pick a Python interpreter argv prefix that can import Shinka deps."""
    override = (os.environ.get("SHINKA_OSINT_PYTHON") or "").strip()
    if override:
        return override.split()

    def _can_import_anthropic(argv: list[str]) -> bool:
        try:
            proc = subprocess.run(
                [*argv, "-c", "import anthropic"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return proc.returncode == 0
        except Exception:
            return False

    candidates: list[list[str]] = [
        [sys.executable],
        ["py", "-3"],
        ["python3"],
        ["python"],
    ]
    for argv in candidates:
        if _can_import_anthropic(argv):
            return argv
    return [sys.executable]


def _call_tool_isolated(tool_name: str, arguments: dict[str, Any]) -> Any:
    root = resolve_root()
    example = (arguments.get("example") or resolve_default_example()).strip()
    example_dir = root / "examples" / example
    if not example_dir.is_dir():
        raise FileNotFoundError(f"Example not found: {example} (under {root / 'examples'})")

    env = os.environ.copy()
    env.update(providers.build_env_overlay())
    env["SHINKA_OSINT_ROOT"] = str(root)
    env["MCP_QUIET_STDERR"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(example_dir), str(root)])
    env.pop("PYTHONSAFEPATH", None)

    python_argv = _resolve_isolated_python()
    proc = subprocess.run(
        [*python_argv, "-c", _ISOLATED_RUNNER],
        input=json.dumps({"tool": tool_name, "arguments": arguments}, ensure_ascii=False),
        cwd=str(example_dir),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Shinka tool {tool_name} failed (exit {proc.returncode}): {detail[:2000]}"
        )
    stdout = (proc.stdout or "").strip()
    if not stdout:
        return {}
    return json.loads(stdout)


def call_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    """Invoke a Shinka MCP tool handler."""
    args = dict(arguments or {})
    if tool_name in _ISOLATED_TOOLS:
        return _call_tool_isolated(tool_name, args)

    module = _load_mcp_module()
    handlers: dict[str, Callable[[dict[str, Any]], Any]] = getattr(module, "TOOL_HANDLERS", {})
    handler = handlers.get(tool_name)
    if handler is None:
        raise KeyError(f"Unknown Shinka MCP tool: {tool_name}")
    return handler(args)


def check_available() -> bool:
    status = root_status()
    if not status["root_exists"] or not status["mcp_server_exists"]:
        return False
    try:
        module = _load_mcp_module()
        status["importable"] = True
        status["tool_count"] = len(getattr(module, "TOOL_HANDLERS", {}))
        return status["tool_count"] > 0
    except Exception:
        return False
