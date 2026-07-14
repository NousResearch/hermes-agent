from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path


def _literal_path(path: Path) -> str:
    return repr(str(path).replace("\\", "/"))


def _copy_source(source_root: Path, checkout: Path, workspace_root: Path) -> None:
    shutil.copytree(
        source_root,
        checkout,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "tests", ".github", "assets", "workspace"),
    )
    config_path = checkout / "app" / "config.py"
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"def get_project_root\(\) -> Path:\n.*?PROJECT_ROOT = get_project_root\(\)\nWORKSPACE_ROOT = PROJECT_ROOT / \"workspace\"",
        re.DOTALL,
    )
    replacement = (
        "def get_project_root() -> Path:\n"
        f"    return Path({_literal_path(checkout)})\n\n\n"
        "PROJECT_ROOT = get_project_root()\n"
        f"WORKSPACE_ROOT = Path({_literal_path(workspace_root)})"
    )
    patched, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError("unsupported OpenManus config.py layout; refusing to run")
    config_path.write_text(patched, encoding="utf-8")


def _write_config(checkout: Path, args: argparse.Namespace) -> None:
    api_key = "" if args.no_secret_env else os.environ.get(args.api_key_env, "")
    if not args.model or not args.base_url or (not args.no_secret_env and not api_key):
        raise RuntimeError("OpenManus LLM configuration is incomplete")
    config_dir = checkout / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": api_key,
        "max_tokens": 8192,
        "temperature": 0.0,
        "api_type": args.api_type,
        "api_version": "",
    }
    lines = ["[llm]"]
    for key, value in values.items():
        if isinstance(value, int):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {json.dumps(str(value))}")
    lines.extend(
        [
            "",
            "[browser]",
            "headless = true",
            "disable_security = false",
            "max_content_length = 4000",
            "",
            "[mcp]",
            'server_reference = "app.mcp.server"',
            "",
            "[runflow]",
            "use_data_analysis_agent = false",
            "",
            "[daytona]",
            "daytona_api_key = \"\"",
        ]
    )
    (config_dir / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _disable_network_imports(checkout: Path) -> None:
    search_init = checkout / "app" / "tool" / "search" / "__init__.py"
    search_text = search_init.read_text(encoding="utf-8")
    search_text = search_text.replace(
        "from app.tool.search.baidu_search import BaiduSearchEngine\n", ""
    )
    search_text = search_text.replace('    "BaiduSearchEngine",\n', "")
    search_init.write_text(search_text, encoding="utf-8")

    web_search = checkout / "app" / "tool" / "web_search.py"
    web_text = web_search.read_text(encoding="utf-8")
    web_text = web_text.replace("    BaiduSearchEngine,\n", "")
    web_text = web_text.replace('        "baidu": BaiduSearchEngine(),\n', "")
    web_search.write_text(web_text, encoding="utf-8")

    browser_tool = checkout / "app" / "tool" / "browser_use_tool.py"
    browser_text = browser_tool.read_text(encoding="utf-8")
    browser_text = browser_text.replace(
        "from app.tool.web_search import WebSearch\n", "WebSearch = object\n"
    )
    browser_tool.write_text(browser_text, encoding="utf-8")

    tool_init = checkout / "app" / "tool" / "__init__.py"
    tool_text = tool_init.read_text(encoding="utf-8")
    tool_text = tool_text.replace("from app.tool.web_search import WebSearch\n", "")
    tool_text = tool_text.replace('    "WebSearch",\n', "")
    tool_init.write_text(tool_text, encoding="utf-8")


def _disable_optional_sandbox_imports(checkout: Path) -> None:
    sandbox_client = checkout / "app" / "sandbox" / "client.py"
    sandbox_client.write_text(
        '''from abc import ABC, abstractmethod
from typing import Dict, Optional

from app.config import SandboxSettings


class BaseSandboxClient(ABC):
    @abstractmethod
    async def create(
        self,
        config: Optional[SandboxSettings] = None,
        volume_bindings: Optional[Dict[str, str]] = None,
    ) -> None:
        ...

    @abstractmethod
    async def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        ...

    @abstractmethod
    async def copy_from(self, container_path: str, local_path: str) -> None:
        ...

    @abstractmethod
    async def copy_to(self, local_path: str, container_path: str) -> None:
        ...

    @abstractmethod
    async def read_file(self, path: str) -> str:
        ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None:
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        ...


class LocalSandboxClient(BaseSandboxClient):
    def __init__(self):
        self.sandbox = None

    async def create(
        self,
        config: Optional[SandboxSettings] = None,
        volume_bindings: Optional[Dict[str, str]] = None,
    ) -> None:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    async def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    async def copy_from(self, container_path: str, local_path: str) -> None:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    async def copy_to(self, local_path: str, container_path: str) -> None:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    async def read_file(self, path: str) -> str:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    async def write_file(self, path: str, content: str) -> None:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    async def cleanup(self) -> None:
        self.sandbox = None


def create_sandbox_client() -> LocalSandboxClient:
    return LocalSandboxClient()


SANDBOX_CLIENT = create_sandbox_client()
''',
        encoding="utf-8",
    )

    tool_base = checkout / "app" / "daytona" / "tool_base.py"
    tool_base.write_text(
        '''from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, Optional

from pydantic import Field

from app.tool.base import BaseTool
from app.utils.files_utils import clean_path


class Sandbox:
    pass


@dataclass
class ThreadMessage:
    type: str
    content: Dict[str, Any]
    is_llm_message: bool = False
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = field(
        default_factory=lambda: datetime.now().timestamp()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "is_llm_message": self.is_llm_message,
            "metadata": self.metadata or {},
            "timestamp": self.timestamp,
        }


class SandboxToolsBase(BaseTool):
    _urls_printed: ClassVar[bool] = False
    project_id: Optional[str] = None
    _sandbox: Optional[Sandbox] = None
    _sandbox_id: Optional[str] = None
    _sandbox_pass: Optional[str] = None
    workspace_path: str = Field(default="/workspace", exclude=True)
    _sessions: dict[str, str] = {}

    class Config:
        arbitrary_types_allowed = True

    async def _ensure_sandbox(self) -> Sandbox:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    @property
    def sandbox(self) -> Sandbox:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    @property
    def sandbox_id(self) -> str:
        raise RuntimeError("sandbox unavailable in Hermes no-network OpenManus run")

    def clean_path(self, path: str) -> str:
        return clean_path(path, self.workspace_path)
''',
        encoding="utf-8",
    )


async def _run(args: argparse.Namespace) -> str:
    source_root = Path(args.source_root).resolve()
    run_root = Path(args.run_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    checkout = run_root / "openmanus-checkout"
    run_root.mkdir(parents=True, exist_ok=True)
    _copy_source(source_root, checkout, workspace_root)
    if args.network_scope != "full":
        _disable_network_imports(checkout)
        _disable_optional_sandbox_imports(checkout)
    config_path = checkout / "config" / "config.toml"
    agent = None
    try:
        _write_config(checkout, args)
        sys.path.insert(0, str(checkout))
        os.chdir(checkout)
        if args.agent_mode == "data_analysis":
            from app.agent.data_analysis import DataAnalysis  # pyright: ignore[reportMissingImports]

            agent = DataAnalysis()
        else:
            from app.agent.manus import Manus  # pyright: ignore[reportMissingImports]

            agent = await Manus.create()
        agent.max_steps = max(1, int(args.max_steps))
        if args.network_scope != "full":
            from app.tool import ToolCollection  # pyright: ignore[reportMissingImports]

            blocked = ("browser", "mcp", "web")
            agent.available_tools = ToolCollection(
                *(tool for tool in agent.available_tools.tools if not any(word in tool.name.lower() for word in blocked))
            )
        result = await agent.run(args.prompt)
        return str(result or "")
    finally:
        if agent is not None:
            cleanup = getattr(agent, "cleanup", None)
            if cleanup is not None:
                await cleanup()
        config_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-type", default="openai")
    parser.add_argument("--api-key-env", default="OPENMANUS_API_KEY")
    parser.add_argument("--no-secret-env", action="store_true")
    parser.add_argument("--agent-mode", choices=["manus", "data_analysis"], default="manus")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-network", dest="allow_network", action="store_false")
    parser.add_argument("--network-scope", choices=["none", "llm_only", "full"], default="none")
    parser.set_defaults(allow_network=False)
    args = parser.parse_args()
    args.allow_network = args.network_scope != "none"
    args.prompt = sys.stdin.read()
    try:
        result = asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"OpenManus runner failed: {exc}", file=sys.stderr)
        return 1
    print("__HERMES_OPENMANUS_RESULT__:" + json.dumps({"result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
