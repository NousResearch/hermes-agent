from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import os
import sys
from pathlib import Path
from typing import Any


LOCAL_DEV_PROFILE = "local-dev"
REMOTE_PROFILE = "remote"
UNSAFE_REMOTE_ACK = "--i-understand-this-is-unsafe"
UNSAFE_REMOTE_ENV = "HERMES_GPT_UNSAFE_REMOTE_NOAUTH"
ENABLE_WRITE_ENV = "HERMES_GPT_ENABLE_WRITE"
ENABLE_MEMORY_WRITE_ENV = "HERMES_GPT_ENABLE_MEMORY_WRITE"
ENABLE_SESSION_SEARCH_ENV = "HERMES_GPT_ENABLE_SESSION_SEARCH"
ENABLE_TERMINAL_ENV = "HERMES_GPT_ENABLE_TERMINAL"
NOAUTH_META = {"securitySchemes": [{"type": "noauth"}]}

HERMES_ROOT: Path | None = None
IMPORT_ERROR: str | None = None
file_tools: Any = None
terminal_tool: Any = None
memory_tool: Any = None
skill_manager_tool: Any = None
SessionDB: Any = None
get_hermes_home: Any = None


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def env_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def is_loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def is_hermes_root(path: Path) -> bool:
    return path.exists() and ((path / "tools").is_dir() or (path / "hermes_state.py").exists())


def candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    for parent in Path(__file__).resolve().parents:
        if is_hermes_root(parent):
            candidates.append(parent)
            break

    env_home = os.environ.get("HERMES_HOME")
    if env_home:
        env_path = Path(env_home).expanduser()
        candidates.extend([env_path, env_path / "hermes-agent"])

    home = Path.home()
    candidates.extend(
        [
            home / "AppData" / "Local" / "hermes" / "hermes-agent",
            home / ".hermes" / "hermes-agent",
        ]
    )

    for package in ("hermes-agent", "hermes_agent"):
        try:
            dist = importlib.metadata.distribution(package)
            base = Path(dist.locate_file("")).resolve()
        except Exception:
            continue
        for parent in [base, *base.parents]:
            if parent.name == "hermes-agent":
                candidates.append(parent)
                break

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        key = str(resolved).lower()
        if key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique


def find_hermes_root() -> Path:
    for candidate in candidate_roots():
        if is_hermes_root(candidate):
            return candidate
    raise RuntimeError("Could not find a Hermes Agent source root with a tools directory.")


def add_path_once(path: Path, *, prepend: bool = True) -> None:
    value = str(path)
    existing = {str(Path(p).resolve()).lower() for p in sys.path if p}
    if str(path.resolve()).lower() not in existing:
        if prepend:
            sys.path.insert(0, value)
        else:
            sys.path.append(value)


def add_hermes_to_syspath(root: Path) -> None:
    add_path_once(root)
    if os.name == "nt":
        venv_roots = [root / ".venv", root / "venv"]
        site_packages = next(
            (
                candidate / "Lib" / "site-packages"
                for candidate in venv_roots
                if (candidate / "Lib" / "site-packages").exists()
            ),
            root / ".venv" / "Lib" / "site-packages",
        )
    else:
        lib_roots = [root / ".venv" / "lib", root / "venv" / "lib"]
        candidates = []
        for lib_root in lib_roots:
            if lib_root.exists():
                candidates.extend(sorted(lib_root.glob("python*/site-packages")))
        site_packages = candidates[0] if candidates else root / "venv" / "lib" / "site-packages"
    if site_packages.exists():
        # Keep Hermes' bundled dependencies available for Hermes internals, but do
        # not let them shadow the MCP SDK used to run this sidecar.
        add_path_once(site_packages, prepend=False)


def import_hermes() -> None:
    global HERMES_ROOT, IMPORT_ERROR, file_tools, terminal_tool, memory_tool
    global skill_manager_tool, SessionDB, get_hermes_home
    try:
        HERMES_ROOT = find_hermes_root()
        add_hermes_to_syspath(HERMES_ROOT)
        from tools import file_tools as ft
        from tools import memory_tool as mt
        from tools import terminal_tool as tt

        file_tools = ft
        terminal_tool = tt
        memory_tool = mt

        try:
            from tools import skill_manager_tool as smt

            skill_manager_tool = smt
        except Exception as exc:
            eprint(f"hermes-gpt: skill manager unavailable: {exc}")

        try:
            from hermes_state import SessionDB as SDB
            from hermes_constants import get_hermes_home as ghh

            SessionDB = SDB
            get_hermes_home = ghh
        except Exception as exc:
            eprint(f"hermes-gpt: session search unavailable: {exc}")
    except Exception as exc:
        IMPORT_ERROR = str(exc)
        eprint(f"hermes-gpt: Hermes imports failed: {exc}")


def call_with_supported_kwargs(func: Any, **kwargs: Any) -> Any:
    params = inspect.signature(func).parameters
    supported = {key: value for key, value in kwargs.items() if key in params}
    return func(**supported)


def expand_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


def require_imports() -> None:
    if IMPORT_ERROR:
        raise RuntimeError(f"Hermes imports are unavailable: {IMPORT_ERROR}")
    missing = [
        name
        for name, module in {
            "file_tools": file_tools,
            "terminal_tool": terminal_tool,
            "memory_tool": memory_tool,
        }.items()
        if module is None
    ]
    if missing:
        raise RuntimeError(f"Hermes imports are unavailable: missing {', '.join(missing)}")


def skill_roots() -> list[Path]:
    roots: list[Path] = []
    hermes_home = None
    if callable(get_hermes_home):
        try:
            hermes_home = Path(get_hermes_home())
        except Exception:
            hermes_home = None
    if hermes_home is None:
        env_home = os.environ.get("HERMES_HOME")
        hermes_home = Path(env_home).expanduser() if env_home else Path.home() / ".hermes"

    roots.append(hermes_home / "skills")
    profiles = hermes_home / "profiles"
    if profiles.exists():
        roots.extend(path / "skills" for path in profiles.iterdir() if path.is_dir())
    if HERMES_ROOT:
        roots.append(HERMES_ROOT / "skills")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except Exception:
            continue
        key = str(resolved).lower()
        if resolved.exists() and key not in seen:
            unique.append(resolved)
            seen.add(key)
    return unique


def parse_skill_doc(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    name = path.parent.name
    description = ""
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
            for line in parts[1].splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip().strip("'\"")
                if key == "name" and value:
                    name = value
                elif key == "description" and value:
                    description = value
    if not description:
        for line in body.splitlines():
            clean = line.strip().lstrip("#").strip()
            if clean:
                description = clean[:180]
                break
    return {"name": name, "description": description, "path": str(path)}


def discover_skills() -> list[dict[str, str]]:
    skills: list[dict[str, str]] = []
    for root in skill_roots():
        for skill_md in root.rglob("SKILL.md"):
            try:
                skills.append(parse_skill_doc(skill_md))
            except Exception as exc:
                eprint(f"hermes-gpt: could not read skill {skill_md}: {exc}")
    return sorted(skills, key=lambda item: (item["name"].lower(), item["path"].lower()))


def clean_error(tool_name: str, exc: Exception) -> RuntimeError:
    eprint(f"hermes-gpt: {tool_name} failed: {exc}")
    return RuntimeError(f"{tool_name} failed: {exc}")


from mcp.server.fastmcp import FastMCP

import_hermes()


def tool_meta(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = dict(NOAUTH_META)
    if extra:
        meta.update(extra)
    return meta


def hermes_read_file(path: str, offset: int = 1, limit: int = 500) -> str:
    try:
        require_imports()
        return file_tools.read_file_tool(path=expand_path(path), offset=offset, limit=limit)
    except Exception as exc:
        raise clean_error("hermes_read_file", exc) from exc


def hermes_write_file(path: str, content: str) -> str:
    try:
        require_imports()
        return file_tools.write_file_tool(path=expand_path(path), content=content)
    except Exception as exc:
        raise clean_error("hermes_write_file", exc) from exc


def hermes_patch(
    path: str,
    old_string: str,
    new_string: str,
    mode: str = "replace",
    replace_all: bool = False,
) -> str:
    try:
        require_imports()
        return call_with_supported_kwargs(
            file_tools.patch_tool,
            mode=mode,
            path=expand_path(path),
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
    except Exception as exc:
        raise clean_error("hermes_patch", exc) from exc


def hermes_search_files(
    pattern: str,
    target: str = "content",
    path: str = ".",
    file_glob: str | None = None,
    limit: int = 50,
) -> str:
    try:
        require_imports()
        return call_with_supported_kwargs(
            file_tools.search_tool,
            pattern=pattern,
            target=target,
            path=expand_path(path),
            file_glob=file_glob,
            limit=limit,
        )
    except Exception as exc:
        raise clean_error("hermes_search_files", exc) from exc


def hermes_run_command(command: str, timeout: int = 30, workdir: str | None = None) -> str:
    try:
        require_imports()
        if not env_enabled(ENABLE_TERMINAL_ENV):
            raise RuntimeError(f"Terminal execution is disabled. Set {ENABLE_TERMINAL_ENV}=1 to enable it.")
        capped_timeout = max(1, min(int(timeout), 120))
        return call_with_supported_kwargs(
            terminal_tool.terminal_tool,
            command=command,
            timeout=capped_timeout,
            workdir=expand_path(workdir),
        )
    except Exception as exc:
        raise clean_error("hermes_run_command", exc) from exc


def hermes_memory(
    action: str,
    target: str = "memory",
    content: str | None = None,
    old_text: str | None = None,
) -> str:
    try:
        require_imports()
        if action not in {"add", "replace", "remove", "search"}:
            raise RuntimeError("Unsupported memory action. Use add, replace, remove, or search.")
        if action in {"add", "replace", "remove"} and not env_enabled(ENABLE_MEMORY_WRITE_ENV):
            raise RuntimeError(f"Memory write actions are disabled. Set {ENABLE_MEMORY_WRITE_ENV}=1 to enable them.")
        return memory_tool.memory_tool(action=action, target=target, content=content, old_text=old_text)
    except Exception as exc:
        raise clean_error("hermes_memory", exc) from exc


def hermes_skill_list() -> str:
    try:
        require_imports()
        skills = discover_skills()
        if not skills:
            return "No Hermes skills found."
        lines = []
        for skill in skills:
            desc = f" - {skill['description']}" if skill["description"] else ""
            lines.append(f"- {skill['name']}{desc}\n  {skill['path']}")
        return "\n".join(lines)
    except Exception as exc:
        raise clean_error("hermes_skill_list", exc) from exc


def hermes_skill_view(name: str) -> str:
    try:
        require_imports()
        query = name.strip().lower()
        matches = [
            skill for skill in discover_skills()
            if skill["name"].lower() == query or Path(skill["path"]).parent.name.lower() == query
        ]
        if not matches:
            return f"No skill matched {name!r}."
        if len(matches) > 1:
            return "Multiple skills matched:\n" + "\n".join(f"- {m['name']}: {m['path']}" for m in matches)
        return Path(matches[0]["path"]).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise clean_error("hermes_skill_view", exc) from exc


def hermes_session_search(query: str, limit: int = 20, offset: int = 0) -> str:
    try:
        require_imports()
        if SessionDB is None:
            return "Hermes session search is unavailable in this install: SessionDB import failed."
        db = SessionDB(read_only=True)
        if not hasattr(db, "search_messages"):
            return "Hermes session search is unavailable in this install: search_messages API is missing."
        rows = db.search_messages(query=query, limit=limit, offset=offset)
        if not rows:
            return "No matching Hermes session messages found."
        rendered = []
        for row in rows:
            session_id = row.get("session_id", "")
            role = row.get("role", "")
            content = (row.get("content") or "").replace("\r", " ").replace("\n", " ")
            rendered.append(f"- {session_id} [{role}] {content[:500]}")
        return "\n".join(rendered)
    except Exception as exc:
        message = f"Hermes session search is unavailable in this install: {exc}"
        eprint(f"hermes-gpt: {message}")
        return message


def build_server(
    *,
    host: str = "127.0.0.1",
    port: int = 7677,
    http: bool = False,
    include_local_settings: bool = False,
) -> FastMCP:
    server = FastMCP(
        "hermes-gpt",
        host=host,
        port=port,
        streamable_http_path="/mcp",
        sse_path="/sse",
        message_path="/messages/",
        stateless_http=http,
        json_response=http,
    )
    register_tools(server)
    return server


def register_tools(server: FastMCP) -> None:
    server.add_tool(hermes_read_file, meta=tool_meta())
    server.add_tool(hermes_search_files, meta=tool_meta())
    server.add_tool(hermes_memory, meta=tool_meta())
    server.add_tool(hermes_skill_list, meta=tool_meta())
    server.add_tool(hermes_skill_view, meta=tool_meta())

    if env_enabled(ENABLE_WRITE_ENV):
        server.add_tool(hermes_write_file, meta=tool_meta())
        server.add_tool(hermes_patch, meta=tool_meta())
    if env_enabled(ENABLE_TERMINAL_ENV):
        server.add_tool(hermes_run_command, meta=tool_meta())
    if env_enabled(ENABLE_SESSION_SEARCH_ENV):
        server.add_tool(hermes_session_search, meta=tool_meta())


mcp = build_server()


def status() -> dict[str, Any]:
    try:
        tools = [tool.name for tool in asyncio.run(build_server().list_tools())]
    except Exception as exc:
        tools = [f"tool discovery failed: {exc}"]
    return {
        "ok": IMPORT_ERROR is None,
        "hermes_root": str(HERMES_ROOT) if HERMES_ROOT else None,
        "import_error": IMPORT_ERROR,
        "tools": sorted(tools),
        "gates": {
            ENABLE_WRITE_ENV: env_enabled(ENABLE_WRITE_ENV),
            ENABLE_MEMORY_WRITE_ENV: env_enabled(ENABLE_MEMORY_WRITE_ENV),
            ENABLE_SESSION_SEARCH_ENV: env_enabled(ENABLE_SESSION_SEARCH_ENV),
            ENABLE_TERMINAL_ENV: env_enabled(ENABLE_TERMINAL_ENV),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes Agent MCP sidecar.")
    parser.add_argument("--http", action="store_true", help="Run streamable HTTP transport instead of stdio.")
    parser.add_argument("--sse", action="store_true", help="Run legacy SSE transport instead of stdio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7677)
    parser.add_argument("--cert", help="Path to SSL certificate file (enables HTTPS)")
    parser.add_argument("--key", help="Path to SSL key file (enables HTTPS)")
    parser.add_argument(
        "--profile",
        choices=[LOCAL_DEV_PROFILE, REMOTE_PROFILE],
        default=LOCAL_DEV_PROFILE,
        help="Release safety profile. Remote no-auth is refused unless explicitly acknowledged.",
    )
    parser.add_argument(
        UNSAFE_REMOTE_ACK,
        action="store_true",
        dest="unsafe_remote_ack",
        help="Allow remote profile without auth. For experiments only; not release-safe.",
    )
    args = parser.parse_args(argv)

    if args.http and args.sse:
        raise SystemExit("Choose only one of --http or --sse.")
    if args.profile == REMOTE_PROFILE and not (args.unsafe_remote_ack and env_enabled(UNSAFE_REMOTE_ENV)):
        raise SystemExit(
            "Remote profile requires real authentication, which is not implemented yet. "
            f"For temporary experiments only, pass {UNSAFE_REMOTE_ACK} and set {UNSAFE_REMOTE_ENV}=1."
        )
    if args.profile == LOCAL_DEV_PROFILE and not is_loopback_host(args.host):
        eprint(
            "WARNING: local-dev profile is bound to a non-loopback host. "
            "Do not expose hermes-gpt without real authentication."
        )
    if args.profile == REMOTE_PROFILE:
        eprint("WARNING: remote no-auth mode is explicitly unsafe and intended only for temporary experiments.")

    transport = "streamable-http" if args.http else "sse" if args.sse else "stdio"
    server = build_server(host=args.host, port=args.port, http=args.http)
    if transport == "stdio":
        eprint("hermes-gpt MCP server starting in stdio mode.")
        server.run(transport="stdio")
        return 0
    else:
        path = "/mcp" if args.http else "/sse"
        eprint(f"hermes-gpt MCP server running at http://{args.host}:{args.port}{path}")

        # Run with uvicorn instead of FastMCP.run() so TLS can be enabled for
        # local-only testing when cert/key are provided.
        import uvicorn
        app = server.streamable_http_app() if args.http else server.sse_app()

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_certfile=args.cert if args.cert else None,
            ssl_keyfile=args.key if args.key else None,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
