"""Setup wizard for Mem0 plugin — interactive and flag-based modes."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from ._oss_providers import (
    LLM_PROVIDERS,
    EMBEDDER_PROVIDERS,
    VECTOR_PROVIDERS,
    KNOWN_DIMS,
    validate_oss_config,
)


def _curses_select(title: str, items: list[tuple[str, str]], default: int = 0) -> int:
    """Interactive single-select with arrow keys."""
    from hermes_cli.curses_ui import curses_radiolist
    display_items = [
        f"{label}  {desc}" if desc else label
        for label, desc in items
    ]
    return curses_radiolist(title, display_items, selected=default, cancel_returns=default)


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    """Prompt for a value with optional default and secret masking."""
    suffix = f" [{default}]" if default else ""
    if secret:
        sys.stdout.write(f"  {label}{suffix}: ")
        sys.stdout.flush()
        if sys.stdin.isatty():
            val = getpass.getpass(prompt="")
        else:
            val = sys.stdin.readline().strip()
    else:
        sys.stdout.write(f"  {label}{suffix}: ")
        sys.stdout.flush()
        val = sys.stdin.readline().strip()
    return val or (default or "")


def has_oss_flags() -> bool:
    """Check if OSS-related flags are present in sys.argv."""
    flags = parse_flags(sys.argv[1:])
    if flags["mode"] == "oss":
        return True
    if any(flags.get(k) for k in ("oss_llm_key", "oss_vector_path", "oss_vector_url")):
        return True
    return False


def parse_flags(argv: list[str] | None = None) -> dict[str, str]:
    """Parse CLI flags from argv. Returns dict of flag values."""
    args = argv if argv is not None else sys.argv[1:]
    flags: dict[str, str] = {
        "mode": "",
        "api_key": "",
        "host": "",
        "oss_llm": "openai",
        "oss_llm_key": "",
        "oss_llm_model": "",
        "oss_llm_url": "",
        "oss_embedder": "openai",
        "oss_embedder_key": "",
        "oss_embedder_model": "",
        "oss_embedder_url": "",
        "oss_vector": "qdrant",
        "oss_vector_path": "",
        "oss_vector_url": "",
        "oss_vector_host": "",
        "oss_vector_port": "",
        "oss_vector_user": "",
        "oss_vector_password": "",
        "oss_vector_dbname": "",
        "user_id": "",
        "dry_run": False,
    }

    flag_map = {
        "--mode": "mode",
        "--api-key": "api_key",
        "--host": "host",
        "--oss-llm": "oss_llm",
        "--oss-llm-key": "oss_llm_key",
        "--oss-llm-model": "oss_llm_model",
        "--oss-llm-url": "oss_llm_url",
        "--oss-embedder": "oss_embedder",
        "--oss-embedder-key": "oss_embedder_key",
        "--oss-embedder-model": "oss_embedder_model",
        "--oss-embedder-url": "oss_embedder_url",
        "--oss-vector": "oss_vector",
        "--oss-vector-path": "oss_vector_path",
        "--oss-vector-url": "oss_vector_url",
        "--oss-vector-host": "oss_vector_host",
        "--oss-vector-port": "oss_vector_port",
        "--oss-vector-user": "oss_vector_user",
        "--oss-vector-password": "oss_vector_password",
        "--oss-vector-dbname": "oss_vector_dbname",
        "--user-id": "user_id",
    }

    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            flags["dry_run"] = True
            i += 1
        elif args[i] in flag_map and i + 1 < len(args):
            flags[flag_map[args[i]]] = args[i + 1]
            i += 2
        else:
            i += 1

    return flags


def build_oss_config(flags: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """Build OSS config dict + env_writes from parsed flags.

    Returns (oss_config, env_writes) where oss_config goes into mem0.json
    and env_writes maps env var names to secret values for .env.
    """
    llm_id = flags.get("oss_llm", "openai")
    llm_def = LLM_PROVIDERS[llm_id]
    llm_model = flags.get("oss_llm_model") or llm_def["default_model"]
    llm_config: dict[str, Any] = {"model": llm_model}
    if "default_url" in llm_def:
        llm_config["ollama_base_url"] = flags.get("oss_llm_url") or llm_def["default_url"]

    embedder_id = flags.get("oss_embedder", "openai")
    embedder_def = EMBEDDER_PROVIDERS[embedder_id]
    embedder_model = flags.get("oss_embedder_model") or embedder_def["default_model"]
    embedder_config: dict[str, Any] = {"model": embedder_model}
    if "default_url" in embedder_def:
        embedder_config["ollama_base_url"] = flags.get("oss_embedder_url") or embedder_def["default_url"]
    dims = KNOWN_DIMS.get(embedder_model)
    if dims:
        embedder_config["embedding_dims"] = dims

    vector_id = flags.get("oss_vector", "qdrant")
    vector_def = VECTOR_PROVIDERS[vector_id]
    vector_config = dict(vector_def["default_config"])
    if vector_id == "qdrant":
        if flags.get("oss_vector_path"):
            vector_config["path"] = flags["oss_vector_path"]
        if flags.get("oss_vector_url"):
            vector_config.pop("path", None)
            vector_config["url"] = flags["oss_vector_url"]
    elif vector_id == "pgvector":
        if flags.get("oss_vector_host"):
            vector_config["host"] = flags["oss_vector_host"]
        if flags.get("oss_vector_port"):
            vector_config["port"] = int(flags["oss_vector_port"])
        if flags.get("oss_vector_user"):
            vector_config["user"] = flags["oss_vector_user"]
        if flags.get("oss_vector_password"):
            vector_config["password"] = flags["oss_vector_password"]
        if flags.get("oss_vector_dbname"):
            vector_config["dbname"] = flags["oss_vector_dbname"]

    oss_config = {
        "llm": {"provider": llm_id, "config": llm_config},
        "embedder": {"provider": embedder_id, "config": embedder_config},
        "vector_store": {"provider": vector_id, "config": vector_config},
    }

    env_writes: dict[str, str] = {}
    if llm_def.get("needs_key") and flags.get("oss_llm_key"):
        env_writes[llm_def["env_var"]] = flags["oss_llm_key"]
    if embedder_def.get("needs_key") and flags.get("oss_embedder_key"):
        env_writes[embedder_def["env_var"]] = flags["oss_embedder_key"]
    elif embedder_def.get("needs_key") and embedder_id == llm_id and flags.get("oss_llm_key"):
        env_writes[embedder_def["env_var"]] = flags["oss_llm_key"]

    return oss_config, env_writes


def _write_env(env_path: Path, env_writes: dict[str, str]) -> None:
    """Append or update env vars in .env file."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text().splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        key_match = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
        if key_match and key_match in env_writes:
            new_lines.append(f"{key_match}={env_writes[key_match]}")
            updated_keys.add(key_match)
        else:
            new_lines.append(line)
    for k, v in env_writes.items():
        if k not in updated_keys:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n")


def _save_mem0_json(hermes_home: str, data: dict) -> None:
    """Merge-write to mem0.json."""
    config_path = Path(hermes_home) / "mem0.json"
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(data)
    config_path.write_text(json.dumps(existing, indent=2) + "\n")


def _setup_platform(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """Platform mode setup — uses the framework's schema-based flow.

    Delegates to the same code path the framework uses when post_setup
    doesn't exist, preserving the original platform onboarding experience.
    """
    schema = [
        {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
        {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
        {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
        {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
    ]

    existing_config = {}
    config_path = Path(hermes_home) / "mem0.json"
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text())
        except Exception:
            pass

    provider_config = dict(existing_config)
    env_writes: dict[str, str] = {}

    print("\n  Configuring mem0:\n")

    for field in schema:
        key = field["key"]
        desc = field.get("description", key)
        default = field.get("default")
        is_secret = field.get("secret", False)
        choices = field.get("choices")
        env_var = field.get("env_var")
        url = field.get("url")

        if flags.get("api_key") and key == "api_key":
            env_writes["MEM0_API_KEY"] = flags["api_key"]
            continue

        if choices and not is_secret:
            choice_items = [(c, "") for c in choices]
            current = provider_config.get(key, default)
            current_idx = 0
            if current and str(current).lower() in choices:
                current_idx = choices.index(str(current).lower())
            sel = _curses_select(f"  {desc}", choice_items, default=current_idx)
            provider_config[key] = choices[sel]
        elif is_secret:
            existing = os.environ.get(env_var, "") if env_var else ""
            if existing:
                masked = f"...{existing[-4:]}" if len(existing) > 4 else "set"
                val = _prompt(f"{desc} (current: {masked}, blank to keep)", secret=True)
            else:
                if url:
                    print(f"  Get yours at {url}")
                val = _prompt(desc, secret=True)
            if val and env_var:
                env_writes[env_var] = val
        else:
            current = provider_config.get(key)
            effective_default = current or default
            val = _prompt(desc, default=str(effective_default) if effective_default else None)
            if val:
                provider_config[key] = val

    if flags.get("dry_run"):
        print(f"\n  [dry-run] Would save config: {provider_config}")
        if env_writes:
            print("  [dry-run] Would write API key to .env")
        print("  [dry-run] No files written.\n")
        return

    provider_config["mode"] = "platform"
    # Clear any stale self-hosted host: routing checks ``host`` before platform
    # (see _create_backend), so leaving it would silently keep routing to the
    # self-hosted server even though the user just chose platform mode. Set it
    # to "" rather than pop() — save_config merges into the existing mem0.json
    # (existing.update), so a popped key would survive; an empty value overwrites
    # it and reads as falsy at routing time.
    provider_config["host"] = ""
    # The json-file clear above can't help when the host comes from the
    # environment: _load_config() seeds ``host`` from MEM0_HOST, and the
    # docs tell self-hosted users to put MEM0_HOST in ~/.hermes/.env. Warn
    # so the user knows platform mode won't take effect until it's removed.
    if os.environ.get("MEM0_HOST", "").strip():
        print(
            "\n  ⚠ MEM0_HOST is set in your environment "
            f"({os.environ['MEM0_HOST']}). It overrides platform mode — "
            "remove it from ~/.hermes/.env (or unset it) or Hermes will keep "
            "routing to the self-hosted server."
        )

    from hermes_cli.config import save_config
    config["memory"]["provider"] = "mem0"
    save_config(config)

    from plugins.memory.mem0 import Mem0MemoryProvider
    provider = Mem0MemoryProvider()
    provider.save_config(provider_config, hermes_home)

    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)

    print("\n  Memory provider: mem0")
    print("  Activation saved to config.yaml")
    print("  Provider config saved")
    if env_writes:
        print("  API keys saved to .env")
    print("\n  Start a new session to activate.\n")


def _check_selfhosted_server(host: str) -> None:
    """Best-effort reachability check for a self-hosted Mem0 server (non-fatal)."""
    import urllib.error
    import urllib.request as _urlreq

    try:
        req = _urlreq.Request(f"{host.rstrip('/')}/docs", method="GET")
        _urlreq.urlopen(req, timeout=5)
        print(f"  ✓ Mem0 server reachable at {host}")
    except urllib.error.HTTPError:
        # Any HTTP response (401/403/404) still means something is listening.
        print(f"  ✓ Mem0 server responding at {host}")
    except Exception:
        print(f"  ⚠ Could not reach {host} — check the URL and that the server is running.")


def _setup_selfhosted(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """Self-hosted mode setup — point at an existing Mem0 dashboard server.

    For users already running the Dockerized Mem0 FastAPI server: stores the
    server URL (behavioral -> mem0.json) and an optional API key
    (secret -> .env as MEM0_API_KEY).
    """
    existing_config = {}
    config_path = Path(hermes_home) / "mem0.json"
    if config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text())
        except Exception:
            pass

    provider_config = dict(existing_config)

    print("\n  Configuring mem0 (self-hosted server):\n")

    host = flags.get("host") or _prompt(
        "Mem0 server URL (e.g. http://localhost:8888)",
        default=provider_config.get("host") or None,
    )
    if not host:
        print("  Error: a server URL is required for self-hosted mode.", file=sys.stderr)
        return
    host = host.rstrip("/")

    env_writes: dict[str, str] = {}
    if flags.get("api_key"):
        env_writes["MEM0_API_KEY"] = flags["api_key"]
    else:
        existing_key = os.environ.get("MEM0_API_KEY", "")
        if existing_key:
            masked = f"...{existing_key[-4:]}" if len(existing_key) > 4 else "set"
            val = _prompt(f"Server API key (current: {masked}, blank to keep)", secret=True)
        else:
            val = _prompt("Server API key (blank if AUTH_DISABLED)", secret=True)
        if val:
            env_writes["MEM0_API_KEY"] = val

    user_id = flags.get("user_id") or _prompt(
        "User identifier", default=provider_config.get("user_id") or "hermes-user"
    )
    agent_id = _prompt("Agent identifier", default=provider_config.get("agent_id") or "hermes")

    if flags.get("dry_run"):
        print(f"\n  [dry-run] Would save config: host={host}, user_id={user_id}, agent_id={agent_id}")
        if env_writes:
            print("  [dry-run] Would write API key to .env")
        _check_selfhosted_server(host)
        print("  [dry-run] No files written.\n")
        return

    provider_config["mode"] = "platform"  # routing: oss > host > platform; host wins
    provider_config["host"] = host
    provider_config["user_id"] = user_id
    provider_config["agent_id"] = agent_id

    from hermes_cli.config import save_config
    config["memory"]["provider"] = "mem0"
    save_config(config)

    from plugins.memory.mem0 import Mem0MemoryProvider
    provider = Mem0MemoryProvider()
    provider.save_config(provider_config, hermes_home)

    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)

    _check_selfhosted_server(host)
    print("\n  Memory provider: mem0 (self-hosted)")
    print(f"  Server: {host}")
    print("  Activation saved to config.yaml")
    print("  Provider config saved")
    if env_writes:
        print("  API key saved to .env")
    print("\n  Start a new session to activate.\n")


def _setup_oss(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """OSS mode setup — build config from flags or interactive prompts.

    Non-interactive when --mode was set explicitly via flags (post_setup already
    resolved mode). Interactive only when mode was chosen via curses picker.
    """
    if not flags.get("_mode_from_flag"):
        _setup_oss_interactive(hermes_home, config)
        return

    oss_config, env_writes = build_oss_config(flags)
    errors = validate_oss_config(oss_config)
    if errors:
        for e in errors:
            print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)

    user_id = flags.get("user_id") or os.getenv("USER", "hermes-user")

    llm_id = oss_config["llm"]["provider"]
    embedder_id = oss_config["embedder"]["provider"]
    vector_id = oss_config["vector_store"]["provider"]

    if flags.get("dry_run"):
        print("\n  [dry-run] OSS config would be:")
        print(f"    LLM: {oss_config['llm']['provider']} ({oss_config['llm']['config'].get('model', '')})")
        print(f"    Embedder: {oss_config['embedder']['provider']} ({oss_config['embedder']['config'].get('model', '')})")
        print(f"    Vector: {vector_id}")
        if env_writes:
            print(f"    Env vars: {', '.join(env_writes.keys())}")
        _run_connectivity_checks(oss_config)
        print("  [dry-run] No files written.\n")
        return

    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)
    _save_mem0_json(hermes_home, {"mode": "oss", "user_id": user_id, "agent_id": "hermes", "oss": oss_config})

    _install_provider_deps(llm_id, embedder_id, vector_id)

    from hermes_cli.config import save_config
    config["memory"]["provider"] = "mem0"
    save_config(config)

    _run_connectivity_checks(oss_config)
    print("\n  ✓ Mem0 configured (OSS mode)")
    print(f"    LLM:      {oss_config['llm']['provider']} ({oss_config['llm']['config'].get('model', '')})")
    print(f"    Embedder: {oss_config['embedder']['provider']} ({oss_config['embedder']['config'].get('model', '')})")
    print(f"    Vector:   {vector_id}")
    if env_writes:
        print("    API keys saved to .env")
    print("    Config saved to mem0.json")
    print("    Provider set in config.yaml")
    print("\n  Start a new session to activate.\n")


def _prompt_api_key(label: str, env_var: str, hermes_home: str) -> str:
    """Prompt for API key, showing masked existing value if found."""
    existing = os.environ.get(env_var, "")
    if not existing:
        env_path = Path(hermes_home) / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(f"{env_var}="):
                    existing = line.split("=", 1)[1].strip()
                    break
    if existing:
        masked = f"...{existing[-4:]}" if len(existing) > 4 else "set"
        return getpass.getpass(f"  {label} API key (current: {masked}, blank to keep): ").strip()
    return getpass.getpass(f"  {label} API key: ").strip()


_PGVECTOR_CONTAINER = "hermes-pgvector"
_PGVECTOR_IMAGE = "pgvector/pgvector:pg17"
_PGVECTOR_PASSWORD = "hermes"


def _ensure_pgvector(host: str = "localhost", port: int = 5432) -> dict | None:
    """Ensure pgvector is reachable; offer Docker setup if not.

    Returns updated vector_config dict if Docker was started, None otherwise.
    """
    ok, _ = _check_pgvector(host, port)
    if ok:
        print(f"  ✓ PostgreSQL reachable at {host}:{port}")
        return None

    print(f"  PostgreSQL not reachable at {host}:{port}")

    # Check if our container already exists but is stopped
    if shutil.which("docker"):
        try:
            result = subprocess.run(
                ["docker", "inspect", _PGVECTOR_CONTAINER, "--format", "{{.State.Status}}"],
                capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0 and "exited" in result.stdout:
                print(f"  Found stopped container '{_PGVECTOR_CONTAINER}', restarting...")
                subprocess.run(["docker", "start", _PGVECTOR_CONTAINER],
                               capture_output=True, timeout=15,
                               stdin=subprocess.DEVNULL)
                _wait_for_port(host, port, timeout=15)
                ok, _ = _check_pgvector(host, port)
                if ok:
                    print("  ✓ PostgreSQL container restarted")
                    return None
        except Exception:
            pass

        answer = input("  Start pgvector via Docker? [Y/n]: ").strip().lower()
        if answer in ("", "y", "yes"):
            return _start_pgvector_docker(host, port)
        else:
            print("  Skipping Docker setup. Make sure PostgreSQL with pgvector is running.")
            return None
    else:
        print("  Docker not found. Install Docker to auto-start pgvector,")
        print("  or run PostgreSQL with pgvector manually.")
        return None


def _start_pgvector_docker(host: str, port: int) -> dict | None:
    """Pull and start pgvector Docker container."""
    try:
        print(f"  Pulling {_PGVECTOR_IMAGE}...")
        subprocess.run(["docker", "pull", _PGVECTOR_IMAGE],
                       capture_output=True, timeout=120,
                       stdin=subprocess.DEVNULL)

        # Remove existing container if present
        subprocess.run(["docker", "rm", "-f", _PGVECTOR_CONTAINER],
                       capture_output=True, timeout=10,
                       stdin=subprocess.DEVNULL)

        print(f"  Starting container '{_PGVECTOR_CONTAINER}' on port {port}...")
        subprocess.run([
            "docker", "run", "-d",
            "--name", _PGVECTOR_CONTAINER,
            "-e", f"POSTGRES_PASSWORD={_PGVECTOR_PASSWORD}",
            "-p", f"{port}:5432",
            _PGVECTOR_IMAGE,
        ], capture_output=True, timeout=30, check=True, stdin=subprocess.DEVNULL)

        _wait_for_port(host, port, timeout=20)
        ok, _ = _check_pgvector(host, port)
        if ok:
            print(f"  ✓ pgvector running on {host}:{port}")
            return {
                "host": host, "port": port,
                "user": "postgres", "password": _PGVECTOR_PASSWORD,
                "dbname": "postgres",
            }
        else:
            print("  Warning: Container started but PostgreSQL not yet accepting connections.")
            print("  It may need a few more seconds. Config will be saved; retry later.")
            return {
                "host": host, "port": port,
                "user": "postgres", "password": _PGVECTOR_PASSWORD,
                "dbname": "postgres",
            }
    except subprocess.CalledProcessError as e:
        print(f"  Failed to start Docker container: {e}")
        return None
    except Exception as e:
        print(f"  Docker error: {e}")
        return None


def _ensure_ollama(models: list[str]) -> bool:
    """Ensure Ollama is running and required models are pulled.

    Returns True if Ollama is ready, False if user needs to handle it manually.
    """
    url = "http://localhost:11434"
    ollama_bin = shutil.which("ollama")
    ok, _ = _check_ollama(url)

    if not ok:
        if ollama_bin:
            print("  Ollama installed but not running. Starting...")
            try:
                subprocess.Popen(
                    [ollama_bin, "serve"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                _wait_for_port("localhost", 11434, timeout=10)
                ok, _ = _check_ollama(url)
                if ok:
                    print("  ✓ Ollama started")
            except Exception as e:
                print(f"  Could not start Ollama: {e}")
        else:
            print("  Ollama not found. Install it:")
            print("    curl -fsSL https://ollama.com/install.sh | sh")
            print("  Or on macOS: brew install ollama")
            return False

    if not ok:
        print("  Warning: Ollama not reachable. Models cannot be pulled.")
        return False

    # Pull required models
    for model in models:
        if _ollama_has_model(url, model):
            print(f"  ✓ Model '{model}' available")
        else:
            print(f"  Pulling '{model}'... (this may take a few minutes)")
            try:
                subprocess.run([ollama_bin or "ollama", "pull", model], timeout=600,
                               stdin=subprocess.DEVNULL)
                print(f"  ✓ Model '{model}' pulled")
            except Exception as e:
                print(f"  Warning: Could not pull '{model}': {e}")
                print(f"  Run manually: ollama pull {model}")

    return True


def _ollama_has_model(url: str, model: str) -> bool:
    """Check if Ollama already has a model pulled."""
    try:
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        names = [m.get("name", "") for m in data.get("models", [])]
        base_model = model.split(":")[0]
        return any(model in n or base_model in n for n in names)
    except Exception:
        return False


def _ensure_pgvector_extension(pg_config: dict) -> None:
    """Create the pgvector extension if it doesn't exist."""
    try:
        import psycopg2
    except ImportError:
        return
    conn_params = {
        "host": pg_config.get("host", "localhost"),
        "port": pg_config.get("port", 5432),
        "user": pg_config.get("user", "postgres"),
        "dbname": pg_config.get("dbname", "postgres"),
    }
    if pg_config.get("password"):
        conn_params["password"] = pg_config["password"]
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.close()
        conn.close()
        print("  ✓ pgvector extension enabled")
    except Exception as e:
        print(f"  Warning: Could not enable pgvector extension: {e}")


def _wait_for_port(host: str, port: int, timeout: int = 15) -> None:
    """Wait until a TCP port is accepting connections."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=1)
            sock.close()
            return
        except OSError:
            time.sleep(0.5)


def _provider_description(v: dict) -> str:
    """Description for LLM/embedder picker: model + URL if applicable."""
    model = v.get("default_model", "")
    url = v.get("default_url")
    if url:
        return f"{model} ({url})"
    return model


def _vector_description(pid: str, v: dict) -> str:
    cfg = v.get("default_config", {})
    if pid == "qdrant":
        return cfg.get("path", "local storage")
    if pid == "pgvector":
        return f"{cfg.get('host', 'localhost')}:{cfg.get('port', 5432)}"
    return pid


def _setup_oss_interactive(hermes_home: str, config: dict) -> None:
    """Interactive OSS setup using curses pickers."""
    llm_items = [(v["label"], _provider_description(v)) for pid, v in LLM_PROVIDERS.items()]
    llm_idx = _curses_select("LLM Provider", llm_items, 0)
    llm_id = list(LLM_PROVIDERS.keys())[llm_idx]
    llm_def = LLM_PROVIDERS[llm_id]

    env_writes: dict[str, str] = {}
    llm_model = llm_def["default_model"]
    llm_url = llm_def.get("default_url")
    if llm_def["needs_key"]:
        key = _prompt_api_key(llm_def["label"], llm_def["env_var"], hermes_home)
        if key:
            env_writes[llm_def["env_var"]] = key
    if llm_id == "ollama":
        llm_model = input(f"  LLM model [{llm_def['default_model']}]: ").strip() or llm_def["default_model"]
        llm_url = input(f"  Ollama URL [{llm_def['default_url']}]: ").strip() or llm_def["default_url"]

    embedder_items = [(v["label"], _provider_description(v)) for pid, v in EMBEDDER_PROVIDERS.items()]
    embedder_idx = _curses_select("Embedder Provider", embedder_items, 0)
    embedder_id = list(EMBEDDER_PROVIDERS.keys())[embedder_idx]
    embedder_def = EMBEDDER_PROVIDERS[embedder_id]

    embedder_model = embedder_def["default_model"]
    embedder_url = embedder_def.get("default_url")
    if embedder_def["needs_key"] and embedder_id != llm_id:
        key = _prompt_api_key(f"{embedder_def['label']} embedder", embedder_def["env_var"], hermes_home)
        if key:
            env_writes[embedder_def["env_var"]] = key
    elif embedder_def["needs_key"] and embedder_id == llm_id:
        if llm_def.get("env_var") in env_writes:
            env_writes[embedder_def["env_var"]] = env_writes[llm_def["env_var"]]
    if embedder_id == "ollama":
        embedder_model = input(f"  Embedder model [{embedder_def['default_model']}]: ").strip() or embedder_def["default_model"]
        embedder_url = input(f"  Ollama URL [{embedder_def['default_url']}]: ").strip() or embedder_def["default_url"]

    vector_items = [(v["label"], _vector_description(pid, v)) for pid, v in VECTOR_PROVIDERS.items()]
    vector_idx = _curses_select("Vector Store", vector_items, 0)
    vector_id = list(VECTOR_PROVIDERS.keys())[vector_idx]

    # Auto-setup: ensure Ollama is running and models are pulled
    ollama_models = []
    if llm_id == "ollama":
        ollama_models.append(llm_model)
    if embedder_id == "ollama":
        ollama_models.append(embedder_model)
    if ollama_models:
        _ensure_ollama(ollama_models)

    # Auto-setup: ensure pgvector is reachable (offer Docker if not)
    pgvector_config = None
    if vector_id == "pgvector":
        pgvector_config = _ensure_pgvector()
        if not pgvector_config:
            # Native PostgreSQL — prompt for connection details
            default_user = os.getenv("USER", "postgres")
            pg_user = input(f"  PostgreSQL user [{default_user}]: ").strip() or default_user
            pg_host = input("  PostgreSQL host [localhost]: ").strip() or "localhost"
            pg_port = input("  PostgreSQL port [5432]: ").strip() or "5432"
            pg_dbname = input("  PostgreSQL database [postgres]: ").strip() or "postgres"
            pg_password = getpass.getpass("  PostgreSQL password (blank if none): ").strip()
            pgvector_config = {
                "host": pg_host, "port": int(pg_port),
                "user": pg_user, "dbname": pg_dbname,
            }
            if pg_password:
                pgvector_config["password"] = pg_password

    user_id = input(f"  User ID [{os.getenv('USER', 'hermes-user')}]: ").strip()
    user_id = user_id or os.getenv("USER", "hermes-user")

    agent_id = input("  Agent ID [hermes]: ").strip()
    agent_id = agent_id or "hermes"

    flags = {
        "oss_llm": llm_id,
        "oss_llm_key": env_writes.get(llm_def["env_var"], "") if llm_def.get("env_var") else "",
        "oss_llm_model": llm_model,
        "oss_llm_url": llm_url or "",
        "oss_embedder": embedder_id,
        "oss_embedder_model": embedder_model,
        "oss_embedder_url": embedder_url or "",
        "oss_vector": vector_id,
        "user_id": user_id,
    }

    if pgvector_config:
        flags["oss_vector_host"] = pgvector_config["host"]
        flags["oss_vector_port"] = str(pgvector_config["port"])
        flags["oss_vector_user"] = pgvector_config["user"]
        if pgvector_config.get("password"):
            flags["oss_vector_password"] = pgvector_config["password"]
        flags["oss_vector_dbname"] = pgvector_config["dbname"]

    oss_config, _ = build_oss_config(flags)

    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)
    _save_mem0_json(hermes_home, {"mode": "oss", "user_id": user_id, "agent_id": agent_id, "oss": oss_config})

    _install_provider_deps(llm_id, embedder_id, vector_id)

    if vector_id == "pgvector" and pgvector_config:
        _ensure_pgvector_extension(pgvector_config)

    from hermes_cli.config import save_config
    config["memory"]["provider"] = "mem0"
    save_config(config)

    _run_connectivity_checks(oss_config)
    print("\n  ✓ Mem0 configured (OSS mode)")
    print(f"    LLM:      {oss_config['llm']['provider']} ({oss_config['llm']['config'].get('model', '')})")
    print(f"    Embedder: {oss_config['embedder']['provider']} ({oss_config['embedder']['config'].get('model', '')})")
    print(f"    Vector:   {vector_id}")
    if env_writes:
        print("    API keys saved to .env")
    print("    Config saved to mem0.json")
    print("    Provider set in config.yaml")
    print("\n  Start a new session to activate.\n")


def _install_provider_deps(llm_id: str, embedder_id: str, vector_id: str) -> None:
    """Install all optional pip deps for selected providers."""
    deps: set[str] = set()
    for registry, pid in [(LLM_PROVIDERS, llm_id), (EMBEDDER_PROVIDERS, embedder_id),
                          (VECTOR_PROVIDERS, vector_id)]:
        dep = registry.get(pid, {}).get("pip_dep")
        if dep:
            deps.add(dep)
    for dep in sorted(deps):
        try:
            print(f"  Installing {dep}...")
            subprocess.run(
                ["uv", "pip", "install", "--python", sys.executable, dep],
                capture_output=True, timeout=60,
            )
            print(f"  ✓ Installed {dep}")
        except Exception:
            print(f"  Warning: Could not install {dep}. Install manually: uv pip install {dep}")
    if deps:
        import importlib
        importlib.invalidate_caches()


def _check_qdrant_path(path: str) -> tuple[bool, str]:
    """Check that qdrant local storage parent dir is writable."""
    p = Path(path).expanduser()
    parent = p.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return True, f"Directory writable: {parent}"
    except OSError as e:
        return False, f"Cannot write to {parent}: {e}"


def _check_ollama(url: str) -> tuple[bool, str]:
    """Check Ollama is reachable via /api/tags."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/api/tags", method="GET")
        urllib.request.urlopen(req, timeout=3)
        return True, "Ollama reachable"
    except Exception as e:
        return False, f"Ollama not reachable at {url}: {e}"


def _check_pgvector(host: str, port: int) -> tuple[bool, str]:
    """Check PGVector via TCP socket."""
    try:
        sock = socket.create_connection((host, port), timeout=3)
        sock.close()
        return True, f"PGVector reachable at {host}:{port}"
    except Exception as e:
        return False, f"PGVector not reachable at {host}:{port}: {e}"


def _run_connectivity_checks(oss_config: dict) -> None:
    """Run connectivity checks and print warnings."""
    vs = oss_config.get("vector_store", {})
    if vs.get("provider") == "qdrant":
        path = vs.get("config", {}).get("path")
        url = vs.get("config", {}).get("url")
        if path:
            ok, msg = _check_qdrant_path(path)
            if not ok:
                print(f"  Warning: {msg}")
        elif url:
            try:
                req = urllib.request.Request(f"{url.rstrip('/')}/healthz", method="GET")
                urllib.request.urlopen(req, timeout=3)
            except Exception as e:
                print(f"  Warning: Qdrant not reachable at {url}: {e}")
    elif vs.get("provider") == "pgvector":
        cfg = vs.get("config", {})
        ok, msg = _check_pgvector(cfg.get("host", "localhost"), cfg.get("port", 5432))
        if not ok:
            print(f"  Warning: {msg}")

    llm = oss_config.get("llm", {})
    if llm.get("provider") == "ollama":
        url = llm.get("config", {}).get("ollama_base_url", "http://localhost:11434")
        ok, msg = _check_ollama(url)
        if not ok:
            print(f"  Warning: {msg}")


def _check_min_dep_version() -> None:
    """Ensure mem0ai meets the minimum version from plugin.yaml."""
    try:
        import mem0
        installed_ver = getattr(mem0, "__version__", None)
        if not installed_ver:
            return
        installed_parts = tuple(int(x) for x in installed_ver.split(".")[:3])
        required_parts = (2, 0, 7)
        if installed_parts < required_parts:
            req_str = ".".join(str(x) for x in required_parts)
            print(f"\n  ⚠ mem0ai {installed_ver} installed but >={req_str} required.")
            print(f"  Run: uv pip install --python {sys.executable} 'mem0ai>={req_str}'")
    except ImportError:
        pass
    except Exception:
        pass


def post_setup(hermes_home: str, config: dict) -> None:
    """Entry point called by hermes memory setup framework.

    Routes on --mode (platform / selfhosted / oss); with no flag it shows an
    interactive picker with all three modes. Platform keeps the framework's
    original schema-based onboarding; selfhosted points at an existing Mem0
    server; oss builds a local SDK config.
    """
    _check_min_dep_version()
    flags = parse_flags(sys.argv[1:])

    if flags["mode"] == "oss":
        flags["_mode_from_flag"] = True
        _setup_oss(hermes_home, config, flags)
        return

    if flags["mode"] in ("selfhosted", "self-hosted"):
        _setup_selfhosted(hermes_home, config, flags)
        return

    if flags["mode"] == "platform":
        _setup_platform(hermes_home, config, flags)
        return

    # No --mode flag: show interactive picker
    mode_items = [
        ("Platform", "Mem0 Cloud API (lightweight, just needs an API key)"),
        ("Self-hosted server", "Connect to an existing self-hosted Mem0 server (Docker/FastAPI)"),
        ("Open Source", "Run Mem0 locally (self-hosted LLM + vector store)"),
    ]
    mode_idx = _curses_select("  Select mode", mode_items, 0)
    if mode_idx == 1:
        _setup_selfhosted(hermes_home, config, flags)
    elif mode_idx == 2:
        flags["_mode_from_flag"] = False
        _setup_oss(hermes_home, config, flags)
    else:
        _setup_platform(hermes_home, config, flags)
