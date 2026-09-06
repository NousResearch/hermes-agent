"""Setup wizard for Mem0 plugin — interactive and flag-based modes."""

from __future__ import annotations

import getpass
import json
from contextlib import suppress
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home  # noqa: F401 — patched by tests

from . import _read_mem0_json
from ._oss_providers import EMBEDDER_PROVIDERS, KNOWN_DIMS, LLM_PROVIDERS, SECTION_REGISTRIES, VECTOR_PROVIDERS, validate_oss_config

_OLLAMA_URL = "http://localhost:11434"
_PGVECTOR_CONTAINER, _PGVECTOR_IMAGE, _PGVECTOR_PASSWORD = "hermes-pgvector", "pgvector/pgvector:pg17", "hermes"


def _curses_select(title: str, items: list[tuple[str, str]], default: int = 0) -> int:
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(title, [f"{label}  {desc}" if desc else label for label, desc in items], selected=default, cancel_returns=default)


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    """Prompt for a value with optional default and secret masking."""
    sys.stdout.write(f"  {label}{f' [{default}]' if default else ''}: ")
    sys.stdout.flush()
    val = getpass.getpass(prompt="") if secret and sys.stdin.isatty() else sys.stdin.readline().strip()
    return val or (default or "")


def _input(label: str, default: str) -> str:
    return input(f"  {label} [{default}]: ").strip() or default


def _masked(secret: str) -> str:
    return f"...{secret[-4:]}" if len(secret) > 4 else "set"


def _http_get(url: str, path: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(f"{url.rstrip('/')}{path}", method="GET"), timeout=timeout)


def _prompt_api_key(label: str, env_var: str, hermes_home: str) -> str:
    """Prompt for API key, showing masked existing value if found."""
    existing = os.environ.get(env_var, "")
    env_path = Path(hermes_home) / ".env"
    if not existing and env_path.exists():  # utf-8-sig: a Notepad BOM on line 1 would otherwise defeat the key match
        lines = env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        existing = next((line.split("=", 1)[1].strip() for line in lines if line.startswith(f"{env_var}=")), "")
    hint = f" (current: {_masked(existing)}, blank to keep)" if existing else ""
    return getpass.getpass(f"  {label} API key{hint}: ").strip()


def _api_key_writes(flags: dict, label: str, *, url: str | None = None, fresh_label: str | None = None) -> dict[str, str]:
    """MEM0_API_KEY for .env: from --api-key, else prompt (masking any key already in the environment)."""
    if flags.get("api_key"):
        return {"MEM0_API_KEY": flags["api_key"]}
    existing = os.environ.get("MEM0_API_KEY", "")
    if url and not existing:
        print(f"  Get yours at {url}")
    val = _prompt(f"{label} (current: {_masked(existing)}, blank to keep)" if existing else fresh_label or label, secret=True)
    return {"MEM0_API_KEY": val} if val else {}


def _print_dry_run(summary: str, env_writes: dict, check=None) -> None:
    print(f"\n  [dry-run] Would save config: {summary}")
    if env_writes:
        print("  [dry-run] Would write API key to .env")
    if check:
        check()
    print("  [dry-run] No files written.\n")


# --oss-vector-<key> flags accepted per vector store (also the pgvector key order).
_VECTOR_FLAG_KEYS = {"qdrant": ("path", "url"), "pgvector": ("host", "port", "user", "password", "dbname")}
_FLAG_KEYS = ("mode", "api_key", "host", *(f"oss_{s}{k}" for s in ("llm", "embedder") for k in ("", "_key", "_model", "_url")),
              "oss_vector", *(f"oss_vector_{k}" for ks in _VECTOR_FLAG_KEYS.values() for k in ks), "user_id")
_FLAG_DEFAULTS = {"oss_llm": "openai", "oss_embedder": "openai", "oss_vector": "qdrant"}


def parse_flags(argv: list[str] | None = None) -> dict[str, str]:
    args = argv if argv is not None else sys.argv[1:]
    flags: dict[str, Any] = {**{k: _FLAG_DEFAULTS.get(k, "") for k in _FLAG_KEYS}, "dry_run": False}
    flag_map = {"--" + k.replace("_", "-"): k for k in _FLAG_KEYS}
    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            flags["dry_run"] = True
        elif args[i] in flag_map and i + 1 < len(args):
            flags[flag_map[args[i]]] = args[i + 1]
            i += 1
        i += 1
    return flags


def _model_block(flags: dict, registry: dict, prefix: str) -> tuple[str, dict, dict[str, Any]]:
    """Resolve (provider_id, provider_def, config) for an LLM/embedder section from flags."""
    pid = flags.get(prefix, "openai")
    pdef = registry[pid]
    cfg: dict[str, Any] = {"model": flags.get(f"{prefix}_model") or pdef["default_model"]}
    url = flags.get(f"{prefix}_url") or pdef.get("default_url")
    if url and pdef.get("base_url_key"):
        cfg[pdef["base_url_key"]] = url
    return pid, pdef, cfg


def build_oss_config(flags: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """Build (oss_config for mem0.json, env_writes of secrets for .env) from parsed flags."""
    llm_id, llm_def, llm_config = _model_block(flags, LLM_PROVIDERS, "oss_llm")
    if llm_id == "openai" and llm_config["model"] == "gpt-5-mini":
        llm_config["is_reasoning_model"] = True
    embedder_id, embedder_def, embedder_config = _model_block(flags, EMBEDDER_PROVIDERS, "oss_embedder")
    dims = KNOWN_DIMS.get(embedder_config["model"])
    if dims:
        embedder_config["embedding_dims"] = dims
    vector_id = flags.get("oss_vector", "qdrant")
    vector_config = dict(VECTOR_PROVIDERS[vector_id]["default_config"])
    for key in _VECTOR_FLAG_KEYS.get(vector_id, ()):
        if val := flags.get(f"oss_vector_{key}"):
            vector_config[key] = int(val) if key == "port" else val
    if "url" in vector_config:
        vector_config.pop("path", None)  # a remote Qdrant URL replaces local storage
    oss_config = {"llm": {"provider": llm_id, "config": llm_config}, "embedder": {"provider": embedder_id, "config": embedder_config}, "vector_store": {"provider": vector_id, "config": vector_config}}
    # An embedder sharing the LLM's provider reuses the LLM key when no embedder key was given.
    llm_key = flags.get("oss_llm_key") if llm_def.get("needs_key") else ""
    emb_key = (flags.get("oss_embedder_key") or (flags.get("oss_llm_key") if embedder_id == llm_id else "")) if embedder_def.get("needs_key") else ""
    env_writes = {d["env_var"]: k for d, k in ((llm_def, llm_key), (embedder_def, emb_key)) if k}
    return oss_config, env_writes


def _write_env(env_path: Path, env_writes: dict[str, str]) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig like the canonical .env readers: a BOM'd first line would miss the key match and get duplicated.
    existing_lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
    keys = [line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None for line in existing_lines]
    new_lines = [f"{k}={env_writes[k]}" if k in env_writes else line for k, line in zip(keys, existing_lines)]
    new_lines += [f"{k}={v}" for k, v in env_writes.items() if k not in keys]
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _activate_provider(config: dict) -> None:
    """Point config.yaml's memory.provider at mem0."""
    from hermes_cli.config import save_config
    config["memory"]["provider"] = "mem0"
    save_config(config)


def _persist_provider_config(hermes_home: str, config: dict, provider_config: dict, env_writes: dict[str, str], label: str, key_line: str, server: str | None = None) -> None:
    """Shared platform/self-hosted tail: activate, write mem0.json (0600), then .env, then a saved summary."""
    _activate_provider(config)
    from plugins.memory.mem0 import Mem0MemoryProvider
    Mem0MemoryProvider().save_config(provider_config, hermes_home)
    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)
    if server:
        _check_selfhosted_server(server)
    print("\n".join(["", f"  Memory provider: {label}", *([f"  Server: {server}"] if server else []), "  Activation saved to config.yaml", "  Provider config saved",
                     *([f"  {key_line}"] if env_writes else []), "", "  Start a new session to activate.", ""]))


def _setup_platform(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """Platform mode setup — prompts for API key (secret -> .env), user/agent ids and rerank (-> mem0.json)."""
    provider_config = _read_mem0_json(Path(hermes_home) / "mem0.json")
    print("\n  Configuring mem0:\n")
    env_writes = _api_key_writes(flags, "Mem0 Platform API key", url="https://app.mem0.ai")
    for key, desc, default in (("user_id", "User identifier", "hermes-user"), ("agent_id", "Agent identifier", "hermes")):
        if val := _prompt(desc, default=str(provider_config.get(key) or default)):
            provider_config[key] = val
    choices = ["true", "false"]
    current = str(provider_config.get("rerank", "false") or "").lower()
    provider_config["rerank"] = choices[_curses_select("  Enable reranking for recall", [(c, "") for c in choices], default=choices.index(current) if current in choices else 0)]
    if flags.get("dry_run"):
        _print_dry_run(str(provider_config), env_writes)
        return
    # Routing checks ``host`` before platform, so clear a stale self-hosted host. "" rather than
    # pop(): save_config merges into the existing mem0.json, so a popped key would survive.
    provider_config.update(mode="platform", host="")
    # _load_config() also seeds ``host`` from MEM0_HOST (.env); the file clear can't help there, so warn.
    if os.environ.get("MEM0_HOST", "").strip():
        print(f"\n  ⚠ MEM0_HOST is set in your environment ({os.environ['MEM0_HOST']}). It overrides platform mode — remove it from ~/.hermes/.env (or unset it) or Hermes will keep routing to the self-hosted server.")
    _persist_provider_config(hermes_home, config, provider_config, env_writes, "mem0", "API keys saved to .env")


def _check_selfhosted_server(host: str) -> None:
    """Best-effort reachability check for a self-hosted Mem0 server (non-fatal)."""
    try:
        _http_get(host, "/docs", 5)
        print(f"  ✓ Mem0 server reachable at {host}")
    except urllib.error.HTTPError:
        # Any HTTP response (401/403/404) still means something is listening.
        print(f"  ✓ Mem0 server responding at {host}")
    except Exception:
        print(f"  ⚠ Could not reach {host} — check the URL and that the server is running.")


def _setup_selfhosted(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """Self-hosted mode — point at an existing Mem0 server: URL -> mem0.json, key -> .env (MEM0_API_KEY)."""
    provider_config = _read_mem0_json(Path(hermes_home) / "mem0.json")
    print("\n  Configuring mem0 (self-hosted server):\n")
    host = flags.get("host") or _prompt("Mem0 server URL (e.g. http://localhost:8888)", default=provider_config.get("host") or None)
    if not host:
        print("  Error: a server URL is required for self-hosted mode.", file=sys.stderr)
        return
    host = host.rstrip("/")
    env_writes = _api_key_writes(flags, "Server API key", fresh_label="Server API key (blank if AUTH_DISABLED)")
    user_id = flags.get("user_id") or _prompt("User identifier", default=provider_config.get("user_id") or "hermes-user")
    agent_id = _prompt("Agent identifier", default=provider_config.get("agent_id") or "hermes")
    if flags.get("dry_run"):
        _print_dry_run(f"host={host}, user_id={user_id}, agent_id={agent_id}", env_writes, lambda: _check_selfhosted_server(host))
        return
    provider_config.update(mode="platform", host=host, user_id=user_id, agent_id=agent_id)  # routing: oss > host > platform
    _persist_provider_config(hermes_home, config, provider_config, env_writes, "mem0 (self-hosted)", "API key saved to .env", server=host)


def _print_oss_summary(oss_config: dict, env_writes: dict, dry_run: bool = False) -> None:
    llm, emb = oss_config["llm"], oss_config["embedder"]
    w = 0 if dry_run else 9  # final summary column-aligns the labels
    lines = ["", "  [dry-run] OSS config would be:" if dry_run else "  ✓ Mem0 configured (OSS mode)",
             f"    {'LLM:':<{w}} {llm['provider']} ({llm['config'].get('model', '')})", f"    {'Embedder:':<{w}} {emb['provider']} ({emb['config'].get('model', '')})",
             f"    {'Vector:':<{w}} {oss_config['vector_store']['provider']}"]
    if dry_run:
        lines += [f"    Env vars: {', '.join(env_writes.keys())}"] if env_writes else []
    else:
        lines += [*(["    API keys saved to .env"] if env_writes else []), "    Config saved to mem0.json", "    Provider set in config.yaml", "", "  Start a new session to activate.", ""]
    print("\n".join(lines))


def _finish_oss(hermes_home: str, config: dict, oss_config: dict, env_writes: dict[str, str], user_id: str, agent_id: str, pgvector_config: dict | None = None) -> None:
    """Shared OSS tail: write secrets + mem0.json, install deps, activate, check, summarize."""
    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)
    config_path = Path(hermes_home) / "mem0.json"  # merge-write, plain text (platform path uses save_config's 0600 atomic write)
    config_path.write_text(json.dumps({**_read_mem0_json(config_path), "mode": "oss", "user_id": user_id, "agent_id": agent_id, "oss": oss_config}, indent=2) + "\n", encoding="utf-8")
    _install_provider_deps(oss_config["llm"]["provider"], oss_config["embedder"]["provider"], oss_config["vector_store"]["provider"])
    if pgvector_config:
        _ensure_pgvector_extension(pgvector_config)
    _activate_provider(config)
    _run_connectivity_checks(oss_config)
    _print_oss_summary(oss_config, env_writes)


def _setup_oss(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """OSS mode — non-interactive when --mode was given, otherwise curses pickers."""
    if not flags.get("_mode_from_flag"):
        _setup_oss_interactive(hermes_home, config)
        return
    oss_config, env_writes = build_oss_config(flags)
    if errors := validate_oss_config(oss_config):
        print("".join(f"  Error: {e}\n" for e in errors), end="", file=sys.stderr)
        sys.exit(1)
    if flags.get("dry_run"):
        _print_oss_summary(oss_config, env_writes, dry_run=True)
        _run_connectivity_checks(oss_config)
        print("  [dry-run] No files written.\n")
        return
    _finish_oss(hermes_home, config, oss_config, env_writes, flags.get("user_id") or os.getenv("USER", "hermes-user"), "hermes")


def _docker(*args: str, timeout: int, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL, **kwargs)


def _pg_ready(host: str, port: int, wait: int) -> bool:
    """Wait up to ``wait`` seconds for the port, then report whether PostgreSQL answers."""
    _wait_for_port(host, port, timeout=wait)
    return _check_pgvector(host, port)[0]


def _ensure_pgvector(host: str = "localhost", port: int = 5432) -> dict | None:
    """Ensure pgvector is reachable, offering Docker if not; returns the started container's vector_config, else None."""
    if _check_pgvector(host, port)[0]:
        print(f"  ✓ PostgreSQL reachable at {host}:{port}")
        return None
    print(f"  PostgreSQL not reachable at {host}:{port}")
    if not shutil.which("docker"):
        print("  Docker not found. Install Docker to auto-start pgvector,\n  or run PostgreSQL with pgvector manually.")
        return None
    with suppress(Exception):  # restart our own container if it exists but is stopped
        result = _docker("inspect", _PGVECTOR_CONTAINER, "--format", "{{.State.Status}}", timeout=10, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0 and "exited" in result.stdout:
            print(f"  Found stopped container '{_PGVECTOR_CONTAINER}', restarting...")
            _docker("start", _PGVECTOR_CONTAINER, timeout=15)
            if _pg_ready(host, port, 15):
                print("  ✓ PostgreSQL container restarted")
                return None
    if input("  Start pgvector via Docker? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        print("  Skipping Docker setup. Make sure PostgreSQL with pgvector is running.")
        return None
    try:
        print(f"  Pulling {_PGVECTOR_IMAGE}...")
        _docker("pull", _PGVECTOR_IMAGE, timeout=120)
        _docker("rm", "-f", _PGVECTOR_CONTAINER, timeout=10)  # remove existing container if present
        print(f"  Starting container '{_PGVECTOR_CONTAINER}' on port {port}...")
        _docker("run", "-d", "--name", _PGVECTOR_CONTAINER, "-e", f"POSTGRES_PASSWORD={_PGVECTOR_PASSWORD}", "-p", f"{port}:5432", _PGVECTOR_IMAGE, timeout=30, check=True)
        if _pg_ready(host, port, 20):
            print(f"  ✓ pgvector running on {host}:{port}")
        else:
            print("  Warning: Container started but PostgreSQL not yet accepting connections.\n  It may need a few more seconds. Config will be saved; retry later.")
        return {"host": host, "port": port, "user": "postgres", "password": _PGVECTOR_PASSWORD, "dbname": "postgres"}
    except subprocess.CalledProcessError as e:
        print(f"  Failed to start Docker container: {e}")
    except Exception as e:
        print(f"  Docker error: {e}")
    return None


def _ensure_ollama(models: list[str]) -> bool:
    """Ensure Ollama is running and ``models`` are pulled; False when the user must handle it manually."""
    ollama_bin = shutil.which("ollama")
    if not (ok := _check_ollama(_OLLAMA_URL)[0]):
        if not ollama_bin:
            print("  Ollama not found. Install it:\n    curl -fsSL https://ollama.com/install.sh | sh\n  Or on macOS: brew install ollama")
            return False
        print("  Ollama installed but not running. Starting...")
        try:
            subprocess.Popen([ollama_bin, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _wait_for_port("localhost", 11434, timeout=10)
            if ok := _check_ollama(_OLLAMA_URL)[0]:
                print("  ✓ Ollama started")
        except Exception as e:
            print(f"  Could not start Ollama: {e}")
    if not ok:
        print("  Warning: Ollama not reachable. Models cannot be pulled.")
        return False
    for model in models:
        try:
            names = [m.get("name", "") for m in json.loads(_http_get(_OLLAMA_URL, "/api/tags", 5).read()).get("models", [])]
        except Exception:
            names = []
        if any(model in n or model.split(":")[0] in n for n in names):
            print(f"  ✓ Model '{model}' available")
            continue
        print(f"  Pulling '{model}'... (this may take a few minutes)")
        try:
            subprocess.run([ollama_bin or "ollama", "pull", model], timeout=600, stdin=subprocess.DEVNULL)
            print(f"  ✓ Model '{model}' pulled")
        except Exception as e:
            print(f"  Warning: Could not pull '{model}': {e}\n  Run manually: ollama pull {model}")
    return True


def _ensure_pgvector_extension(pg_config: dict) -> None:
    try:
        import psycopg2
    except ImportError:
        return
    defaults = {"host": "localhost", "port": 5432, "user": "postgres", "dbname": "postgres"}
    try:
        conn = psycopg2.connect(**(defaults | {k: v for k, v in pg_config.items() if k in defaults or (k == "password" and v)}))
        conn.autocommit = True
        conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.close()
        print("  ✓ pgvector extension enabled")
    except Exception as e:
        print(f"  Warning: Could not enable pgvector extension: {e}")


def _wait_for_port(host: str, port: int, timeout: int = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection((host, port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.5)


# Picker descriptions: LLM/embedder show model (+ URL); vector stores by provider id (default: the id itself).
_VECTOR_DESCRIPTIONS = {"qdrant": lambda cfg: cfg.get("path", "local storage"), "pgvector": lambda cfg: f"{cfg.get('host', 'localhost')}:{cfg.get('port', 5432)}"}


def _configure_model_provider(kind: str, registry: dict, hermes_home: str, env_writes: dict[str, str], llm: tuple[str, dict] | None = None) -> tuple[str, dict, str, str | None]:
    """Pick an LLM/embedder provider, collect its key, and (for Ollama) model + URL -> (id, definition, model, url).
    For the embedder (``llm`` given), a provider shared with the LLM reuses the LLM key instead of prompting again."""
    items = [(v["label"], f"{v.get('default_model', '')} ({v['default_url']})" if v.get("default_url") else v.get("default_model", "")) for v in registry.values()]
    pid = list(registry)[_curses_select(f"{kind} Provider", items, 0)]
    pdef = registry[pid]
    model, url = pdef["default_model"], pdef.get("default_url")
    if pdef["needs_key"]:
        if llm is None or pid != llm[0]:
            if key := _prompt_api_key(pdef["label"] if llm is None else f"{pdef['label']} embedder", pdef["env_var"], hermes_home):
                env_writes[pdef["env_var"]] = key
        elif llm[1].get("env_var") in env_writes:
            env_writes[pdef["env_var"]] = env_writes[llm[1]["env_var"]]
    if pid == "ollama":
        model = _input(f"{kind} model", pdef["default_model"])
        url = _input("Ollama URL", pdef["default_url"])
    return pid, pdef, model, url


def _setup_oss_interactive(hermes_home: str, config: dict) -> None:
    env_writes: dict[str, str] = {}
    llm_id, llm_def, llm_model, llm_url = _configure_model_provider("LLM", LLM_PROVIDERS, hermes_home, env_writes)
    embedder_id, _, embedder_model, embedder_url = _configure_model_provider("Embedder", EMBEDDER_PROVIDERS, hermes_home, env_writes, llm=(llm_id, llm_def))
    vector_items = [(v["label"], _VECTOR_DESCRIPTIONS.get(pid, lambda cfg: pid)(v.get("default_config", {}))) for pid, v in VECTOR_PROVIDERS.items()]
    vector_id = list(VECTOR_PROVIDERS)[_curses_select("Vector Store", vector_items, 0)]
    # Auto-setup: ensure Ollama is running and models are pulled; ensure pgvector is reachable (offer Docker if not).
    ollama_models = [m for pid, m in ((llm_id, llm_model), (embedder_id, embedder_model)) if pid == "ollama"]
    if ollama_models:
        _ensure_ollama(ollama_models)
    pgvector_config = _ensure_pgvector() if vector_id == "pgvector" else None
    if vector_id == "pgvector" and not pgvector_config:  # native PostgreSQL: prompt for connection details (user first, historical order)
        pg = {k: _input(f"PostgreSQL {label}", d) for k, label, d in (("user", "user", os.getenv("USER", "postgres")), ("host", "host", "localhost"), ("port", "port", "5432"), ("dbname", "database", "postgres"))}
        pg_password = getpass.getpass("  PostgreSQL password (blank if none): ").strip()
        pgvector_config = {**pg, "port": int(pg["port"]), **({"password": pg_password} if pg_password else {})}
    user_id = _input("User ID", os.getenv("USER", "hermes-user"))
    agent_id = _input("Agent ID", "hermes")
    flags = {
        "oss_llm": llm_id, "oss_llm_model": llm_model, "oss_llm_url": llm_url or "",
        "oss_llm_key": env_writes.get(llm_def["env_var"], "") if llm_def.get("env_var") else "",
        "oss_embedder": embedder_id, "oss_embedder_model": embedder_model, "oss_embedder_url": embedder_url or "",
        "oss_vector": vector_id, "user_id": user_id,
    }
    flags.update({f"oss_vector_{key}": str(val) for key, val in (pgvector_config or {}).items() if val})
    oss_config, _ = build_oss_config(flags)
    _finish_oss(hermes_home, config, oss_config, env_writes, user_id, agent_id, pgvector_config)


def _install_provider_deps(llm_id: str, embedder_id: str, vector_id: str) -> None:
    deps = {registry[pid]["pip_dep"] for (_, registry), pid in zip(SECTION_REGISTRIES, (llm_id, embedder_id, vector_id)) if registry.get(pid, {}).get("pip_dep")}
    for dep in sorted(deps):
        print(f"  Installing {dep}...")
        try:
            # Environment-aware install: sealed hosted venvs redirect to the durable data-volume target instead of /opt/hermes.
            from tools.lazy_deps import install_specs
            outcome = install_specs([dep], timeout=60)
        except Exception:
            outcome = None
        print(f"  ✓ Installed {dep}" if outcome is not None and outcome.ok else f"  Warning: cannot install {dep}: {outcome.reason}" if outcome is not None and outcome.blocked
              else f"  Warning: Could not install {dep}. Install manually: uv pip install {dep}")
    if deps:
        import importlib
        importlib.invalidate_caches()


def _probe(fn, ok: str, fail: str, exc=Exception) -> tuple[bool, str]:
    """Run ``fn``; (True, ok) on success, (False, "fail: <error>") on ``exc``."""
    try:
        fn()
        return True, ok
    except exc as e:
        return False, f"{fail}: {e}"


def _check_qdrant_path(path: str) -> tuple[bool, str]:
    """Check that qdrant local storage parent dir is writable."""
    parent = Path(path).expanduser().parent
    return _probe(lambda: parent.mkdir(parents=True, exist_ok=True), f"Directory writable: {parent}", f"Cannot write to {parent}", OSError)


def _check_ollama(url: str) -> tuple[bool, str]:
    return _probe(lambda: _http_get(url, "/api/tags", 3), "Ollama reachable", f"Ollama not reachable at {url}")


def _check_pgvector(host: str, port: int) -> tuple[bool, str]:
    return _probe(lambda: socket.create_connection((host, port), timeout=3).close(), f"PGVector reachable at {host}:{port}", f"PGVector not reachable at {host}:{port}")


def _warn_unless(check: tuple[bool, str]) -> None:
    ok, msg = check
    if not ok:
        print(f"  Warning: {msg}")


def _run_connectivity_checks(oss_config: dict) -> None:
    vs = oss_config.get("vector_store", {})
    cfg = vs.get("config", {})
    if vs.get("provider") == "qdrant":
        path, url = cfg.get("path"), cfg.get("url")
        if path:
            _warn_unless(_check_qdrant_path(path))
        elif url:
            _warn_unless(_probe(lambda: _http_get(url, "/healthz", 3), "Qdrant reachable", f"Qdrant not reachable at {url}"))
    elif vs.get("provider") == "pgvector":
        _warn_unless(_check_pgvector(cfg.get("host", "localhost"), cfg.get("port", 5432)))
    llm = oss_config.get("llm", {})
    if llm.get("provider") == "ollama":
        _warn_unless(_check_ollama(llm.get("config", {}).get("ollama_base_url", _OLLAMA_URL)))


_MODE_HANDLERS = {"oss": _setup_oss, "selfhosted": _setup_selfhosted, "self-hosted": _setup_selfhosted, "platform": _setup_platform}
# Interactive picker order: Platform, Self-hosted server, Open Source.
_MODE_ITEMS = [("Platform", "Mem0 Cloud API (lightweight, just needs an API key)"), ("Self-hosted server", "Connect to an existing self-hosted Mem0 server (Docker/FastAPI)"), ("Open Source", "Run Mem0 locally (self-hosted LLM + vector store)")]
_MODE_PICKER = (_setup_platform, _setup_selfhosted, _setup_oss)


def post_setup(hermes_home: str, config: dict) -> None:
    """Entry point for `hermes memory setup`: routes on --mode (platform / selfhosted / oss), else shows a picker.
    OSS is non-interactive only when the mode came from the flag."""
    with suppress(ImportError):  # mem0ai must meet the minimum version from plugin.yaml
        import mem0
        installed_ver = getattr(mem0, "__version__", None)
        if installed_ver and tuple(int(x) for x in installed_ver.split(".")[:3]) < (2, 0, 7):
            print(f"\n  ⚠ mem0ai {installed_ver} installed but >=2.0.7 required.\n  Run: uv pip install --python {sys.executable} 'mem0ai>=2.0.7'")
    flags = parse_flags(sys.argv[1:])
    handler = _MODE_HANDLERS.get(flags["mode"])
    flags["_mode_from_flag"] = handler is not None
    if handler is None:
        handler = _MODE_PICKER[_curses_select("  Select mode", _MODE_ITEMS, 0)]
    handler(hermes_home, config, flags)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def has_oss_flags() -> bool:
    """Check if OSS-related flags are present in sys.argv."""
    flags = parse_flags(sys.argv[1:])
    if flags["mode"] == "oss":
        return True
    if any(flags.get(k) for k in ("oss_llm_key", "oss_vector_path", "oss_vector_url")):
        return True
    return False
# ---- END PLUGIN-COMPAT ----
