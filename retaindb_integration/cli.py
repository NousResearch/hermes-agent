"""CLI commands for Hermes' native RetainDB integration."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

from retaindb_integration.client import DEFAULT_BASE_URL, RetainDBClient, RetainDBClientConfig
from retaindb_integration.session import RetainDBSessionManager


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    if secret and sys.stdin.isatty():
        import getpass

        value = getpass.getpass(prompt="")
    else:
        value = sys.stdin.readline().strip()
    return value or (default or "")


def _load_hermes_config() -> dict:
    from hermes_cli.config import load_config

    return load_config() or {}


def _save_hermes_config(cfg: dict) -> None:
    from hermes_cli.config import save_config

    save_config(cfg)


def _config_path() -> Path:
    from hermes_cli.config import get_config_path

    return get_config_path()


def _env_path() -> Path:
    from hermes_cli.config import get_env_path

    return get_env_path()


def _write_runtime_config(
    cfg: dict,
    *,
    project: str,
    base_url: str,
    enabled: bool = True,
) -> None:
    block = cfg.setdefault("retaindb", {})
    block.update(
        {
            "enabled": enabled,
            "base_url": base_url or DEFAULT_BASE_URL,
            "project": project,
            "memory_mode": block.get("memory_mode", "hybrid") or "hybrid",
            "recall_mode": block.get("recall_mode", "hybrid") or "hybrid",
            "write_frequency": block.get("write_frequency", "async") or "async",
            "context_tokens": int(block.get("context_tokens", 1200) or 1200),
            "prefetch_timeout_ms": int(block.get("prefetch_timeout_ms", 1500) or 1500),
            "flush_batch_size": int(block.get("flush_batch_size", 50) or 50),
            "disable_tool_exposure": bool(block.get("disable_tool_exposure", False)),
            "debug_recall_trace": bool(block.get("debug_recall_trace", False)),
            "agent_id": str(block.get("agent_id", "hermes") or "hermes"),
        }
    )
    _save_hermes_config(cfg)


def _save_env(api_key: str, base_url: str) -> None:
    from hermes_cli.config import save_env_value

    save_env_value("RETAINDB_API_KEY", api_key)
    save_env_value(
        "RETAINDB_BASE_URL",
        base_url if base_url and base_url != DEFAULT_BASE_URL else "",
    )


def _select_or_create_project(
    client: RetainDBClient,
    args,
    *,
    interactive: bool,
) -> str:
    projects = client.list_projects()
    explicit_project = str(getattr(args, "project", None) or "").strip()
    if explicit_project:
        for project in projects:
            if explicit_project in {
                str(project.get("id") or ""),
                str(project.get("slug") or ""),
                str(project.get("name") or ""),
            }:
                return str(project.get("slug") or project.get("name") or project.get("id"))
        created = client.create_project(explicit_project)
        return str(created.get("slug") or created.get("name") or created.get("id"))

    if not projects:
        if not interactive:
            raise RuntimeError("No RetainDB projects found. Re-run with --project <name> to auto-create one.")
        project_name = _prompt("Project name", default="hermes-agent")
        created = client.create_project(project_name)
        return str(created.get("slug") or created.get("name") or created.get("id"))

    if not interactive:
        if len(projects) == 1:
            project = projects[0]
            return str(project.get("slug") or project.get("name") or project.get("id"))
        raise RuntimeError("Multiple RetainDB projects found. Re-run with --project <slug>.")

    print("\n  Available RetainDB projects:")
    for idx, project in enumerate(projects, start=1):
        label = project.get("slug") or project.get("name") or project.get("id")
        print(f"    {idx}. {label}")
    print(f"    {len(projects) + 1}. Create new project")

    choice = _prompt("Choose project", default="1")
    try:
        selected = int(choice)
    except ValueError:
        selected = 1

    if selected == len(projects) + 1:
        project_name = _prompt("New project name", default="hermes-agent")
        created = client.create_project(project_name)
        return str(created.get("slug") or created.get("name") or created.get("id"))

    project = projects[max(0, min(len(projects) - 1, selected - 1))]
    return str(project.get("slug") or project.get("name") or project.get("id"))


def _smoke_test(client: RetainDBClient, config: RetainDBClientConfig) -> tuple[bool, str]:
    smoke_config = replace(
        config,
        prefetch_timeout_ms=max(int(config.prefetch_timeout_ms or 0), 8000),
    )
    manager = RetainDBSessionManager(client=RetainDBClient(smoke_config), config=smoke_config)
    identity = manager.resolve_identity("hermes-retaindb-setup")
    try:
        manager.get_profile(identity.session_id)
    except Exception as exc:
        return False, f"Read smoke test failed: {exc}"

    try:
        write_result = manager.remember(
            identity.session_id,
            "Hermes setup smoke test",
            memory_type="factual",
            importance=0.1,
            metadata={"source": "hermes.retaindb.setup"},
        )
        memory_id = (
            write_result.get("memory", {}) or {}
        ).get("id") or write_result.get("memory_id") or write_result.get("id")
        if memory_id:
            try:
                manager.forget(str(memory_id))
            except Exception:
                pass
    except Exception as exc:
        return False, f"Write smoke test failed: {exc}"

    return True, "Read/write smoke test passed."


def cmd_setup(args) -> None:
    cfg = _load_hermes_config()
    current = RetainDBClientConfig.from_global_config()
    interactive = not bool(getattr(args, "yes", False))
    config_path = _config_path()
    env_path = _env_path()

    print("\nRetainDB setup\n" + "-" * 40)
    print("  Native RetainDB memory for Hermes.")
    print(f"  Config: {config_path}")
    print(f"  Env:    {env_path}\n")

    api_key = str(getattr(args, "api_key", None) or current.api_key or "").strip()
    if interactive and not api_key:
        api_key = _prompt("RetainDB API key", secret=True)
    if not api_key:
        print("  No API key provided. Set RETAINDB_API_KEY or re-run with --api-key.\n")
        return

    base_url = str(getattr(args, "base_url", None) or current.base_url or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    if interactive and not getattr(args, "base_url", None):
        base_url = _prompt("RetainDB base URL", default=base_url)

    test_config = RetainDBClientConfig.from_global_config()
    test_config.api_key = api_key
    test_config.base_url = base_url
    client = RetainDBClient(test_config)

    print("  Validating API key... ", end="", flush=True)
    try:
        client.validate_api_key()
    except Exception as exc:
        print("FAILED")
        print(f"  {exc}\n")
        return
    print("OK")

    try:
        project = _select_or_create_project(client, args, interactive=interactive)
    except Exception as exc:
        print(f"  {exc}\n")
        return

    _write_runtime_config(cfg, project=project, base_url=base_url)
    _save_env(api_key, base_url)

    final_config = RetainDBClientConfig.from_global_config()
    final_config.api_key = api_key
    final_config.base_url = base_url
    final_config.project = project

    ok, message = _smoke_test(RetainDBClient(final_config), final_config)
    print(f"\n  Config written to {config_path}")
    print(f"  Env written to {env_path}")
    print(f"  Project: {project}")
    print(f"  Base URL: {base_url}")
    print(f"  Status: {'ready' if ok else 'not ready'}")
    print(f"  {message}\n")


def cmd_status(args) -> None:
    config = RetainDBClientConfig.from_global_config()
    api_key = config.api_key or ""
    masked = f"...{api_key[-8:]}" if len(api_key) > 8 else ("set" if api_key else "not set")
    print("\nRetainDB status\n" + "-" * 40)
    print(f"  Enabled:        {config.enabled}")
    print(f"  API key:        {masked}")
    print(f"  Base URL:       {config.base_url}")
    print(f"  Project:        {config.project or 'not set'}")
    print(f"  Memory mode:    {config.memory_mode}")
    print(f"  Recall mode:    {config.recall_mode}")
    print(f"  Write freq:     {config.write_frequency}")
    print(f"  Context tokens: {config.context_tokens}")
    print(f"  Tool exposure:  {not config.disable_tool_exposure}")

    if not config.should_activate():
        print("\n  Not connected (missing enabled/project/api key).\n")
        return

    status = RetainDBSessionManager(config=config).connection_status()
    if status.get("ok"):
        print(f"\n  Connection... OK ({len(status.get('projects') or [])} project(s) visible)\n")
    else:
        print(f"\n  Connection... FAILED ({status.get('error')})\n")


def cmd_test(args) -> None:
    config = RetainDBClientConfig.from_global_config()
    if not config.should_activate():
        print("  RetainDB is not configured. Run 'hermes retaindb setup' first.\n")
        return
    ok, message = _smoke_test(RetainDBClient(config), config)
    print(f"  {'PASS' if ok else 'FAIL'}: {message}\n")


def cmd_mode(args) -> None:
    cfg = _load_hermes_config()
    current = str((cfg.get("retaindb", {}) or {}).get("memory_mode") or "hybrid")
    mode = getattr(args, "mode", None)
    if not mode:
        print("\nRetainDB memory mode\n" + "-" * 40)
        print(f"  Current: {current}")
        print("  Options: hybrid, retaindb\n")
        return
    if mode not in {"hybrid", "retaindb"}:
        print("  Invalid mode. Options: hybrid, retaindb\n")
        return
    cfg.setdefault("retaindb", {})["memory_mode"] = mode
    _save_hermes_config(cfg)
    print(f"  Memory mode -> {mode}\n")


def cmd_tokens(args) -> None:
    cfg = _load_hermes_config()
    current = int((cfg.get("retaindb", {}) or {}).get("context_tokens") or 1200)
    context_tokens = getattr(args, "context", None)
    if context_tokens is None:
        print("\nRetainDB token budget\n" + "-" * 40)
        print(f"  Context tokens: {current}\n")
        return
    cfg.setdefault("retaindb", {})["context_tokens"] = int(context_tokens)
    _save_hermes_config(cfg)
    print(f"  Context tokens -> {int(context_tokens)}\n")


def cmd_identity(args) -> None:
    config = RetainDBClientConfig.from_global_config()
    session_id = getattr(args, "session_id", None) or "hermes-retaindb-identity"
    runtime_identity = {
        "platform": "cli",
        "user_name": os.getenv("USER") or os.getenv("USERNAME") or "",
    }
    identity = RetainDBSessionManager(config=config, runtime_identity=runtime_identity).resolve_identity(session_id)
    print("\nRetainDB identity\n" + "-" * 40)
    print(f"  user_id:    {identity.user_id}")
    print(f"  session_id: {identity.session_id}")
    print(f"  agent_id:   {identity.agent_id}")
    print(f"  project:    {identity.project or 'not set'}")
    print(f"  source:     {identity.source}")
    print(f"  cache:      {config.identity_cache_path}\n")


def retaindb_command(args) -> None:
    action = getattr(args, "retaindb_command", None)
    if action == "setup":
        cmd_setup(args)
    elif action == "status":
        cmd_status(args)
    elif action == "test":
        cmd_test(args)
    elif action == "mode":
        cmd_mode(args)
    elif action == "tokens":
        cmd_tokens(args)
    elif action == "identity":
        cmd_identity(args)
    else:
        print("Usage: hermes retaindb [setup|status|test|mode|tokens|identity]\n")
