from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import BaseModel

from hermes_cli.config import _secure_dir, _secure_file, ensure_hermes_home, get_hermes_home
from hermes_cli.product_config import load_product_config, runtime_host_access_host
from hermes_cli.product_identity import render_product_soul
from hermes_cli.runtime_provider import resolve_runtime_provider
from toolsets import validate_toolset


class ProductRuntimeRecord(BaseModel):
    user_id: str
    display_name: str | None = None
    session_id: str
    container_name: str
    runtime: str
    runtime_port: int
    runtime_root: str
    hermes_home: str
    workspace_root: str
    env_file: str
    manifest_file: str
    status: str = "staged"


class ProductRuntimeSession(BaseModel):
    session_id: str
    messages: list[dict[str, Any]]
    runtime_mode: str
    runtime_toolsets: list[str]


class ProductRuntimeTurnRequest(BaseModel):
    user_message: str


class ProductRuntimeEvent(BaseModel):
    event: str
    payload: dict[str, Any]


def _user_id(user: dict[str, Any]) -> str:
    username = str(user.get("preferred_username") or user.get("sub") or "").strip()
    if not username:
        raise ValueError("Signed-in user is missing a usable username")
    return username


def product_runtime_session_id(user_id: str) -> str:
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:12]
    return f"product_{user_id}_{digest}"


def _product_storage_root(config: dict[str, Any]) -> Path:
    return get_hermes_home() / str(config.get("storage", {}).get("root", "product"))


def _product_users_root(config: dict[str, Any]) -> Path:
    return get_hermes_home() / str(config.get("storage", {}).get("users_root", "product/users"))


def _runtime_root(config: dict[str, Any], user_id: str) -> Path:
    return _product_users_root(config) / user_id / "runtime"


def _workspace_root(config: dict[str, Any], user_id: str) -> Path:
    return _product_users_root(config) / user_id / "workspace"


def _hermes_home(config: dict[str, Any], user_id: str) -> Path:
    return _runtime_root(config, user_id) / "hermes"


def _manifest_path(config: dict[str, Any], user_id: str) -> Path:
    return _runtime_root(config, user_id) / "launch-spec.json"


def _env_path(config: dict[str, Any], user_id: str) -> Path:
    return _runtime_root(config, user_id) / "runtime.env"


def _runtime_toolsets(config: dict[str, Any]) -> list[str]:
    configured = config.get("tools", {}).get("hermes_toolsets", [])
    if isinstance(configured, list):
        normalized = [
            str(item).strip()
            for item in configured
            if str(item).strip() and validate_toolset(str(item).strip())
        ]
        if normalized:
            return normalized
    return ["memory", "session_search"]


def _runtime_port_range(config: dict[str, Any]) -> tuple[int, int]:
    runtime_config = config.get("runtime", {})
    start = int(runtime_config.get("host_port_start", 18091))
    end = int(runtime_config.get("host_port_end", 18150))
    return start, end


def _runtime_image(config: dict[str, Any]) -> str:
    runtime_config = config.get("runtime", {})
    return str(runtime_config.get("image", "ghcr.io/erniconcepts/hermes-agent-core:main")).strip() or "ghcr.io/erniconcepts/hermes-agent-core:main"


def _runtime_binary(config: dict[str, Any]) -> str:
    runtime_config = config.get("runtime", {})
    return str(runtime_config.get("isolation_runtime", "runsc")).strip() or "runsc"


def _runtime_internal_port(config: dict[str, Any]) -> int:
    return int(config.get("runtime", {}).get("internal_port", 8091))


def _resolve_runtime_model_base_url(config: dict[str, Any], base_url: str) -> str:
    normalized = str(base_url or "").strip()
    if not normalized:
        return normalized
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        return normalized
    hostname = (parsed.hostname or "").strip().lower()
    if hostname not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return normalized.rstrip("/")
    replacement_host = runtime_host_access_host(config)
    netloc = replacement_host
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    rewritten = parsed._replace(netloc=netloc)
    return urlunparse(rewritten).rstrip("/")


def _resolve_runtime_port(config: dict[str, Any], user_id: str) -> int:
    existing = load_runtime_record(user_id, config=config)
    if existing is not None:
        return existing.runtime_port
    used_ports: set[int] = set()
    users_root = _product_users_root(config)
    if users_root.exists():
        for manifest in users_root.glob("*/runtime/launch-spec.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                used_ports.add(int(payload["runtime_port"]))
            except Exception:
                continue
    start, end = _runtime_port_range(config)
    for port in range(start, end + 1):
        if port not in used_ports:
            return port
    raise RuntimeError("No runtime ports are available in the configured product range")


def load_runtime_record(user_id: str, *, config: dict[str, Any] | None = None) -> ProductRuntimeRecord | None:
    product_config = config or load_product_config()
    manifest_path = _manifest_path(product_config, user_id)
    if not manifest_path.exists():
        return None
    return ProductRuntimeRecord.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def _write_runtime_record(record: ProductRuntimeRecord) -> None:
    manifest_path = Path(record.manifest_file)
    manifest_path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _secure_file(manifest_path)


def stage_product_runtime(user: dict[str, Any], *, config: dict[str, Any] | None = None) -> ProductRuntimeRecord:
    product_config = config or load_product_config()
    ensure_hermes_home()
    user_id = _user_id(user)
    runtime_root = _runtime_root(product_config, user_id)
    hermes_home = _hermes_home(product_config, user_id)
    workspace_root = _workspace_root(product_config, user_id)
    for path in (
        _product_storage_root(product_config),
        _product_users_root(product_config),
        runtime_root,
        hermes_home,
        hermes_home / "memories",
        workspace_root,
    ):
        path.mkdir(parents=True, exist_ok=True)
        _secure_dir(path)

    soul_path = hermes_home / "SOUL.md"
    soul_path.write_text(render_product_soul(product_config), encoding="utf-8")
    _secure_file(soul_path)

    route = product_config.get("models", {}).get("default_route", {})
    base_url = str(route.get("base_url") or "").strip()
    provider = str(route.get("provider") or "custom").strip() or "custom"
    api_mode = str(route.get("api_mode") or "chat_completions").strip() or "chat_completions"
    model = str(route.get("model") or "qwen3.5-9b-local").strip() or "qwen3.5-9b-local"
    api_key = "product-local-route"
    if not base_url:
        resolved = resolve_runtime_provider(requested=provider)
        base_url = str(resolved.get("base_url") or "").strip()
        api_key = str(resolved.get("api_key") or "").strip() or api_key
        provider = str(resolved.get("provider") or provider).strip() or provider
        api_mode = str(resolved.get("api_mode") or api_mode).strip() or api_mode
    if not base_url:
        raise RuntimeError("Product runtime requires a resolved base URL for the configured model route")
    base_url = _resolve_runtime_model_base_url(product_config, base_url)
    session_id = product_runtime_session_id(user_id)
    runtime_port = _resolve_runtime_port(product_config, user_id)
    container_name = f"hermes-product-runtime-{user_id}"
    toolsets = _runtime_toolsets(product_config)

    env = {
        "HERMES_HOME": "/srv/hermes",
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": api_key,
        "HERMES_PRODUCT_RUNTIME_MODE": "product",
        "MYNAH_RUNTIME_HOST": "0.0.0.0",
        "MYNAH_RUNTIME_PORT": str(_runtime_internal_port(product_config)),
        "MYNAH_PRODUCT_SESSION_ID": session_id,
        "HERMES_PRODUCT_TOOLSETS": ",".join(toolsets),
        "HERMES_PRODUCT_PROVIDER": provider,
        "HERMES_PRODUCT_API_MODE": api_mode,
        "HERMES_PRODUCT_MODEL": model,
    }
    env_path = _env_path(product_config, user_id)
    env_path.write_text("".join(f"{key}={value}\n" for key, value in sorted(env.items())), encoding="utf-8")
    _secure_file(env_path)

    record = ProductRuntimeRecord(
        user_id=user_id,
        display_name=str(user.get("name") or user.get("preferred_username") or "").strip() or None,
        session_id=session_id,
        container_name=container_name,
        runtime=_runtime_binary(product_config),
        runtime_port=runtime_port,
        runtime_root=str(runtime_root),
        hermes_home=str(hermes_home),
        workspace_root=str(workspace_root),
        env_file=str(env_path),
        manifest_file=str(_manifest_path(product_config, user_id)),
        status="staged",
    )
    _write_runtime_record(record)
    return record


def _docker_run_command(record: ProductRuntimeRecord, config: dict[str, Any]) -> list[str]:
    internal_port = _runtime_internal_port(config)
    command = [
        "docker",
        "run",
        "--detach",
        "--restart",
        "unless-stopped",
        "--runtime",
        record.runtime,
        "--name",
        record.container_name,
        "--publish",
        f"127.0.0.1:{record.runtime_port}:{internal_port}",
        "--env-file",
        record.env_file,
        "--add-host",
        f"{runtime_host_access_host(config)}:host-gateway",
        "--mount",
        f"type=bind,src={Path(record.hermes_home).as_posix()},dst=/srv/hermes",
        "--mount",
        f"type=bind,src={Path(record.workspace_root).as_posix()},dst=/srv/workspace",
        "--label",
        f"ch.hermes.product.user_id={record.user_id}",
        "--label",
        "ch.hermes.product.role=runtime",
        _runtime_image(config),
        "python",
        "-m",
        "hermes_cli.product_runtime_service",
    ]
    return command


def _docker_inspect_state(container_name: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["docker", "inspect", container_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    if not payload:
        return None
    return payload[0]


def _remove_container_if_exists(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_product_runtime(user: dict[str, Any], *, config: dict[str, Any] | None = None) -> ProductRuntimeRecord:
    product_config = config or load_product_config()
    record = stage_product_runtime(user, config=product_config)
    container_state = _docker_inspect_state(record.container_name)
    if container_state and bool(container_state.get("State", {}).get("Running")):
        return ProductRuntimeRecord(**{**record.model_dump(), "status": "running"})

    _remove_container_if_exists(record.container_name)
    result = subprocess.run(
        _docker_run_command(record, product_config),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip() or "docker run failed")
    return ProductRuntimeRecord(**{**record.model_dump(), "status": "running"})


def runtime_base_url(record: ProductRuntimeRecord) -> str:
    return f"http://127.0.0.1:{record.runtime_port}"


def get_product_runtime_session(user: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    record = ensure_product_runtime(user, config=config)
    response = httpx.get(f"{runtime_base_url(record)}/runtime/session", timeout=60.0)
    response.raise_for_status()
    return ProductRuntimeSession.model_validate(response.json()).model_dump(mode="json")


def stream_product_runtime_turn(
    user: dict[str, Any],
    user_message: str,
    *,
    config: dict[str, Any] | None = None,
) -> Iterator[str]:
    message = user_message.strip()
    if not message:
        raise ValueError("User message must not be empty")
    record = ensure_product_runtime(user, config=config)
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{runtime_base_url(record)}/runtime/turn/stream",
            json=ProductRuntimeTurnRequest(user_message=message).model_dump(),
            headers={"Accept": "text/event-stream"},
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_text():
                yield chunk


def delete_product_runtime(user_id: str, *, config: dict[str, Any] | None = None) -> None:
    product_config = config or load_product_config()
    record = load_runtime_record(user_id, config=product_config)
    if record is not None:
        _remove_container_if_exists(record.container_name)
        runtime_root = Path(record.runtime_root)
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
