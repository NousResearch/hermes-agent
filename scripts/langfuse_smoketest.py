#!/usr/bin/env python3
"""Verify Langfuse credentials and that the user plugin can emit a trace.

Loads ``~/.hermes/.env`` (and optional repo ``.env``) like Hermes. Run from repo:

  uv run python scripts/langfuse_smoketest.py

Exit codes: 0 ok, 1 connectivity/plugin failure, 2 missing keys/plugin files.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pick(*keys: str) -> str:
    for k in keys:
        v = os.getenv(k, "").strip()
        if v:
            return v
    return ""


def _load_hermes_env() -> None:
    repo = _repo_root()
    sys.path.insert(0, str(repo))
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_constants import get_hermes_home

    load_hermes_dotenv(hermes_home=get_hermes_home(), project_env=repo / ".env")


def _sdk_smoke() -> str:
    from langfuse import Langfuse

    pk = _pick("HERMES_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY", "CC_LANGFUSE_PUBLIC_KEY")
    sk = _pick("HERMES_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY", "CC_LANGFUSE_SECRET_KEY")
    base = _pick("HERMES_LANGFUSE_BASE_URL", "LANGFUSE_BASE_URL", "CC_LANGFUSE_BASE_URL")
    if not base:
        base = "https://cloud.langfuse.com"
    if not pk or not sk:
        print("ERROR: set HERMES_LANGFUSE_PUBLIC_KEY and HERMES_LANGFUSE_SECRET_KEY (or LANGFUSE_* aliases).")
        sys.exit(2)

    lf = Langfuse(public_key=pk, secret_key=sk, base_url=base)
    if not lf.auth_check():
        print("ERROR: Langfuse auth_check() returned False.")
        sys.exit(1)

    trace_id = lf.create_trace_id(seed="hermes-langfuse-smoketest")
    root = lf.start_observation(
        trace_context={"trace_id": trace_id},
        name="Hermes langfuse_smoketest (SDK)",
        as_type="chain",
        input={"check": "sdk"},
        metadata={"source": "scripts/langfuse_smoketest.py"},
    )
    child = root.start_observation(
        name="sub-span",
        as_type="generation",
        input={"ping": True},
        model="smoke/test",
    )
    child.update(output={"pong": True})
    child.end()
    root.end()
    lf.flush()
    try:
        url = lf.get_trace_url(trace_id=trace_id)
    except Exception:
        url = f"{base.rstrip('/')}/traces/{trace_id}"
    print("SDK smoke: OK")
    print("  trace_id:", trace_id)
    print("  url:", url)
    return trace_id


def _plugin_smoke() -> None:
    plugin_path = Path.home() / ".hermes" / "plugins" / "langfuse_tracing" / "__init__.py"
    if not plugin_path.is_file():
        print("SKIP plugin smoke: no file at", plugin_path)
        return

    spec = importlib.util.spec_from_file_location("langfuse_tracing_smoke", plugin_path)
    if spec is None or spec.loader is None:
        print("ERROR: cannot load plugin module spec")
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["langfuse_tracing_smoke"] = mod
    spec.loader.exec_module(mod)

    mod._TRACE_STATE.clear()
    mod._LANGFUSE_CLIENT = None

    session_id = f"smoke_sess_{uuid.uuid4().hex[:8]}"
    effective_task_id = str(uuid.uuid4())
    user_msg = "Langfuse plugin smoketest message."

    mod.on_pre_llm_call(
        session_id=session_id,
        user_message=user_msg,
        conversation_history=[],
        model="smoke/model",
        platform="cli",
    )
    mod.on_pre_api_request(
        task_id=effective_task_id,
        session_id=session_id,
        platform="cli",
        model="smoke/model",
        provider="test",
        base_url="http://localhost",
        api_mode="chat_completions",
        api_call_count=1,
        message_count=1,
        tool_count=0,
        approx_input_tokens=10,
        request_char_count=40,
        max_tokens=256,
    )
    mod.on_post_api_request(
        task_id=effective_task_id,
        session_id=session_id,
        provider="test",
        base_url="http://localhost",
        api_mode="chat_completions",
        model="smoke/model",
        api_call_count=1,
        api_duration=0.01,
        finish_reason="stop",
        usage={
            "input_tokens": 5,
            "output_tokens": 5,
            "total_tokens": 10,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
        assistant_content_chars=4,
        assistant_tool_call_count=0,
        response_model="smoke/model",
    )
    mod.on_post_llm_call(
        session_id=session_id,
        user_message=user_msg,
        assistant_response="pong",
        conversation_history=[],
        model="smoke/model",
        platform="cli",
    )

    client = mod._get_langfuse()
    if client is None:
        print("SKIP plugin smoke: Langfuse disabled or keys missing (_get_langfuse is None).")
        return
    client.flush()
    print("Plugin hook chain: OK (flushed)")
    print("  session_id:", session_id)


def _api_list_traces(limit: int = 2) -> None:
    pk = _pick("HERMES_LANGFUSE_PUBLIC_KEY", "LANGFUSE_PUBLIC_KEY", "CC_LANGFUSE_PUBLIC_KEY")
    sk = _pick("HERMES_LANGFUSE_SECRET_KEY", "LANGFUSE_SECRET_KEY", "CC_LANGFUSE_SECRET_KEY")
    base = _pick("HERMES_LANGFUSE_BASE_URL", "LANGFUSE_BASE_URL", "CC_LANGFUSE_BASE_URL")
    if not base or not pk or not sk:
        return
    base = base.rstrip("/")
    auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
    req = Request(
        f"{base}/api/public/traces?limit={limit}",
        headers={"Authorization": f"Basic {auth}"},
    )
    try:
        with urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode())
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print("REST list traces: failed:", exc)
        return
    rows = payload.get("data") or []
    print(f"REST /api/public/traces?limit={limit}: {len(rows)} row(s)")
    for row in rows:
        name = row.get("name")
        tid = row.get("id")
        ts = row.get("timestamp")
        print(f"  - {ts}  {name!r}  id={tid}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plugin", action="store_true", help="Only run SDK smoke + REST list")
    args = parser.parse_args()

    _load_hermes_env()
    _sdk_smoke()
    if not args.no_plugin:
        _plugin_smoke()
    _api_list_traces(limit=3)
    print("Done.")


if __name__ == "__main__":
    main()
