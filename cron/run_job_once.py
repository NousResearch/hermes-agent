"""Run a single cron job in the current profile and print JSON to stdout.

Used by cron.scheduler to execute profile-scoped jobs under an alternate
HERMES_HOME. Input is a JSON object on stdin with keys:
- job: full job dict
- prompt: already-built prompt string
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_time import now as _hermes_now
from run_agent import AIAgent
from hermes_cli.config import load_config
from hermes_constants import parse_reasoning_effort
from cron.cron_timeout import resolve_cron_inactivity_timeout_seconds
from agent.smart_model_routing import resolve_turn_route
from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error

logger = logging.getLogger(__name__)


def main() -> int:
    payload = json.load(sys.stdin)
    job = payload["job"]
    prompt = payload["prompt"]

    # Session store optional
    session_db = None
    cron_session_id = f"cron_{job['id']}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"
    try:
        from hermes_state import SessionDB
        session_db = SessionDB()
    except Exception:
        session_db = None

    try:
        origin = job.get("origin") or {}
        if origin.get("platform") and origin.get("chat_id"):
            os.environ["HERMES_SESSION_PLATFORM"] = str(origin["platform"])
            os.environ["HERMES_SESSION_CHAT_ID"] = str(origin["chat_id"])
            if origin.get("chat_name"):
                os.environ["HERMES_SESSION_CHAT_NAME"] = str(origin["chat_name"])
            if origin.get("thread_id"):
                os.environ["HERMES_SESSION_THREAD_ID"] = str(origin["thread_id"])

        # Silence rich/tool progress output so the parent scheduler gets clean JSON on stdout.
        os.environ.setdefault("HERMES_QUIET", "1")
        os.environ.setdefault("NO_COLOR", "1")

        from dotenv import load_dotenv
        from hermes_cli.config import get_shared_env_path
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
        shared_env_path = get_shared_env_path()
        if shared_env_path.exists() and shared_env_path != (hermes_home / ".env"):
            try:
                load_dotenv(str(shared_env_path), override=True, encoding="utf-8")
            except UnicodeDecodeError:
                load_dotenv(str(shared_env_path), override=True, encoding="latin-1")
        try:
            load_dotenv(str(hermes_home / ".env"), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            load_dotenv(str(hermes_home / ".env"), override=True, encoding="latin-1")

        cfg = load_config() or {}
        model_spec = job.get("model")
        provider_override = job.get("provider")
        model = model_spec or ""
        if isinstance(model_spec, dict):
            provider_override = model_spec.get("provider") or provider_override
            model = model_spec.get("model") or model_spec.get("default") or ""
        model_cfg = cfg.get("model", {})
        if not model:
            if isinstance(model_cfg, str):
                model = model_cfg
            elif isinstance(model_cfg, dict):
                model = model_cfg.get("default", model)

        effort = str(cfg.get("agent", {}).get("reasoning_effort", "")).strip()
        reasoning_config = parse_reasoning_effort(effort)
        max_iterations = cfg.get("agent", {}).get("max_turns") or cfg.get("max_turns") or 90
        pr = cfg.get("provider_routing", {})
        smart_routing = cfg.get("smart_model_routing", {}) or {}

        try:
            runtime_kwargs = {"requested": provider_override or os.getenv("HERMES_INFERENCE_PROVIDER")}
            if job.get("base_url"):
                runtime_kwargs["explicit_base_url"] = job.get("base_url")
            runtime = resolve_runtime_provider(**runtime_kwargs)
        except Exception as exc:
            raise RuntimeError(format_runtime_provider_error(exc)) from exc

        target_home = os.environ.get("HOME") or str(Path.home())
        cookie_override = os.environ.get("V2EX_CHROME_COOKIE_FILE")
        if not cookie_override and "v2ex_fetch.py" in prompt:
            cookie_override = str(Path(target_home) / "Library/Application Support/Google/Chrome/Profile 1/Cookies")

        agent_env = {
            "HOME": target_home,
            "MCP_QUIET": "1",
            "CHROME_DEVTOOLS_MCP_QUIET": "1",
        }
        if cookie_override:
            agent_env["V2EX_CHROME_COOKIE_FILE"] = cookie_override

        turn_route = resolve_turn_route(
            prompt,
            smart_routing,
            {
                "model": model,
                "api_key": runtime.get("api_key"),
                "base_url": runtime.get("base_url"),
                "provider": runtime.get("provider"),
                "api_mode": runtime.get("api_mode"),
                "command": runtime.get("command"),
                "args": list(runtime.get("args") or []),
            },
        )

        fallback_model = cfg.get("fallback_providers") or cfg.get("fallback_model") or None
        credential_pool = None
        runtime_provider = str(turn_route["runtime"].get("provider") or "").strip().lower()
        if runtime_provider:
            try:
                from agent.credential_pool import load_pool
                pool = load_pool(runtime_provider)
                if pool.has_credentials():
                    credential_pool = pool
            except Exception:
                credential_pool = None

        agent = AIAgent(
            model=turn_route["model"],
            api_key=turn_route["runtime"].get("api_key"),
            base_url=turn_route["runtime"].get("base_url"),
            provider=turn_route["runtime"].get("provider"),
            api_mode=turn_route["runtime"].get("api_mode"),
            acp_command=turn_route["runtime"].get("command"),
            acp_args=turn_route["runtime"].get("args"),
            max_iterations=max_iterations,
            reasoning_config=reasoning_config,
            fallback_model=fallback_model,
            credential_pool=credential_pool,
            providers_allowed=pr.get("only"),
            providers_ignored=pr.get("ignore"),
            providers_order=pr.get("order"),
            provider_sort=pr.get("sort"),
            disabled_toolsets=["cronjob", "messaging", "clarify"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cron",
            session_id=cron_session_id,
            session_db=session_db,
        )

        cron_timeout = resolve_cron_inactivity_timeout_seconds(cfg)
        inactivity_limit = cron_timeout if cron_timeout > 0 else None
        poll_interval = 5.0
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(agent.run_conversation, prompt)
        inactivity_timeout = False
        try:
            if inactivity_limit is None:
                result = future.result()
            else:
                result = None
                while True:
                    done, _ = concurrent.futures.wait({future}, timeout=poll_interval)
                    if done:
                        result = future.result()
                        break
                    idle_secs = 0.0
                    if hasattr(agent, "get_activity_summary"):
                        try:
                            idle_secs = agent.get_activity_summary().get("seconds_since_activity", 0.0)
                        except Exception:
                            idle_secs = 0.0
                    if idle_secs >= inactivity_limit:
                        inactivity_timeout = True
                        break
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        if inactivity_timeout:
            if hasattr(agent, "interrupt"):
                agent.interrupt("Cron job timed out (inactivity)")
            raise TimeoutError(f"Cron job '{job.get('name', job['id'])}' idle too long")

        final_response = result.get("final_response", "") or ""
        print(json.dumps({"success": True, "final_response": final_response}, ensure_ascii=False))
        return 0

    except Exception as exc:
        print(json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    finally:
        for key in (
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
            "HERMES_SESSION_THREAD_ID",
            "HERMES_CRON_AUTO_DELIVER_PLATFORM",
            "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
            "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
        ):
            os.environ.pop(key, None)
        if session_db:
            try:
                session_db.end_session(cron_session_id, "cron_complete")
            except Exception:
                pass
            try:
                session_db.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
