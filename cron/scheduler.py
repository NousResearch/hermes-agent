"""
Cron job scheduler - executes due jobs.

Provides tick() which checks for due jobs and runs them. The gateway
calls this every 60 seconds from a background thread.

Uses a file-based lock (~/.hermes/cron/.tick.lock) so only one tick
runs at a time if multiple processes overlap.
"""

import asyncio
import logging
import os
import sys
import traceback

# fcntl is Unix-only; on Windows use msvcrt for file locking
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cron.jobs import get_due_jobs, mark_job_run, save_job_output

# File-based lock prevents concurrent ticks from gateway + daemon + systemd timer
_LOCK_DIR = Path.home() / ".hermes" / "cron"
_LOCK_FILE = _LOCK_DIR / ".tick.lock"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job, returning {platform, chat_id, chat_name} or None."""
    origin = job.get("origin")
    if not origin:
        return None
    platform = origin.get("platform")
    chat_id = origin.get("chat_id")
    if platform and chat_id:
        return origin
    return None


def _deliver_result(job: dict, content: str) -> None:
    """
    Deliver job output to the configured target (origin chat, specific platform, etc.).

    Uses the standalone platform send functions from send_message_tool so delivery
    works whether or not the gateway is running.
    """
    deliver = job.get("deliver", "local")
    origin = _resolve_origin(job)

    if deliver == "local":
        return

    # Resolve target platform + chat_id
    if deliver == "origin":
        if not origin:
            logger.warning("Job '%s' deliver=origin but no origin stored, skipping delivery", job["id"])
            return
        platform_name = origin["platform"]
        chat_id = origin["chat_id"]
    elif ":" in deliver:
        platform_name, chat_id = deliver.split(":", 1)
    else:
        # Bare platform name like "telegram" — need to resolve to origin or home channel
        platform_name = deliver
        if origin and origin.get("platform") == platform_name:
            chat_id = origin["chat_id"]
        else:
            # Fall back to home channel
            chat_id = os.getenv(f"{platform_name.upper()}_HOME_CHANNEL", "")
            if not chat_id:
                logger.warning("Job '%s' deliver=%s but no chat_id or home channel. Set via: hermes config set %s_HOME_CHANNEL <channel_id>", job["id"], deliver, platform_name.upper())
                return

    from tools.send_message_tool import _send_to_platform
    from gateway.config import load_gateway_config, Platform

    platform_map = {
        "telegram": Platform.TELEGRAM,
        "discord": Platform.DISCORD,
        "slack": Platform.SLACK,
        "whatsapp": Platform.WHATSAPP,
    }
    platform = platform_map.get(platform_name.lower())
    if not platform:
        logger.warning("Job '%s': unknown platform '%s' for delivery", job["id"], platform_name)
        return

    try:
        config = load_gateway_config()
    except Exception as e:
        logger.error("Job '%s': failed to load gateway config for delivery: %s", job["id"], e)
        return

    pconfig = config.platforms.get(platform)
    if not pconfig or not pconfig.enabled:
        logger.warning("Job '%s': platform '%s' not configured/enabled", job["id"], platform_name)
        return

    # Run the async send in a fresh event loop (safe from any thread)
    try:
        result = asyncio.run(_send_to_platform(platform, pconfig, chat_id, content))
    except RuntimeError:
        # asyncio.run() fails if there's already a running loop in this thread;
        # spin up a new thread to avoid that.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _send_to_platform(platform, pconfig, chat_id, content))
            result = future.result(timeout=30)
    except Exception as e:
        logger.error("Job '%s': delivery to %s:%s failed: %s", job["id"], platform_name, chat_id, e)
        return

    if result and result.get("error"):
        logger.error("Job '%s': delivery error: %s", job["id"], result["error"])
    else:
        logger.info("Job '%s': delivered to %s:%s", job["id"], platform_name, chat_id)
        # Mirror the delivered content into the target's gateway session
        try:
            from gateway.mirror import mirror_to_session
            mirror_to_session(platform_name, chat_id, content, source_label="cron")
        except Exception:
            pass


def run_job(job: dict) -> tuple[bool, str, str, Optional[str]]:
    """
    Execute a single cron job.
    
    Returns:
        Tuple of (success, full_output_doc, final_response, error_message)
    """
    from run_agent import AIAgent
    
    job_id = job["id"]
    job_name = job["name"]
    prompt = job["prompt"]
    origin = _resolve_origin(job)
    
    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    # Inject origin context so the agent's send_message tool knows the chat
    if origin:
        os.environ["HERMES_SESSION_PLATFORM"] = origin["platform"]
        os.environ["HERMES_SESSION_CHAT_ID"] = str(origin["chat_id"])
        if origin.get("chat_name"):
            os.environ["HERMES_SESSION_CHAT_NAME"] = origin["chat_name"]

    try:
        # Re-read .env and config.yaml fresh every run so provider/key
        # changes take effect without a gateway restart.
        from dotenv import load_dotenv
        try:
            load_dotenv(os.path.expanduser("~/.hermes/.env"), override=True, encoding="utf-8")
        except UnicodeDecodeError:
            load_dotenv(os.path.expanduser("~/.hermes/.env"), override=True, encoding="latin-1")

        model = os.getenv("HERMES_MODEL", "anthropic/claude-opus-4.6")
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        try:
            import yaml
            _cfg_path = os.path.expanduser("~/.hermes/config.yaml")
            if os.path.exists(_cfg_path):
                with open(_cfg_path) as _f:
                    _cfg = yaml.safe_load(_f) or {}
                _model_cfg = _cfg.get("model", {})
                if isinstance(_model_cfg, str):
                    model = _model_cfg
                elif isinstance(_model_cfg, dict):
                    model = _model_cfg.get("default", model)
                    base_url = _model_cfg.get("base_url", base_url)
        except Exception:
            pass

        agent = AIAgent(
            model=model,
            api_key=api_key,
            base_url=base_url,
            quiet_mode=True,
            session_id=f"cron_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        result = agent.run_conversation(prompt)
        
        final_response = result.get("final_response", "")
        if not final_response:
            final_response = "(No response generated)"
        
        output = f"""# Cron Job: {job_name}

**Job ID:** {job_id}
**Run Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Response

{final_response}
"""
        
        logger.info("Job '%s' completed successfully", job_name)
        return True, output, final_response, None
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error("Job '%s' failed: %s", job_name, error_msg)
        
        output = f"""# Cron Job: {job_name} (FAILED)

**Job ID:** {job_id}
**Run Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Schedule:** {job.get('schedule_display', 'N/A')}

## Prompt

{prompt}

## Error

```
{error_msg}

{traceback.format_exc()}
```
"""
        return False, output, "", error_msg

    finally:
        # Clean up injected env vars so they don't leak to other jobs
        for key in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID", "HERMES_SESSION_CHAT_NAME"):
            os.environ.pop(key, None)


def tick(verbose: bool = True) -> int:
    """
    Check and run all due jobs.
    
    Uses a file lock so only one tick runs at a time, even if the gateway's
    in-process ticker and a standalone daemon or manual tick overlap.
    
    Args:
        verbose: Whether to print status messages
    
    Returns:
        Number of jobs executed (0 if another tick is already running)
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-platform file locking: fcntl on Unix, msvcrt on Windows
    try:
        lock_fd = open(_LOCK_FILE, "w")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
    except (OSError, IOError):
        logger.debug("Tick skipped — another instance holds the lock")
        return 0

    try:
        due_jobs = get_due_jobs()

        if verbose and not due_jobs:
            logger.info("%s - No jobs due", datetime.now().strftime('%H:%M:%S'))
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", datetime.now().strftime('%H:%M:%S'), len(due_jobs))

        executed = 0
        for job in due_jobs:
            try:
                success, output, final_response, error = run_job(job)

                output_file = save_job_output(job["id"], output)
                if verbose:
                    logger.info("Output saved to: %s", output_file)

                # Deliver the final response to the origin/target chat
                deliver_content = final_response if success else f"⚠️ Cron job '{job.get('name', job['id'])}' failed:\n{error}"
                if deliver_content:
                    try:
                        _deliver_result(job, deliver_content)
                    except Exception as de:
                        logger.error("Delivery failed for job %s: %s", job["id"], de)

                mark_job_run(job["id"], success, error)
                executed += 1

            except Exception as e:
                logger.error("Error processing job %s: %s", job['id'], e)
                mark_job_run(job["id"], False, str(e))

        return executed
    finally:
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        elif msvcrt:
            try:
                msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, IOError):
                pass
        lock_fd.close()


if __name__ == "__main__":
    tick(verbose=True)
