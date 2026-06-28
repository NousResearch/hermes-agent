"""ollama-cloud (and other API-key providers) periodic failback.

The credential pool already fails *over* to the backup on 429 (see
``agent.credential_pool.CredentialPool.mark_exhausted_and_rotate``), and
the TTL on a 429 EXHAUSTED entry is 1 hour — so under normal load the
pool re-evaluates the primary on every ``select()`` call once the cooldown
elapses.  But that's a passive "wait until you happen to select()" model.

The user wanted an active "every 5 hours, if I'm on backup, force rotation
back to primary" cron job.  This module is the active half.

Contract (see kpi_test_ollama_failover.py KPI-4):
    - On the backup entry AND the primary is currently EXHAUSTED with
      cooldown elapsed: reset the primary to OK, return
      ``{"action": "failback_triggered", "from": ..., "to": ...}``.
    - On the primary entry: return ``{"action": "no_failback_needed"}``
      AND emit NOTHING on stdout (the cron wrapper — see
      ``scripts/failback.py`` — uses the --no-agent pattern, so empty
      stdout means the operator sees nothing).
    - On any setup error (pool not loadable, no primary, etc.):
      return ``{"action": "error", "error": "..."}`` and let the wrapper
      decide whether to surface that to the user.

Why a JSON dict?  The wrapper script (scripts/failback.py) calls
``run()`` via Python and writes the dict to stdout only when the action
is ``failback_triggered`` — that gives the operator a single
human-readable line in the chat when the rare event actually happens,
and silence otherwise.  See kpi_test_ollama_failover.py KPI-4b.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_FAILBACK_PROVIDER = "ollama-cloud"


def _primary_source(provider: str) -> Optional[str]:
    """Return the env-var source string for the primary key of *provider*.

    Currently the only provider with a primary/backup split in its
    ``api_key_env_vars`` tuple is ollama-cloud.  This helper keeps the
    failback logic honest: if someone adds the same pattern to a new
    provider, the failback only triggers for that provider's primary.
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        cfg = PROVIDER_REGISTRY.get(provider)
    except Exception as exc:
        logger.debug("failback: cannot import PROVIDER_REGISTRY: %s", exc)
        return None
    if cfg is None:
        return None
    env_vars = tuple(getattr(cfg, "api_key_env_vars", ()) or ())
    if not env_vars:
        return None
    # Primary is the first var in the tuple (same convention as
    # _seed_from_env: it iterates in order, lowest-priority first).
    return f"env:{env_vars[0]}"


def run(
    provider: str = DEFAULT_FAILBACK_PROVIDER,
    *,
    env: Optional[Dict[str, str]] = None,
    pool: Optional[Any] = None,
) -> Dict[str, Any]:
    """Active failback for *provider*.  See module docstring for the contract.

    Parameters
    ----------
    provider:
        Provider id (default ``"ollama-cloud"``).
    env:
        Optional environment dict to load the pool under.  In normal
        use we want the same HERMES_HOME as the running Hermes process;
        the wrapper passes ``os.environ`` so this works out of the box.
    pool:
        Optional pre-loaded ``CredentialPool`` to use.  When None,
        ``run()`` calls ``load_pool(provider)`` itself.  Tests that
        have already set up a pool with a known ``current_id`` pass
        it in directly so the failback sees the right "are we on
        backup?" answer (the pool's ``_current_id`` is per-process
        state and isn't persisted to auth.json).
    """
    if env is not None:
        # Caller explicitly passed an env (test isolation, subshell, etc.).
        # Don't mutate the caller's dict.
        run_env = env
    else:
        run_env = None  # signal "use current process env"

    primary_source = _primary_source(provider)
    if primary_source is None:
        return {
            "action": "error",
            "error": f"provider {provider!r} has no api_key_env_vars[0] — cannot determine primary",
        }

    if pool is None:
        try:
            # Import lazily so the module is importable even if the agent
            # package isn't on sys.path (e.g. during a thin test import).
            from agent.credential_pool import load_pool
        except Exception as exc:
            return {
                "action": "error",
                "error": f"cannot import credential_pool: {exc}",
            }

        if run_env is not None and run_env is not os.environ:
            # The caller (e.g. the wrapper subprocess) gave us a custom env.
            # Propagate it into the current process so the pool's load_env()
            # helper sees the right HERMES_HOME / .env file.
            for key, value in run_env.items():
                os.environ[key] = value

        try:
            pool = load_pool(provider)
        except Exception as exc:
            return {
                "action": "error",
                "error": f"load_pool({provider!r}) raised: {exc}",
            }

    entries = pool.entries()
    if not entries:
        return {
            "action": "error",
            "error": f"pool {provider!r} has no entries",
        }

    primary = next((e for e in entries if e.source == primary_source), None)
    if primary is None:
        return {
            "action": "error",
            "error": f"primary entry {primary_source!r} not found in pool",
        }

    current = pool.current()
    current_source = current.source if current else None

    # Decision matrix:
    #
    #   current is None             → no_failback_needed
    #     (pool was loaded but select() was never called; nothing to
    #      fail back from.  This is the fresh-process / first-API-call
    #      state, not the "we're on backup" state.)
    #   on_primary, primary_ok      → no_failback_needed
    #   on_primary, primary_exhausted → no_failback_needed
    #   on_backup, primary_ok       → trigger failback
    #   on_backup, primary_exhausted → trigger failback
    #     (the 1h 429 TTL is the *passive* recovery path; the 5h active
    #      failback exists precisely to override it.  If the primary is
    #      still 429'd when we retry, mark_exhausted_and_rotate will
    #      rotate us right back to the backup on the next API call —
    #      no harm done.  See kpi_test_ollama_failover.py KPI-4.)
    if current is None:
        return {
            "action": "no_failback_needed",
            "current": None,
            "reason": "no current selection (pool not yet used in this process)",
        }

    current_source = current.source
    if current_source == primary_source:
        return {
            "action": "no_failback_needed",
            "current": current_source,
            "primary_status": primary.last_status,
        }

    return _trigger_failback(pool, primary, current)


def _trigger_failback(pool, primary, current) -> Dict[str, Any]:
    """Reset primary to OK and force-rotate the pool's current pointer to it.

    Uses the same machinery as the live failover path so the persisted
    state in auth.json is consistent.  Falls back to a direct dataclass
    replace if the pool's public surface doesn't expose a "reset entry"
    helper (it doesn't, so we use ``_replace_entry`` + ``_persist``).
    """
    from dataclasses import replace
    from agent.credential_pool import STATUS_OK

    updated = replace(
        primary,
        last_status=STATUS_OK,
        last_status_at=None,
        last_error_code=None,
        last_error_reason=None,
        last_error_message=None,
        last_error_reset_at=None,
    )
    pool._replace_entry(primary, updated)
    pool._persist()
    # Make the primary the current entry so the next select() returns it
    # immediately, even if other entries also have status=ok.
    pool._current_id = updated.id

    from_source = current.source if current else None
    return {
        "action": "failback_triggered",
        "from": from_source,
        "to": updated.source,
        "primary_id": updated.id,
    }


# --------------------------------------------------------------------------- #
# CLI entry point (also used by scripts/failback.py)
# --------------------------------------------------------------------------- #


def main(argv: Optional[list] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Force-rotate the credential pool back to its primary entry "
        "if the primary is healthy and we're currently on the backup."
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_FAILBACK_PROVIDER,
        help="Provider id (default: ollama-cloud)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=True,
        help="Suppress stdout unless failback_triggered (default: True)",
    )
    args = parser.parse_args(argv)

    result = run(provider=args.provider)
    if not args.quiet or result.get("action") == "failback_triggered":
        # Only emit on the interesting case.  Errors also emit so the
        # operator can see why the periodic check failed.
        print(json.dumps(result))
    return 0 if result.get("action") != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
