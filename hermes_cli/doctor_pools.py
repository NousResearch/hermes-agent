"""``hermes doctor`` — credential-pool state rows (sibling of ``hermes_cli.doctor``)."""

from __future__ import annotations

import time

from hermes_cli.doctor_report import Finding, _section, check_info, check_ok, check_warn, doctor_check


def _pool_provider_ids() -> list[str]:
    """API-key pool ids to scan: every ``api_key`` registry provider plus openrouter, which is
    absent from PROVIDER_REGISTRY on purpose (#109397) yet owns the pool the chat path burns (#119533)."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    return ["openrouter", *(
        pid
        for pid, pconfig in PROVIDER_REGISTRY.items()
        if getattr(pconfig, "auth_type", "") == "api_key"
    )]


@doctor_check(on_error="Credential pools", detail="(could not check: {e})")
def _check_credential_pools(should_fix: bool, f: Finding) -> None:
    """Benched/exhausted pool rows: distinguish "pool burned (wait or ``hermes auth reset``)" from
    actual config loss — the generic ``No LLM provider configured`` turn-death hides which one it
    was (#119533). Snapshot only: no token refresh and no network calls (load_pool may normalize
    stale pool rows exactly as any model call does); unconfigured pools print nothing."""
    from agent.credential_pool import STATUS_DEAD, STATUS_EXHAUSTED, load_pool

    rows: list = []
    for pid in _pool_provider_ids():
        try:
            pool = load_pool(pid)
        except Exception as exc:
            rows.append((
                check_warn, f"Credential pool: {pid}", f"(unreadable: {exc})",
                f"Could not read this pool. Fix: check auth.json or run `hermes auth reset {pid}`",
            ))
            f.manual_issues.append(f"Credential pool {pid} is unreadable ({exc})")
            continue
        if not pool.has_credentials():
            continue  # not configured: the env/connectivity checks own that verdict
        total = len(pool.entries())
        if pool.has_available():
            rows.append((
                check_ok, f"Credential pool: {pid}",
                f"({total} entries, at least one available)", None,
            ))
            continue
        burned = sum(1 for entry in pool.entries() if entry.last_status in (STATUS_EXHAUSTED, STATUS_DEAD))
        if not burned or burned < total:
            # Unusable rows without burn state (env-sourced references whose secret does not
            # resolve in this process): key-presence verdicts belong to the env/connectivity
            # checks, and accusing them here would flood the summary with false burns (#119533).
            continue
        nxt = pool.next_available_at()
        wait = None if nxt is None else max(0, int(nxt - time.time()))
        detail = (
            f"(all {total} entries unavailable, no recovery time)" if wait is None
            else f"(all {total} entries benched, back in ~{wait}s)"
        )
        rows.append((
            check_warn, f"Credential pool: {pid}", detail,
            "Pool state lives on disk (auth.json) — a gateway restart will NOT clear it. "
            "Wait it out or run `hermes auth reset <provider>`",
        ))
        if wait is None:
            f.manual_issues.append(
                f"Credential pool {pid}: all {total} entries unavailable with no recovery time. "
                f"Fix: run `hermes auth reset {pid}`"
            )
        else:
            f.manual_issues.append(
                f"Credential pool {pid}: all {total} entries benched for ~{wait}s more "
                f"(on disk — a restart will not clear it). Fix: wait it out or run `hermes auth reset {pid}`"
            )

    if not rows:
        return
    _section("Credential Pools")
    for mark, label, row_detail, info in rows:
        mark(label, row_detail)
        if info:
            check_info(info)
