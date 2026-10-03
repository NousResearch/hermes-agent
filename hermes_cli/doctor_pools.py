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
    """Proactive, offline triage of every credential pool: which providers are burned, and
    which recovery actually applies to them.

    Not the fix for #119533's turn-death — ``agent_init._routed_client_kwargs`` already raises a
    provider-specific billing/cooldown verdict there. What that reactive path cannot do is list
    the pools you are NOT currently using: a pool benched an hour ago while you were working on
    something else stays invisible until it becomes the one you pick. This is that view.
    Unconfigured pools print nothing.

    Never triggers a token refresh and never makes a network call. Note it is NOT read-only:
    ``load_pool`` persists exactly what any model call would (env-key ingestion into auth.json,
    singleton seeding, stale-row pruning), so this is not a safe way to inspect a machine you
    must not touch — read ``auth.json`` directly instead."""
    from agent.credential_pool import STATUS_DEAD, STATUS_EXHAUSTED, load_pool

    rows: list = []
    for pid in _pool_provider_ids():
        try:
            pool = load_pool(pid)
        except OSError as exc:
            rows.append((
                check_warn, f"Credential pool: {pid}", f"(unreadable: {exc})",
                f"Could not read this pool. Fix: check auth.json or run `hermes auth reset {pid}`",
            ))
            f.manual_issues.append(f"Credential pool {pid} is unreadable ({exc})")
            continue
        entries = pool.entries()
        total = len(entries)
        if pool.has_available():
            rows.append((
                check_ok, f"Credential pool: {pid}",
                f"({total} entries, at least one available)", None,
            ))
            continue
        burned = sum(1 for entry in entries if entry.last_status in (STATUS_EXHAUSTED, STATUS_DEAD))
        if not burned:
            # No burn state at all: env-sourced references whose secret does not resolve in
            # this process. Key-presence belongs to the env/connectivity checks, so accusing
            # them here would flood the summary with false burns (#119533).
            continue
        if burned < total:
            # Partly burned. The unhydrated remainder must NOT make this look healthy — say
            # how many are benched and how many are merely unresolvable, and still name the
            # pool: a real burn behind one env reference used to print nothing at all.
            nxt_partial = pool.next_available_at()
            wait_partial = None if nxt_partial is None else max(0, int(nxt_partial - time.time()))
            when_partial = (
                "unavailable with no recovery time" if wait_partial is None
                else f"benched for ~{wait_partial}s more"
            )
            rows.append((
                check_warn, f"Credential pool: {pid}",
                f"({burned} of {total} entries {when_partial}, "
                f"{total - burned} unavailable with no burn state)",
                "The unavailable entries carry no burn state — an env-sourced reference whose "
                "secret does not resolve here. Key presence is reported by the env checks.",
            ))
            f.manual_issues.append(
                f"Credential pool {pid}: {burned} of {total} entries {when_partial} while "
                f"{total - burned} are unavailable with no burn state. "
                f"Fix: wait it out, or `hermes auth reset {pid}`"
            )
            continue
        nxt = pool.next_available_at()
        wait = None if nxt is None else max(0, int(nxt - time.time()))
        # The remedy depends on WHY it burned. `hermes auth reset` only clears local
        # stamps: on a 402 (billing) the very next call re-402s and re-benches for another
        # hour, and DEAD never re-enters via TTL at all (credential_pool._available_entries
        # prunes it after 24h and tells the user to `hermes auth add`). Teaching the wrong
        # remedy is worse than saying nothing, so pick per class.
        codes = {e.last_error_code for e in entries if e.last_error_code}
        dead_only = all(e.last_status == STATUS_DEAD for e in entries)
        if 402 in codes and not dead_only:
            remedy = (
                "out of credits (402). `hermes auth reset` only clears the local stamp — the "
                f"next call re-402s. Fix: add credits, or rotate a key with `hermes auth add {pid}`"
            )
        elif dead_only:
            remedy = (
                f"dead (revoked); DEAD never recovers on its own. Fix: `hermes auth add {pid}`, "
                "or `hermes auth reset` to retry the existing key"
            )
        else:
            remedy = f"benched with a cooldown. Fix: wait it out, or `hermes auth reset {pid}`"
        detail = (
            f"(all {total} entries unavailable, no recovery time)" if wait is None
            else f"(all {total} entries benched, back in ~{wait}s)"
        )
        rows.append((
            check_warn, f"Credential pool: {pid}", detail,
            f"Pool state lives on disk (auth.json) — a gateway restart will NOT clear it. {remedy}",
        ))
        when = "unavailable with no recovery time" if wait is None else f"benched for ~{wait}s more"
        f.manual_issues.append(
            f"Credential pool {pid}: all {total} entries {when} "
            f"(on disk — a restart will not clear it). {remedy}"
        )

    if not rows:
        return
    _section("Credential Pools")
    for mark, label, row_detail, info in rows:
        mark(label, row_detail)
        if info:
            check_info(info)
