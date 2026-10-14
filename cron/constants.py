"""Fire-claim timing bounds and the job-record schema, shared by the job store and its siblings.

Import-free on purpose: a sibling loaded fresh from a newer on-disk tree must resolve
these without going through the ``cron.jobs`` object a long-lived process cached at boot.
"""

# A fire_claim younger than this is a live run (heartbeat cadence is 60 s). One value
# for claiming, one-shot re-arm, and stale-error recovery so they cannot disagree.
FIRE_CLAIM_TTL_SECONDS = 300
# A hosted/webhook fire for the armed slot can arrive a few seconds before the stored
# ``next_run_at`` (the fire scheduler's clock runs ahead of ours). Claims that early still own
# the slot; only claims further ahead are off-tick manual/dashboard fires.
FIRE_CLAIM_SKEW_SECONDS = 60
# Multiplier over HERMES_CRON_TIMEOUT for claim TTLs: the timeout is an *inactivity* limit, not a
# wall-clock cap, so healthy runs may legitimately exceed it and a TTL of exactly the timeout
# would expire live claims.
CLAIM_TTL_INACTIVITY_HEADROOM = 3

# Persisted fields the operator AUTHORS (create_job / profile distributions). Cron owns the split:
# importers merging an authored store into a live one refresh exactly these (cron/job_definition.py),
# and the job store keeps them in jobs.json while everything the scheduler writes goes to
# cron/runtime.db (cron/jobs.py::_is_runtime_field).
JOB_DEFINITION_FIELDS = frozenset({
    "name", "prompt", "skills", "skill", "model", "provider", "base_url",
    "script", "no_agent", "monitor_script", "monitor_url", "context_from",
    "schedule", "schedule_display", "deliver", "origin", "enabled_toolsets",
    "workdir", "attach_to_session", "reasoning_effort", "failure_deliver",
})
# Also kept in jobs.json, though profile distributions do not refresh them: identity and operator
# intent (``enabled`` flips only on pause/resume/trigger and terminal completion, never per fire).
DECLARATIVE_JOB_EXTRAS = frozenset({"id", "enabled", "created_at"})

def is_recurring(job: dict) -> bool:
    """A cron/interval job (vs a one-shot); a null ``schedule`` counts as not recurring."""
    return (job.get("schedule") or {}).get("kind") in {"cron", "interval"}
