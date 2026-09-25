"""Fire-claim timing bounds shared by the job store and its siblings.

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
