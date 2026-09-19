"""Fire-claim timing bounds shared by the job store and its siblings.

Import-free on purpose: a sibling loaded fresh from a newer on-disk tree must resolve
these without going through the ``cron.jobs`` object a long-lived process cached at boot.
"""

# A fire_claim younger than this is a live run. The heartbeat thread's nominal cadence is 60 s,
# but it refreshes through the shared jobs store, so cross-process write contention can stall
# the refresh for minutes while the run is genuinely alive — at 300 s the next tick re-claims
# and interrupts actively-working jobs (#116136). One value for claiming, one-shot re-arm, and
# stale-error recovery so they cannot disagree. That coupling is also the worst-case cost: a
# run that died without reclaiming holds its slot up to 600 s (was 300 s) before stale-error
# recovery or one-shot re-arm can take it back.
FIRE_CLAIM_TTL_SECONDS = 600
# A hosted/webhook fire for the armed slot can arrive a few seconds before the stored
# ``next_run_at`` (the fire scheduler's clock runs ahead of ours). Claims that early still own
# the slot; only claims further ahead are off-tick manual/dashboard fires.
FIRE_CLAIM_SKEW_SECONDS = 60
