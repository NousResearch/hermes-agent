"""Fire-claim timing bounds shared by the job store and its siblings.

Kept in a dependency-free leaf on purpose: ``cron/occurrences.py`` is loaded
fresh from disk by long-lived processes that booted on an older tree, and its
constants must resolve without going through the ``cron.jobs`` facade already
cached in ``sys.modules`` from that older boot. A facade import there mixes
code generations in one interpreter and turns a routine in-place update into
a silent skip of every due job.
"""

# A fire_claim younger than this is a live run (heartbeat cadence is 60 s). One value
# for claiming, one-shot re-arm, and stale-error recovery so they cannot disagree.
FIRE_CLAIM_TTL_SECONDS = 300
# A hosted/webhook fire for the armed slot can arrive a few seconds before the stored
# ``next_run_at`` (the fire scheduler's clock runs ahead of ours). Claims that early still own
# the slot; only claims further ahead are off-tick manual/dashboard fires.
FIRE_CLAIM_SKEW_SECONDS = 60
