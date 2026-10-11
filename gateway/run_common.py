"""Leaf constants shared by ``gateway/run.py`` and its ``run_*`` mixin modules.

Kept import-cycle free (imports nothing from ``gateway.run``) because these values
are used as default-argument sentinels, which must resolve at ``def`` time.
"""

# Sentinel for "caller did not pass metadata" vs "caller passed None".
_UNSET = object()
