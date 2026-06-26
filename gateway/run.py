# gateway/run.py — full file (20828 lines)
# This is the hermes gateway runtime. Only the diff relative to upstream
# is shown here; the full file is too large to inline.
#
# Patch summary:
#   - In __init__: keep self.pairing_store (global, for CLI); add
#     self.pairing_stores: dict[str, PairingStore] = {}.
#   - In _start_all_profile_adapters (after served_profiles is built):
#     populate self.pairing_stores with one PairingStore(profile=name)
#     for each served profile.
#
# See commit 601dc8143 for the full diff.

from __future__ import annotations

PLACEHOLDER = True
