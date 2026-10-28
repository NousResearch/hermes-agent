"""Executed with the pre-authority checkout on PYTHONPATH, not the new writer."""
import sys
from pathlib import Path
from tools.bot_live_delivery import deliver_to_live_owner, claim_pending_delivery

home, sid = Path(sys.argv[1]), sys.argv[2]
owner = dict(profile_home=str(home), session_id=sid, lease_id='departed-owner', live_session_id='old-native')
deliver_to_live_owner(home, owner, 'LEGACY_CLAIMED_NEVER', delivery_id='c' * 32)
assert claim_pending_delivery(home, owner)['delivery_id'] == 'c' * 32
deliver_to_live_owner(home, owner, 'LEGACY_QUEUED_ONCE', delivery_id='b' * 32)
foreign = dict(owner, session_id='unrelated-history')
deliver_to_live_owner(home, foreign, 'UNRELATED_NEVER', delivery_id='d' * 32)
assert claim_pending_delivery(home, foreign)['delivery_id'] == 'd' * 32
