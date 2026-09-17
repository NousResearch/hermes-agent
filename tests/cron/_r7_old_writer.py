"""Load exact R4 fixture before upgrade; publish only on parent signal."""
import json, sys, time
from pathlib import Path
from tests.cron.test_r5_salt_legacy_and_races import prior, receipt
from cron import delivery_queue as dq
home=Path(sys.argv[1]);eid=sys.argv[2];barrier=Path(sys.argv[3])
old=prior();old.EXECUTIONS_FILE=home/'cron'/'executions.db'
dq.DELIVERY_DB=home/'cron'/'deliveries.db'
assert old.get_execution(eid)['delivery_manifest']=='{"external": true}'
assert dq.claim_next()['execution_id']==eid
(barrier/'ready').write_text('loaded old module')
deadline=time.monotonic()+20
while not (barrier/'go').exists():
    if time.monotonic()>deadline: raise TimeoutError('writer release')
    time.sleep(.01)
manifest=receipt(home,eid)
old.journal_manifest(eid,manifest)
assert dq._finish(eid,error=None)
print(json.dumps({'manifest':manifest,'external':dq.get_status(eid)}))
