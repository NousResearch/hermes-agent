"""Evidence subprocess: real old journal writer / current replay / fresh read."""
import json, os, sys, time
from pathlib import Path
from cron import executions as ex, delivery_queue as dq
from tests.cron.test_r5_salt_legacy_and_races import prior, receipt, snap
home=Path(sys.argv[2]); eid=sys.argv[3]
ex.EXECUTIONS_FILE=home/'cron'/'executions.db'
dq.DELIVERY_DB=home/'cron'/'deliveries.db'
mode=sys.argv[1]
if mode=='replay':
    ex._recover_unrecorded_manifests()
    print(json.dumps(snap(eid)))
elif mode=='read':
    ex.reconcile_delivery_projections()
    print(json.dumps(snap(eid)))
elif mode=='writer':
    barrier=Path(sys.argv[4]); old=prior();old.EXECUTIONS_FILE=ex.EXECUTIONS_FILE
    assert old.get_execution(eid)
    assert dq.claim_next()['execution_id']==eid
    (barrier/'ready').write_text('old module loaded, external claimed')
    deadline=time.monotonic()+25
    while not (barrier/'publish').exists():
        assert time.monotonic()<deadline
        time.sleep(.005)
    manifest=receipt(home,eid)
    old.journal_manifest(eid,manifest)
    assert (old._journal_root()/f'{eid}.json').is_file()
    (barrier/'published').write_text(json.dumps(manifest))
    while not (barrier/'complete').exists():
        assert time.monotonic()<deadline
        time.sleep(.005)
    assert dq._finish(eid,error=None)
    print(json.dumps({'manifest':manifest,'external':dq.get_status(eid)}))
else: raise ValueError(mode)
