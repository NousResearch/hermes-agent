import json, sys
from pathlib import Path
from cron import executions as ex, delivery_queue as dq, jobs
home=Path(sys.argv[1]);eid=sys.argv[2]
ex.EXECUTIONS_FILE=home/'cron'/'executions.db';dq.DELIVERY_DB=home/'cron'/'deliveries.db'
ex.reconcile_delivery_projections()
print(json.dumps({'execution':ex.get_execution(eid),'job':jobs.get_job('child-job'),'journal':ex.journaled_manifests()}))
