"""Fresh-process bookkeeping only; never sends."""
import sys
from cron import executions
if int(sys.argv[1]):
    original=executions._store_manifest
    def failed(conn,eid,manifest):
        if 'bot' in manifest:raise OSError('injected persistent SQLite manifest write failure after restart')
        return original(conn,eid,manifest)
    executions._store_manifest=failed
for _ in range(int(sys.argv[2])):executions.reconcile_delivery_projections()
print('RECONCILED')
