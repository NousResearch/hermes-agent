"""Independent initializer process, synchronization only at inventory boundary."""
import json, os, sys, time
from pathlib import Path
from cron import executions as ex
home, barrier, name = map(Path, sys.argv[1:])
ex.EXECUTIONS_FILE=home/'cron'/'executions.db'
real_inventory=ex._journal_inventory_strict
real_adopt=ex._try_adopt_legacy_intent
trace=[]; entry=[]
def inventory():
    result=real_inventory()
    (barrier/(str(name)+'.ready')).write_text('ready')
    deadline=time.monotonic()+20
    while not (barrier/'go').exists():
        if time.monotonic()>deadline: raise TimeoutError('barrier')
        time.sleep(.01)
    return result
def adopt(conn):
    entry.append({'in_transaction':conn.in_transaction,'isolation_level':conn.isolation_level,'journal_mode':conn.execute('PRAGMA journal_mode').fetchone()[0]})
    conn.set_trace_callback(trace.append)
    return real_adopt(conn)
ex._journal_inventory_strict=inventory
ex._try_adopt_legacy_intent=adopt
rows=ex.list_executions()
print(json.dumps({'name':str(name),'entry':entry,'trace':trace,'pending':[r['delivery_manifest_pending'] for r in rows],'adopted':ex._legacy_intent_adopted()}))
