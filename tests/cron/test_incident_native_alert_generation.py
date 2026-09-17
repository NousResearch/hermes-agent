"""Salt R3 S4: an immediate native alert for occurrence N must not mark a reopened occurrence N+1."""
import json
import os
import subprocess
import sys
from cron import incidents, executions
from tests.cron.test_mixed_delivery_retention import setup


def test_direct_native_old_output_does_not_alert_reopened_occurrence(setup,monkeypatch):
    home,cfg,run,sent=setup
    original=incidents.upsert_incident
    hits=[]
    def after_commit(*args,**kw):
        result=original(*args,**kw)
        iid=result[0]
        hits.append(iid)
        subprocess.run([sys.executable,'-c', '''
from cron import incidents
import sys
i=incidents.get_incident(sys.argv[1])
assert incidents.close_incidents_for_recovered_job(i['job_id'])==1
incidents.upsert_incident(i['job_id'],i['error'])
''',iid],env={**os.environ,'PYTHONPATH':os.getcwd()},check=True,timeout=30)
        return result
    monkeypatch.setattr(incidents,'upsert_incident',after_commit)
    job,execution=run(False,deliver='telegram:test')
    assert len(hits)==1 and len(sent)==1
    i=incidents.get_incident(execution['incident_id'])
    assert execution['incident_generation']==1 and i['generation']==2
    print('DIRECT_SIBLING',json.dumps({'execution':execution,'incident':i,'wire_count':len(sent)}))
    assert i['state']=='detected'
