"""Called by actual outer job script; inner executes through manual-run API."""
import json, os, sys
from pathlib import Path
REPO=str(next(p for p in Path(__file__).resolve().parents if (p/'cron'/'executions.py').exists()))
sys.path.insert(0,REPO)
os.environ['PYTHONPATH']=REPO
from cron import jobs, executions
assert Path(executions.__file__).is_relative_to(REPO), executions.__file__
from tools.cronjob_tools import _execute_job_now
home=Path(os.environ['HERMES_HOME'])
inner=jobs.create_job(prompt=None,schedule='every 1h',script='fail.sh',no_agent=True,deliver='bot-chat')
with executions._transaction() as conn:
    conn.execute('''CREATE TRIGGER nested_reject_manifest BEFORE UPDATE OF delivery_manifest ON executions
    WHEN NEW.delivery_manifest LIKE '%"bot"%' BEGIN SELECT RAISE(ABORT,'nested manifest fault'); END''')
result=_execute_job_now(inner)
(home/'inner-proof.json').write_text(json.dumps({'outer_env':os.environ.get('_HERMES_CRON_EXTERNAL_WORKER'),'inner':executions.latest_execution(inner['id']),'result':result}))
