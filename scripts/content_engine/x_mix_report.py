"""Read-only weekly staging mix. This is not published-post growth analytics."""
import html
import json
import sqlite3
import sys
import uuid
from datetime import datetime,timedelta,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'content_engine'))
from config import DB_PATH
from x_analytics import load_growth_section
from hermes_constants import get_hermes_home


def render_mix():
    counts=[]
    if DB_PATH.exists():
        with sqlite3.connect(f'file:{DB_PATH}?mode=ro',uri=True) as conn:
            exists=conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='x_manager_artifacts'").fetchone()
            if exists:
                counts=conn.execute('SELECT lane,status,COUNT(*) FROM x_manager_artifacts WHERE created_at>=? GROUP BY lane,status',((datetime.now(timezone.utc)-timedelta(days=7)).isoformat(),)).fetchall()
    report_dir=get_hermes_home()/'document_cache'/'x-manager'
    report_dir.mkdir(parents=True,exist_ok=True)
    path=report_dir/f'mix-{uuid.uuid4().hex}.html'
    path.write_text('<!doctype html><meta charset="utf-8"><h1>X staging mix</h1><p>Last seven days. Pending, approved and rejected drafts are not published posts; these counts do not measure reach or growth. No activity is a valid result. No fixed ratio is a ranking promise.</p><pre>'+html.escape(json.dumps(counts,indent=2))+'</pre>'+load_growth_section(get_hermes_home()/'data'/'x-analytics'/'observations.json'))
    return str(path)


if __name__=='__main__':
    print('X Manager · weekly staging mix (not growth metrics)')
    print('MEDIA:'+render_mix())
