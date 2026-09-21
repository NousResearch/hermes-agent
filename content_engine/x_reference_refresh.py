"""Refresh living own-post references and metrics; no publishing capability."""
from pathlib import Path
import json
import os
import tempfile
import fcntl
from hermes_constants import get_hermes_home


def refresh_references(*, cookies_path=None, collector=None):
    from x_analytics_browser import collect_own_analytics
    from x_analytics import validate, merge_observations
    collector = collector or collect_own_analytics
    fresh = collector('Sahil_Saghir', cookies_path=Path(cookies_path or Path.home()/'.x-browser-state/cookies.json'),
                      max_posts=40,max_scrolls=5,timeout_seconds=90,include_replies=True)
    fresh = validate(fresh)
    if fresh['account'] != 'Sahil_Saghir':
        raise ValueError('wrong owner account')
    target = Path(os.environ.get('X_OBSERVATIONS_PATH', str(get_hermes_home()/'data/x-analytics/observations.json')))
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent/'refresh.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if target.exists():
            old=validate(json.loads(target.read_text()))
            fresh=merge_observations(old,fresh)
        fd,name=tempfile.mkstemp(prefix='.observations-',suffix='.json',dir=target.parent)
        try:
            with os.fdopen(fd,'w') as f:
                json.dump(fresh,f,ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(name,target)
        finally:
            if Path(name).exists(): Path(name).unlink()
    return {'path':str(target),'posts':len(fresh['posts']),'capabilities':fresh.get('capabilities',{})}
