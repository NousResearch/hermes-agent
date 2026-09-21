"""Read-only, bounded GitHub grounding across the authenticated owner's repos."""
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from x_knowledge import safe_excerpt
from x_voice_gate import _tokens


def _api(endpoint, *, paginate=False):
    command=['gh','api',endpoint]
    if paginate: command += ['--paginate','--slurp']
    result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=True)
    if len(result.stdout)>20_000_000: raise ValueError('GitHub response too large')
    return json.loads(result.stdout)


def _cached(path, ttl, fetch):
    now=time.time()
    if path.is_file() and path.stat().st_size<3_000_000:
        try:
            saved=json.loads(path.read_text())
            if 0 <= now-float(saved['observed_at']) < ttl:
                return saved['data'],saved['observed_at']
        except (ValueError,KeyError,TypeError,OSError): pass
    data=fetch()
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.github-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump({'observed_at':now,'data':data},f)
        os.replace(name,path)
    finally:
        if Path(name).exists():Path(name).unlink()
    return data,now


def remote_knowledge(query, cache_dir, *, limit=3, api=None):
    if os.environ.get('X_GITHUB_KNOWLEDGE')=='0':
        return {'references':[],'coverage':{'status':'disabled'}}
    api=api or _api
    cache=Path(cache_dir)
    def catalog():
        pages=api('user/repos?per_page=100&affiliation=owner',paginate=True)
        if not isinstance(pages,list) or not all(isinstance(p,list) for p in pages):
            raise ValueError('invalid paginated catalog')
        indexed={}
        for page in pages:
            for item in page:
                name=item.get('full_name','')
                if re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',name):
                    indexed[name]={'name':name,'private':bool(item.get('private')),
                                   'description':safe_excerpt(str(item.get('description') or ''))}
        return list(indexed.values())
    try:
        repos,observed=_cached(cache/'catalog.json',86400,catalog)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
        return {'references':[],'coverage':{'status':'unavailable'}}
    terms=_tokens(query)
    ranked=sorted(repos,key=lambda r:len(terms&_tokens(r['name']+' '+r['description'])),reverse=True)
    selected=[r for r in ranked if terms&_tokens(r['name']+' '+r['description'])][:limit]
    references=[];unavailable=[]
    for repo in selected:
        name=repo['name']
        def commits():
            rows=api(f'repos/{name}/commits?per_page=30')
            if not isinstance(rows,list):raise ValueError('invalid commits')
            return [{'sha':r.get('sha',''),'text':safe_excerpt(str(r.get('commit',{}).get('message','')).split('\n')[0]),
                     'date':r.get('commit',{}).get('committer',{}).get('date')}
                    for r in rows if re.fullmatch('[0-9a-f]{40}',str(r.get('sha','')))]
        try:
            rows,commit_observed=_cached(cache/(name.replace('/','--')+'.json'),3600,commits)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
            unavailable.append(name);continue
        ranked_commits=sorted(rows,key=lambda r:len(terms&_tokens(r['text'])),reverse=True)
        matching=[r for r in ranked_commits if len(terms&_tokens(r['text'])) >= 2 and r['text']][:2]
        for commit in matching:
            references.append({'kind':'repository_commit','repository':name,'commit':commit['sha'],
                'url':f'https://github.com/{name}/commit/{commit["sha"]}',
                'text':commit['text'],'committed_at':commit['date'],'observed_at':commit_observed,
                'private':repo['private'],'disclosure':'internal_only','deployment_status':'unknown'})
    return {'references':references,'coverage':{'status':'partial' if unavailable else 'available',
        'repository_count':len(repos),'private':sum(r['private'] for r in repos),
        'public':sum(not r['private'] for r in repos),'catalog_observed_at':observed,
        'selected':len(selected),'unavailable_repositories':unavailable}}
