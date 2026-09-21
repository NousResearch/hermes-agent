"""Relevant internal knowledge, not permission to publish repository details."""
from pathlib import Path
import json
import re
import subprocess
from x_voice_gate import _tokens

_SECRET = re.compile(r'(?i)(?:api[_ -]?key|password|secret|bearer|token\s*[:=]|https?://[^\s/]+:[^\s@]+@|-----BEGIN .*PRIVATE KEY)')


def safe_excerpt(text):
    text = str(text)
    if _SECRET.search(text) or re.search(r'(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})', text):
        return ''
    return text[:1600]


def repository_knowledge(query, roots, *, per_repo=15, limit=8):
    wanted = _tokens(query)
    scored = []
    seen = set()
    for root in roots:
        root = Path(root).expanduser()
        # Bounded discovery includes grouping directories; no arbitrary deep crawl.
        candidates = [root] + sorted(root.glob('*')) + sorted(root.glob('*/*')) if root.is_dir() else []
        for repo in candidates:
            if not repo.is_dir() or not (repo/'.git').exists() or repo.resolve() in seen:
                continue
            seen.add(repo.resolve())
            try:
                out = subprocess.run(['git','-C',str(repo),'log',f'-{per_repo}','--format=%H%x09%cI%x09%s'],capture_output=True,text=True,timeout=5,check=True)
            except (OSError, subprocess.SubprocessError):
                continue
            for line in out.stdout.splitlines():
                fields = line.split('\t',2)
                if len(fields) != 3:
                    continue
                sha, at, subject = fields
                text = safe_excerpt(subject)
                score = len(wanted & _tokens(text))
                if score < 2 or not text:
                    continue
                scored.append((score, {'text':text,'provenance':{'kind':'git_commit','repository':str(repo),'commit':sha,'committed_at':at},
                    'status':'committed; not proof of tests, deployment or measured benefit',
                    'disclosure':'internal_only; human review before mentioning private details'}))
    scored.sort(key=lambda x:-x[0])
    return [item for _,item in scored[:limit]]


def memory_knowledge(query, path, *, limit=3):
    """Curated memory hints are not independent implementation evidence."""
    path = Path(path)
    if not path.is_file():
        # Native stable memory remains available when no curated retrieval cache
        # exists. Treat it as fallible internal hints, never verified outcomes.
        import hashlib
        notes = path.parents[2] / 'memories/MEMORY.md'
        if not notes.is_file() or notes.stat().st_size > 1_000_000:
            return []
        raw = notes.read_text()
        items = [{'memory_id':'file:' + hashlib.sha256(part.encode()).hexdigest(), 'text':part}
                 for part in re.split(r'\n\n|§', raw) if part.strip()]
    else:
        if path.stat().st_size > 1_000_000:
            return []
        try:
            items=json.loads(path.read_text())
            if not isinstance(items,list): return []
        except (OSError,ValueError):
            return []
    wanted=_tokens(query)
    found=[]
    for item in items:
        if not isinstance(item,dict) or not item.get('memory_id'): continue
        text=safe_excerpt(item.get('text',''))
        score=len(wanted&_tokens(text))
        if score>=2:
            found.append((score,{'text':text,'provenance':{'kind':'memory_hint','memory_id':item['memory_id']},
                                'status':'unverified memory; corroborate before autobiographical claim',
                                'disclosure':'internal_only'}))
    return [item for _,item in sorted(found,key=lambda p:-p[0])[:limit]]
