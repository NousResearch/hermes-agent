"""Bounded local retrieval; evidence stays data, never generation instructions.

Approved blog writing supports a stated view, not proof of implementation.
Implementation receipts require separate public-use approval and byte integrity.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from x_voice_gate import _tokens


def runtime_grounding(query):
    import os
    from hermes_constants import get_hermes_home
    from x_voice_gate import load_voice_corpus
    from x_knowledge import repository_knowledge, memory_knowledge
    home = get_hermes_home()
    result = retrieve_grounding(query, own_posts=load_voice_corpus(limit=1000),
        blog_dir=os.environ.get('SAHILBLOG_CONTENT_DIR', str(Path.home()/'repos/SahilBlog/src/content/blog')),
        experience_path=home/'research/x-voice/public-implementation-evidence.json')
    # Private repositories are authorised internal sources. Their metadata never
    # establishes permission to expose private implementation details publicly.
    roots = json.loads(os.environ.get('X_KNOWLEDGE_ROOTS_JSON', json.dumps([str(Path.home()/'repos'), str(Path.home()/'worktrees')])) )
    result['repositories'] = repository_knowledge(query, roots)
    from x_github_knowledge import remote_knowledge
    remote = remote_knowledge(query, get_hermes_home()/'research/x-voice/github-cache')
    result['repositories'].extend(remote['references'])
    result['github_coverage'] = remote['coverage']
    result['memory'] = memory_knowledge(query, home/'research/x-voice/memory-hints.json')
    result['coverage']['repositories'] = 'matched' if result['repositories'] else 'no_relevant_matches'
    result['coverage']['memory'] = 'matched' if result['memory'] else 'no_relevant_matches'
    return result


def retrieve_grounding(query, *, own_posts, blog_dir, experience_path, limit=3):
    wanted = _tokens(query)
    def rank(items):
        scored = [(len(wanted & _tokens(str(i.get('text', '')))), i) for i in items]
        return [i for score, i in sorted(scored, key=lambda pair: -pair[0]) if score >= 2][:limit]
    own = rank([i for i in own_posts if i.get('approved') is True])
    blogs = []
    directory = Path(blog_dir)
    for path in sorted(directory.glob('*.mdx'))[:2000]:
        if path.is_symlink() or path.stat().st_size > 200_000:
            continue
        raw = path.read_bytes()
        parts = raw.decode('utf-8').split('---', 2)
        if len(parts) != 3 or parts[0].strip() or not re.search(r'^approved:\s*true\s*$', parts[1], re.M):
            continue
        paragraphs = [p.strip() for p in parts[2].split('\n\n') if p.strip()]
        selected = sorted(paragraphs, key=lambda p: -len(wanted & _tokens(p)))[:3]
        blogs.append({'text': '\n\n'.join(selected)[:5000],
                      'provenance': {'kind': 'approved_blog_file', 'path': str(path),
                                     'sha256': hashlib.sha256(raw).hexdigest()},
                      'usage': 'Author viewpoint only; not implementation proof'})
    experiences = []
    path = Path(experience_path)
    status = 'unavailable'
    if path.is_file() and path.stat().st_size <= 2_000_000:
        try:
            records = json.loads(path.read_text())
            if not isinstance(records, list):
                raise ValueError('experience manifest must be a list')
            status = 'available'
            for item in records:
                if not isinstance(item, dict) or item.get('verified') is not True or item.get('public_approved') is not True:
                    continue
                receipt = Path(str(item.get('receipt_path', '')))
                if not receipt.is_file() or receipt.is_symlink() or receipt.stat().st_size > 2_000_000:
                    continue
                raw = receipt.read_bytes()
                if hashlib.sha256(raw).hexdigest() != item.get('sha256'):
                    continue
                # A hash cannot establish semantic entailment. Require the supplied
                # public excerpt to occur literally in the verified receipt.
                text = item.get('text')
                if not isinstance(text, str) or not text.strip() or text not in raw.decode('utf-8'):
                    continue
                experiences.append({'text': text[:5000], 'provenance': {'kind':'verified_public_excerpt',
                                    'path':str(receipt),'sha256':item['sha256']}})
        except (ValueError, OSError, UnicodeError):
            status = 'invalid'
    return {'own_posts': own, 'blog': rank(blogs), 'implementation':rank(experiences),
            'coverage': {'own_posts': 'available' if own_posts else 'unavailable',
                         'blog': 'available' if blogs else 'unavailable', 'implementation':status}}
