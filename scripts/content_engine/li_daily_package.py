#!/usr/bin/env python3
"""Deterministic, approval-only LinkedIn daily package adapter."""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from pathlib import Path

CE = Path('/home/kensei/repos/KenseiAgent/content_engine')
BLOG = Path('/home/kensei/repos/SahilBlog/src/content/blog')
VOICE = Path('/home/kensei/.hermes/skills/social-media/sahil-linkedin-voice/SKILL.md')
STATE = CE / 'data' / 'li_daily_last_run.json'
sys.path.insert(0, str(CE))

from llm_generate import _call_llm_chain
import li_manager as li
from li_manager_report import render_report

SYSTEM = """Write one approval-only LinkedIn draft for Sahil. Return exactly one JSON object with keys body, claim, evidence, mechanism, position. Ground every claim only in the supplied source. Never invent first-hand experience, metrics or employer details. No em-dashes, engagement bait, generic AI slogans or corporate boilerplate. Write as a credible, opinionated Senior Product Manager who builds real AI systems. The body must be native LinkedIn prose, not a blog excerpt."""


def _load_env() -> None:
    path = Path('/home/kensei/.hermes/.env')
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())


def _frontmatter(text: str) -> dict:
    if not text.startswith('---'):
        return {}
    block = text.split('---', 2)[1]
    out = {}
    for line in block.splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        out[key.strip()] = value.strip().strip('"\'')
    return out


def collect_pm_sources() -> list[dict]:
    rows = []
    for path in BLOG.glob('*.mdx'):
        text = path.read_text(errors='replace')
        meta = _frontmatter(text)
        if meta.get('approved', '').lower() != 'true' or meta.get('tier') != 'pm':
            continue
        body = text.split('---', 2)[-1].strip()
        rows.append({
            'slug': path.stem,
            'title': meta.get('title') or path.stem.replace('-', ' ').title(),
            'description': meta.get('description') or '',
            'pubDate': meta.get('pubDate') or '',
            'body': re.sub(r'!\[[^]]*\]\([^)]*\)', '', body)[:6000],
            'url': f'https://algorithmiccompass.com/blog/{path.stem}',
        })
    rows.sort(key=lambda row: (row['pubDate'], row['slug']), reverse=True)
    return rows


def _voice() -> str:
    try:
        text = VOICE.read_text()
        # The voice skill is intentionally bounded to avoid crowding out source evidence.
        return text[:24000]
    except Exception:
        return ''


def _json_call(instruction: str, source: dict) -> dict:
    user = (
        f"{instruction}\n\nSOURCE TITLE: {source['title']}\n"
        f"SOURCE URL: {source['url']}\nSOURCE DESCRIPTION: {source['description']}\n"
        f"SOURCE BODY:\n{source['body']}"
    )
    output = _call_llm_chain(SYSTEM + '\n\n' + _voice(), user, timeout=120, max_tokens=4000)
    if not output:
        raise RuntimeError('empty LLM output')
    start, end = output.find('{'), output.rfind('}')
    if start < 0 or end <= start:
        raise RuntimeError('no JSON object in LLM output')
    return json.loads(output[start:end + 1])


def _pack(data: dict, source: dict) -> li.ArgumentPack:
    return li.ArgumentPack(
        claim=(data.get('claim') or '').strip(),
        evidence=(data.get('evidence') or '').strip(),
        mechanism=(data.get('mechanism') or '').strip(),
        position=(data.get('position') or '').strip(),
        context={
            'source_text': source['description'] or source['body'][:1000],
            'source_url': source['url'],
            'author': 'Algorithmic Compass',
            'source_slug': source['slug'],
        },
    )


def build_package(sources: list[dict]) -> li.DailyPackage:
    if len(sources) < 3:
        raise RuntimeError('fewer than three approved PM sources')
    primary, source_a, source_b = sources[:3]
    insight_data = _json_call(
        "Write a 40-100 word opinion-led summary of this PM Insight. Include the supplied direct URL in the body. Do not use hashtags.",
        primary,
    )
    insight = li.LiArtifact(
        id=li._new_id(li.KIND_PM_INSIGHT), lane=li.LANE_DAILY_PACKAGE,
        kind=li.KIND_PM_INSIGHT, body=(insight_data.get('body') or '').strip(),
        pack=_pack(insight_data, primary), source_url=primary['url'],
    )
    posts = []
    for source in (source_a, source_b):
        data = _json_call(
            "Write one independent 120-220 word evidence-backed AI/Product Management post. Explain a specific mechanism and land a clear position. Use 3-5 semantically relevant Title Case hashtags at the end. Do not mention or link the source in the body.",
            source,
        )
        posts.append(li.LiArtifact(
            id=li._new_id(li.KIND_AI_PM_POST), lane=li.LANE_DAILY_PACKAGE,
            kind=li.KIND_AI_PM_POST, body=(data.get('body') or '').strip(),
            pack=_pack(data, source), source_url=source['url'],
        ))
    return li.build_daily_package(insight, posts)


def _same_sources(sources: list[dict]) -> bool:
    try:
        state = json.loads(STATE.read_text())
    except Exception:
        return False
    return state.get('slugs') == [row['slug'] for row in sources[:3]]


def main() -> None:
    _load_env()
    sources = collect_pm_sources()
    if len(sources) < 3 or _same_sources(sources):
        return
    try:
        package = build_package(sources)
    except Exception as exc:
        print(f'[li-manager] package failed: {exc}', file=sys.stderr)
        return
    artifacts = [package.insight, *package.posts]
    for artifact in artifacts:
        li.stage_for_approval(artifact)
    report = render_report(artifacts, lane='daily-package', title='Daily LinkedIn recommendations')
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({'date': date.today().isoformat(), 'slugs': [row['slug'] for row in sources[:3]]}))
    print('LinkedIn Manager · 3 recommendations')
    print('One PM Insight summary and two independent AI/PM drafts are in the attached review.')
    print(f'MEDIA:{report}')


if __name__ == '__main__':
    main()
