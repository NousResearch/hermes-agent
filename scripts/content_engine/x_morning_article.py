#!/usr/bin/env python3
"""Approval-only morning X Article source adapter.

Collects a bounded set of verified internal/research signals, drafts 1-2
finished text-first X Articles grounded in a specific claim + evidence via
the LLM, stages them as pending through x_manager, and renders approval
cards for #x-twitter-manager.

Never publishes, never wires to blog management, no image generation.
"""
import json
import os
import sys
from pathlib import Path

CE = Path('/home/kensei/repos/KenseiAgent/content_engine')
sys.path.insert(0, str(CE))

from llm_generate import _call_llm_chain
import x_manager as xm
from x_manager_report import render_report

# Bounded, deterministic source set. Each is an evidence-bearing internal
# signal the router already treats as verified/claim-based.
SIGNAL_SOURCES = [
    ("harness_change", 8),
    ("research_signal", 7),
    ("research_tool", 6),
    ("github_push", 6),
]
MAX_ARTICLE_WORDS = 1400
STATE_FILE = Path('/home/kensei/repos/KenseiAgent/content_engine/data/x_article_last_run.json')
VOICE_SKILL = Path('/home/kensei/.hermes/skills/social-media/sahil-twitter-voice/SKILL.md')

ARTICLE_SYSTEM = (
    "You write one finished, text-first X Article for Sahil in the same "
    "opinionated, personal and conversational voice he uses on X. It must "
    "interrogate a concrete claim/artefact, test its real user value, "
    "assumptions, limitations, and application to the reader's own workflow. "
    "It may challenge hype or constructively explain a useful idea, but it "
    "must make the reader think. A practical test is welcome when natural, "
    "but do not force a consultant-style framework or adoption checklist. Reply "
    "with exactly one JSON object: {\"title\": \"...\", \"claim\": \"...\", "
    "\"evidence\": \"...\", \"mechanism\": \"...\", \"weak_practice\": \"...\", "
    "\"position\": \"...\", \"body\": \"...\"}. body is the full markdown "
    "article (no credentials, no image generation, no invented facts)."
)


def _runtime_voice() -> str:
    try:
        text = VOICE_SKILL.read_text()
        marker = '## Corpus-Calibrated Runtime Voice'
        start = text.index(marker)
        return text[start:start + 7000]
    except Exception:
        return ''


def _load_env():
    env = Path('/home/kensei/.hermes/.env')
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())


def _collect_signals() -> list:
    try:
        import activity_collector as ac
        return ac.collect_all().get('signals', [])
    except Exception as exc:
        print(f"[x-morning-article] collector unavailable: {exc}")
        # Fallback: local, non-sensitive seed signals from RecentChanges only.
        return [{
            'signal_id': f'seed-{i}', 'summary': s['summary'],
            'signal_type': s['type'], 'priority': s['priority'],
        } for i, s in enumerate([
            {'summary': 'Design-driven velocity now exceeds problem-validation depth in AI product teams; fast-looking output is outpacing quality gates.',
             'type': 'research_signal', 'priority': 7},
        ])]


def _valid(signal: dict) -> bool:
    return bool((signal.get('summary') or '').strip()) and len((signal.get('summary') or '').strip()) >= 60


def _draft_article(signal: dict) -> dict:
    user = f"Source signal ({signal.get('signal_type','')}):\n{(signal.get('summary') or '')[:2000]}"
    system = ARTICLE_SYSTEM + "\n\n" + _runtime_voice()
    out = _call_llm_chain(system, user, timeout=120, max_tokens=4000)
    if not out:
        raise RuntimeError('empty LLM output')
    text = out.strip()
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        raise RuntimeError('no JSON object in LLM output')
    return json.loads(text[start:end + 1])


def _build_artifact(signal: dict) -> xm.XArtifact:
    data = _draft_article(signal)
    pack = xm.ArgumentPack(
        claim=(data.get('claim') or '').strip(),
        evidence=(data.get('evidence') or '').strip(),
        mechanism=(data.get('mechanism') or '').strip(),
        position=(data.get('position') or '').strip(),
        context={'signal_id': signal.get('signal_id', ''), 'signal_type': signal.get('signal_type', ''),
                 'source': (signal.get('summary') or '')[:500]},
    )
    body = (data.get('body') or '').strip()
    if not pack.is_complete() or not body:
        raise RuntimeError('incomplete article pack')
    return xm.XArtifact(id=xm._new_id(xm.LANE_ARTICLE), lane=xm.LANE_ARTICLE,
                        brand='sahil_twitter', body=body, pack=pack)



def _already_reported(artifacts) -> bool:
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        return False
    return any(a.id in state.get('ids', []) for a in artifacts)


def _record_reported(artifacts) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE_FILE.read_text())
    except Exception:
        state = {'ids': []}
    ids = [a.id for a in artifacts]
    state.setdefault('ids', [])
    state['ids'] = list(dict.fromkeys(state['ids'] + ids))[-20:]
    STATE_FILE.write_text(json.dumps(state))


def main():
    _load_env()
    signals = [s for s in _collect_signals() if _valid(s)]
    signals.sort(key=lambda s: -s.get('priority', 0))
    artifacts = []
    for sig in signals[:4]:
        try:
            artifacts.append(_build_artifact(sig))
        except Exception as exc:
            print(f"[x-morning-article] draft skip: {exc}", file=sys.stderr)
        if len(artifacts) >= 2:
            break
    if not artifacts:
        return
    if _already_reported(artifacts):
        return
    staged = []
    for art in artifacts:
        try:
            xm.stage_for_approval(art)
            staged.append(art)
        except Exception as exc:
            print(f"[x-morning-article] stage failed: {exc}", file=sys.stderr)
    if not staged:
        return
    report = render_report(staged, lane='morning-articles', title='Morning X Article recommendations')
    _record_reported(staged)
    print(f"X Manager · {len(staged)} morning Article recommendation{'s' if len(staged) != 1 else ''}")
    print("Source evidence and full drafts are in the attached review.")
    print(f"MEDIA:{report}")


if __name__ == "__main__":
    main()
