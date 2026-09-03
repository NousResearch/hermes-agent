#!/usr/bin/env python3
"""Daily X thesis incubator — approval-only original-thesis drafts.

Runs every morning (07:30) after the article lane. Draws from two evidence
sources:
  1. The experience inbox (/x captures in #x-twitter-manager)
  2. Standalone seeds accumulated by the quote scout verdict layer

Emits 1-2 deliberately original thesis posts (the identity-building lane)
with a proposed post, why-it-works framing, and the raw evidence. Silent
when there is not enough material — no filler. Never publishes.
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

MIN_CAPTURES = 2
MAX_DRAFTS = 2
THESIS_STATE = Path('/home/kensei/repos/KenseiAgent/content_engine/data/x_thesis_last_run.json')
VOICE_SKILL = Path('/home/kensei/.hermes/skills/social-media/sahil-twitter-voice/SKILL.md')

THESIS_SYSTEM = (
    "You write one deliberately original X post for Sahil — an identity-"
    "building thesis, not a reaction to anyone. Base it strictly on the "
    "supplied captures/seeds. Form: 'I've been thinking about X…' or a very "
    "strong statement developed from repeated observation. It should not be "
    "a generic take; it should be a claim Sahil owns and can defend from "
    "experience.\n"
    "Reply with exactly one JSON object: "
    '{"claim": "...", "evidence": "...", "mechanism": "...", "position": "...", '
    '"post": "..."}. The post may be 25-280 characters. No em-dashes, '
    "hashtags, engagement bait, invented experience or generic praise. "
    "Do not reveal credentials or private data."
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


def _load_inbox() -> list:
    try:
        import x_inbox  # type: ignore[import-not-found]
        return x_inbox.list_captures(used=False, limit=20)
    except Exception as exc:
        print(f"[x-thesis] inbox load failed: {exc}", file=sys.stderr)
        return []


def _load_seeds() -> list:
    p = Path('/home/kensei/repos/KenseiAgent/content_engine/data/x_standalone_seeds.json')
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _mark_inbox_used(captures: list) -> None:
    import x_inbox  # type: ignore[import-not-found]
    ids = [c["id"] for c in captures]
    try:
        x_inbox.mark_used(ids)
    except Exception as exc:
        print(f"[x-thesis] mark used failed: {exc}", file=sys.stderr)


def _draft_thesis(material: str) -> dict:
    system = THESIS_SYSTEM + "\n\n" + _runtime_voice()
    out = _call_llm_chain(system, material[:3000], timeout=90, max_tokens=4000)
    if not out:
        raise RuntimeError('empty LLM output')
    text = out.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError('no JSON object in LLM output')
    return json.loads(text[start:end + 1])


def _already_reported(artifact_id: str) -> bool:
    try:
        state = json.loads(THESIS_STATE.read_text())
    except Exception:
        return False
    return artifact_id in state.get('ids', [])


def _record_reported(ids: list) -> None:
    THESIS_STATE.parent.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(THESIS_STATE.read_text())
    except Exception:
        state = {'ids': []}
    state.setdefault('ids', [])
    state['ids'] = list(dict.fromkeys(state['ids'] + ids))[-20:]
    THESIS_STATE.write_text(json.dumps(state))


def main():
    _load_env()
    inbox = _load_inbox()
    seeds = _load_seeds()
    captures = []
    for c in inbox:
        captures.append(f"[capture] {c.get('text', '')}")
    for s in seeds:
        captures.append(f"[seed] {s.get('claim', '')} — {s.get('evidence', '')}")
    if len(captures) < MIN_CAPTURES:
        print("[x-thesis] insufficient material; silent", file=sys.stderr)
        return
    material = "\n".join(captures)

    drafts = []
    for i in range(MAX_DRAFTS):
        try:
            data = _draft_thesis(material)
        except Exception as exc:
            print(f"[x-thesis] draft skip: {exc}", file=sys.stderr)
            continue
        post = (data.get('post') or '').strip()
        if not post:
            continue
        pack = xm.ArgumentPack(
            claim=(data.get('claim') or '').strip(),
            evidence=(data.get('evidence') or '').strip(),
            mechanism=(data.get('mechanism') or '').strip(),
            position=(data.get('position') or '').strip(),
            context={'source': 'thesis-incubator', 'material': material[:1200]},
        )
        if not pack.is_complete():
            continue
        art = xm.XArtifact(
            id=xm._new_id(xm.LANE_THESIS),
            lane=xm.LANE_THESIS,
            brand='sahil_twitter',
            body=post[:280],
            pack=pack,
        )
        try:
            xm.stage_for_approval(art)
            drafts.append(art)
        except Exception as exc:
            print(f"[x-thesis] stage failed: {exc}", file=sys.stderr)
    if not drafts:
        return
    _mark_inbox_used(inbox)
    report = render_report(drafts, lane='thesis-incubator', title='Thesis ideas (original posts)')
    _record_reported([a.id for a in drafts])
    print(f"X Manager · {len(drafts)} original thesis ideas")
    print("Proposed posts with evidence are in the attached review.")
    print(f"MEDIA:{report}")


if __name__ == "__main__":
    main()
