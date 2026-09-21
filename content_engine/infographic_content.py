"""Turn a draft's flat text into STRUCTURED infographic labels automatically.

The model (Seedream) renders whatever text it's given, so the quality of an
infographic depends on the labels being structured to the layout (two named
sides for a comparison, a DO/DON'T split, numbered points for a list) rather
than a single paragraph. This derives that structure from the draft so the
pipeline is hands-off; if it can't parse cleanly it falls back to title + body,
which still renders (just less precisely).
"""
from __future__ import annotations

import re

from content_router import suggest_layout

_SPLIT = re.compile(r"\s*[;\n]\s*|\.\s+")


def _parts(text: str) -> list[str]:
    return [p.strip(" .-•") for p in _SPLIT.split(text or "") if len(p.strip(" .-•")) > 1]


def _sides(text: str):
    """Best-effort split of 'A vs B' style text into (left_name, right_name)."""
    m = re.search(r"(.{2,40}?)\s+(?:vs\.?|versus)\s+(.{2,40})", text, re.I)
    if m:
        return m.group(1).strip(" .,"), m.group(2).strip(" .,")
    return None


def build_ig_fields(draft: dict) -> dict:
    """Return {'layout', 'labels'} structured for the post, derived from the draft."""
    title = (draft.get("title") or draft.get("topic") or "").strip()
    body = (draft.get("body_text") or "").strip()
    layout = draft.get("ig_layout") or suggest_layout(draft)
    takeaway = title or (_parts(body)[:1] or [""])[0]

    if layout == "binary-comparison":
        sides = _sides(title) or _sides(body)
        rows = _parts(body)
        rows = [r for r in rows if not _sides(r)][:4]
        left, right = sides or ("Option A", "Option B")
        labels = (f"Title: {title}. Left column heading: {left}. Right column heading: {right}. "
                  f"Rows comparing the two: {'; '.join(rows) if rows else body}. "
                  f"Bottom takeaway: {takeaway}.")
        return {"layout": "binary-comparison", "labels": labels}

    if layout == "do-dont":
        do, dont = [], []
        for p in _parts(body):
            stripped = re.sub(r"^(don'?t|avoid|never|do|always)\b[:\-\s]*", "", p, flags=re.I).strip()
            if re.match(r"(don'?t|avoid|never)\b", p, re.I):
                dont.append(stripped)
            elif re.match(r"(do|always)\b", p, re.I):
                do.append(stripped)
            else:
                (do if len(do) <= len(dont) else dont).append(stripped)
        labels = (f"Title: {title}. DO column (green ticks): {'; '.join(do[:4]) or 'best practice'}. "
                  f"DON'T column (red crosses): {'; '.join(dont[:4]) or 'common mistakes'}. "
                  f"Bottom takeaway: {takeaway}.")
        return {"layout": "do-dont", "labels": labels}

    if layout in ("feature-list", "funnel", "pyramid"):
        pts = _parts(body)[:5] or [title]
        labels = (f"Title: {title}. Points in order: {'; '.join(pts)}. Bottom takeaway: {takeaway}.")
        return {"layout": layout, "labels": labels}

    # default: still give the model a clean title + concise points + takeaway
    pts = _parts(body)[:5]
    labels = f"Title: {title}. {('Key points: ' + '; '.join(pts) + '. ') if pts else ''}Takeaway: {takeaway}."
    return {"layout": layout, "labels": labels}


def build_structured(draft: dict, ctype: str) -> dict:
    """Return layout-aware STRUCTURED data (not a model-prompt string) for
    prompt_engine to render. Shapes: comparison->items[], flowchart/timeline->
    steps[], framework->nodes[], infographic->points[]."""
    title = (draft.get("title") or draft.get("topic") or "").strip()
    body = (draft.get("body_text") or "").strip()
    parts = _parts(body)
    takeaway = title or (parts[:1] or [""])[0]
    base = {"type": ctype, "title": title, "takeaway": takeaway}
    if ctype == "comparison":
        items = []
        for p in parts[:4]:
            if ":" in p:
                name, val = p.split(":", 1)
                items.append({"name": name.strip()[:24], "value": val.strip()[:60]})
        if not items:
            items = [{"name": s, "value": ""} for s in (_parts(title) or [title])[:3]]
        return {**base, "items": items}
    if ctype in ("flowchart", "timeline"):
        return {**base, "steps": [p[:70] for p in parts[:6]] or [title]}
    if ctype == "framework":
        return {**base, "nodes": [p[:40] for p in parts[:6]] or [title]}
    return {**base, "points": [p[:70] for p in parts[:6]]}
