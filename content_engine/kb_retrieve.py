"""Retrieve Sahil's prior takes on a topic from his knowledge stores so
fully-automated posts reflect his actual thinking. Keyword scoring over GBrain
markdown (+ optional wiki). Read-only; degrades to [] when a source is absent."""
from __future__ import annotations
import os, re
from pathlib import Path

BRAIN_DIR = Path(os.path.expanduser("~/brain"))


def _wiki_dir() -> Path:
    """Resolve the canonical wiki root while preserving explicit overrides."""
    return Path(
        os.environ.get("WIKI_PATH", str(Path.home() / "docs" / "wiki"))
    ).expanduser()


WIKI_DIR = _wiki_dir()
_STOP = {"the","a","an","and","or","to","of","in","is","for","with","on","my","i"}


def _keywords(topic: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", topic.lower()) if w not in _STOP and len(w) > 2}


def _scan(d: Path, kws: set, limit: int) -> list:
    hits = []
    if not d.exists():
        return hits
    for f in list(d.rglob("*.md"))[:2000]:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        low = text.lower()
        score = sum(low.count(k) for k in kws)
        if score:
            for para in text.split("\n\n"):
                if any(k in para.lower() for k in kws) and len(para.strip()) > 40:
                    hits.append((score, para.strip()[:400]))
                    break
    hits.sort(key=lambda x: -x[0])
    return [p for _, p in hits[:limit]]


def retrieve(topic: str, limit: int = 3) -> list:
    kws = _keywords(topic)
    if not kws:
        return []
    out = _scan(BRAIN_DIR, kws, limit)
    if len(out) < limit:
        out += _scan(WIKI_DIR, kws, limit - len(out))
    return out[:limit]


# TODO(mnemosyne): integrate Mnemosyne working/episodic memory read API
# when a simple read interface is available. For v1, brain+wiki suffice.
