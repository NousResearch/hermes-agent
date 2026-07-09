"""
TwelveLabs (https://twelvelabs.io) integration — optional, opt-in helpers.

This module wraps two TwelveLabs models so MoneyPrinterTurbo can make better
use of the stock/B-roll footage it downloads:

  * Marengo (multimodal embeddings, 512-dim) — used to *semantically reorder*
    the LLM-generated search terms against the video subject, so that when the
    timeline budget runs out the most on-topic footage is the footage that made
    it in (instead of whatever the LLM happened to list first).

  * Pegasus (video understanding) — used to QA / describe a generated clip from
    a public URL, e.g. to sanity-check that a downloaded clip actually matches
    the script before it ships.

The integration is fully opt-in and non-breaking:
  * If `twelvelabs_api_keys` is not configured, every public function here is a
    no-op that returns its input unchanged (or None), so default behavior is
    identical to a build without TwelveLabs.
  * The `twelvelabs` SDK is imported lazily, so the dependency is only required
    when the feature is actually used.

Config (config.toml, [app] section):
    twelvelabs_api_keys = ["tlk_xxx"]   # required to enable
    twelvelabs_rerank_terms = true      # opt-in: reorder search terms by relevance
    twelvelabs_marengo_model = "marengo3.0"   # optional override
    twelvelabs_pegasus_model = "pegasus1.5"   # optional override

Configure a TwelveLabs API key from the TwelveLabs dashboard (https://twelvelabs.io) to enable this optional integration.
"""

import math
from functools import lru_cache
from typing import List, Optional

from loguru import logger

from app.config import config
from app.services import material

DEFAULT_MARENGO_MODEL = "marengo3.0"
DEFAULT_PEGASUS_MODEL = "pegasus1.5"
# Pegasus requires max_tokens in [512, 98304]; 512 is plenty for a one-line QA.
_PEGASUS_MIN_MAX_TOKENS = 512


def is_enabled() -> bool:
    """True only when at least one TwelveLabs API key is configured."""
    keys = config.app.get("twelvelabs_api_keys")
    return bool(keys)


def _client():
    # Lazy import + rotated key reuse mirrors the other providers in
    # material.py (get_api_key rotates across configured keys).
    from twelvelabs import TwelveLabs

    api_key = material.get_api_key("twelvelabs_api_keys")
    return TwelveLabs(api_key=api_key)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def embed_text(text: str, model: Optional[str] = None) -> Optional[List[float]]:
    """
    Return a 512-dim Marengo text embedding, or None on failure / when disabled.

    Cached so repeated terms across a session don't re-hit the API.
    """
    if not is_enabled() or not text or not text.strip():
        return None
    model = model or config.app.get("twelvelabs_marengo_model", DEFAULT_MARENGO_MODEL)
    try:
        # lru_cache only memoizes successful returns; a raised exception is not
        # cached, so a transient API error never poisons the cache.
        return _embed_text_cached(text.strip(), model)
    except Exception as e:  # noqa: BLE001 - never break the pipeline on TL errors
        logger.warning(f"TwelveLabs embed_text failed, skipping rerank: {e}")
        return None


@lru_cache(maxsize=512)
def _embed_text_cached(text: str, model: str) -> List[float]:
    client = _client()
    resp = client.embed.create(model_name=model, text=text)
    # SDK aliases the raw JSON 'float' vector key to `float_`.
    return list(resp.text_embedding.segments[0].float_)


def rerank_terms_by_subject(
    video_subject: str,
    search_terms: List[str],
    model: Optional[str] = None,
) -> List[str]:
    """
    Reorder `search_terms` so the terms most semantically relevant to
    `video_subject` come first (Marengo cosine similarity).

    Opt-in: only runs when TwelveLabs is enabled AND
    `twelvelabs_rerank_terms` is truthy. Falls back to the original order on
    any failure, so it can never make the pipeline worse.
    """
    if not is_enabled() or not config.app.get("twelvelabs_rerank_terms"):
        return search_terms
    if not video_subject or len(search_terms) < 2:
        return search_terms

    subject_vec = embed_text(video_subject, model)
    if subject_vec is None:
        return search_terms

    scored = []
    for term in search_terms:
        vec = embed_text(term, model)
        if vec is None:
            # If any term can't be embedded, don't risk a partial reorder.
            return search_terms
        scored.append((term, _cosine(subject_vec, vec)))

    ranked = [term for term, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    logger.info(
        f"TwelveLabs Marengo reranked {len(ranked)} search terms by relevance "
        f"to subject '{video_subject}': {ranked}"
    )
    return ranked


def analyze_clip(
    video_url: str,
    prompt: str = "Describe what happens in this video in one sentence.",
    model: Optional[str] = None,
    max_tokens: int = _PEGASUS_MIN_MAX_TOKENS,
) -> Optional[str]:
    """
    QA / describe a clip from a public URL with Pegasus, returning the model's
    text answer (or None when disabled / on failure).

    Notes (TwelveLabs API constraints):
      * Pegasus needs a publicly reachable URL (or an uploaded asset), not a
        bare local path; the analyzed window must be >= 4s.
      * max_tokens must be >= 512 for this model.
    """
    if not is_enabled() or not video_url:
        return None
    model = model or config.app.get("twelvelabs_pegasus_model", DEFAULT_PEGASUS_MODEL)
    try:
        from twelvelabs.types import VideoContext_Url

        client = _client()
        resp = client.analyze(
            model_name=model,
            video=VideoContext_Url(url=video_url),
            prompt=prompt,
            max_tokens=max(max_tokens, _PEGASUS_MIN_MAX_TOKENS),
        )
        return resp.data
    except Exception as e:  # noqa: BLE001
        logger.warning(f"TwelveLabs analyze_clip failed: {e}")
        return None
