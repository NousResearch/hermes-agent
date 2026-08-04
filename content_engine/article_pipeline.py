"""Article pipeline — end-to-end orchestrator for the X Articles track.

Order of operations:
  1. article_router.choose()  -> plan or None (skip)
  2. article_generator.write()  -> ArticleDraft (or None on LLM dead)
  3. article_gates.check()  -> GateResult; if fail, retry once with
     feedback threaded into build_article_prompt; else skip
  4. article_illustrator.illustrate()  -> illustrated body
  5. article_assembler.bundle()  -> ArticleBundle on disk
  6. database.insert_draft(content_type='article')  -> DB row
  7. discord_digest.post_article()  -> preview card

All external reads are read-only. No writes to ~/.hermes state.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from config import ARTICLE_ENABLED, ARTICLE_TOPIC_RECENCY_DAYS
import article_assembler as aa
import article_gates as ag
import article_generator as gen
import article_illustrator as ai
import article_router as ar
import database as db
import discord_digest as dd


# Article topic recency is tracked under its own brand key so it is independent
# of the per-platform short-post topic log. The router fans one plan to every
# brand, so dedup is decided once at the article level, not per platform.
_ARTICLE_TOPIC_BRAND = "article"


def _recent_article_topic_ids() -> list[str]:
    """Signal ids that became an article within the recency window.

    Read-only and defensive: a DB hiccup must never block today's article, it
    just degrades to the in-run-only dedup we had before.
    """
    try:
        return db.get_recently_used_topics(
            _ARTICLE_TOPIC_BRAND, days=ARTICLE_TOPIC_RECENCY_DAYS,
        )
    except Exception:
        return []


def _record_article_topics(plan: dict) -> None:
    """Persist the chosen signal ids so future runs cannot re-pick them.

    Called only after at least one brand published, so a failed generation
    leaves the topic available to retry tomorrow.
    """
    for sig in plan.get("signals", []):
        sid = sig.get("signal_id")
        if not sid:
            continue
        try:
            db.log_topic_usage(
                topic_id=sid, brand=_ARTICLE_TOPIC_BRAND,
                topic_text=sig.get("summary", "")[:200], platform="article",
            )
        except Exception:
            pass


def router_choose(state: dict) -> Optional[dict]:
    return ar.choose(state)


def generate_draft(plan: dict, brand: str) -> Optional[dict]:
    return gen.write(plan, brand=brand)


def gate_draft(draft: dict) -> tuple[str, list[str]]:
    """Wrap article_gates.check; return (status, issues). status in
    'ok' | 'fail' | 'skip' (skip when the body is so broken a retry
    would just regenerate from scratch)."""
    res = ag.check(draft)
    return ("ok" if res.passed else "fail"), res.issues


def illustrate(draft: dict, out_dir: Path, **kw) -> str:
    return ai.illustrate(draft, out_dir=out_dir, **kw)


def assemble(illustrated_body: str, draft: dict, out_root: Path, dry_run: bool) -> dd.ArticleBundle:
    illustrated = {"body_md": illustrated_body, "images": [], "outline_path": None}
    bundle = aa.bundle(illustrated, draft, out_root=out_root, dry_run=dry_run)
    # Re-wrap as discord_digest's ArticleBundle so the preview sees the
    # redacted body without the orchestrator reaching into the dataclass.
    return dd.ArticleBundle(
        dir=bundle.dir, article_md=bundle.article_md,
        article_md_path=bundle.article_md_path, image_paths=bundle.image_paths,
        title=bundle.title, lede=bundle.lede, mode=bundle.mode, pillar=bundle.pillar,
    )


def persist_article_draft(**kwargs) -> str:
    """Persist the article and its separate durable approval record."""
    draft_id = kwargs.get("id") or f"art_{abs(hash(kwargs.get('title',''))) % 10**8:08d}"
    bundle_path = kwargs.get("bundle_path")
    if bundle_path is None:
        raise ValueError("bundle_path is required for long-form article approval")
    db.insert_draft(
        draft_id=draft_id,
        brand=kwargs.get("brand", "sahil_twitter"),
        platform=kwargs.get("platform", "twitter"),
        pillar=kwargs.get("pillar", "general"),
        topic=kwargs.get("title", "Article"),
        title=kwargs.get("title", "Article"),
        body_text=kwargs.get("body_md", ""),
        content_type="article",
        visual_description=kwargs.get("visual_description", ""),
        slop_score=kwargs.get("slop_score", 0),
        slop_issues=kwargs.get("slop_issues", ""),
    )
    db.register_article_approval(
        article_id=draft_id, bundle_path=bundle_path,
        brand=kwargs.get("brand", "sahil_twitter"),
        platform=kwargs.get("platform", "twitter"),
    )
    return draft_id


def deliver_preview(bundle: dd.ArticleBundle) -> Optional[str]:
    return dd.post_article(bundle)


def _redact_at_boundary(text: str) -> tuple[str, int]:
    """Final-pass secret redaction at the output boundary.

    Re-uses article_gates' regex set so the contract is "no body that
    leaves the pipeline contains a credential-shaped token". Returns
    (redacted_text, count). This is the spec's mandatory secret-scan
    gate enforced right before DB write + Discord send.
    """
    return ag._redact(text or "")


def _run_for_brand(plan: dict, brand: str, out_root: Path,
                   deliver: bool) -> dict:
    """Generate -> gate -> illustrate -> assemble -> persist -> deliver one
    article for a single brand from an already-chosen router plan.

    Same status contract as run(); never re-runs the router so a shared plan
    can fan out to multiple brands without re-consuming signals.
    """
    platform = gen.platform_for(brand)

    # 1. Generate the draft. Retry once with feedback on gate failure.
    draft = generate_draft(plan, brand=brand)
    if not draft:
        return {"status": "skipped_llm_dead", "plan": plan, "brand": brand}

    # 2. Gate. Reuse build_article_prompt's retry_feedback when fail.
    status, issues = gate_draft(draft)
    if status == "fail":
        from article_generator import build_article_prompt, _call_llm_first, _extract_title, _slug_from_title
        context_blob = "\n\n---\n\n".join(
            gen.enrich_signal(s) for s in plan["signals"][:3]
        ) or ""
        kb = gen.retrieve_kb(plan.get("title_hint", "") or plan["signals"][0].get("summary", ""))[:3]
        feedback = "; ".join(issues) or "rejected by quality gate"
        prompts = build_article_prompt(
            brand, plan, context_blob, kb, retry_feedback=feedback,
        )
        body = _call_llm_first(prompts["system"], prompts["user"])
        if not body:
            return {"status": "skipped_llm_dead", "plan": plan, "issues": issues, "brand": brand}
        title = _extract_title(body) or plan.get("title_hint", "Article")
        draft = {
            "title": title, "body_md": body.strip(),
            "mode": plan.get("mode"), "pillar": plan.get("pillar"),
            "slug": _slug_from_title(title), "signals": plan["signals"],
            "context": context_blob, "kb_snippets": kb,
        }
        status, issues = gate_draft(draft)
        if status == "fail":
            return {"status": "skipped_gate", "plan": plan, "issues": issues, "brand": brand}

    # 3. Illustrate. Uses the redacted body so no secret leaks into prompts.
    gate_res = ag.check(draft)
    redacted_body = gate_res.redacted_body
    work_draft = {**draft, "body_md": redacted_body}
    work_dir = out_root / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    illustrated_body = illustrate(work_draft, work_dir, density="per-section",
                                  max_images=6)

    # 4. Assemble on disk.
    bundle = assemble(illustrated_body, work_draft, out_root=out_root, dry_run=False)

    # Final redaction at the output boundary. Whatever intermediate stages may
    # have done, the body that hits the DB and Discord must not carry
    # credential-shaped tokens. This is the spec's mandatory secret-scan gate.
    bundle.article_md, _stripped = _redact_at_boundary(bundle.article_md)

    # 5. Persist a drafts row (platform derived from the brand).
    persist_article_draft(
        brand=brand, platform=platform, pillar=draft.get("pillar", "general"),
        title=draft.get("title", "Article"), body_md=bundle.article_md,
        bundle_path=bundle.dir,
        slop_issues="; ".join(issues) if issues else "",
    )

    # 6. Deliver the preview.
    if deliver:
        deliver_preview(bundle)
    return {"status": "ok", "bundle": bundle, "plan": plan, "brand": brand}


def run(out_root: Optional[Path] = None, deliver: bool = True,
        brand: str = "sahil_twitter") -> dict:
    """Drive one article through the pipeline for a single brand.

    Status values:
      - "skipped_disabled" — ARTICLE_ENABLED is False
      - "skipped_router"  — router returned None
      - "skipped_llm_dead" — generator returned None
      - "skipped_gate"     — gate failed after retry
      - "ok"               — bundle + DB row + preview delivered
    """
    if not ARTICLE_ENABLED:
        return {"status": "skipped_disabled"}

    state: dict = {"used": _recent_article_topic_ids()}
    plan = router_choose(state)
    if not plan:
        return {"status": "skipped_router", "state": state}

    result = _run_for_brand(plan, brand, out_root or aa.OUTPUT_ROOT, deliver)
    if result.get("status") == "ok":
        _record_article_topics(plan)
    return result


def run_all(out_root: Optional[Path] = None, deliver: bool = True,
            brands: tuple = ("sahil_twitter", "sahil_linkedin")) -> dict:
    """Pick ONE router plan, then produce a platform-tailored article per brand.

    The router runs once so the day's chosen signals are consumed once and both
    the X and LinkedIn articles cover the same story in each platform's voice.
    Returns {"status": "ok"|"skipped_*", "results": {brand: result-dict}}.
    """
    if not ARTICLE_ENABLED:
        return {"status": "skipped_disabled", "results": {}}

    state: dict = {"used": _recent_article_topic_ids()}
    plan = router_choose(state)
    if not plan:
        return {"status": "skipped_router", "state": state, "results": {}}

    out_root = out_root or aa.OUTPUT_ROOT
    results = {brand: _run_for_brand(plan, brand, out_root, deliver)
               for brand in brands}
    any_ok = any(r.get("status") == "ok" for r in results.values())
    if any_ok:
        _record_article_topics(plan)
    return {"status": "ok" if any_ok else "skipped_all", "results": results, "plan": plan}
