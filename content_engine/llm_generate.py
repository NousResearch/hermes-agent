"""LLM draft generator for personal brands using governed Hermes routes.

The active profile's model and fallback_providers are resolved by the same
Hermes runtime APIs as CLI/gateway/cron, including native credential pools.

Returns the identical draft dict shape as llm_drafts.generate_drafts for drop-in use.

ARCHITECTURE NOTE:
    The actual LLM call does NOT happen inside this Python module. Instead:
    - The cron prompt IS the generation step (the cron agent writes the posts)
    - build_generation_prompt() produces the system+user prompt that the cron agent follows
    - For --self-call testing from this session, the calling agent (me) generates output
    - For cron execution, the cron agent's own model produces the output

    This avoids subprocess calls, nested agent loops, and hermes -z timeouts.
"""

import json
import os
import random
import re
import sys
import time
import uuid
from datetime import datetime
from typing import Optional

# ── Voice skill paths ──────────────────────────────────────────────────

SKILL_DIR = os.path.expanduser("~/.hermes/skills/social-media")

VOICE_SKILL_PATHS = {
    "sahil_twitter": os.path.join(SKILL_DIR, "sahil-twitter-voice", "SKILL.md"),
    "sahil_linkedin": os.path.join(SKILL_DIR, "sahil-linkedin-voice", "SKILL.md"),
    # These three brands have full voice docs on disk that were never wired in,
    # so they were silently running on the 2-sentence BRAND_VOICE_FALLBACK.
    "plenishd": os.path.join(SKILL_DIR, "plenishd-voice", "SKILL.md"),
    "coachos": os.path.join(SKILL_DIR, "coachos-voice", "SKILL.md"),
    "matchdaymaestro": os.path.join(SKILL_DIR, "matchdaymaestro-voice", "SKILL.md"),
}


def _load_voice_skill(brand: str) -> str:
    """Read brand voice SKILL.md verbatim for context injection."""
    path = VOICE_SKILL_PATHS.get(brand)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            # Strip YAML frontmatter
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    content = content[end + 3:].strip()
            return content
    return ""


def _load_exemplars(brand: str, pillar: str, platform: str, max_exemplars: int = 3) -> list[str]:
    """Load exemplar templates from llm_drafts.BRAND_TEMPLATES.

    We import here to avoid circular dependency. Uses the narrative template
    lists as few-shot exemplars (style reference, not content source).
    """
    from llm_drafts import BRAND_TEMPLATES
    pillar_templates = BRAND_TEMPLATES.get(brand, {}).get(pillar, [])
    if not pillar_templates:
        # Fall back to any pillar for this brand
        for p, ts in BRAND_TEMPLATES.get(brand, {}).items():
            pillar_templates.extend(ts)
    if not pillar_templates:
        return ["Write a short, specific, story-driven social post."]
    random.shuffle(pillar_templates)
    return pillar_templates[:max_exemplars]


def _load_live_variables(brand: str, topic: dict) -> dict:
    """Inject live signal variables using the existing mechanism."""
    from llm_drafts import _build_topic_variables, _is_future_fixture
    if not _is_future_fixture(topic):
        return {}
    return _build_topic_variables(brand, topic)


# ── Platform length norms ──────────────────────────────────────────────

PLATFORM_LENGTH = {
    "twitter": {"max": 280, "min": 180},
    # X Articles are long-form: a wide window so the shared gate's length check
    # never penalises a full article. The word floor is enforced separately.
    "article": {"max": 60000, "min": 1500},
    "linkedin": {"max": 3000, "min": 600, "story_floor": 400},
    # First ~125 chars show before "more" on IG; the hook must land there.
    "instagram": {"max": 1500, "min": 120},
    "tiktok": {"max": 300, "min": 60},
}

# Positioning fallback for product brands that have no voice SKILL.md.
BRAND_VOICE_FALLBACK = {
    "plenishd": (
        "Plenishd: UK voice-first kitchen inventory app. 'Snap it. Say it. Stock it.' "
        "Voice of a real busy UK parent: warm, wry, practical, zero corporate polish. "
        "Talk about real kitchen chaos, food waste guilt, takeaway-money regret. "
        "Never invent statistics or user numbers."
    ),
    "coachos": (
        "CoachSense (formerly CoachOS): grassroots football coaching platform. "
        "Voice of a thoughtful grassroots coach: calm, craft-obsessed, generous with "
        "knowledge. Editorial restraint, no hype. Beta opens August 2026; never claim it is live."
    ),
    "matchdaymaestro": (
        "MatchdayMaestro: football prediction and trivia companion. Voice of the "
        "sharpest mate in the group chat: quick, confident, banter without cruelty. "
        "Predictions and bragging rights, never betting or odds talk."
    ),
    "kicktionary": (
        "Kick-tionary: football tactics education for ages 6-18. Voice of an "
        "encouraging junior coach: clear, playful, jargon explained simply. Safe for kids."
    ),
}


# ── Prompt builder ─────────────────────────────────────────────────────

def build_generation_prompt(
    brand: str,
    topic: dict,
    platform: str,
    retry_feedback: Optional[str] = None,
) -> dict:
    """Build system + user prompt for LLM generation.

    Args:
        brand: Brand key (sahil_twitter, sahil_linkedin)
        topic: Topic dict with pillar, topic text, optional activity_data
        platform: Platform name (twitter, linkedin)
        retry_feedback: If regenerating, feedback from the gate to fix

    Returns:
        {"system": str, "user": str}
    """
    pillar = topic.get("pillar", "general")
    topic_text = topic.get("topic", "")

    # Load brand voice
    voice_skill = _load_voice_skill(brand)
    if not voice_skill:
        voice_skill = BRAND_VOICE_FALLBACK.get(brand, f"Write as {brand}, direct and specific.")

    # Load exemplars for pillar
    exemplars = _load_exemplars(brand, pillar, platform)
    exemplar_text = "\n\n---\n".join(exemplars)

    # Load live variables
    variables = _load_live_variables(brand, topic)

    # Build variable hint
    var_hint = ""
    if variables:
        var_items = "; ".join(f"{k}={v}" for k, v in variables.items() if v and len(str(v)) < 80)
        if var_items:
            var_hint = f"\n\nContext variables available: {var_items}"

    # Platform-specific instructions
    plat_info = PLATFORM_LENGTH.get(platform, {})
    max_chars = plat_info.get("max", 280)
    min_chars = plat_info.get("min", 100)

    if platform == "twitter":
        length_rule = f"Write a SINGLE tweet ({min_chars}-{max_chars} characters). One focused idea. No thread markers."
    elif platform == "linkedin":
        story_floor = plat_info.get("story_floor", 300)
        length_rule = f"Write a LinkedIn post ({min_chars}-{max_chars} characters, at least {story_floor} chars for proper storytelling). Setup-Evidence-Frame structure. Hashtags at the end."
    elif platform == "instagram":
        length_rule = (f"Write an Instagram caption ({min_chars}-{max_chars} characters). "
                       "The first 125 characters must work as a standalone hook (shown before 'more'). "
                       "Short paragraphs, line breaks between them, 3-5 niche hashtags at the end.")
    elif platform == "tiktok":
        length_rule = (f"Write a TikTok caption ({min_chars}-{max_chars} characters). "
                       "Punchy hook, conversational, 2-4 hashtags. The video carries the story; "
                       "the caption sets up curiosity.")
    else:
        length_rule = f"Write a social post ({min_chars}-{max_chars} characters)."

    # Platform-specific formatting
    if platform == "twitter":
        format_rules = "\n- No thread markers (1/3, 2/3, etc.)\n- Max 2 hashtags at end\n- Line breaks for readable flow (short lines work well on X)"
    elif platform == "linkedin":
        format_rules = "\n- 3-5 hashtags at end, Title Case\n- Paragraphs with line breaks\n- No emoji chains or engagement bait"
    else:
        format_rules = ""

    # Build system prompt
    segments = [
        f"You are writing a social media post for the brand '{brand}' on {platform}.",
        "",
        "## Brand Voice (use this EXACTLY — do not smooth it out)",
        voice_skill,
        "",
        "## Style Reference (pattern-match, DO NOT copy content)",
        "Here are approved example posts in this brand's voice for this topic pillar:",
        exemplar_text,
        var_hint,
        "",
        "## Content Requirements",
        length_rule,
        "- Storytelling: specific, concrete details. Real numbers, real tools, real moments.",
        "- Do NOT invent statistics, user counts, growth numbers, dates, or metrics. "
        "Use ONLY numbers supplied in the topic/context; if none are given, write "
        "without invented figures.",
        "- British English spelling (organise, colour, behaviour, centre).",
        "- No em-dashes (use commas or line breaks instead).",
        "- No AI-isms: no 'Let's dive in', 'In today's world', 'Great question'.",
        "- No boilerplate mantras or template-itis.",
        format_rules,
    ]

    if retry_feedback:
        segments.append("")
        segments.append("## THIS IS A RETRY — the previous draft was rejected for:")
        segments.append(retry_feedback)
        segments.append("Fix ALL of these issues in your new draft.")

    system_prompt = "\n".join(segments)

    # Build user prompt
    user_parts = [
        f"Write a {platform} post for brand '{brand}' on pillar '{pillar}'.",
    ]
    if topic_text:
        user_parts.append(f"Topic: {topic_text}")
    if variables:
        for k, v in variables.items():
            if v and len(str(v)) < 80:
                user_parts.append(f"{k}: {v}")

    user_prompt = "\n".join(user_parts)

    return {"system": system_prompt, "user": user_prompt}


# ── Educational prompt builder ─────────────────────────────────────────────

_EDU_FRAME = (
    "You are writing EDUCATIONAL content for the AI builder community. Teach one "
    "concrete idea from the author's real work: explain the mechanism, cite the "
    "specific detail (a real number, tool, or technique), and end by inspiring "
    "the reader to try or rethink something. Builders-leaning but accessible to "
    "technical peers. Never generic hype. Never invent facts beyond the context."
)

def build_educational_prompt(brand, pillar, platform, signal, context, kb_snippets, retry_feedback=None):
    """System+user prompt for a fully-automated educational post grounded in real
    context + Sahil's prior takes."""
    voice = _load_voice_skill(brand) or BRAND_VOICE_FALLBACK.get(brand, "")
    plat = PLATFORM_LENGTH.get(platform, {})
    length = f"{plat.get('min',100)}-{plat.get('max',280)} characters"
    takes = "\n".join(f"- {s}" for s in (kb_snippets or [])) or "(none on file)"
    rules = [
        f"- Length: {length}.", "- British English. No em-dashes.",
        "- No AI-isms. Must include a concrete specific from the context.",
    ]
    if retry_feedback:
        rules.append("- Previous attempt rejected: " + retry_feedback)
    system = "\n".join([
        f"You are writing a {platform} post for '{brand}', pillar '{pillar}'.",
        "## Brand voice (use exactly)", voice,
        "## Educational frame", _EDU_FRAME,
        "## Rules",
    ] + rules)
    user = "\n".join([
        f"Pillar: {pillar}", f"Signal: {signal.get('summary','')}",
        "## Real context (ground the post in this)", context or "(none)",
        "## Author's prior takes (reflect this thinking)", takes,
        "Write the post now.",
    ])
    return {"system": system, "user": user}


def _specificity_ok(body: str, context: str) -> bool:
    """Reject generic auto-content: the draft must share a concrete token
    (number, tool name, or notable word) with the real context."""
    import re as _re
    if _re.search(r"\d", body) and _re.search(r"\d", context):
        return True
    ctx_tokens = {w.lower() for w in _re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}", context)}
    body_tokens = {w.lower() for w in _re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{3,}", body)}
    generic = {"this","that","with","from","your","about","into","more","just","they","here"}
    overlap = (ctx_tokens & body_tokens) - generic
    return len(overlap) >= 2


# ── Quality gate ───────────────────────────────────────────────────────

def _check_em_dashes(body: str) -> tuple[bool, str]:
    """Check for em-dashes (—) which are banned."""
    if "—" in body:
        count = body.count("—")
        return False, f"Contains {count} em-dash(es) — replace with commas or line breaks"
    return True, ""


def _check_length(body: str, platform: str) -> tuple[bool, str]:
    """Check post length against platform limits."""
    plat_info = PLATFORM_LENGTH.get(platform, {})
    max_chars = plat_info.get("max", 280)
    min_chars = plat_info.get("min", 100)
    char_count = len(body)

    if char_count > max_chars:
        return False, f"Too long: {char_count} chars (max {max_chars})"
    if platform == "linkedin":
        story_floor = plat_info.get("story_floor", 300)
        if char_count < story_floor:
            return False, f"Too short for LinkedIn storytelling: {char_count} chars (min recommended {story_floor})"
    elif char_count < min_chars:
        return False, f"Too short: {char_count} chars (min {min_chars})"
    return True, ""


def gate_post(body_text: str, platform: str, context: str = "") -> dict:
    """Quality gate: slop audit + stale tech + unfilled placeholders + em-dashes + length.

    Returns the same contract as llm_drafts._audit_slop for compatibility.
    """
    from llm_drafts import _audit_slop, _reject_stale_tech, _has_unfilled_placeholders

    issues = []
    slop_score = 0

    # 1. Slop audit
    audit = _audit_slop(body_text)
    slop_score = max(slop_score, audit["slop_score"])
    issues.extend(audit["issues"])

    # 2. Stale tech check
    passed_tech, stale_match = _reject_stale_tech(body_text)
    if not passed_tech:
        issues.append(f"Stale tech reference: {stale_match}")
        slop_score += 3

    # 3. Unfilled placeholders
    if _has_unfilled_placeholders(body_text):
        issues.append("Unfilled placeholder patterns detected")
        slop_score += 4

    # 4. Em-dash check
    dash_ok, dash_msg = _check_em_dashes(body_text)
    if not dash_ok:
        issues.append(dash_msg)
        slop_score += 2

    # 5. Length check
    len_ok, len_msg = _check_length(body_text, platform)
    if not len_ok:
        issues.append(len_msg)
        slop_score += 1

    # 6. Specificity check (educational posts only)
    if context and not _specificity_ok(body_text, context):
        issues.append("Too generic, no concrete from source")
        slop_score += 3

    # 7. Fabricated metrics: audience/growth numbers (users, %, large figures)
    #    that are not grounded in the provided context. Lazy import avoids the
    #    article_gates <-> llm_generate import cycle.
    if context:
        try:
            from article_gates import _fabricated_numbers
            bad = _fabricated_numbers(body_text, context)
            if bad:
                issues.append(f"Fabricated numbers not in context: {', '.join(bad)}")
                slop_score += 3
        except Exception:
            pass

    return {
        "slop_score": min(slop_score, 10),
        "issues": issues,
        "passed": slop_score < 6,
    }


# ── Model call ─────────────────────────────────────────────────────────

_SELF_CALL = False
"""Set to True when the calling agent will generate text directly.
When False (cron mode), the cron agent handles generation via its own prompt."""


CONTENT_ROUTING_SURFACE = "content-strategist"


def _content_surface_home():
    """Resolve the governed content surface independently of the caller."""
    from hermes_cli.profiles import get_profile_dir, profile_exists

    if not profile_exists(CONTENT_ROUTING_SURFACE):
        raise RuntimeError(
            f"Governed content routing surface {CONTENT_ROUTING_SURFACE!r} is unavailable"
        )
    return get_profile_dir(CONTENT_ROUTING_SURFACE)


def _load_hermes_config(*, home_resolver=None, config_loader=None) -> dict:
    """Load content-strategist through canonical Hermes override APIs."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.config import load_config

    home = (home_resolver or _content_surface_home)()
    loader = config_loader or load_config
    token = set_hermes_home_override(home)
    try:
        config = loader()
    finally:
        reset_hermes_home_override(token)
    if not isinstance(config, dict):
        raise RuntimeError(
            f"Governed content routing surface {CONTENT_ROUTING_SURFACE!r} has invalid config"
        )
    return config


def _resolve_runtime(**kwargs) -> dict:
    """Late-bound seam for Hermes' shared provider + credential-pool resolver."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(**kwargs)


def _llm_configs(longform: bool = False, *, config=None) -> list[dict]:
    """Resolve the governed, exact-model route chain using Hermes runtime APIs.

    ``longform`` is retained for API compatibility; route policy belongs to the
    governed content surface's generated config, never to this application module.
    """
    del longform
    config = _load_hermes_config() if config is None else config
    model_cfg = config.get("model") if isinstance(config, dict) else {}
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    primary_model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
    primary_provider = str(model_cfg.get("provider") or "auto").strip()
    if not primary_model or not primary_provider:
        raise RuntimeError(
            f"Governed content routing surface {CONTENT_ROUTING_SURFACE!r} has no valid model route"
        )

    from hermes_cli.fallback_config import get_fallback_chain, resolve_entry_api_key

    routes = [{
        "provider": primary_provider,
        "model": primary_model,
        "base_url": model_cfg.get("base_url"),
        "api_key": model_cfg.get("api_key"),
    }, *get_fallback_chain(config)]
    configs: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for route in routes:
        exact_model = str(route.get("model") or "").strip()
        provider = str(route.get("provider") or "").strip()
        base_url = str(route.get("base_url") or "").strip().rstrip("/") or None
        identity = (provider.lower(), exact_model.lower(), (base_url or "").lower())
        if not provider or not exact_model or identity in seen:
            continue
        route_model_id = str(route.get("route_model_id") or exact_model).strip()
        if route_model_id != primary_model:
            continue
        seen.add(identity)
        try:
            runtime = _resolve_runtime(
                requested=provider,
                explicit_api_key=resolve_entry_api_key(route),
                explicit_base_url=base_url,
                target_model=exact_model,
            )
        except Exception as exc:
            print(f"[llm_generate] route resolution failed for {provider}: {exc}", file=sys.stderr)
            continue
        cfg = {
            "base": str(runtime.get("base_url") or "").rstrip("/"),
            "model": exact_model,
            "key": runtime.get("api_key") or "",
            "provider": provider,
            "api_mode": runtime.get("api_mode") or "chat_completions",
        }
        pool = runtime.get("credential_pool")
        if pool is not None:
            cfg["_credential_pool"] = pool
            current = pool.peek()
            if current is not None:
                cfg["_credential_id"] = current.id
        if cfg["base"]:
            configs.append(cfg)
    return configs


def _classify_failure_reason(status_code, body_text) -> Optional[str]:
    """Classify a provider HTTP failure from status + response body.

    Returns a stable reason string for the credential pool, or None when the
    failure is not a credential-level condition. Zero-credit/quota-exhausted
    responses often arrive as HTTP 400 with provider-specific wording, so the
    body must be inspected rather than trusting the status code alone.
    """
    if status_code in (401, 403):
        return "auth"
    if status_code == 402:
        return "insufficient_credits"
    if status_code == 429:
        return "rate_limit"
    if status_code == 400:
        lowered = (body_text or "").lower()
        if any(marker in lowered for marker in (
            "insufficient balance", "insufficient credit", "credit insufficient",
            "balance=0", "balance is 0", "quota exhausted", "quota exceeded",
        )):
            return "insufficient_credits"
    return None


def _call_llm_chain(system: str, user: str, *, timeout: int = 90,
                    max_tokens: int = 3000, longform: bool = False, configs=None) -> Optional[str]:
    """Try every governed route, rotating a native pool after failed calls."""
    for cfg in (_llm_configs(longform=longform) if configs is None else configs):
        attempted_ids: set[str] = set()
        while True:
            credential_id = str(cfg.get("_credential_id") or "")
            if credential_id:
                if credential_id in attempted_ids:
                    break
                attempted_ids.add(credential_id)
            result = _call_llm(system, user, cfg, timeout=timeout, max_tokens=max_tokens)
            if result:
                return result
            pool = cfg.get("_credential_pool")
            failure_status = cfg.get("_failure_status")
            failure_reason = cfg.get("_failure_reason")
            # Rotate on credential-level failures: auth (401/403), payment
            # (402), rate limit (429), and zero-credit responses that surface
            # as HTTP 400. A plain 400 without a credit-classified body is a
            # request problem — retrying the same or another credential of the
            # same provider will not help, so stop here.
            rotatable = failure_reason in {"auth", "insufficient_credits", "rate_limit"}
            if pool is None or not credential_id or failure_status is None or not rotatable:
                break
            next_entry = pool.mark_exhausted_and_rotate(
                status_code=failure_status, credential_id=credential_id,
                api_key_hint=cfg.get("key"),
                failure_reason=failure_reason,
            )
            if next_entry is None:
                break
            cfg = dict(cfg)
            cfg["key"] = next_entry.runtime_api_key
            cfg["_credential_id"] = next_entry.id
    return None


def _llm_config() -> Optional[dict]:
    """First endpoint in the chain (kept for callers gating on LLM availability)."""
    configs = _llm_configs()
    return configs[0] if configs else None


def _call_llm_first(system: str, user: str, *, timeout: int = 90,
                    max_tokens: int = 3000, longform: bool = False) -> Optional[str]:
    """First non-empty body across the governed chain, with pool rotation.

    Callers that do not need per-attempt control should prefer this over
    hand-rolled ``for cfg in _llm_configs(...)`` loops: it delegates to
    ``_call_llm_chain`` so credential-level failures (auth, 402, 429 and
    zero-credit HTTP 400s) rotate the provider's credential pool before the
    next route is attempted.
    """
    return _call_llm_chain(system, user, timeout=timeout, max_tokens=max_tokens,
                           longform=longform)


_FALLBACK_ALERTED = False


def _alert_static_fallback(brand: str, platform: str) -> None:
    """Loudly flag that every LLM endpoint failed and we are serving a frozen
    canned template. Silent fallback here is exactly what made all output read
    stale and off-voice, so make it impossible to miss in the run logs."""
    global _FALLBACK_ALERTED
    msg = (f"[llm_generate] ⚠️  ALL LLM endpoints failed for {brand}/{platform} — "
           f"serving a STATIC CANNED TEMPLATE (not generated copy). "
           f"Check endpoint quota/config: {[c['base'] + ' ' + c['model'] for c in _llm_configs()]}")
    print(msg, file=sys.stderr)
    if not _FALLBACK_ALERTED:
        print("=" * 78, file=sys.stderr)
        print("CONTENT ENGINE DEGRADED: text falling back to frozen templates. "
              "Generated copy is OFF until an LLM endpoint is reachable.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        _FALLBACK_ALERTED = True


def _call_llm(system: str, user: str, cfg: dict, timeout: int = 90,
              max_tokens: int = 3000) -> Optional[str]:
    """Call one endpoint, retry transient HTTP once, and degrade on failure.

    ``max_tokens`` remains overridable for long-form callers.
    """
    cfg.pop("_failure_status", None)
    if (cfg.get("api_mode") or "chat_completions") != "chat_completions":
        try:
            from agent.auxiliary_client import resolve_provider_client

            client, resolved_model = resolve_provider_client(
                cfg.get("provider", "custom"), model=cfg["model"],
                explicit_base_url=cfg.get("base"), explicit_api_key=cfg.get("key"),
                api_mode=cfg.get("api_mode"),
            )
            if client is None:
                return None
            response = client.chat.completions.create(
                model=resolved_model or cfg["model"],
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text = (response.choices[0].message.content or "").strip()
            return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip() or None
        except Exception as exc:
            cfg["_failure_status"] = getattr(exc, "status_code", None)
            cfg["_failure_reason"] = _classify_failure_reason(
                cfg["_failure_status"], str(exc),
            )
            print(f"[llm_generate] Hermes provider call failed: {exc}", file=sys.stderr)
            return None

    try:
        import requests
    except ImportError:
        return None
    headers = {"Content-Type": "application/json"}
    # OpenCode.ai and some other providers use Cloudflare which blocks
    # Python's default User-Agent. Use a browser-like UA to avoid 403/1010.
    headers["User-Agent"] = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    if cfg.get("key"):
        headers["Authorization"] = f"Bearer {cfg['key']}"
    # run_daily.sh sources ~/.hermes/.env with `set -a`, which exports a Privoxy
    # HTTP(S)_PROXY (127.0.0.1:8118). That proxy is not always up, and these are
    # public generation APIs that must not route through it — a refused proxy was
    # silently collapsing all copy to canned templates. trust_env=False makes the
    # session ignore inherited proxy/netrc env entirely.
    session = requests.Session()
    session.trust_env = False
    for attempt in range(2):
        try:
            r = session.post(
                f"{cfg['base']}/chat/completions",
                json={
                    "model": cfg["model"],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.8,
                    # Reasoning models burn tokens on hidden thinking before the
                    # answer; a tight cap returns empty content.
                    "max_tokens": max_tokens,
                },
                headers=headers, timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt == 0:
                print(
                    f"[llm_generate] transient LLM transport failure: {exc}; "
                    "retrying once",
                    file=sys.stderr,
                )
                time.sleep(1.0)
                continue
            print(f"[llm_generate] LLM call failed: {exc}", file=sys.stderr)
            return None
        except Exception as exc:  # noqa: BLE001 (generation must degrade, not crash the cron)
            print(f"[llm_generate] LLM call failed: {exc}", file=sys.stderr)
            return None

        if r.status_code != 200:
            cfg["_failure_status"] = r.status_code
            cfg["_failure_reason"] = _classify_failure_reason(r.status_code, r.text[:400])
            transient = r.status_code == 429 or r.status_code >= 500
            if transient and attempt == 0:
                print(
                    f"[llm_generate] transient LLM HTTP {r.status_code}; retrying once",
                    file=sys.stderr,
                )
                time.sleep(1.0)
                continue
            print(f"[llm_generate] LLM HTTP {r.status_code}: {r.text[:160]}", file=sys.stderr)
            return None

        try:
            payload = r.json()
            text = (payload.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            # Reasoning models (deepseek-v4-flash etc.) put the answer in
            # `reasoning_content` when `content` is empty. Fallback to that.
            if not text.strip():
                text = (payload.get("choices") or [{}])[0].get("message", {}).get("reasoning_content", "") or ""
            # Some models inline their reasoning as <think> blocks in content.
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            return text.strip() or None
        except Exception as exc:  # noqa: BLE001 (malformed providers must degrade)
            print(f"[llm_generate] LLM response parse failed: {exc}", file=sys.stderr)
            return None
    return None


def generate_one(
    brand: str,
    topic: dict,
    platform: str,
    max_retries: int = 2,
) -> Optional[dict]:
    """Generate one draft: build prompt → call model → gate; retry on fail.

    Generation uses the active profile's governed exact-model route chain and
    is gated by gate_post with one retry carrying the gate's feedback.

    Returns a draft dict (same shape as llm_drafts.generate_drafts items)
    or None.
    """
    configs = _llm_configs()
    if configs:
        for cfg in configs:
            feedback = None
            endpoint_reachable = False
            for _ in range(max_retries + 1):
                if topic.get("educational"):
                    context = topic.get("context", "")
                    kb = topic.get("kb_snippets", [])
                    prompts = build_educational_prompt(
                        brand, pillar=topic.get("pillar","general"), platform=platform,
                        signal=topic, context=context, kb_snippets=kb,
                        retry_feedback=feedback)
                else:
                    prompts = build_generation_prompt(brand, topic, platform, retry_feedback=feedback)
                body = _call_llm(prompts["system"], prompts["user"], cfg)
                if not body:
                    # Endpoint dead/quota'd — stop retrying it, try the next one.
                    break
                endpoint_reachable = True
                if topic.get("educational"):
                    edu_ctx = topic.get("context", "")
                    draft = generate_with_text(brand, topic, platform, body, context=edu_ctx)
                else:
                    draft = generate_with_text(brand, topic, platform, body)
                if draft:
                    return draft
                gate = gate_post(body, platform, context=topic.get("context", "") if topic.get("educational") else "")
                feedback = "; ".join(gate.get("issues", [])) or "rejected by quality gate"
            if endpoint_reachable:
                print(f"[llm_generate] {brand}/{platform}: {cfg['model']} reachable but all "
                      f"retries failed the gate; trying next endpoint", file=sys.stderr)
        # Every endpoint exhausted.
        return None

    if not _SELF_CALL:
        # Cron mode — return None; the cron agent generates text from the prompt
        return None

    # Self-call mode — the caller provides body_text via generate_one(body_text=...)
    raise ValueError(
        "Self-call not supported via this path. "
        "Use generate_with_text(brand, topic, platform, body_text) instead."
    )


def generate_with_text(
    brand: str,
    topic: dict,
    platform: str,
    body_text: str,
    context: str = "",
) -> Optional[dict]:
    """Validate and wrap externally-generated text into a draft dict.

    Used by --self-call: the calling agent generates text, then passes it here
    for gating and wrapping into the standard draft shape.
    """
    pillar = topic.get("pillar", "general")
    topic_text = topic.get("topic", "")

    # Gate check
    gate_result = gate_post(body_text, platform, context=context)
    if not gate_result["passed"]:
        return None

    # Determine content type. Prefer the source-aware router when the topic
    # carries a real source signal; fall back to the existing static mapping.
    from llm_drafts import _choose_content_type
    content_type = None
    source_provenance = topic.get("source_provenance")
    editorial_rationale = ""
    try:
        from content_router import source_aware_content_type
        from editorial_router import route_signal
        signal = _signal_from_topic(topic)
        # Manual blog queues can opt into source-aware routing by setting
        # curated/evidence/source_notes/context/source_override.
        draft_stub = {
            "brand": brand, "platform": platform, "pillar": pillar,
            "title": topic_text, "body_text": body_text,
        }
        content_type = source_aware_content_type(draft_stub, signal)
        route = route_signal(signal, candidate_surfaces=[_surface_for_platform(platform)])
        source_provenance = source_provenance or route.get("source_provenance")
        editorial_rationale = route.get("rationale", "")
    except Exception:
        content_type = None
    if not content_type:
        content_type = _choose_content_type(brand, pillar, platform)

    visual_descs = {
        "sahil_twitter": "Dark terminal aesthetic. Code overlay, monospace font, build-in-public style.",
        "sahil_linkedin": "Professional graphic with clean typography. Quote-style layout, grey tones with red accent.",
    }

    draft_id = f"{brand[:4]}_{str(uuid.uuid4())[:8]}"

    drafted = {
        "id": draft_id,
        "brand": brand,
        "platform": platform,
        "pillar": pillar,
        "topic": topic_text,
        "title": topic_text,
        "body_text": body_text,
        "content_type": content_type,
        "visual_description": visual_descs.get(brand, ""),
        "slop_audit": gate_result,
        "source_provenance": source_provenance or {},
        "editorial_rationale": editorial_rationale,
    }

    # For educational topics, derive an art-directed visual scene so the image
    # generator doesn't default to literalism (e.g. "clipboard upgrade" -> a
    # clipboard photo). Falls back to the brand default on any failure.
    if context:
        scene = derive_visual_subject(brand, drafted)
        if scene:
            drafted["visual_description"] = scene

    return drafted




def _signal_from_topic(topic: dict) -> dict:
    """Normalise a topic/activity object into an editorial_router signal."""
    signal = dict(topic or {})
    activity = signal.get("activity_data") or {}
    if activity and not signal.get("signal_type"):
        signal["signal_type"] = activity.get("signal_type", "")
        signal["variables"] = activity.get("variables", {})
        signal["signal_id"] = activity.get("signal_id", "")
    return signal


def _has_source_context(signal: dict) -> bool:
    """Whether a topic is source-aware enough for editorial skip/routing."""
    return bool(
        signal.get("signal_type")
        or signal.get("source_provenance")
        or signal.get("source_override")
        or signal.get("curated") is True
        or signal.get("evidence")
        or signal.get("source_notes")
        or signal.get("context")
    )


def _surface_for_platform(platform: str) -> str:
    return {
        "twitter": "x_post",
        "linkedin": "linkedin_post",
        "blog": "blog",
        "article": "article",
    }.get(platform, "x_post")

# ── Visual subject derivation (art direction) ──────────────────────────

# What a compelling, non-literal visual scene looks like per brand. Steers the
# deriver away from photographing the literal nouns in a headline (the cause of
# "clipboard upgrade" -> a clipboard).
_VISUAL_BRAND_HINT = {
    "plenishd": "a warm, real UK kitchen moment — food, hands, worktops, fridge light; never app screenshots or text",
    "coachos": "a quiet, premium grassroots-football scene — pitch, boots, nets, dawn light, a lone figure; near-monochrome with a hint of green",
    "matchdaymaestro": "a dramatic matchday-night football scene — floodlights, crowd energy, motion; bold and cinematic",
    "sahil_twitter": "a moody build-in-public developer scene or a striking tech metaphor; dark, cinematic, high craft",
    "sahil_linkedin": "a restrained, professional editorial metaphor; uncluttered, premium, one clear idea",
}


def derive_visual_subject(brand: str, draft: dict) -> Optional[str]:
    """Translate a post into a single vivid, photographable VISUAL scene.

    Returns a concrete scene phrase (no rendered words, no literal headline props)
    to feed prompt_engine as the subject hook, or None if every endpoint is down
    (callers then keep their existing literal subject). Low volume: only runs for
    approved drafts at enrich time.
    """
    configs = _llm_configs()
    if not configs:
        return None
    brand_l = (brand or "").lower()
    hint = _VISUAL_BRAND_HINT.get(brand_l, "a striking, on-brand editorial scene; one clear focal idea")
    topic = (draft.get("title") or draft.get("topic") or "").strip()
    body = (draft.get("body_text") or "").strip()[:400]

    system = (
        "You are an art director. Turn a social post into ONE vivid, photographable "
        "scene for an image generator. Rules: describe a real visual scene (subject, "
        "setting, action, mood) in 12-25 words. Do NOT restate the post's words. Do NOT "
        "depict any text, letters, logos, UI, charts or app screenshots. Avoid literally "
        "drawing abstract nouns (e.g. for 'clipboard upgrade story' do not draw a clipboard). "
        f"For this brand, aim for: {hint}. Output the scene only, no preamble."
    )
    user = f"Brand: {brand}\nPost headline: {topic}\nPost body: {body}\n\nVisual scene:"
    for cfg in configs:
        out = _call_llm(system, user, cfg, timeout=45)
        if out:
            scene = out.strip().strip('"').splitlines()[0].strip()
            # Guard against the model echoing instructions or going long.
            if 4 <= len(scene) <= 240:
                return scene
    return None


# ── Batch generation ───────────────────────────────────────────────────

def generate_drafts_llm(
    brand: str,
    topics: list,
    platform: Optional[str] = None,
    self_call_body_texts: Optional[list[str]] = None,
) -> list[dict]:
    """Batch generate LLM drafts for personal brands.

    Drop-in replacement for llm_drafts.generate_drafts.
    Mirrors the same calling convention and return shape.

    In cron mode: returns [] because the cron agent generates text from
    build_generation_prompt() output. Each topic's prompt is available
    via build_generation_prompt(brand, topic, platform).

    In self-call mode: self_call_body_texts provides the LLM-generated text.
    Each entry is gated via generate_with_text() before being accepted.
    """
    from config import BRANDS

    # Derive platform from brand config when not specified
    if platform:
        platforms = [platform]
    else:
        brand_config = BRANDS.get(brand, {})
        brand_platforms = brand_config.get("platforms", [])
        platforms = [brand_platforms[0]] if brand_platforms else ["twitter"]

    drafts = []
    text_idx = 0

    for topic in topics:
        for plat in platforms:
            # Active editorial routing gate. Source-aware topics can be skipped
            # before any LLM spend or static fallback. Static app-brand topics
            # without provenance continue through the legacy path.
            try:
                from editorial_router import route_signal
                signal = _signal_from_topic(topic)
                if _has_source_context(signal):
                    route = route_signal(signal, candidate_surfaces=[_surface_for_platform(plat)])
                    if route.get("decision") == "skip":
                        print(f"[llm_generate] editorial skip {brand}/{plat}: {route.get('rationale','')}", file=sys.stderr)
                        continue
            except Exception as exc:
                print(f"[llm_generate] editorial routing failed open for {brand}/{plat}: {exc}", file=sys.stderr)

            if self_call_body_texts and text_idx < len(self_call_body_texts):
                body_text = self_call_body_texts[text_idx]
                text_idx += 1
                draft = generate_with_text(brand, topic, plat, body_text)
            else:
                draft = generate_one(brand, topic, plat)

            if draft:
                draft["brand"] = brand
                draft["pillar"] = topic.get("pillar", "")
                draft["topic"] = topic.get("topic", "")
                drafts.append(draft)
            else:
                # ── sahil_twitter anti-scripted guardrail ──
                # No static template fallback. Sahil personal X drafts MUST
                # start from a current internal activity/live signal. If LLM
                # generation failed and there's no activity_data, skip entirely.
                if brand == "sahil_twitter":
                    print(f"[llm_generate] sahil_twitter/{plat}: LLM failed, no activity signal; skipping (anti-scripted guard)")
                    continue

                # Fall back to static template if LLM path fails — but make the
                # degradation loud, never silent.
                _alert_static_fallback(brand, plat)
                from llm_drafts import generate_drafts as fallback_gen
                fallbacks = fallback_gen(brand, [topic], platform=plat)
                if fallbacks:
                    fb = fallbacks[0]
                    fb["id"] = f"{brand[:4]}_{str(uuid.uuid4())[:8]}"
                    fb["brand"] = brand
                    drafts.append(fb)

    return drafts
