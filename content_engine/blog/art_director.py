"""Art director — one rich art brief per article, shared across hero + sections.

The blog image path is Codex CLI only. This module owns the creative layer
before Codex sees a prompt:

  - choose one visual mode for the whole post
  - apply creative-concept-direction: layered scene, tension, medium-as-concept
  - expose Baoyu-derived layout/workflow modes as Codex-compatible prompt modes
  - allow text/labels ONLY for styles designed for graphical information design
  - compile each image into a vivid prompt, not a long checklist

The illustrator fills each compiled prompt into Codex. Palette, layout grammar,
and motif are shared so the post reads as one designed set; hero and section
concepts differ so the images are not repetitive.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Mapping, Optional

from blog.asset_manifest import AssetManifest, build_asset_manifest
from blog.concept_direction import (
    INSPIRATION_CATEGORIES,
    ConceptPlanError,
    build_concept_plan,
)
from blog.reference_catalog import ReferenceCatalog
from blog.visual_plan import VisualPlan, build_visual_plan

# ── Creative direction contract ─────────────────────────────────────────────

CREATIVE_DIRECTION_RULES = (
    "Before choosing a style, apply the Creative Concept Direction layer: "
    "extract 3-5 complementary or conflicting metaphors, blend an unexpected "
    "era/genre/medium, build a scene with 5-12 concrete interacting elements, "
    "and pick a medium-as-concept format. The image must be scroll-stopping, "
    "not merely tasteful. Avoid single-symbol metaphors and safe stock scenes. "
    "Compile the final image idea into 3-5 vivid sentences, not a checklist."
)

REFERENCE_IMAGE_STANDARD = (
    "Quality bar: Sahil's local reference images favour dense editorial design, "
    "poster/infographic craft, deliberate composition, visible structure, and "
    "graphical detail. Treat that as the taste target: designed artefact, not "
    "generic illustration."
)

TEXT_POLICIES = {
    "none": (
        "No readable text, letters, numbers, captions, logos, or UI screenshots. "
        "Use abstract glyphs, marks, panels, arrows, ticks, and symbolic blocks instead."
    ),
    "labels": (
        "Text is allowed as graphical design detail. Use only short, deliberate, "
        "large, legible English labels: 1-6 words each, max 6 labels total. "
        "No paragraphs, no tiny illegible UI text, no lorem ipsum, no gibberish. "
        "If text is included, it must be spelled exactly as requested."
    ),
    "typography": (
        "Typography is the image. Use a short exact phrase as the main visual form, "
        "with large crisp lettering, poster-grade hierarchy, and no misspellings. "
        "Keep supporting labels minimal and legible."
    ),
}

# ── Curated style/workflow library ──────────────────────────────────────────
# Includes direct visual styles AND Codex-adapted workflow styles. Workflow
# skills (Baoyu, data, typography, diorama, etc.) are represented as prompt
# modes here; the blog path never calls their original FAL/Pollinations tools.
STYLE_LIBRARY = [
    {
        "id": "mythic-tech-codex",
        "label": "Mythic Tech Codex",
        "kind": "direct-style",
        "look": "Edwardian/Victorian scientific-illustration plate; ink linework, antique watercolour, sepia and aged-paper tones, museum encyclopedia feel.",
        "best_for": "frameworks, theory, first-principles mechanics, taxonomies.",
        "layout": "annotated scientific plate, cutaway, taxonomy grid, specimen card, marginalia map",
        "text_policy": "labels",
    },
    {
        "id": "ninth-observatory",
        "label": "The Ninth Observatory",
        "kind": "direct-style",
        "look": "systems-as-architecture; vast built spaces, cross-sections of halls, conveyors, archives; muted stone-and-brass palette; one warm focal light; awe of scale.",
        "best_for": "infrastructure, pipelines, routing, memory systems, orchestration, spatial systems.",
        "layout": "architectural cross-section, control hall, vault map, conveyor network, layered city-section",
        "text_policy": "labels",
    },
    {
        "id": "chromatic-institute",
        "label": "Chromatic Institute",
        "kind": "direct-style",
        "look": "clean modern research abstraction; networked nodes and fields; confident saturated colour on light ground; Bauhaus-ish geometry; optimistic and precise.",
        "best_for": "research findings, model behaviour, networks, abstract conceptual pieces.",
        "layout": "node field, matrix, research wall, layered geometric map, signal landscape",
        "text_policy": "labels",
    },
    {
        "id": "signal-hud",
        "label": "Dark Cyberpunk HUD",
        "kind": "direct-style",
        "look": "dark technical HUD; neon data-flows and instrument panels on near-black; one glowing focal element; diagnostic command-layer mood; restrained, not rainbow.",
        "best_for": "observability, evals, metrics, diagnostics, debugging, production reliability.",
        "layout": "mission-control HUD, telemetry strips, fault-map, diagnostic cockpit, dark systems topology",
        "text_policy": "labels",
    },
    {
        "id": "baoyu-article-illustrator",
        "label": "Baoyu Article Illustrator",
        "kind": "workflow-adapter",
        "look": "article-illustration system built from Type × Style × Palette: infographic, scene, flowchart, comparison, framework, timeline; consistent visual language across a post.",
        "best_for": "choosing the right information structure for article images rather than only a painterly style.",
        "layout": "type-driven: infographic, scene, flowchart, comparison, framework, timeline",
        "text_policy": "labels",
    },
    {
        "id": "baoyu-infographic",
        "label": "Baoyu Infographic",
        "kind": "workflow-adapter",
        "look": "high-density infographic system: 21 layout families × 21 styles; clean panels, bento grids, arrows, comparisons, hierarchy, matrices, visual summaries.",
        "best_for": "step-by-step explainers, comparisons, frameworks, decision flows, PM/product/AI systems.",
        "layout": "bento-grid, linear progression, binary comparison, comparison matrix, hierarchical layers, radial map, process flow, quadrant chart",
        "text_policy": "labels",
    },
    {
        "id": "baoyu-comic",
        "label": "Baoyu Knowledge Comic",
        "kind": "workflow-adapter",
        "look": "single-card educational comic adapted from Baoyu comic workflow; 2-4 panels inside one 16:9 composition, expressive characters, visual explanation, concise captions.",
        "best_for": "teaching concepts, failure modes, before/after explanations, memorable mental models.",
        "layout": "single-card comic, 2-panel contrast, 3-panel progression, 4-panel grid, annotated character scene",
        "text_policy": "labels",
    },
    {
        "id": "technical-diorama",
        "label": "Technical Diorama",
        "kind": "direct-style",
        "look": "isometric exploded views, miniature worlds, cutaways, hardware/software systems as physical mechanisms; precise, dimensional, tactile.",
        "best_for": "architecture, agents, pipelines, infrastructure, product mechanics, how-it-works breakdowns.",
        "layout": "isometric exploded view, cutaway box, miniature factory, sectional model, labelled parts diagram",
        "text_policy": "labels",
    },
    {
        "id": "data-atlas",
        "label": "Data Atlas",
        "kind": "direct-style",
        "look": "editorial data visualisation as art: chord diagrams, Sankey flows, treemaps, network graphs, heatmaps, small multiples; publication-grade and narrative-led.",
        "best_for": "quantitative claims, benchmarks, market maps, costs, trade-offs, relationships, distributions.",
        "layout": "Sankey, chord diagram, treemap, heatmap, network graph, small-multiples grid, annotated chart spread",
        "text_policy": "labels",
    },
    {
        "id": "typographic-poster-design",
        "label": "Typographic Poster Design",
        "kind": "direct-style",
        "look": "typography-forward poster where words are the image: Saul Bass minimalism, Swiss grid, propaganda, pulp masthead, concert flyer, concrete poetry.",
        "best_for": "sharp theses, principles, manifestos, titles with strong wording, punchy PM/AI claims.",
        "layout": "poster masthead, Swiss grid, typographic collage, propaganda placard, concrete-poetry layout",
        "text_policy": "typography",
    },
    {
        "id": "vintage-print-atelier",
        "label": "Vintage Print Atelier",
        "kind": "direct-style",
        "look": "carnival posters, pulp covers, propaganda prints, concert flyers, woodcut/risograph texture, aged paper, bold handmade lettering.",
        "best_for": "contrarian takes, satire, dramatic claims, weird blends, memorable editorial hooks.",
        "layout": "pulp cover, carnival poster, propaganda broadside, wanted poster, flyposted street sheet",
        "text_policy": "typography",
    },
    {
        "id": "photographic-realism",
        "label": "Photographic Realism",
        "kind": "direct-style",
        "look": "candid iPhone, 35mm documentary, surveillance still, archival press photo; grounded, believable, human context.",
        "best_for": "workplace/product stories, human adoption, operations, real-world consequences.",
        "layout": "documentary still, desk scene, control room photo, surveillance frame, archival press shot",
        "text_policy": "none",
    },
    {
        "id": "cosmic-postcard",
        "label": "Cosmic Postcard Atelier",
        "kind": "direct-style",
        "look": "mid-century Space Age retro-futurism; optimistic travel-poster composition; warm oranges/teals/cream; cinematic horizon.",
        "best_for": "futures, adoption curves, trajectories, opportunity, forward-looking pieces.",
        "layout": "travel poster, horizon scene, retro launch pad, destination map, cinematic vista",
        "text_policy": "typography",
    },
    {
        "id": "ink-ember-studio",
        "label": "Ink & Ember Studio",
        "kind": "direct-style",
        "look": "painterly, human-centred contemporary fine art; warm emotive palette; real people and gesture; editorial-magazine feel.",
        "best_for": "leadership, culture, careers, trust, human-in-the-loop, reflective essays.",
        "layout": "human scene, workshop, conversation tableau, symbolic portrait, warm editorial spread",
        "text_policy": "none",
    },
    {
        "id": "saga-noir",
        "label": "Saga Noir Studio",
        "kind": "direct-style",
        "look": "bold graphic-novel mythology; high-contrast ink, dramatic silhouettes, limited spot colour; powerful and decisive.",
        "best_for": "high-stakes decisions, competitive strategy, transformation, pivotal moments.",
        "layout": "comic splash page, dramatic confrontation, mythic threshold, split battlefield, symbolic duel",
        "text_policy": "labels",
    },
    {
        "id": "pixel-art",
        "label": "Pixel Art",
        "kind": "direct-style",
        "look": "deliberate pixel-art; visible pixel blocks, limited SNES/PICO-8 palette, optional CRT glow; playful but crafted.",
        "best_for": "tooling, CLIs, indie/retro, small-and-scrappy builder notes. Rare specialist.",
        "layout": "pixel UI map, side-scroller system, inventory screen, retro dashboard, isometric pixel room",
        "text_policy": "labels",
    },
]

# Rejected from SahilBlog after visual review: their output was too repetitive
# and too far from the editorial direction. Keep their definitions for legacy
# plan readability, but never offer or validate them for new blog generation.
BLOG_EXCLUDED_STYLE_IDS = frozenset({"ninth-observatory", "chromatic-institute"})
STYLE_IDS = {s["id"] for s in STYLE_LIBRARY if s["id"] not in BLOG_EXCLUDED_STYLE_IDS}
STYLE_BY_ID = {s["id"]: s for s in STYLE_LIBRARY}

STYLE_NATIVE_COMPILERS = {
    "baoyu-article-illustrator": "Redraft the concept as an article-illustration system: choose a clear Type (infographic, scene, flowchart, comparison, framework, or timeline), visible information hierarchy, section cards, arrows, and exact short labels.",
    "baoyu-infographic": "Redraft the concept as a dense infographic: bento panels, comparison blocks, numbered steps, icon-like objects, arrows, category labels, and a strong top-to-bottom reading path.",
    "baoyu-comic": "Redraft the concept as a single-card knowledge comic: 2-4 panels inside one 16:9 image, expressive characters or objects, before/after contrast, concise caption labels, and a memorable punchline moment.",
    "technical-diorama": "Redraft the concept as a physical mechanism: isometric cutaway, exploded parts, labelled components, miniature operators, cables, levers, gauges, material textures, and a visible cause-effect path.",
    "data-atlas": "Redraft the concept as editorial data cartography: Sankey/chord/treemap/heatmap/network logic, labelled flows, scale cues, clusters, legends, and publication-grade annotation.",
    "typographic-poster-design": "Redraft the concept as a poster where typography carries the argument: one dominant exact phrase, grid hierarchy, expressive type placement, sharp negative space, and minimal supporting symbols.",
    "vintage-print-atelier": "Redraft the concept as a vintage printed artefact: pulp/carnival/propaganda composition, bold headline lettering, halftone or woodcut texture, dramatic central emblem, and aged paper craft.",
    "photographic-realism": "Redraft the concept as a plausible documentary photograph: real environment, human behaviour, specific props, camera/lens feel, imperfect lighting, and no symbolic floating UI unless physically present.",
    "mythic-tech-codex": "Redraft the concept as an antique scientific plate: specimen labels, taxonomy grid, cutaway annotations, marginal diagrams, brass instruments, and museum-catalogue precision.",
    "ninth-observatory": "Redraft the concept as architecture: control hall, vault, machine room, layered city-section, gantries, archives, operators, labelled chambers, and one warm focal system.",
    "chromatic-institute": "Redraft the concept as modern research graphics: geometric node fields, matrix panels, clean diagram layers, colour-coded clusters, and lab-wall precision.",
    "signal-hud": "Redraft the concept as diagnostic instrumentation: telemetry strips, fault maps, dashboard panes, command labels, warning lights, and a single high-signal focal readout.",
    "cosmic-postcard": "Redraft the concept as retro-future travel poster: destination horizon, bold title treatment, Space Age composition, route lines, optimistic scale, and cinematic atmosphere.",
    "ink-ember-studio": "Redraft the concept as a human editorial scene: gesture, expression, warm light, real stakes, tactile setting, and symbolic props embedded naturally in the room.",
    "saga-noir": "Redraft the concept as graphic-novel mythology: dramatic threshold, silhouettes, decisive conflict, symbolic duel, strong spot colour, and cinematic panel energy.",
    "pixel-art": "Redraft the concept as a crafted retro system: pixel UI rooms, inventory/map/dashboard metaphors, limited palette, blocky readable labels, and playful implementation detail.",
}

def _styles_catalogue() -> str:
    lines = []
    for s in STYLE_LIBRARY:
        if s["id"] in BLOG_EXCLUDED_STYLE_IDS:
            continue
        lines.append(
            f"- {s['id']} ({s['label']}): kind={s['kind']}; look={s['look']} "
            f"Best for: {s['best_for']} Layout grammar: {s['layout']} "
            f"Text policy: {s['text_policy']} — {TEXT_POLICIES[s['text_policy']]}"
        )
    return "\n".join(lines)




def _selection_seed(draft: dict) -> str:
    """Stable selector seed for debugging/reproducibility logs."""
    seed_src = "|".join([
        str(draft.get("title", "")),
        str(draft.get("description", "")),
        str(draft.get("stream", "")),
    ])
    return hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:16]


def _layout_variants_for(style_id: str, base_layout: str = "") -> list[str]:
    raw = base_layout or STYLE_BY_ID.get(style_id, {}).get("layout", "")
    variants = [x.strip() for x in re.split(r",|;|\bor\b", raw) if x.strip()]
    return variants[:6] or [raw or "editorial composition"]


def _normalise_layout(brief: dict, style_id: str) -> tuple[str, list[str]]:
    variants = brief.get("layout_variants") or []
    if isinstance(variants, str):
        variants = _layout_variants_for(style_id, variants)
    elif isinstance(variants, list):
        variants = [str(v).strip() for v in variants if str(v).strip()]
    base = str(brief.get("layout") or STYLE_BY_ID.get(style_id, {}).get("layout", "")).strip()
    merged = []
    for item in [base, *_layout_variants_for(style_id, base), *variants]:
        if item and item not in merged:
            merged.append(item)
    return (base or merged[0], merged[:6])


def _native_compiler_for(style_id: str) -> str:
    return STYLE_NATIVE_COMPILERS.get(style_id, "Redraft the concept in the native language of the chosen visual style, with concrete composition, material, label, and reading-path decisions.")

def _brief_system_prompt(
    recent_styles: list[str], recent_concept_fingerprints: list[str] | None = None
) -> str:
    recent_style_rule = (
        "Recently used rendering styles (prefer a materially different one unless article fit requires it): "
        + ", ".join(recent_styles)
        + ".\n\n"
        if recent_styles else ""
    )
    recent_world_rule = (
        "Recently used creative-world fingerprints (do not repeat these worlds): "
        + "; ".join(recent_concept_fingerprints or [])
        + ".\n\n"
        if recent_concept_fingerprints else ""
    )
    return (
        "You are the concept director for a technical publication. Each article earns its own "
        "rendering language: cinematic scene, comic, infographic, retro artefact, documentary "
        "photo, technical cutaway, or another style from the supplied catalogue. Do not default "
        "to one house style. Choose visual variation only when it makes the article clearer, more "
        "memorable, or more emotionally precise — never as random decoration.\n\n"
        f"Choose one inspiration_category per candidate from: {', '.join(sorted(INSPIRATION_CATEGORIES))}.\n\n"
        f"Available rendering styles (use the exact id in the style field):\n{_styles_catalogue()}\n\n"
        + recent_style_rule
        + recent_world_rule
        + "Your job is to create controlled conceptual variation. First identify the article truth. "
        "Then produce 3-5 candidate shared worlds by combining it with an unexpected world, a "
        "narrative rule, and a story moment. Every candidate must visibly explain the article; "
        "randomness is only the translation, never the reason for the image. The three asset "
        "scenes must be materially different: hero=thesis, each section=its local mechanism or consequence.\n\n"
        "For every scene use the exact asset_key and source_target_id supplied by the user. Include: "
        "the article claim, its visual translation, a precise scene, a unique composition, and a "
        "unique creative technique. No candidate is valid if it cannot supply every requested asset.\n\n"
        "Return ONLY a JSON object, no prose, with this exact shape:\n"
        '{\n'
        '  "style": "<one style id from the supplied catalogue>",\n'
        '  "layout": "<legacy compatibility layout>",\n'
        '  "text_policy": "none",\n'
        '  "text_elements": [],\n'
        '  "palette": "<legacy compatibility palette>",\n'
        '  "motif": "<recurring object or shape>",\n'
        '  "art_direction": "<shared rendering rules>",\n'
        '  "hero_prompt": "<legacy compatibility placeholder; it will be compiled from the scene>",\n'
        '  "section_prompts": [],\n'
        '  "concept_candidates": [\n'
        '    {"candidate_id":"<stable short id>","world":"<one shared creative world>",'
        '"fingerprint":["<3 concrete world terms>"],"creative_score":<0-100>,\n'
        '     "relevance_rationale":"<why every scene explains the article>",'
        '"inspiration_category":"<one approved inspiration category>",'
        '"specialist_skill":"<optional specialist override; normally omit>",\n'
        '     "scenes":[{"asset_key":"hero","source_target_id":"hero",'
        '"source_claim":"<claim from that target>","visual_translation":"<claim made visible>",'
        '"scene":"<specific moment>","composition":"<specific layout>",'
        '"creative_technique":"<distinct story device>"}]}\n'
        "  ]\n"
        "}\n"
    )


def _brief_user_prompt(title: str, description: str, body_md: str,
                       stream: str, headings: list[str]) -> str:
    body = body_md.strip()
    if len(body) > 6000:
        body = body[:6000] + "\n...[truncated]"
    heads = "\n".join(f"- section-{index:02d}: {heading}" for index, heading in enumerate(headings, start=1)) if headings else "(none; hero only)"
    return (
        f"STREAM: {stream}\n"
        f"TITLE: {title}\n"
        f"DECK: {description}\n\n"
        "REQUIRED SOURCE TARGET IDS (use these exact ids in every candidate):\n"
        "- hero: title/deck and article thesis\n"
        f"{heads}\n\n"
        f"FULL ARTICLE:\n{body}\n\n"
        "Design 3-5 candidates. Each candidate must provide exactly one hero scene and "
        "one scene for every section target above, using the matching source_target_id."
    )


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced JSON object out of an LLM response."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    break
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _normalise_text_policy(style_id: str, supplied: str) -> str:
    default = STYLE_BY_ID.get(style_id, {}).get("text_policy", "none")
    supplied = (supplied or "").strip().lower()
    if supplied in TEXT_POLICIES:
        # Never let the model silently ban text on a style whose value is text.
        if default in {"labels", "typography"} and supplied == "none":
            return default
        # Never let labels/typography leak into styles not designed for readable text.
        if default == "none" and supplied != "none":
            return default
        # Keep typography-forward styles typography-forward, not merely labelled.
        if default == "typography" and supplied != "typography":
            return default
        # Never let typography leak into styles not designed for it.
        if supplied == "typography" and default != "typography":
            return default
        return supplied
    return default


def _scene_image_concept(world: str, scene) -> str:
    """Compile a reviewed scene into the concise prompt language a renderer needs."""
    return (
        f"Shared world: {world} "
        f"Article claim to make visible: {scene.source_claim} "
        f"Visual translation: {scene.visual_translation} "
        f"Scene: {scene.scene}"
    )


def _validate(brief: dict, headings: list[str], draft: Optional[dict] = None,
              recent_styles: Optional[list[str]] = None,
              recent_concept_fingerprints: Optional[list[str]] = None) -> Optional[dict]:
    """Coerce/validate an LLM brief into a usable shape, or None if unusable."""
    if not isinstance(brief, dict):
        return None
    style = str(brief.get("style", "")).strip()
    if style:
        low = style.lower()
        style = next((sid for sid in STYLE_IDS if sid == style or sid in low or low in sid), "")
        if not style:
            return None
    preferred_style = style
    hero = str(brief.get("hero_prompt", "")).strip()

    selection_seed = _selection_seed(draft) if draft is not None else ""
    concept_plan = None
    selected_candidate: Mapping[str, object] = {}
    if draft is not None:
        raw_candidates = brief.get("concept_candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            return None
        try:
            concept_plan = build_concept_plan(
                draft,
                headings,
                candidates=raw_candidates,
                recent_concept_fingerprints=recent_concept_fingerprints or [],
            )
        except ConceptPlanError:
            return None
        selected_candidate = next(
            (candidate for candidate in raw_candidates
             if isinstance(candidate, Mapping) and candidate.get("candidate_id") == concept_plan.candidate_id),
            {},
        )
        # Some providers place all rendering fields inside each candidate. Accept
        # that equivalent shape, but still validate the selected style locally.
        style = style or str(selected_candidate.get("style", "")).strip()
        hero = hero or str(selected_candidate.get("hero_prompt", "")).strip()
        low = style.lower()
        style = next((sid for sid in STYLE_IDS if sid == style or sid in low or low in sid), "")
        if not style or not hero:
            return None
        preferred_style = style
        # The scene and article select the rendering language. A specialist
        # candidate is a deliberate override; otherwise retain the validated
        # article-fit style supplied by the art director.
        style = concept_plan.specialist_skill or preferred_style

    direction_source = {**selected_candidate, **brief}
    palette = str(direction_source.get("palette", "")).strip()
    motif = str(direction_source.get("motif", "")).strip()
    direction = str(direction_source.get("art_direction", "")).strip()
    asset_layouts: dict[str, str] = {}
    if concept_plan is not None:
        asset_layouts = {scene.asset_key: scene.composition for scene in concept_plan.scenes}
        layout = asset_layouts["hero"]
        layout_variants = list(asset_layouts.values())
    else:
        layout_source = dict(brief)
        if style != preferred_style:
            layout_source["layout"] = STYLE_BY_ID.get(style, {}).get("layout", "")
            layout_source["layout_variants"] = []
        layout, layout_variants = _normalise_layout(layout_source, style)
    text_policy = _normalise_text_policy(style, str(direction_source.get("text_policy", "")))

    raw_text = direction_source.get("text_elements") or []
    text_elements: list[str] = []
    if isinstance(raw_text, list) and text_policy != "none":
        for item in raw_text[:6]:
            label = str(item).strip()
            if label:
                text_elements.append(label[:48])

    raw_sections = brief.get("section_prompts") or []
    section_prompts: dict[str, str] = {}
    if concept_plan is not None:
        hero = _scene_image_concept(
            concept_plan.world,
            next(scene for scene in concept_plan.scenes if scene.asset_key == "hero"),
        )
        for index, heading in enumerate(headings, start=1):
            scene = next(scene for scene in concept_plan.scenes if scene.asset_key == f"section-{index:02d}")
            section_prompts[heading] = _scene_image_concept(concept_plan.world, scene)
    elif isinstance(raw_sections, list):
        for i, h in enumerate(headings):
            if i < len(raw_sections) and isinstance(raw_sections[i], dict):
                p = str(raw_sections[i].get("prompt", "")).strip()
                if p:
                    section_prompts[h] = p
    return {
        "style": style,
        "layout": layout,
        "layout_variants": layout_variants,
        "asset_layouts": asset_layouts,
        "selection_seed": selection_seed,
        "style_candidates": [style],
        "style_native_compiler": _native_compiler_for(style),
        "text_policy": text_policy,
        "text_elements": text_elements,
        "palette": palette,
        "motif": motif,
        "art_direction": direction,
        "hero_prompt": hero,
        "section_prompts": section_prompts,
        **({"concept_plan": concept_plan.to_dict()} if concept_plan is not None else {}),
    }


def _text_rule_for(brief: dict) -> str:
    policy = brief.get("text_policy") or STYLE_BY_ID.get(brief.get("style", ""), {}).get("text_policy", "none")
    rule = TEXT_POLICIES.get(policy, TEXT_POLICIES["none"])
    labels = brief.get("text_elements") or []
    if policy != "none" and labels:
        quoted = ", ".join(json.dumps(x) for x in labels[:6])
        rule += f" Exact text elements to include if visually useful: {quoted}."
    return rule


def compose_prompt(image_concept: str, brief: dict, *, assigned_layout: str | None = None) -> str:
    """Compile one source-grounded scene into its deliberately assigned layout."""
    style = STYLE_BY_ID.get(brief["style"], {})
    active_layout = (assigned_layout or brief.get("layout") or style.get("layout", "")).strip()
    parts = [
        f"Create a 16:9 landscape editorial image in the {style.get('label', brief['style'])} mode.",
        f"Visual language: {style.get('look', '')}",
        f"Assigned layout/composition for this asset: {active_layout}.",
        f"Style-native prompt redraft rule: {brief.get('style_native_compiler') or _native_compiler_for(brief['style'])}",
    ]
    if brief.get("palette"):
        parts.append(f"Colour palette (use consistently): {brief['palette']}.")
    if brief.get("motif"):
        parts.append(f"Recurring visual motif to include: {brief['motif']}.")
    if brief.get("art_direction"):
        parts.append(f"Shared art direction: {brief['art_direction']}")
    # Extended style menu (blog path): additive neutral trait fragment from
    # the ~9,000-style registry, merged into every composed prompt so the
    # whole post shares one deterministic blend (coherent within, varied
    # across posts). Never a protected name, never SREF.
    extended = brief.get("extended_traits") or {}
    if extended.get("fragment"):
        parts.append(f"Additional neutral style traits: {extended['fragment']}.")
    parts.extend([
        f"Compiled image concept: {image_concept}",
        CREATIVE_DIRECTION_RULES,
        REFERENCE_IMAGE_STANDARD,
        _text_rule_for(brief),
        "High detail, intentional editorial composition, graphical design detail, strong reading path, not generic AI art, no logos, no watermarks.",
    ])
    return "\n\n".join(p for p in parts if p)


def build_art_brief(
    draft: dict,
    headings: list[str],
    recent_styles: Optional[list[str]] = None,
    recent_concept_fingerprints: Optional[list[str]] = None,
    llm=None,
) -> Optional[dict]:
    """Produce a validated art brief for a post via one LLM pass."""
    recent_styles = recent_styles or []
    if llm is None:
        try:
            from llm_generate import _call_llm, _llm_configs

            def default_llm(system: str, user: str) -> Optional[str]:
                for cfg in _llm_configs(longform=True):
                    # The creative contract now asks for multiple fully grounded
                    # scenes. Reasoning providers need enough room for their
                    # private deliberation plus the complete nested JSON payload.
                    # Retry one contract-invalid transport success on the same
                    # provider. Structured output is stochastic, while transport
                    # failures are already retried inside _call_llm and should
                    # advance to the next configured provider.
                    for _contract_attempt in range(2):
                        body = _call_llm(system, user, cfg, timeout=180, max_tokens=16384)
                        if not body:
                            break
                        # A transport success is not enough: providers often return
                        # prose or near-JSON that cannot drive the image pipeline.
                        # Accept only responses that are both parseable and
                        # contract-valid; otherwise consume the bounded retry.
                        parsed = _extract_json(body)
                        if parsed and _validate(parsed, headings, draft=draft, recent_styles=recent_styles,
                                                recent_concept_fingerprints=recent_concept_fingerprints):
                            return body
                return None

            llm = default_llm
        except Exception:
            return None

    system = _brief_system_prompt(recent_styles, recent_concept_fingerprints)
    user = _brief_user_prompt(
        draft.get("title", ""), draft.get("description", ""),
        draft.get("body_md", ""), draft.get("stream", "ai"), headings,
    )
    try:
        raw = llm(system, user)
    except Exception:
        return None
    brief = _extract_json(raw or "")
    if brief is None:
        return None
    return _validate(brief, headings, draft=draft, recent_styles=recent_styles,
                     recent_concept_fingerprints=recent_concept_fingerprints)


def fallback_brief(draft: dict, headings: list[str],
                   recent_styles: Optional[list[str]] = None) -> dict:
    """Deterministic brief retained for tests/manual inspection only.

    `blog_illustrator` deliberately does NOT call this when the LLM fails.
    """
    recent = recent_styles or []
    ordered = [
        s["id"] for s in STYLE_LIBRARY
        if s["id"] not in recent and s["id"] not in BLOG_EXCLUDED_STYLE_IDS
    ] or list(STYLE_IDS)
    style = ordered[0]
    style_meta = STYLE_BY_ID.get(style, {})
    title = draft.get("title", "")
    desc = draft.get("description", "")
    concept = f"{title}. {desc}".strip(". ")
    text_policy = style_meta.get("text_policy", "none")
    layout, layout_variants = _normalise_layout({"layout": style_meta.get("layout", "")}, style)
    return {
        "style": style,
        "layout": layout,
        "layout_variants": layout_variants,
        "selection_seed": _selection_seed(draft),
        "style_candidates": [style],
        "style_native_compiler": _native_compiler_for(style),
        "text_policy": text_policy,
        "text_elements": [title[:40]] if text_policy != "none" and title else [],
        "palette": "",
        "motif": "",
        "art_direction": "",
        "hero_prompt": concept,
        "section_prompts": {h: f"{h} — in the context of: {concept}" for h in headings},
    }


# ── Provider-free P11 contract seam ─────────────────────────────────────────

_PLAN_DIMENSIONS = {"width": 1600, "height": 900}


def _article_id_for_plan(draft: Mapping[str, object], brief: Mapping[str, object]) -> str:
    slug = str(draft.get("slug", "")).strip()
    if slug:
        return slug
    seed = str(brief.get("selection_seed", "")).strip()
    if seed:
        return f"article-{seed[:16]}"
    stable_title = str(draft.get("title", "")).strip() or "untitled"
    return f"article-{hashlib.sha256(stable_title.encode('utf-8')).hexdigest()[:16]}"


def _planned_layouts(
    brief: Mapping[str, object], count: int, fallback_layout_ids: list[str]
) -> list[str]:
    """Use brief layouts when supplied; otherwise name reviewed layout inputs.

    Older brief producers only supplied the shared art family. Naming the
    deterministically selected reviewed layout references keeps that public
    brief shape compatible without inventing a visual direction.
    """
    candidates: list[str] = []
    raw_variants = brief.get("layout_variants")
    variants = raw_variants if isinstance(raw_variants, list) else []
    for value in [brief.get("layout", ""), *variants]:
        text = str(value).strip()
        if text and text not in candidates:
            candidates.append(text)
    for reference_id in fallback_layout_ids:
        fallback = f"reviewed-layout:{reference_id}"
        if fallback not in candidates:
            candidates.append(fallback)
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        raise ValueError("canonical core pack does not provide enough distinct layouts")
    return candidates[:count]


def _core_record_ids(
    catalog: ReferenceCatalog,
    visual_role: str,
    count: int,
    *,
    selection_seed: str = "",
    excluded_ids: set[str] | None = None,
) -> list[str]:
    excluded = excluded_ids or set()
    candidates = [
        record.reference_id
        for record in catalog.records_for_contract()
        if visual_role in record.allowed_roles and record.reference_id not in excluded
    ]
    if not candidates:
        raise ValueError(f"canonical core pack has no unused {visual_role!r} reference")
    offset = int(selection_seed or "0", 16) % len(candidates)
    return [candidates[(offset + index) % len(candidates)] for index in range(count)]


def build_visual_plan_from_brief(
    draft: Mapping[str, object],
    headings: list[str],
    brief: Mapping[str, object],
    *,
    catalog_root: Path,
) -> VisualPlan:
    """Bind one existing art brief to reviewed core references before generation.

    This is deliberately deterministic and provider-free: the plan can be
    reviewed or persisted without authorising any reference for generation.
    """
    catalog = ReferenceCatalog.load(Path(catalog_root))
    asset_count = 1 + len(headings)
    selection_seed = str(brief.get("selection_seed", ""))
    layout_ids = _core_record_ids(catalog, "layout", asset_count, selection_seed=selection_seed)
    layouts = _planned_layouts(brief, asset_count, layout_ids)
    style_ids = _core_record_ids(
        catalog, "style", asset_count, selection_seed=selection_seed, excluded_ids=set(layout_ids)
    )
    composition_ids = _core_record_ids(
        catalog,
        "composition",
        asset_count,
        selection_seed=selection_seed,
        excluded_ids=set(layout_ids) | set(style_ids),
    )
    style = str(brief.get("style", "")).strip()
    palette = str(brief.get("palette", "")).strip()
    motif = str(brief.get("motif", "")).strip()
    direction = str(brief.get("art_direction", "")).strip() or str(brief.get("hero_prompt", "")).strip()
    if not all((style, palette, motif, direction)):
        raise ValueError("art brief is missing shared style, palette, motif or direction")
    asset_rows: list[Mapping[str, object]] = []
    for index in range(asset_count):
        is_hero = index == 0
        asset_rows.append(
            {
                "role": "hero" if is_hero else "section",
                "key": "hero" if is_hero else f"section-{index:02d}",
                "layout": layouts[index],
                "style": style,
                "palette": palette,
                "motif": motif,
                "reference_assignments": {
                    "layout": [layout_ids[index]],
                    "style": [style_ids[index]],
                    "composition": [composition_ids[index]],
                },
                **({} if is_hero else {"section_heading": headings[index - 1]}),
            }
        )
    return build_visual_plan(
        article_id=_article_id_for_plan(draft, brief),
        art_brief=direction,
        assets=asset_rows,
        catalog=catalog,
    )


def build_planned_asset_manifest_from_plan(
    plan: VisualPlan,
    prompts_by_key: Mapping[str, str],
    output_paths_by_key: Mapping[str, str],
    *,
    text_policy: str,
) -> AssetManifest:
    """Create the immutable planned state before an unchanged generator runs."""
    records: list[Mapping[str, object]] = []
    for asset in plan.assets:
        prompt = str(prompts_by_key.get(asset.key, "")).strip()
        output_path = str(output_paths_by_key.get(asset.key, "")).strip()
        if not prompt or not output_path:
            raise ValueError(f"missing prompt or output path for planned asset {asset.key!r}")
        records.append(
            {
                "asset_key": asset.key,
                "article_id": plan.article_id,
                "state": "planned",
                "visual_plan_schema_version": plan.version,
                "visual_plan_digest": plan.digest(),
                "prompt": prompt,
                "prompt_digest": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "reference_inputs": [
                    {
                        "reference_id": assignment.reference_id,
                        "sha256": assignment.sha256,
                        "provenance_class": assignment.provenance_class,
                        "visual_role": assignment.visual_role,
                    }
                    for assignment in asset.reference_assignments
                ],
                "provider": None,
                "model": None,
                "output_path": output_path,
                "output_digest": None,
                "requested_dimensions": dict(_PLAN_DIMENSIONS),
                "actual_dimensions": None,
                "generated_at": None,
                "text_ocr": {"policy": text_policy or "none", "result": "not-run"},
                "visual_qa": {"status": "pending", "rejection_reasons": []},
                "review_status": "pending",
            }
        )
    return build_asset_manifest(plan.article_id, records)
