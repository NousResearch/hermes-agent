"""
Meta-Router v2.0 — Message classification for PiClaw / SoM Council.

Classifies any incoming message into one of 7 categories:
  code | audit | research | production | integration | design | config

Returns: { "type": str, "mode": str, "directive": str, "confidence": float }

Pure Python module — no FastAPI dependency. Import directly in tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ── Category type alias ──────────────────────────────────────────────────────
Category = Literal["code", "audit", "research", "production", "integration", "design", "config"]

# ── Keyword maps (order matters — first match wins on tie-break) ─────────────
_RULES: list[tuple[Category, list[str]]] = [
    ("audit", [
        r"\baudit\b", r"\bsecurity\b", r"\bvulner",
        r"\bpentest\b", r"\bscan\b", r"\bcve\b", r"\bleak\b", r"\bexploit\b",
        r"\bthreat\b", r"\bcompli", r"\bsast\b", r"\bdast\b",
    ]),
    ("design", [
        r"\bui\b", r"\bux\b", r"\bdesign\b", r"\blayout\b", r"\bcomponent\b",
        r"\bstyle\b", r"\bcss\b", r"\btheme\b", r"\bfigma\b", r"\bwireframe\b",
        r"\bmockup\b", r"\bfont\b", r"\bcolor\b", r"\bresponsive\b", r"\banimation\b",
        r"\btailwind\b", r"\bdark.?mode\b", r"\bdashboard\b", r"\bvisual\b",
        r"\bpanel\b", r"\bcard\b", r"\bmodal\b", r"\btooltip\b",
        r"\barchitecture\b", r"\bdiagram\b", r"\bblueprint\b", r"\bschema\b",
    ]),
    ("research", [
        r"\bresearch\b", r"\bfind\b", r"\bwhat is\b", r"\bhow does\b",
        r"\bexplain\b", r"\bsummar", r"\bdocument", r"\blearn\b",
        r"\bcompare\b", r"\banalys", r"\binvestigat", r"\blook.?up\b",
    ]),
    ("production", [
        r"\bdeploy\b", r"\brelease\b", r"\brollout\b",
        r"\bprod\b", r"\bproduction\b",
        r"\bmonitor\b", r"\bdowntime\b", r"\bincident\b", r"\balert\b",
        r"\bpager\b", r"\bhotfix\b", r"\brollback\b", r"\bscale\b",
        r"\bk8s\b", r"\bkubernetes\b", r"\bdocker\b", r"\bcicd\b",
        r"\bcutover\b", r"\blive_env\b", r"\bpagerduty\b", r"\boncall\b",
        r"\bsla\b", r"\bpostmortem\b", r"\bfailover\b", r"\bcanary\b",
        r"\bblue.?green\b", r"\bprod.deploy\b",
    ]),
    ("integration", [
        r"\bapi\b", r"\bwebhook\b", r"\bintegrat", r"\bconnect\b",
        r"\boauth\d*\b", r"\bauth(?:entication|orization|\.json|\.yaml)\b", r"\btoken\b",
        r"\bsdk\b", r"\bmcp\b", r"\bplugin\b", r"\bmiddleware\b", r"\bbridge\b",
        r"\binterface\b",
    ]),
    ("config", [
        r"\bconfig", r"\bsetup\b", r"\bset.?up\b", r"\binstall\b", r"\benv\b",
        r"\bsetting\b", r"\byaml\b", r"\bjson\b", r"\btoml\b",
        r"\bdotenv\b", r"\binit\b", r"\bbootstrap\b", r"\bsecret\b",
        r"\bcredential\b",
        # Web servers / reverse proxies
        r"\bnginx\b", r"\bapache\b", r"\bcaddy\b", r"\btraefik\b",
        r"\bproxy\b", r"\breverse.proxy\b", r"\bvhost\b", r"\bvirtual.host\b",
        # TLS/SSL
        r"\bssl\b", r"\btls\b", r"\bcertificate\b", r"\bcertbot\b",
        r"\bhttps\b", r"\bacme\b",
        # System services
        r"\bsystemd\b", r"\bservice\b", r"\bdaemon\b", r"\bsystemctl\b",
        r"\bsupervisor\b", r"\bpm2\b",
        # Firewall / network
        r"\bfirewall\b", r"\biptables\b", r"\bufw\b", r"\bport\b",
        # File extensions
        r"\.conf\b", r"\.ini\b", r"\.env\b",
    ]),
    ("code", [
        r"\bcode\b", r"\bfunction\b", r"\bclass\b", r"\bscript\b",
        r"\bbug\b", r"\bfix\b", r"\brefactor\b", r"\btest\b",
        r"\bfeature\b", r"\bimplement\b", r"\bwrite\b", r"\bbuild\b",
        r"\bmodule\b", r"\bpython\b", r"\btypescript\b", r"\bjavascript\b",
        r"\brust\b", r"\bgo\b", r"\bsql\b", r"\bquery\b",
        r"\bnull\b", r"\bpointer\b", r"\bexception\b", r"\berror\b",
        r"\bcrash\b", r"\.py\b", r"\.ts\b", r"\.js\b",
        r"\bparse\b", r"\bloop\b", r"\barray\b", r"\bobject\b",
    ]),
]

# ── Mode inference ───────────────────────────────────────────────────────────
_MODE_RULES: list[tuple[str, list[str]]] = [
    ("urgent",  [r"\burgent\b", r"\basap\b", r"\bcrash\b", r"\bdown\b", r"\bbroken\b"]),
    ("review",  [r"\breview\b", r"\bcheck\b", r"\blook.?at\b", r"\bverify\b"]),
    ("plan",    [r"\bplan\b", r"\barchitect\b", r"\bpropose\b", r"\bstrategy\b",
                r"\bbest.approach\b", r"\bbest.way\b", r"\brecommend\b", r"\badvise\b",
                r"\bapproach\b", r"\bshould.i\b", r"\bwhat.would\b", r"\bhow.to.approach\b",
                r"\boutline\b", r"\bsteps.to\b", r"\bhow.to.build\b", r"\bhow.to.set.?up\b"]),
    ("execute", []),   # default
]


@dataclass
class RouteResult:
    type: Category
    mode: str
    confidence: float
    directive: str


def classify(text: str) -> RouteResult:
    """
    Classify a message into a category and mode.

    :param text: raw message text
    :return: RouteResult with type, mode, confidence, and directive string
    """
    if not text or not text.strip():
        return RouteResult(type="code", mode="execute", confidence=0.0,
                           directive="[META-ROUTER | code | execute]")

    lower = text.lower()
    scores: dict[Category, int] = {cat: 0 for cat, _ in _RULES}

    for category, patterns in _RULES:
        for pattern in patterns:
            if re.search(pattern, lower):
                scores[category] += 1

    best_cat: Category = max(scores, key=lambda c: scores[c])  # type: ignore[arg-type]
    best_score = scores[best_cat]
    total_hits = sum(scores.values()) or 1
    confidence = round(best_score / total_hits, 3)

    # Fall back to "code" when nothing matched;
    # confidence=0.0 ensures the LLM fallback is always consulted
    # for unrecognised tasks (the threshold is < 0.5, not <= 0.5)
    if best_score == 0:
        best_cat = "code"
        confidence = 0.0

    # Infer mode
    mode = "execute"
    for m, patterns in _MODE_RULES[:-1]:  # skip the default "execute" entry
        if any(re.search(p, lower) for p in patterns):
            mode = m
            break

    directive = f"[META-ROUTER | {best_cat} | {mode}]"
    result = RouteResult(type=best_cat, mode=mode, confidence=confidence, directive=directive)
    if result.confidence < 0.5:
        try:
            from gateway.meta_router_llm import llm_classify
        except Exception:
            llm_classify = None
        if llm_classify is not None:
            llm_result = llm_classify(text, result)
            if llm_result is not None:
                return llm_result
    return result


def prepend_directive(text: str, result: RouteResult) -> str:
    """Return the original text with the meta-router directive prepended."""
    return f"{result.directive}\n\n{text}"
