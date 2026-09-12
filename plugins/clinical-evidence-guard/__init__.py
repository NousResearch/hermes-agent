"""Clinical evidence certification for Alfred on Hermes.

The plugin records ephemeral evidence-provider activity and certifies clinical
answers at the common pre-delivery boundary. A failed certification returns
structured repair reasons to the agent instead of silently replacing completed
work after synthesis.

No prompt text, tool result or clinical content is persisted to disk.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any



_MAX_EVIDENCE_CHARS = 500_000
_MAX_STATES = 256
_STATE_TTL_SECONDS = 15 * 60

_CLINICAL_RE = re.compile(
    r"(?:"
    r"\b(?:cirug|quir[uú]rgic|rinoplast|mamoplast|mastopex|abdominoplast|blefaroplast|"
    r"lipoaspir|liposucci|anestes|sutur|cicatri|necros|infecci|antibi[oó]tic|"
    r"reconstru|complicaci|pr[oó]tes|implant|injert|colgaj|seroma|hematoma|"
    r"postoperat|posoperat|perioperat|subplatism|farmac|pedi[aá]tric|cefazolin|"
    r"apixab|eliquis|rivarox|xarelto|enoxapar|lovenox|anticoagul|dosis|mortalidad|mortality|cl[ií]nic\w*|meta[- ]?an[aá]lis|evaluador|"
    r"clustering|maude|motiva|ergonomix|diep|supercharg|diagn[oó]stic|tratamiento|"
    r"manejo|profilaxis|morbilidad|incidencia|prevalencia|tasa)\w*|"
    r"\b(?:PMID|DOI)\b|\bvenous congestion\b|"
    r"\bcuello\w*[^.!?]{0,100}\b(?:plano|subplatism|liberaci[oó]n)\w*|"
    r"\btecho\b[^.!?]{0,100}\b(?:desviaci[oó]n|conservar\w* en bloque)\b|"
    r"\brelleno\w*[^.!?]{0,120}\b(?:dolor|piel|p[aá]lid|reticular|vascular)\w*|"
    r"\b(?:estudio|an[aá]lisis)\b[^.!?]{0,140}\b(?:pre[/-]post|evaluador|"
    r"clustering|residente|p[eé]rdida)\w*|"
    r"\b(?:lifting|simclip|evidencia|abstract|gu[ií]a|clinical|dose|treatment|"
    r"management|surgery|surgical|postoperative|preoperative|guideline)\b"
    r")",
    re.IGNORECASE,
)
_DOSE_ORDER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|μg|µg|g|ml|mL|ui|iu|u)"
    r"(?:\s+(?:cada|every)\s+\d+\s*(?:h|hr|hora|horas))?\b",
    re.IGNORECASE,
)
_NONCLINICAL_LABEL_RE = re.compile(
    r"\b(?:no\s+cl[ií]nic[oa]s?|non[- ]?clinical)\b",
    re.IGNORECASE,
)
_SOFTWARE_ACTION_RE = re.compile(
    r"\b(?:audit\w*|revis\w*|depur\w*|implement\w*|correg\w*|edit\w*|"
    r"refactor\w*|test\w*|probar)\b",
    re.IGNORECASE,
)
_SOFTWARE_ARTIFACT_RE = re.compile(
    r"\b(?:plugin|c[oó]digo|code|repo(?:sitorio)?|runtime|hook|tests?|suite|"
    r"gateway|clasificador|m[oó]dulo|funci[oó]n|archivo|typescript|python|pytest|vitest)\b",
    re.IGNORECASE,
)
_BUSINESS_WORKFLOW_RE = re.compile(
    r"\b(?:marketing|comercial|ventas?|negocio|monetiz|seo|landing|campa[nñ]a|"
    r"publicidad|contenido|redes? sociales|producto|infoproducto|crm|emr|software|"
    r"sitio|web|bot|forma agent|ko-fi|hotmart)\w*\b",
    re.IGNORECASE,
)
_META_CLINICAL_PROCESS_RE = re.compile(
    r"\b(?:alfred|sistema|modelo|guard|respuesta\w*|negaci\w*|rechaz\w*|"
    r"reh[uú]s\w*|aprendi\w*|mejor\w*|ajust\w*|calibr\w*|correcci\w*|"
    r"auditor[ií]a|persist\w*|memoria)\b",
    re.IGNORECASE,
)
_ASSISTANT_FAILURE_COMPLAINT_RE = re.compile(
    r"\b(?:respuesta\w*\s+(?:gen[eé]ric\w*|autom[aá]tic\w*|in[uú]til\w*)|"
    r"no\s+responde\w*|no\s+sirve\w*|in[uú]til\w*|decepcion\w*|"
    r"no\s+quiero\s+volver\s+a\s+verl\w*|elimin\w*\s+(?:esa|ese|este|esta)\s+"
    r"(?:frase|respuesta)|se\s+repite\w*)\b",
    re.IGNORECASE,
)
_CLINICAL_SUBJECT_RE = re.compile(
    r"\b(?:paciente\w*|caso\w*|diagn[oó]stic\w*|s[ií]ntoma\w*|signo\w*|"
    r"procedimiento\w*|cirug[ií]a\w*|postoperat\w*|posoperat\w*|perioperat\w*|"
    r"ritidectom\w*|rinoplast\w*|mamoplast\w*|mastopex\w*|abdominoplast\w*|"
    r"blefaroplast\w*|liposucci\w*|lipoaspir\w*|hematoma\w*|seroma\w*|infecci[oó]n\w*|"
    r"necrosis\w*|pseudoaneurisma\w*|f[aá]rmaco\w*|medicamento\w*|"
    r"apixab\w*|cefazolin\w*|dosis\w*|tratamiento\w*|manejo\w*|"
    r"fisiopatolog[ií]a\w*|causalidad\w*|seguimiento\w*)\b",
    re.IGNORECASE,
)
_DIRECT_CARE_REQUEST_RE = re.compile(
    r"\b(?:dime|indica|recomienda|prescribe|administra|dar|usar)\b[^\n]{0,80}"
    r"\b(?:dosis|tratamiento|manejo|pauta|apixab|eliquis|rivarox|xarelto|"
    r"enoxapar|lovenox|anticoagul)\w*\b|\bqu[eé]\s+dosis\b|"
    r"\b(?:tratamiento|manejo|dosis|pauta)\w*\b[^\n]{0,80}\b(?:mi|este|esta|el|la|un|una)\s+paciente\b|"
    r"\b(?:mi|este|esta|el|la|un|una)\s+paciente\b[^\n]{0,80}\b(?:tratamiento|manejo|dosis|pauta)\w*\b",
    re.IGNORECASE,
)

_SAFE_DEGRADATION_RE = re.compile(
    r"(?:no pude|no se pudo|sin|falt[óo])[^.\n]{0,90}"
    r"(?:evidencia|fuente|literatura)[^.\n]{0,60}"
    r"(?:verific|recuper|respald|sustent)",
    re.IGNORECASE,
)
_DEGRADATION_TAIL_RE = re.compile(
    r"\b(?:sin embargo|pero|aunque|a[uú]n as[ií]|por otra parte|"
    r"administr\w*|indic\w*|reinici\w*|inici\w*|suspend\w*|"
    r"trat\w*|prescrib\w*|aplic\w*|retir\w*|mant\w*|"
    r"recomiend\w*|sugier\w*|debe\w*|deber[ií]a\w*)\b",
    re.IGNORECASE,
)

_SPECIFIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*[-–—]\s*\d+(?:[.,]\d+)?)?\s*"
    r"(?:%|mg|g|mcg|μg|µg|kg|ml|mL|l|L|ui|iu|u|"
    r"mmhg|cmh2o|mm|cm|h|hr|hrs|hora(?:s)?|min(?:uto)?s?|"
    r"d[ií]a(?:s)?|semana(?:s)?|mes(?:es)?|año(?:s)?)(?=$|\W)",
    re.IGNORECASE,
)
_WORD_NUMBER_UNIT_RE = re.compile(
    r"\b(?:cero|un[oa]?|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|"
    r"once|doce|trece|catorce|quince|veinte|treinta|one|two|three|four|five|"
    r"six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:miligram\w*|microgram\w*|gram\w*|mililitr\w*|unidad\w*|hora\w*|"
    r"minut\w*|d[ií]a\w*|semana\w*|mes\w*|año\w*|milligram\w*|microgram\w*|"
    r"gram\w*|milliliter\w*|unit\w*|hour\w*|minute\w*|day\w*|week\w*|month\w*|year\w*)\b",
    re.IGNORECASE,
)
_EFFECT_RE = re.compile(
    r"\b(?:RR|OR|HR|IRR|PRR|ROR|SMD|MD)\s*[:=]?\s*\d+(?:[.,]\d+)?\b",
    re.IGNORECASE,
)
_P_VALUE_RE = re.compile(r"\bp\s*[<=>]\s*\d+(?:[.,]\d+)?\b", re.IGNORECASE)
_PER_DENOMINATOR_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s+(?:por|per)\s+(?:(?:cada|each)\s+)?"
    r"\d+(?:[.,]\d+)?\b",
    re.IGNORECASE,
)
_RATIO_COUNT_RE = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*/\s*\d+(?:[.,]\d+)?\b|"
    r"\b\d+(?:[.,]\d+)?\s+(?:de|of)\s+\d+(?:[.,]\d+)?\s+"
    r"(?:pacientes?|patients?|casos?|cases?)\b)",
    re.IGNORECASE,
)
_CI_RE = re.compile(
    r"\b(?:IC|CI)\s*\d+(?:[.,]\d+)?\s*%?\s*(?:[:=]?\s*)"
    r"\d+(?:[.,]\d+)?\s*[-–—]\s*\d+(?:[.,]\d+)?\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s<>\]\[\"']+", re.IGNORECASE)
_UNICODE_FRACTION_RE = re.compile(
    r"\b\d*[¼½¾⅓⅔]\s*(?:mg|g|mcg|μg|µg|kg|ml|mL|l|L|ui|iu|u)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b", re.IGNORECASE)
_PMID_RE = re.compile(r"\bPM(?:ID|C)\s*:?\s*(\d{4,})\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_AUTHOR_YEAR_RE = re.compile(
    r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})\s+(?:et al\.?\s+)?(20\d{2})\b"
)
_NO_AUTHOR = {
    "en", "la", "el", "los", "las", "una", "uno", "para", "según",
    "guía", "estudio", "revisión", "desde", "hasta", "durante", "antes",
    "después", "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
}

_REAL_EVIDENCE_TOOLS = {
    "web_search",
    "web_extract",
    "mcp__consensus__search",
    "mcp__elicit__search_papers",
    "mcp__elicit__search_trials",
    "mcp__elicit__get_report",
    "mcp__elicit__get_systematic_review",
    "mcp__paperguide__search",
    "mcp__paperguide__fetch",
    "mcp__notebooklm__notebook_query",
    "mcp__notebooklm__notebook_query_status",
    "mcp__notebooklm__source_get_content",

    "mcp__braintools__buscar_evidencia",
    "mcp__braintools__buscar_biblioteca",
    "mcp__braintools__buscar_protocolo",
    "mcp__braintools__obtener_fulltext",
    "mcp__braintools__sintetizar_paper",
}
_ROUTER_ONLY_TOOLS = {
    "mcp__braintools__deep_research_route",
    "deep_research_route",
}
_GBRAIN_CORE_TOOLS = {
    "mcp__gbrain__query",
    "mcp__gbrain__search",
    "mcp__gbrain__recall",
    "mcp__gbrain__get_page",
    "mcp__gbrain__get_chunks",
    "mcp__gbrain__think",
}
_CORE_PROVENANCE_KEYS = {
    "collection", "corpus", "namespace", "path", "provenance",
    "slug", "source", "source_id", "source_type", "tags",
}
_ANCHOR_STOP = {
    "actual", "administrar", "clinical", "clinica", "clinico", "conducta",
    "dosis", "evidencia", "evaluacion", "existe", "fuente", "fuentes",
    "guideline", "guia", "indicar", "manejo", "paciente", "pacientes",
    "paper", "recomendacion", "recomienda", "reiniciar", "reinicie", "requiere",
    "riesgo", "segun", "tratamiento", "usar", "utilizar", "vigente",
    "corresponde", "escenario", "respuesta",
}
_EVIDENCE_CONTENT_KEYS = {
    "abstract", "answer", "body", "content", "description", "fulltext",
    "markdown", "passage", "passages", "quote", "result", "results",
    "snippet", "text",
}
_EVIDENCE_CONTAINER_KEYS = {"data", "documents", "hits", "items", "papers"}


@dataclass(frozen=True)
class _EvidenceRecord:
    provider: str
    tool_name: str
    document_ids: tuple[str, ...]
    passages: tuple[str, ...]
    session_id: str
    turn_id: str


@dataclass
class _TurnState:
    clinical: bool
    user_message: str = ""
    started_at: float = field(default_factory=time.monotonic)
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    evidence: list[_EvidenceRecord] = field(default_factory=list)
    evidence_chars: int = 0


_STATES: dict[tuple[str, str], _TurnState] = {}
_LOCK = threading.RLock()


def _session_key(session_id: str, turn_id: str = "") -> tuple[str, str]:
    value = str(session_id or "").strip()
    turn = str(turn_id or "").strip()
    if value:
        return value, turn
    return f"__thread__:{threading.get_ident()}", turn


def _thread_key(turn_id: str = "") -> tuple[str, str]:
    return f"__thread__:{threading.get_ident()}", str(turn_id or "").strip()


def _get_or_migrate_state(
    session_id: str, turn_id: str = "",
) -> tuple[tuple[str, str], _TurnState | None]:
    """Resolve the turn state when Hermes assigns session_id after pre_llm_call."""
    key = _session_key(session_id, turn_id)
    state = _STATES.get(key)
    if state is not None or not str(session_id or "").strip():
        return key, state
    provisional = _thread_key(turn_id)
    state = _STATES.pop(provisional, None)
    if state is not None:
        _STATES[key] = state
    return key, state


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace(",", ".")
    value = re.sub(r"[–—]", "-", value)
    return " ".join(value.split())


def _strip_reply_context(message: str) -> str:
    """Remove Telegram's quoted reply wrapper before classifying the new request."""
    text = str(message or "")
    if not text.lstrip().startswith("[Replying to:"):
        return text
    prefix, separator, authored = text.partition("\n\n")
    if separator and prefix.rstrip().endswith("]"):
        return authored.lstrip()
    return text


def _prune_states(now: float) -> None:
    stale = [
        key for key, state in _STATES.items()
        if now - state.started_at > _STATE_TTL_SECONDS
    ]
    for key in stale:
        _STATES.pop(key, None)
    if len(_STATES) <= _MAX_STATES:
        return
    oldest = sorted(_STATES, key=lambda key: _STATES[key].started_at)
    for key in oldest[: len(_STATES) - _MAX_STATES]:
        _STATES.pop(key, None)


def _is_clinical(message: str) -> bool:
    candidate = _NONCLINICAL_LABEL_RE.sub(" ", message or "")
    if not _DIRECT_CARE_REQUEST_RE.search(candidate):
        if _ASSISTANT_FAILURE_COMPLAINT_RE.search(candidate):
            return False
        if (
            _SOFTWARE_ACTION_RE.search(candidate)
            and _SOFTWARE_ARTIFACT_RE.search(candidate)
        ):
            return False
        if (
            _SOFTWARE_ACTION_RE.search(candidate)
            and _BUSINESS_WORKFLOW_RE.search(candidate)
        ):
            return False
        if (
            _META_CLINICAL_PROCESS_RE.search(candidate)
            and not _CLINICAL_SUBJECT_RE.search(candidate)
        ):
            return False
    return bool(
        _CLINICAL_RE.search(candidate)
        or _DOSE_ORDER_RE.search(candidate)
    )


def _parsed_result_has_error(result: str, status: str) -> bool:
    if str(status or "").lower() in {"error", "failed", "blocked"}:
        return True
    try:
        parsed = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if not isinstance(parsed, dict):
        return False
    if parsed.get("error"):
        return True
    if parsed.get("success") is False or parsed.get("ok") is False:
        return True
    if parsed.get("results") == [] and not any(
        parsed.get(key) for key in ("content", "data", "answer", "items")
    ):
        return True
    return False


def _has_gbrain_core_provenance(result: str) -> bool:
    try:
        parsed = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return False

    def visit(value: Any, *, trusted: bool = False) -> bool:
        if isinstance(value, str):
            return trusted and bool(re.fullmatch(
                r"(?:clinical[-_/\s]?)?core(?:$|[/_\s].*)",
                value.lower().strip(),
            ))
        if isinstance(value, list):
            return any(visit(item, trusted=trusted) for item in value)
        if not isinstance(value, dict):
            return False
        return any(
            visit(item, trusted=trusted or str(key).lower() in _CORE_PROVENANCE_KEYS)
            for key, item in value.items()
        )

    return visit(parsed)


def _is_real_evidence_tool(tool_name: str, result: str) -> bool:
    name = str(tool_name or "")
    if name in _ROUTER_ONLY_TOOLS:
        return False
    if name in _REAL_EVIDENCE_TOOLS:
        return True
    if name in _GBRAIN_CORE_TOOLS:
        return _has_gbrain_core_provenance(result)
    if name.startswith("mcp__scite__search_") or name.startswith("mcp__scite__get_"):
        return True
    return False


def _provider_from_tool(tool_name: str, result: str) -> str:
    if tool_name in _GBRAIN_CORE_TOOLS and _has_gbrain_core_provenance(result):
        return "gbrain-core"
    parts = str(tool_name or "").split("__")
    if len(parts) >= 3 and parts[0] == "mcp":
        return parts[1]
    if "openevidence" in (result or "").lower():
        return "open-evidence"
    return str(tool_name or "unknown")


def _document_ids(result: str) -> tuple[str, ...]:
    identifiers = [match.group(0) for match in _DOI_RE.finditer(result or "")]
    identifiers.extend(match.group(0) for match in _PMID_RE.finditer(result or ""))
    identifiers.extend(match.group(0).rstrip(".,;)") for match in _URL_RE.finditer(result or ""))
    return tuple(dict.fromkeys(identifiers))


def _append_evidence(
    state: _TurnState,
    *,
    session_id: str,
    tool_name: str,
    result: str,
    evidence_text: str,
) -> None:
    if state.evidence_chars >= _MAX_EVIDENCE_CHARS:
        return
    remaining = _MAX_EVIDENCE_CHARS - state.evidence_chars
    fragment = (evidence_text or "")[:remaining]
    passages = tuple(line.strip() for line in fragment.splitlines() if line.strip())
    if not passages:
        return
    state.evidence.append(_EvidenceRecord(
        provider=_provider_from_tool(tool_name, result),
        tool_name=tool_name,
        document_ids=_document_ids(result),
        passages=passages,
        session_id=session_id,
        turn_id=state.turn_id,
    ))
    state.evidence_chars += len(fragment)


def _extract_evidence_text(result: str) -> str:
    """Return substantive provider content while excluding query/metadata echo."""
    try:
        parsed = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return str(result or "").strip()

    fragments: list[str] = []

    def visit(value: Any, *, content_context: bool = False) -> None:
        if isinstance(value, str):
            if content_context and value.strip():
                fragments.append(value.strip())
            return
        if isinstance(value, list):
            for item in value:
                visit(item, content_context=content_context)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _EVIDENCE_CONTENT_KEYS:
                visit(item, content_context=True)
            elif normalized_key in _EVIDENCE_CONTAINER_KEYS:
                visit(item, content_context=False)

    visit(parsed)
    extracted = "\n".join(fragments)
    # Provider question/answer payloads are generated answer echoes, not
    # independently retrieved document passages. Reject them before records
    # are split into lines, which would otherwise erase the Q/A relationship.
    if _QUERY_ANSWER_ECHO_RE.search(extracted):
        return ""
    return extracted


def _specific_tokens(text: str) -> set[str]:
    patterns = (
        _SPECIFIC_RE,
        _WORD_NUMBER_UNIT_RE,
        _EFFECT_RE,
        _P_VALUE_RE,
        _PER_DENOMINATOR_RE,
        _RATIO_COUNT_RE,
        _CI_RE,
        _UNICODE_FRACTION_RE,
        _YEAR_RE,
    )
    return {
        _normalize(match.group(0))
        for pattern in patterns
        for match in pattern.finditer(text or "")
    }


_CLINICAL_TOKEN_ALIASES = (
    (("ritidect", "rhytidect", "facelift"), "facelift"),
    (("posoperat", "postoperat"), "postoperative"),
    (("complicacion", "complication"), "complication"),
    (("riesgo", "risk"), "risk"),
    (("tempran", "early"), "early"),
    (("principal", "main"), "main"),
    (("frecuent", "common"), "common"),
    (("evacuacion", "evacuation"), "evacuation"),
    (("urgent", "urgente"), "urgent"),
    (("aerea", "airway"), "airway"),
    (("colgajo", "flap"), "flap"),
    (("piel", "skin"), "skin"),
    (("infeccion", "infection"), "infection"),
    (("cicatriz", "healing"), "healing"),
    (("coleccion", "collection"), "collection"),
    (("expansiv", "expanding"), "expanding"),
    (("hipertens", "hypertens"), "hypertension"),
)


def _canonical_content_token(token: str) -> str:
    for prefixes, canonical in _CLINICAL_TOKEN_ALIASES:
        if token.startswith(prefixes):
            return canonical
    return token


def _content_tokens(text: str) -> set[str]:
    normalized = _normalize(text)
    return {
        _canonical_content_token(token)
        for token in re.findall(r"\b[a-z][a-z0-9_-]{3,}\b", normalized)
        if token not in _ANCHOR_STOP
    }


def _has_sufficient_overlap(response: str, evidence: str) -> bool:
    """Require substantive topical overlap without demanding verbatim copying.

    Clinical synthesis necessarily introduces connective and evaluative language
    that will not appear literally in one abstract. Exact numbers, citations and
    other specific claims are checked separately by ``_unsupported_specifics``.
    """
    response_tokens = _content_tokens(response)
    evidence_tokens = _content_tokens(evidence)
    if not response_tokens or not evidence_tokens:
        return False
    shared = response_tokens & evidence_tokens
    required = max(1, min(5, (len(response_tokens) + 4) // 5))
    return len(shared) >= required


def _evidence_is_relevant(user_message: str, response: str, evidence: str) -> bool:
    response_tokens = _content_tokens(response)
    evidence_tokens = _content_tokens(evidence)
    anchors = _content_tokens(user_message) & response_tokens
    if anchors and not anchors & evidence_tokens:
        return False
    shared = response_tokens & evidence_tokens
    return len(shared) >= min(2, len(response_tokens))


_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|contraindicad[oa]s?|evitar|avoid)\b",
    re.IGNORECASE,
)
_QUERY_ANSWER_ECHO_RE = re.compile(
    r"(?:^|\n)\s*(?:q|question|pregunta)\s*:.*(?:\n|\r)+\s*(?:a|answer|respuesta)\s*:",
    re.IGNORECASE | re.DOTALL,
)


def _polarity_matches(response: str, evidence: str) -> bool:
    """Require every evidence-linked response sentence to have matching polarity."""
    response_sentences = _claim_sentences(response)
    evidence_sentences = _claim_sentences(evidence)
    for response_sentence in response_sentences:
        response_tokens = _content_tokens(response_sentence)
        if not response_tokens:
            continue
        relevant = [
            sentence for sentence in evidence_sentences
            if len(response_tokens & _content_tokens(sentence)) >= min(2, len(response_tokens))
        ]
        if not relevant:
            continue
        response_negative = bool(_NEGATION_RE.search(response_sentence))
        if not any(
            bool(_NEGATION_RE.search(sentence)) == response_negative
            for sentence in relevant
        ):
            return False
    return True


_RELATION_GROUPS = (
    ("reduce", "reduced", "reduces", "reduccion", "disminuye", "disminucion", "lower", "lowered"),
    ("increase", "increased", "increases", "aumenta", "aumento", "eleva", "elevated"),
    ("cause", "caused", "causes", "causa", "causo", "provoca", "provoco"),
    ("prevent", "prevented", "prevents", "previene", "previno", "evita", "evito"),
)


def _relation_matches(response: str, evidence: str) -> bool:
    response_tokens = _content_tokens(response)
    evidence_tokens = _content_tokens(evidence)
    asserted_groups = [set(group) for group in _RELATION_GROUPS if response_tokens & set(group)]
    return all(group & evidence_tokens for group in asserted_groups)


def _is_query_answer_echo(user_message: str, response: str, evidence: str) -> bool:
    if not _QUERY_ANSWER_ECHO_RE.search(evidence or ""):
        return False
    evidence_tokens = _content_tokens(evidence)
    return (
        bool(_content_tokens(response))
        and _content_tokens(user_message) <= evidence_tokens
        and _content_tokens(response) <= evidence_tokens
    )


def _unsupported_specifics(response: str, evidence: str) -> bool:
    evidence_specifics = _specific_tokens(evidence)
    for token in _specific_tokens(response):
        if token not in evidence_specifics:
            return True
    for match in _DOI_RE.finditer(response or ""):
        if match.group(0).lower() not in evidence.lower():
            return True
    for match in _PMID_RE.finditer(response or ""):
        if match.group(1) not in evidence:
            return True
    for match in _URL_RE.finditer(response or ""):
        url = match.group(0).rstrip(".,;:)")
        if url not in evidence:
            return True
    for match in _AUTHOR_YEAR_RE.finditer(response or ""):
        author = match.group(1)
        if author.lower() in _NO_AUTHOR:
            continue
        if author.lower() not in evidence.lower():
            return True
    return False


def _is_safe_degradation(response: str) -> bool:
    match = _SAFE_DEGRADATION_RE.search(response or "")
    if not match or _unsupported_specifics(response, ""):
        return False
    return not _DEGRADATION_TAIL_RE.search((response or "")[match.end():])


def _on_pre_llm_call(
    session_id: str = "",
    turn_id: str = "",
    user_message: str = "",
    conversation_history: Any = None,
    **_: Any,
) -> None:
    key = _session_key(session_id, turn_id)
    now = time.monotonic()
    authored_message = _strip_reply_context(user_message)
    if "[PRE_DELIVERY_REPAIR]" in str(user_message or ""):
        with _LOCK:
            _resolved_key, state = _get_or_migrate_state(session_id, turn_id)
            if state is not None and state.clinical:
                state.started_at = now
                return None
    try:
        clinical = _classify_turn(authored_message, conversation_history)
    except Exception:
        # Classification uncertainty at a clinical boundary is not permission to deliver.
        clinical = True
    with _LOCK:
        _STATES[key] = _TurnState(
            clinical=clinical,
            user_message=authored_message or "",
        )
        _prune_states(now)
    return None


def _classify_turn(user_message: str, conversation_history: Any = None) -> bool:
    clinical = _is_clinical(user_message)
    if (
        clinical
        or len(user_message or "") > 120
        or _ASSISTANT_FAILURE_COMPLAINT_RE.search(user_message or "")
        or (
            _META_CLINICAL_PROCESS_RE.search(user_message or "")
            and not _CLINICAL_SUBJECT_RE.search(user_message or "")
        )
    ):
        return clinical
    recent: list[str] = []
    if isinstance(conversation_history, list):
        for item in conversation_history[-6:]:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                recent.append(item["content"])
    if recent and re.search(
        r"^\s*(?:[¿?]?\s*(?:y|entonces|pero|cu[aá]l|c[oó]mo|qu[eé])\b|"
        r"(?:la|el|esa|ese|eso)\s+)",
        user_message or "",
        re.IGNORECASE,
    ):
        return _is_clinical("\n".join(recent))
    return False


def _on_pre_delivery_scope(
    user_message: str = "",
    conversation_history: Any = None,
    **_: Any,
) -> bool:
    """Protect only clinical turns; workflow/meta turns retain ordinary streaming."""
    try:
        return _classify_turn(_strip_reply_context(user_message), conversation_history)
    except Exception:
        # Uncertainty at an emission boundary is not permission to stream a potentially clinical turn.
        return True


def _on_post_tool_call(
    session_id: str = "",
    turn_id: str = "",
    tool_name: str = "",
    result: str = "",
    status: str = "ok",
    **_: Any,
) -> None:
    with _LOCK:
        _resolved_key, state = _get_or_migrate_state(session_id, turn_id)
        if state is None or not state.clinical:
            return None
        if not _is_real_evidence_tool(tool_name, result):
            return None
        if _parsed_result_has_error(result, status):
            return None
        evidence_text = _extract_evidence_text(result)
        if not evidence_text:
            return None
        _append_evidence(
            state,
            session_id=session_id,
            tool_name=tool_name,
            result=result,
            evidence_text=evidence_text,
        )
    return None


def _claim_sentences(response: str) -> tuple[str, ...]:
    return tuple(
        claim.strip()
        for claim in re.split(r"(?<=[.!?])\s+|\n+", response or "")
        if claim.strip()
    )


def _claim_is_supported(
    user_message: str,
    claim: str,
    records: list[_EvidenceRecord],
) -> bool:
    fragments = [
        fragment
        for record in records
        for passage in record.passages
        for fragment in _claim_sentences(passage)
    ]
    has_exact_marker = bool(
        _specific_tokens(claim)
        or _DOI_RE.search(claim or "")
        or _PMID_RE.search(claim or "")
        or _URL_RE.search(claim or "")
        or _AUTHOR_YEAR_RE.search(claim or "")
    )
    if has_exact_marker:
        candidates = fragments
    else:
        claim_tokens = _content_tokens(claim)
        relevant = [
            fragment for fragment in fragments
            if claim_tokens & _content_tokens(fragment)
        ]
        candidates = ["\n".join(relevant)] if relevant else []
    return any(
        _has_sufficient_overlap(claim, evidence)
        and _polarity_matches(claim, evidence)
        and _relation_matches(claim, evidence)
        and not _is_query_answer_echo(user_message, claim, evidence)
        and not _unsupported_specifics(claim, evidence)
        for evidence in candidates
    )


def _clinical_body(response: str) -> str:
    """Exclude an optional provenance appendix from clinical-claim matching."""
    return re.split(r"(?im)^\s*(?:\*\*)?fuentes\s+declaradas(?:\*\*)?\s*$", response or "", maxsplit=1)[0].strip()


_PRIVATE_BLOCK_MARKER = "pre_delivery_blocked"


def _pre_delivery_block(reason: str) -> dict[str, str]:
    """Return a repairable, structured delivery decision.

    The agent consumes ``reason`` as private repair context. ``message`` is an opaque contract
    marker required by the plugin dispatcher and must never become user-visible content.
    """
    return {
        "action": "block",
        "message": _PRIVATE_BLOCK_MARKER,
        "reason": reason,
    }


def _on_pre_delivery(
    response_text: str = "",
    session_id: str = "",
    turn_id: str = "",
    **_: Any,
) -> dict[str, str]:
    """Certify a clinical answer or request in-turn evidence repair.

    Non-clinical turns are never gated. Clinical turns remain fail-closed, but a rejection
    is now actionable: the common agent loop retains the candidate, retrieves or corrects
    evidence, regenerates, and invokes this hook again before any channel can deliver it.
    """
    with _LOCK:
        key, state = _get_or_migrate_state(session_id, turn_id)

    if state is None:
        if _is_clinical(response_text):
            return _pre_delivery_block(
                "Falta el estado verificable del turno clínico; reconstruye la intención, "
                "recupera evidencia y vuelve a generar la respuesta."
            )
        return {"action": "allow"}
    if not state.clinical:
        with _LOCK:
            _STATES.pop(key, None)
        return {"action": "allow"}
    if _is_safe_degradation(response_text):
        return _pre_delivery_block(
            "La respuesta se limitó a abstenerse. Continúa la recuperación por rutas alternativas "
            "y responde la consulta clínica con evidencia verificable."
        )
    if not state.evidence:
        return _pre_delivery_block(
            "No se recuperó evidencia documental utilizable. Ejecuta la ruta clínica seleccionada, "
            "prueba proveedores alternativos cuando uno falle y regenera la respuesta."
        )

    body = _clinical_body(response_text)
    passages = [passage for record in state.evidence for passage in record.passages]
    evidence = "\n".join(passages)
    reasons: list[str] = []
    if not body:
        reasons.append(
            "coloca primero la respuesta clínica sustantiva antes del encabezado exacto "
            "'Fuentes declaradas'; no pongas la conclusión dentro del ledger"
        )
    elif not _evidence_is_relevant(state.user_message, body, evidence):
        reasons.append("la evidencia recuperada no cubre la pregunta y la respuesta")
    if body and not _polarity_matches(body, evidence):
        reasons.append("la polaridad de una afirmación contradice la evidencia recuperada")
    if body and _is_query_answer_echo(state.user_message, body, evidence):
        reasons.append("el resultado recuperado es un eco de pregunta/respuesta, no una fuente documental")
    if body and _unsupported_specifics(body, evidence):
        reasons.append("hay cifras, citas o identificadores que no aparecen en la evidencia recuperada")
    unsupported_claims = [
        claim for claim in _claim_sentences(body)
        if not _claim_is_supported(state.user_message, claim, state.evidence)
    ]
    if unsupported_claims:
        reasons.append("hay afirmaciones sin respaldo suficiente en la evidencia recuperada")

    if reasons:
        return _pre_delivery_block(
            "; ".join(reasons)
            + ". Conserva lo respaldado, corrige o elimina sólo estas afirmaciones, recupera "
            "la evidencia faltante y vuelve a someter la respuesta completa."
        )

    with _LOCK:
        _STATES.pop(key, None)
    return {"action": "allow"}


def _reset_for_tests() -> None:
    with _LOCK:
        _STATES.clear()


def register(ctx: Any) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_delivery_scope", _on_pre_delivery_scope)
    ctx.register_hook("pre_delivery", _on_pre_delivery)
