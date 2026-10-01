"""``litkit_jev``: typed first-pass judgments on matter documents from TypeSafe's Jev.

Jev (TypeSafe's System One model) answers typed questions about a text with calibrated
probabilities: a ``noul`` is the probability of yes, a ``choice`` picks one option with a
distribution and a confidence, a ``score`` places the text on ordered levels. It writes no
prose. Two actions use it:

* ``screen`` sends each document (at most :data:`SCREEN_MAX_DOCS`) one request that fans out
  over ``responsive``, one noul per criterion (``c_0``, ``c_1``, ...), ``privileged_signal`` and
  ``junk``. The cull rule runs here, in code (:func:`decide`): a document is set aside only when
  ``responsive`` and every criterion are at or below ``low`` and ``privileged_signal`` is at or
  below ``priv``. Anything else, including a document Jev could not see, is read in full.
* ``ask`` sends one document (or pasted text) and up to :data:`ASK_MAX_QUESTIONS` typed questions
  and returns Jev's answers as given.

Document text comes from LitKit through the same client as every other LitKit tool, so the
acting lawyer's assertion goes on each call and LitKit's walls decide what Jev may see. The
TypeSafe key is the host's ``TYPESAFE_API_KEY``; without it the tool says Jev is not configured.
Document text is never logged; the per-document results are saved under ``jev/``.

The handlers are wrapped and registered by :mod:`litco.litkit.tools`. They import its document
helpers at call time, because ``tools`` imports this module to build its registration table.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import httpx

from litco.litkit.client import _secret
from litco.litkit.files import relative

logger = logging.getLogger("litco.litkit.jev")

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
KEY_ENV = "TYPESAFE_API_KEY"
NOT_CONFIGURED = ("Jev not configured on this host: TYPESAFE_API_KEY is not set. Screen the documents with "
                  "an ordinary review run, or ask the firm's LitKit admin to configure Jev for the agent hosts.")

SCREEN_MAX_DOCS = 200
SCREEN_MAX_CRITERIA = 40
ASK_MAX_QUESTIONS = 20
CONCURRENCY = 8
DEFAULT_LOW = 0.15
DEFAULT_PRIV = 0.30
UNCERTAIN_BAND = 0.10
# TypeSafe's limit is 32k tokens for state plus the longest question; we aim at 30k.
STATE_TOKEN_BUDGET = 30_000
CHARS_PER_TOKEN = 4
MIN_TEXT_CHARS = 2_000
CRITERION_DESCRIPTION_MAX = 1_500
ASK_TEXT_MAX = 200_000
PRICE_USD_PER_MTOK = 0.042
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})
MAX_ATTEMPTS = 5
MAX_RETRY_AFTER = 60.0
QUESTION_TYPES = ("noul", "choice", "score")
_QUESTION_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_LENGTH_ERROR = re.compile(r"token|context|too long|length", re.I)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class JevError(Exception):
    """A Jev call failed. ``status`` is the HTTP status (0 for a transport failure)."""

    def __init__(self, message: str, *, status: int = 0, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"error": str(self), "status": self.status}
        if self.detail:
            out["detail"] = self.detail
        return out

    @property
    def too_long(self) -> bool:
        return self.status in (400, 413, 422) and bool(_LENGTH_ERROR.search(f"{self} {self.detail or ''}"))


def _scrub(detail: Any) -> Any:
    """A validation error may echo the request; drop any ``input`` it carries and clip the rest."""
    if isinstance(detail, dict):
        return {k: _scrub(v) for k, v in detail.items() if k != "input"}
    if isinstance(detail, list):
        return [_scrub(v) for v in detail[:5]]
    if isinstance(detail, str) and len(detail) > 500:
        return detail[:499] + "…"
    return detail


def _retry_after(resp: Optional[httpx.Response]) -> Optional[float]:
    if resp is None:
        return None
    raw = resp.headers.get("retry-after")
    try:
        return max(0.0, min(float(raw), MAX_RETRY_AFTER)) if raw is not None else None
    except ValueError:
        return None


class JevClient:
    """``POST /v1/systemone`` with the host's key. Retries 429, 529 and 5xx with exponential
    backoff, honoring ``retry-after``. Never logs the state it sends."""

    def __init__(self, api_key: str, *, transport: Optional[httpx.BaseTransport] = None,
                 sleep: Callable[[float], None] = time.sleep, timeout: float = 120.0,
                 max_attempts: int = MAX_ATTEMPTS):
        self._key = api_key
        self._sleep = sleep
        self._max_attempts = max_attempts
        self._http = httpx.Client(transport=transport, timeout=timeout)

    @classmethod
    def from_env(cls, **kwargs: Any) -> Optional["JevClient"]:
        key = _secret(KEY_ENV)
        return cls(key, **kwargs) if key else None

    def __repr__(self) -> str:  # never print the key
        return "JevClient(key=set)"

    def close(self) -> None:
        self._http.close()

    def _delay(self, attempt: int, resp: Optional[httpx.Response]) -> float:
        hinted = _retry_after(resp)
        if hinted is not None:
            return hinted
        return min(30.0, 2.0 ** attempt) * (0.5 + random.random() / 2)

    def evaluate(self, state: Any, questions: Dict[str, Any]) -> Dict[str, Any]:
        body = {"state": state, "model": JEV_MODEL, "questions": questions}
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        for attempt in range(self._max_attempts):
            last = attempt == self._max_attempts - 1
            try:
                resp = self._http.post(JEV_URL, json=body, headers=headers)
            except httpx.TransportError as exc:
                if last:
                    raise JevError(f"Jev unreachable: {type(exc).__name__}") from None
                self._sleep(self._delay(attempt, None))
                continue
            if resp.status_code in RETRY_STATUSES and not last:
                logger.info("jev %s, retrying (attempt %d)", resp.status_code, attempt + 1)
                self._sleep(self._delay(attempt, resp))
                continue
            if resp.status_code >= 400:
                try:
                    detail = _scrub(resp.json())
                except ValueError:
                    detail = None
                if resp.status_code == 401:
                    raise JevError("Jev refused this host's TypeSafe key (401)", status=401)
                if resp.status_code in RETRY_STATUSES:
                    raise JevError(f"Jev is rate-limited or overloaded ({resp.status_code}) after "
                                   f"{self._max_attempts} attempts", status=resp.status_code)
                raise JevError(f"Jev rejected the request ({resp.status_code})", status=resp.status_code,
                               detail=detail)
            try:
                out = resp.json()
            except ValueError:
                raise JevError("Jev answered with something other than JSON", status=resp.status_code) from None
            if not isinstance(out, dict) or not isinstance(out.get("answers"), dict):
                raise JevError("Jev's answer carried no answers", status=resp.status_code)
            return out
        raise JevError("Jev call did not complete")  # unreachable: the last attempt returns or raises


_DEFAULT: Optional[JevClient] = None
_OVERRIDE = False
_LOCK = threading.Lock()


def default_jev() -> Optional[JevClient]:
    """The host's Jev client, or None when TYPESAFE_API_KEY is unset."""
    global _DEFAULT
    with _LOCK:
        if _OVERRIDE:
            return _DEFAULT
        if _DEFAULT is None or _DEFAULT._key != _secret(KEY_ENV):
            _DEFAULT = JevClient.from_env()
        return _DEFAULT


def set_default_jev(client: Optional[JevClient], *, override: bool = True) -> None:
    """Tests: pin the client (``None`` pins "not configured"); ``override=False`` restores env lookup."""
    global _DEFAULT, _OVERRIDE
    with _LOCK:
        _DEFAULT, _OVERRIDE = client, override


# ---------------------------------------------------------------------------
# state, questions, truncation
# ---------------------------------------------------------------------------

def estimate_tokens(value: Any) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return -(-len(text) // CHARS_PER_TOKEN)


def head_tail(text: str, budget_chars: int) -> Tuple[str, bool]:
    """Keep the head and the tail of a text longer than the budget, with a marker between."""
    if len(text) <= budget_chars:
        return text, False
    marker = f"\n\n[... {len(text) - budget_chars:,} characters omitted to fit Jev's limit ...]\n\n"
    room = max(0, budget_chars - len(marker))
    head = room * 5 // 6
    tail = room - head
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def document_state(meta: Mapping[str, Any], text: str, questions: Mapping[str, Any],
                   budget_tokens: int = STATE_TOKEN_BUDGET) -> Tuple[Dict[str, Any], bool]:
    """``{"document": {bates, custodian, date, ..., text}}`` cut to fit state plus the longest question."""
    doc = {k: v for k, v in meta.items() if v not in (None, "")}
    longest = max((estimate_tokens(q) for q in questions.values()), default=0)
    overhead = estimate_tokens({"document": {**doc, "text": ""}})
    budget_chars = (budget_tokens - longest - overhead) * CHARS_PER_TOKEN
    if budget_chars < MIN_TEXT_CHARS:
        raise ValueError("the questions leave too little room for the document under Jev's 32k-token limit; "
                         "shorten the criteria descriptions or ask fewer, shorter questions")
    # Newlines and quotes grow when the state is encoded; cut until the encoded text fits.
    limit = budget_chars
    for _ in range(4):
        cut, truncated = head_tail(text, limit)
        excess = len(json.dumps(cut, ensure_ascii=False)) - 2 - budget_chars
        if excess <= 0:
            break
        limit -= excess
    return {"document": {**doc, "text": cut}}, truncated


def screen_criteria(criteria: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Criteria as Jev sees them: an id for code, a title and a clipped description for the model."""
    out = []
    for n, crit in enumerate(criteria):
        row = {"id": f"c_{n}", "title": str(crit.get("title") or "").strip()}
        desc = str(crit.get("description") or "").strip()
        if desc:
            row["description"] = desc if len(desc) <= CRITERION_DESCRIPTION_MAX else \
                desc[:CRITERION_DESCRIPTION_MAX - 1] + "…"
        out.append(row)
    return out


def screen_questions(criteria: List[Dict[str, str]]) -> Dict[str, Dict[str, Any]]:
    """One request's fan-out: responsive, one noul per criterion, privileged_signal, junk."""
    requests = [{k: v for k, v in c.items() if k != "id"} for c in criteria]
    questions: Dict[str, Dict[str, Any]] = {
        "responsive": {
            "type": "noul",
            "instructions": {
                "question": "Does any part of `document` fall within at least one of the requests in `requests`?",
                "requests": requests,
            },
            "criteria": {
                "true": "Some content of the document falls within at least one request in `requests`.",
                "false": "Nothing in the document falls within any request in `requests`.",
            },
        },
    }
    for crit, request in zip(criteria, requests):
        questions[crit["id"]] = {
            "type": "noul",
            "instructions": {"question": "Does any part of `document` fall within `request`?", "request": request},
            "criteria": {
                "true": "Some content of the document falls within `request`.",
                "false": "Nothing in the document falls within `request`.",
            },
        }
    questions["privileged_signal"] = {
        "type": "noul",
        "instructions": ("Does `document` show signs that it may be privileged: a communication to or from a "
                         "lawyer, legal advice requested or given, or material prepared because of litigation?"),
        "criteria": {
            "true": "A lawyer, legal advice, or litigation preparation appears in the document.",
            "false": "No lawyer, legal advice, or litigation preparation appears in the document.",
        },
    }
    questions["junk"] = {
        "type": "noul",
        "instructions": ("Is `document` without substance for a legal review: an automated system notice, a "
                         "calendar entry with no content, a newsletter or mass mailing, or a blank or near-empty "
                         "file?"),
        "criteria": {
            "true": "The document carries no substantive content a reviewer would need to read.",
            "false": "The document carries substantive content written by or for the people involved.",
        },
    }
    return questions


# ---------------------------------------------------------------------------
# the cull rule
# ---------------------------------------------------------------------------

def thresholds(value: Any) -> Dict[str, float]:
    """``{low, priv}``, each strictly between 0 and 1; missing keys take the defaults."""
    value = value if value is not None else {}
    if not isinstance(value, dict):
        raise ValueError("'thresholds' must be an object {low, priv}")
    unknown = sorted(set(value) - {"low", "priv"})
    if unknown:
        raise ValueError(f"'thresholds' takes only low and priv; not {', '.join(unknown)}")
    out = {"low": DEFAULT_LOW, "priv": DEFAULT_PRIV}
    for key in ("low", "priv"):
        if value.get(key) is not None:
            try:
                number = float(value[key])
            except (TypeError, ValueError):
                raise ValueError(f"thresholds.{key} must be a number between 0 and 1") from None
            if not 0 < number < 1:
                raise ValueError(f"thresholds.{key} must be between 0 and 1")
            out[key] = number
    return out


def decide(probs: Mapping[str, float], criterion_ids: List[str], limits: Mapping[str, float]) -> Dict[str, Any]:
    """Set aside only when responsive and every criterion are <= low and privileged_signal <= priv.

    ``uncertain`` marks a decision a 0.1 move in the probabilities would flip: a set-aside document
    with a topical probability above ``low - 0.1`` or a privilege signal above ``priv - 0.1``, or a
    document read in full whose probabilities all sit within 0.1 above the thresholds."""
    topical = max([probs["responsive"], *(probs[c] for c in criterion_ids)])
    priv = probs["privileged_signal"]
    set_aside = topical <= limits["low"] and priv <= limits["priv"]
    if set_aside:
        uncertain = topical > limits["low"] - UNCERTAIN_BAND or priv > limits["priv"] - UNCERTAIN_BAND
    else:
        uncertain = topical <= limits["low"] + UNCERTAIN_BAND and priv <= limits["priv"] + UNCERTAIN_BAND
    return {"decision": "set_aside" if set_aside else "read_in_full", "uncertain": uncertain}


def _noul(answers: Mapping[str, Any], key: str) -> float:
    answer = answers.get(key)
    value = answer.get("noul") if isinstance(answer, dict) else None
    if not isinstance(value, (int, float)):
        raise JevError(f"Jev returned no probability for {key}")
    return float(value)


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def _tools():
    from litco.litkit import tools
    return tools


def _screen_doc_ids(client: Any, mid: str, args: Dict[str, Any]) -> List[str]:
    T = _tools()
    given = [k for k in ("documentIds", "bates", "workSetId") if args.get(k) not in (None, "", [])]
    if len(given) != 1:
        raise ValueError("give exactly one of documentIds, bates, or workSetId")
    kind = given[0]
    if kind == "workSetId":
        body = client.get(f"/api/work-sets/{T._uuid(args, 'workSetId')}")
        raw = body.get("docIds") if isinstance(body, dict) else None
        if not isinstance(raw, list):
            raise ValueError("that work set carries no document list")
        ids = [str(d) for d in raw]
    elif kind == "documentIds":
        if not isinstance(args["documentIds"], list):
            raise ValueError("documentIds must be a list of LitKit ids (uuid)")
        ids = [str(d).strip() for d in args["documentIds"]]
        bad = [d for d in ids if not T._UUID.match(d)]
        if bad:
            raise ValueError(f"documentIds must be LitKit ids (uuid); not ids: {bad[:3]}")
    else:
        raw = args["bates"] if isinstance(args["bates"], list) else [args["bates"]]
        numbers = [str(b).strip() for b in raw if str(b).strip()]
        if len(numbers) > SCREEN_MAX_DOCS:
            raise ValueError(_too_many(len(numbers)))
        ids = [T._resolve_doc_id(client, {"bates": b}) for b in numbers]
    ids = list(dict.fromkeys(ids))
    if not ids:
        raise ValueError("no documents to screen")
    if len(ids) > SCREEN_MAX_DOCS:
        raise ValueError(_too_many(len(ids)))
    return ids


def _too_many(count: int) -> str:
    return (f"{count:,} documents is more than litkit_jev screens in one call ({SCREEN_MAX_DOCS}). Propose a review "
            "run with firstPass={enabled: true} instead (litkit_review action=create): LitKit runs the Jev "
            "first pass over the whole scope on the server and lists what it set aside on the job page.")


def _screen_criteria(client: Any, mid: str, args: Dict[str, Any]) -> List[Dict[str, Any]]:
    T = _tools()
    has_set, has_list = bool(args.get("criteriaSetId")), args.get("criteria") is not None
    if has_set == has_list:
        raise ValueError("give exactly one of criteria or criteriaSetId")
    if has_set:
        body = client.get(f"/api/matters/{mid}/criteria-sets/{T._set_id(args)}")
        criteria = body.get("criteria") if isinstance(body, dict) else None
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("that criteria set holds no criteria")
    else:
        criteria = args.get("criteria")
    criteria = T._criteria(criteria)
    if len(criteria) > SCREEN_MAX_CRITERIA:
        raise ValueError(f"litkit_jev screens against at most {SCREEN_MAX_CRITERIA} criteria per call; propose a "
                         "review run with firstPass instead, or screen against the criteria that decide scope")
    return criteria


def _export_texts(client: Any, mid: str, ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Text and Bates metadata per document through LitKit's ACL-scoped export (one call, <= 500 ids)."""
    rows: Dict[str, Dict[str, Any]] = {}
    for line in client.stream_ndjson("POST", f"/api/matters/{mid}/export/text", json_body={"docIds": ids},
                                     timeout=600):
        doc_id = str(line.get("docId") or "")
        if doc_id:
            rows[doc_id] = line
    return rows


def _meta(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {"bates": row.get("batesStart"), "custodian": row.get("custodian"), "date": row.get("date"),
            "author": row.get("author"), "subject": row.get("subject")}


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _evaluate(jev: JevClient, meta: Mapping[str, Any], text: str,
              questions: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], bool]:
    """One Jev call; if Jev says the request is too long after all, retry once with half the room."""
    state, truncated = document_state(meta, text, questions)
    try:
        return jev.evaluate(state, questions), truncated
    except JevError as exc:
        if not exc.too_long:
            raise
    state, _ = document_state(meta, text, questions, budget_tokens=STATE_TOKEN_BUDGET // 2)
    return jev.evaluate(state, questions), True


def _round(p: float) -> float:
    return round(p, 3)


def screen(args: Dict[str, Any], jev: JevClient) -> Dict[str, Any]:
    T = _tools()
    client = T._client()
    mid = T._mid(client)
    limits = thresholds(args.get("thresholds"))
    criteria = screen_criteria(_screen_criteria(client, mid, args))
    ids = _screen_doc_ids(client, mid, args)
    questions = screen_questions(criteria)
    document_state({}, "", questions)  # criteria too long for Jev fail here, before any request
    cids = [c["id"] for c in criteria]
    titles = {c["id"]: c["title"] for c in criteria}
    texts = _export_texts(client, mid, ids)

    def one(doc_id: str) -> Dict[str, Any]:
        row = texts.get(doc_id)
        if row is None or row.get("error") == "not_found":
            return {"docId": doc_id, "decision": "not_found"}
        if row.get("error"):
            return {"docId": doc_id, "decision": "read_in_full", "error": f"LitKit: {row['error']}"}
        meta = _meta(row)
        base = {"docId": doc_id, "bates": meta["bates"]}
        text = str(row.get("text") or "")
        if not text.strip():
            return {**base, "decision": "read_in_full", "noText": True}
        try:
            out, cut = _evaluate(jev, meta, text, questions)
            answers = out["answers"]
            probs = {k: _noul(answers, k) for k in ("responsive", "privileged_signal", "junk", *cids)}
        except (JevError, ValueError) as exc:
            logger.warning("jev screen failed for %s: status %s", doc_id, getattr(exc, "status", None))
            return {**base, "decision": "read_in_full", "error": str(exc)}
        verdict = decide(probs, cids, limits)
        top = sorted(cids, key=lambda c: -probs[c])[:3]
        result = {**base, **verdict, "responsive": _round(probs["responsive"]),
                  "top": [[c, _round(probs[c])] for c in top],
                  "privileged_signal": _round(probs["privileged_signal"]), "junk": _round(probs["junk"])}
        if cut or row.get("truncated"):
            result["truncated"] = True
        usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
        result["_tokens"] = int(usage.get("input_tokens") or 0)
        result["_probs"] = {k: _round(v) for k, v in probs.items()}
        return result

    with ThreadPoolExecutor(max_workers=min(CONCURRENCY, len(ids))) as pool:
        results = list(pool.map(one, ids))

    tokens = sum(r.pop("_tokens", 0) for r in results)
    full = []
    for r in results:
        probs = r.pop("_probs", None)
        full.append({**r, "probabilities": probs} if probs else dict(r))
    for r in results:
        if not r.get("uncertain"):
            r.pop("uncertain", None)
    counts = {
        "screened": len(ids),
        "read_in_full": sum(r["decision"] == "read_in_full" for r in results),
        "set_aside": sum(r["decision"] == "set_aside" for r in results),
        "uncertain": sum(bool(r.get("uncertain")) for r in results),
        "uncertain_set_aside": sum(bool(r.get("uncertain")) and r["decision"] == "set_aside" for r in results),
        "not_found": sum(r["decision"] == "not_found" for r in results),
        "no_text": sum(bool(r.get("noText")) for r in results),
        "errors": sum("error" in r for r in results),
        "truncated": sum(bool(r.get("truncated")) for r in results),
    }
    saved = T._save_json("jev", "screen", {"thresholds": limits, "criteria": criteria, "counts": counts,
                                            "documents": full})
    summary = (f"Jev set aside {counts['set_aside']:,} of {counts['screened']:,}; {counts['read_in_full']:,} "
               f"read in full; {counts['uncertain']:,} uncertain ({counts['uncertain_set_aside']:,} of them "
               "set aside).")
    out: Dict[str, Any] = {**counts, "summary": summary, "thresholds": limits,
                           "usage": {"inputTokens": tokens,
                                     "estCostUsd": round(tokens * PRICE_USD_PER_MTOK / 1_000_000, 4)},
                           "criteria": {c: titles[c] for c in cids}, "saved": relative(saved),
                           "note": ("Set aside means not read yet, not non-responsive: Jev's numbers are a screen, "
                                    "never a finding. Read the uncertain ones in full. Documents with no text, "
                                    "not found, or an error were not screened.")}
    if counts["not_found"]:
        out["notFoundNote"] = ("not found means LitKit returned no text for that id: it is not in this matter or "
                               "this lawyer may not see it")
    out["rows"] = results
    return out


def _validate_questions(value: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("'questions' must be an object {id: {type, instructions, criteria?}}")
    if len(value) > ASK_MAX_QUESTIONS:
        raise ValueError(f"ask takes at most {ASK_MAX_QUESTIONS} questions per call")
    out: Dict[str, Dict[str, Any]] = {}
    for qid, q in value.items():
        if not _QUESTION_ID.match(str(qid)):
            raise ValueError(f"question id {str(qid)[:40]!r} must be letters, digits, _ or -, starting with a letter")
        if not isinstance(q, dict):
            raise ValueError(f"question {qid} must be an object {{type, instructions, criteria?}}")
        kind = q.get("type")
        if kind not in QUESTION_TYPES:
            raise ValueError(f"question {qid}: type must be one of {', '.join(QUESTION_TYPES)}")
        if q.get("instructions") in (None, "", [], {}):
            raise ValueError(f"question {qid} needs instructions")
        criteria = q.get("criteria")
        if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
            raise ValueError(f"question {qid}: a choice needs criteria {{option: description}} with two or more "
                             "options")
        if kind == "score" and (not isinstance(criteria, list) or not 2 <= len(criteria) <= 10):
            raise ValueError(f"question {qid}: a score needs criteria as an ordered list of 2 to 10 levels")
        if kind == "noul" and criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"true", "false"}):
            raise ValueError(f"question {qid}: noul criteria, if given, are {{true, false}}")
        out[str(qid)] = {"type": kind, "instructions": q["instructions"],
                         **({"criteria": criteria} if criteria is not None else {})}
    return out


def ask(args: Dict[str, Any], jev: JevClient) -> Dict[str, Any]:
    T = _tools()
    questions = _validate_questions(args.get("questions"))
    base: Dict[str, Any] = {}
    if args.get("text") is not None and not (args.get("documentId") or args.get("bates")):
        text = str(args["text"])
        if not text.strip():
            raise ValueError("'text' is empty")
        if len(text) > ASK_TEXT_MAX:
            raise ValueError(f"pasted text is limited to {ASK_TEXT_MAX:,} characters; ask about the document instead")
        meta: Dict[str, Any] = {}
        capped = False
    else:
        client = T._client()
        bates = args.get("bates")
        if isinstance(bates, list):
            if len(bates) != 1:
                raise ValueError("ask reads one document: give one documentId or one Bates number")
            bates = bates[0]
        doc_id = T._resolve_doc_id(client, {"documentId": args.get("documentId"), "bates": bates})
        doc = T._metadata(client, doc_id)
        body = client.get(f"/api/documents/{doc_id}/text")
        text, trunc = T._text_from_payload(body if isinstance(body, dict) else {})
        if not text.strip():
            return {"docId": doc_id, "bates": doc.get("batesStart"), "answers": None,
                    "note": "LitKit holds no extracted text for this document, so Jev cannot read it."}
        meta = {"bates": doc.get("batesStart"), "custodian": doc.get("custodian"), "date": T._date_of(doc),
                "author": doc.get("author"), "subject": doc.get("subject"), "fileName": doc.get("fileName")}
        base = {"docId": doc_id, "bates": doc.get("batesStart")}
        capped = bool(trunc)
    out, cut = _evaluate(jev, meta, text, questions)
    usage = out.get("usage") if isinstance(out.get("usage"), dict) else {}
    result = {**base, "model": out.get("model"), "answers": out["answers"],
              "usage": {"inputTokens": usage.get("input_tokens")},
              "note": "Calibrated probabilities from Jev: a signal for routing your own reading, never a finding."}
    if cut or capped:
        result["truncated"] = True
    return result


ACTIONS = ("screen", "ask")


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    action = str(args.get("action") or "screen")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {', '.join(ACTIONS)}")
    jev = default_jev()
    if jev is None:
        return {"error": NOT_CONFIGURED, "configured": False}
    try:
        return screen(args, jev) if action == "screen" else ask(args, jev)
    except JevError as exc:
        return {**exc.to_dict(), "jev": True}


_QUESTION = {"type": "object", "properties": {
    "type": {"type": "string", "enum": list(QUESTION_TYPES)},
    "instructions": {"description": "the question: a string, or an object with the question in one field and data it "
                     "refers to (in backticks) in others"},
    "criteria": {"description": "noul: optional {true, false}; choice: {option: description}; score: ordered list "
                 "of 2-10 levels"}},
    "required": ["type", "instructions"]}

SCHEMA: Dict[str, Any] = {
    "name": "litkit_jev",
    "description": (
        "Jev (TypeSafe) first-pass screen: calibrated probabilities, never findings or tags. screen: up to "
        f"{SCREEN_MAX_DOCS} documents vs criteria; sets aside only when responsive and every criterion <= low and "
        "privileged_signal <= priv. More documents: a review run with firstPass. ask: up to "
        f"{ASK_MAX_QUESTIONS} typed questions on one document (`document` in state). Skill: jev-first-pass-review."),
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "documentIds": {"type": "array", "items": {"type": "string"}, "description": "screen: LitKit ids"},
        "bates": {"type": "array", "items": {"type": "string"},
                  "description": "screen: Bates numbers (one string for ask)"},
        "workSetId": {"type": "string", "description": "screen: a work set's documents"},
        "criteria": {"type": "array", "description": "screen: [{title, description}] (or a criteriaSetId)",
                     "items": {"type": "object", "properties": {"title": {"type": "string"},
                                                                "description": {"type": "string"}},
                               "required": ["title"]}},
        "criteriaSetId": {"type": "string", "description": "screen: a criteria set id (uuid) or builtin:<name>"},
        "thresholds": {"type": "object", "description": "screen: {low (default 0.15), priv (default 0.30)}",
                       "properties": {"low": {"type": "number"}, "priv": {"type": "number"}}},
        "documentId": {"type": "string", "description": "ask: one document"},
        "text": {"type": "string", "description": "ask: pasted text instead of a document"},
        "questions": {"type": "object", "additionalProperties": _QUESTION,
                      "description": "ask: {id: {type, instructions, criteria?}}"}},
        "required": ["action"]},
}
