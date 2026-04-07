"""
LLM-as-Judge for Memory Benchmark

Evaluates whether actual memory recall matches the expected gold answer.
Supports both LLM judges (Haiku) and heuristic fallback.

Two judge modes
---------------
Binary (judge_answer):
    Returns a simple CORRECT/INCORRECT verdict via JUDGE_SYSTEM_PROMPT.
    Backward-compatible; all existing callers continue to work unchanged.

Structured (judge_answer_structured — MemoryJudge only):
    Returns a multi-dimensional rubric score alongside the binary verdict.
    Uses STRUCTURED_JUDGE_SYSTEM_PROMPT and expects a JSON response.
    Falls back to the binary judge_answer on any parse error.
    HeuristicJudge approximates the same rubric dimensions from its
    heuristic signals so both judges produce comparable output.

Timing
------
Every public judge_answer* call populates JudgeResult.latency_ms with
the wall-clock duration (time.monotonic, milliseconds).
"""

import os
import json
import re
import time
import logging
from typing import Optional

from benchmarks.interface import JudgeResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a judge evaluating a memory retrieval system's recall accuracy.

The memory system retrieves stored facts relevant to a question. The gold answer shows what a correct response requires. Your job is to determine whether the retrieved facts are SUFFICIENT to answer the question correctly.

Rules:
- CORRECT: The retrieved facts contain the key information needed to produce the gold answer. This includes:
  * Direct matches (facts explicitly state the gold answer)
  * Inferential matches (facts contain the component data that logically implies the gold answer, even if the computation isn't shown — e.g., if gold says "500 > 200 limit" and facts show "50 instances × 10 connections" and "max_connections = 200", that's CORRECT)
  * Paraphrasing and minor wording differences are acceptable
- INCORRECT: The retrieved facts are missing critical information needed to answer the question, or contain wrong information.
- Extra information: If retrieved facts include correct extra details beyond what's needed, still mark CORRECT.
- No answer: If the retrieved facts are empty or completely irrelevant, mark INCORRECT.

Key insight: The memory system's job is fact retrieval, not reasoning. If the retrieved facts contain all the pieces needed to derive the correct answer, mark CORRECT.

Respond with exactly one word: CORRECT or INCORRECT
Then on a new line, a brief explanation (one sentence max)."""


STRUCTURED_JUDGE_SYSTEM_PROMPT = """You are a judge evaluating a memory retrieval system's recall accuracy.

Score the retrieved facts on four dimensions (each 0-10) and then give a binary verdict.

Scoring dimensions:
- relevance (0-10): Are the retrieved facts relevant to the question? 0 = completely off-topic, 10 = perfectly targeted.
- factual_accuracy (0-10): Do the retrieved facts correctly match the gold answer? 0 = factually wrong or contradictory, 10 = exact match.
- completeness (0-10): Are ALL parts of the gold answer covered by the retrieved facts? 0 = nothing covered, 10 = every detail present.
- temporal_correctness (0-10): If temporal or versioned information is involved, is the most current/correct version retrieved? If no temporal element is relevant, default to 10.

Verdict rules (same as always):
- CORRECT: The retrieved facts are sufficient to derive the gold answer (direct matches, inferential matches, paraphrasing all count).
- INCORRECT: Critical information is missing or wrong.

Respond ONLY with valid JSON — no markdown fences, no extra text — in this exact schema:
{"relevance": N, "factual_accuracy": N, "completeness": N, "temporal_correctness": N, "verdict": "CORRECT", "explanation": "One sentence."}"""


JUDGE_USER_TEMPLATE = """Question: {question}
Expected answer: {gold_answer}
Actual answer: {actual_answer}

Verdict:"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scores_to_01(raw: dict) -> dict:
    """Convert 0-10 integer scores to 0.0-1.0 floats, clamping to [0, 1]."""
    keys = ("relevance", "factual_accuracy", "completeness", "temporal_correctness")
    return {k: max(0.0, min(1.0, float(raw.get(k, 0)) / 10.0)) for k in keys}


# ---------------------------------------------------------------------------
# MemoryJudge
# ---------------------------------------------------------------------------

class MemoryJudge:
    """Evaluates memory recall answers using an LLM judge (Claude Haiku).

    Public methods
    --------------
    judge_answer(question, gold_answer, actual_answer) -> JudgeResult
        Binary CORRECT/INCORRECT verdict.  Backward-compatible.

    judge_answer_structured(question, gold_answer, actual_answer) -> JudgeResult
        Multi-dimensional rubric + binary verdict.  Falls back to
        judge_answer if the LLM returns malformed JSON.
    """

    def __init__(self, model: str = "claude-haiku-4-5",
                 api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._client = None

    # ------------------------------------------------------------------
    # Proxy / client setup
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_proxy_port() -> int:
        """Read the aegis proxy port from its pid file, fall back to 8444."""
        import json as _json
        for path in [
            os.path.expanduser("~/.hermes-aegis/proxy.pid"),
            "/Users/evinova/.hermes-aegis/proxy.pid",
        ]:
            try:
                with open(path) as f:
                    data = _json.load(f)
                    return data.get("port", 8444)
            except (FileNotFoundError, ValueError, KeyError):
                continue
        return 8444

    def _get_client(self):
        """Lazy-init the Anthropic client.

        Routes through aegis proxy if available (container environment).
        The proxy injects API keys, so we don't need them in the container.
        """
        if self._client is None:
            import anthropic
            import httpx

            # Try aegis proxy first (container environment)
            ca_cert = "/certs/mitmproxy-ca-cert.pem"

            if os.path.exists(ca_cert):
                port = self._detect_proxy_port()
                proxy_url = f"http://host.docker.internal:{port}"
                # Route through aegis proxy — it injects the API key
                http_client = httpx.Client(
                    proxy=proxy_url,
                    verify=ca_cert,
                )
                self._client = anthropic.Anthropic(
                    api_key=self.api_key or "<aegis-injected>",
                    http_client=http_client,
                )
                logger.info("LLM judge using aegis proxy on port %d", port)
            else:
                # Direct connection (host environment)
                self._client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("LLM judge using direct API")
        return self._client

    # ------------------------------------------------------------------
    # Public judge interface
    # ------------------------------------------------------------------

    def judge_answer(self, question: str, gold_answer: str,
                     actual_answer: str) -> JudgeResult:
        """Binary judge: returns CORRECT/INCORRECT verdict.

        Backward-compatible with all existing callers.
        Populates latency_ms; scores and confidence are left empty/0.
        """
        t0 = time.monotonic()

        if not actual_answer or not actual_answer.strip():
            return JudgeResult(
                correct=False,
                raw_response="Empty answer",
                question_type="",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            actual_answer=actual_answer,
        )

        response = self._call_llm(prompt, system=JUDGE_SYSTEM_PROMPT)
        self._call_count += 1
        correct = self._parse_verdict(response)

        return JudgeResult(
            correct=correct,
            raw_response=response,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def judge_answer_structured(self, question: str, gold_answer: str,
                                actual_answer: str) -> JudgeResult:
        """Structured rubric judge: scores 4 dimensions + binary verdict.

        Sends the structured system prompt and expects a JSON response.
        On any parse failure falls back transparently to judge_answer so
        callers always receive a valid JudgeResult.

        Returns
        -------
        JudgeResult with:
            correct      — binary verdict
            scores       — {relevance, factual_accuracy, completeness,
                            temporal_correctness} each in [0.0, 1.0]
            confidence   — mean of the four dimension scores
            latency_ms   — wall-clock ms for this call
            tokens_used  — input + output tokens consumed
        """
        t0 = time.monotonic()

        if not actual_answer or not actual_answer.strip():
            return JudgeResult(
                correct=False,
                raw_response="Empty answer",
                question_type="",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            actual_answer=actual_answer,
        )

        raw = self._call_llm(prompt, system=STRUCTURED_JUDGE_SYSTEM_PROMPT,
                             max_tokens=300)
        self._call_count += 1
        latency_ms = (time.monotonic() - t0) * 1000

        # --- Parse JSON response ---
        try:
            # Strip optional markdown fences the model might add despite instructions
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)

            data = json.loads(cleaned)

            scores = _scores_to_01(data)
            verdict_str = str(data.get("verdict", "")).upper()
            correct = "CORRECT" in verdict_str and "INCORRECT" not in verdict_str
            explanation = data.get("explanation", "")
            confidence = sum(scores.values()) / len(scores) if scores else 0.0

            tokens = (
                self._total_input_tokens + self._total_output_tokens
                if self._call_count > 0 else 0
            )

            return JudgeResult(
                correct=correct,
                raw_response=raw,
                scores=scores,
                confidence=confidence,
                latency_ms=latency_ms,
                tokens_used=tokens,
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Structured judge failed to parse JSON response (%s); "
                "falling back to binary judge_answer. Raw: %.200s",
                exc, raw,
            )
            # Fall back — re-use binary verdict, timing already captured
            fallback = self.judge_answer(question, gold_answer, actual_answer)
            fallback.latency_ms += latency_ms  # cumulative
            return fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, system: str = JUDGE_SYSTEM_PROMPT,
                  max_tokens: int = 150) -> str:
        """Call the configured model for judgment.

        Parameters
        ----------
        prompt:     Formatted user message.
        system:     System prompt to use (binary or structured).
        max_tokens: Token budget for the response.
        """
        try:
            client = self._get_client()
            msg = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            # Track token usage
            self._total_input_tokens += msg.usage.input_tokens
            self._total_output_tokens += msg.usage.output_tokens
            return msg.content[0].text
        except Exception as e:
            logger.warning("LLM judge call failed: %s", e)
            # Fall back to heuristic if LLM fails
            return self._heuristic_fallback(prompt)

    def _heuristic_fallback(self, prompt: str) -> str:
        """Heuristic judge using keyword matching + substring containment.

        Uses multiple signals:
        1. Exact substring containment of gold in actual
        2. Keyword overlap (significant words)
        3. Number/identifier matching (port numbers, versions, names)
        4. Number + entity co-occurrence (for cross-reference reasoning answers)
        """
        lines = prompt.strip().split("\n")
        gold = ""
        actual = ""
        for line in lines:
            if line.startswith("Expected answer:"):
                gold = line.split(":", 1)[1].strip()
            elif line.startswith("Actual answer:"):
                actual = line.split(":", 1)[1].strip()

        if not actual:
            return "INCORRECT\nEmpty answer"

        gold_lower = gold.lower()
        actual_lower = actual.lower()

        # Signal 0: compound gold answers (e.g., "A (x) / B (y)" or "A, plus B")
        # Split on common compound separators and check if all parts match
        compound_seps = [" / ", ", plus ", " and ", " + "]
        for sep in compound_seps:
            if sep in gold_lower:
                parts = [p.strip() for p in gold_lower.split(sep)]
                if all(any(kw in actual_lower for kw in part.split()
                          if len(kw) > 3) for part in parts):
                    return "CORRECT\nCompound answer parts all found"

        # Signal 1: exact substring match (strongest signal)
        if gold_lower in actual_lower:
            return "CORRECT\nExact substring match"

        # Signal 2: extract key identifiers (numbers, proper nouns, technical terms)
        identifiers = re.findall(
            r'\b(?:\d+(?:\.\d+)*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b', gold
        )
        if identifiers:
            id_matches = sum(1 for ident in identifiers if ident.lower() in actual_lower)
            id_ratio = id_matches / len(identifiers)
            if id_ratio >= 0.8:
                return f"CORRECT\n{id_ratio:.0%} identifier match ({identifiers})"

        # Signal 3: number + entity co-occurrence for cross-reference answers
        gold_numbers = re.findall(r'\d+(?:\.\d+)?(?:\s*%)?', gold)
        if gold_numbers:
            num_matches = sum(1 for n in gold_numbers if n.strip() in actual_lower)
            num_ratio = num_matches / len(gold_numbers)

            tech_terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b', gold)
            stop_words_ext = {
                "the", "a", "an", "is", "are", "was", "were", "with",
                "and", "or", "for", "in", "on", "at", "to", "of", "it",
                "its", "by", "as", "that", "this", "from", "has", "have",
                "be", "been", "we", "our", "they", "do", "does", "not",
                "but", "so", "if", "than", "then", "about", "up", "out",
                "yes", "no", "each", "per", "would", "well", "within",
                "after", "before", "between", "both", "because", "since",
                "there", "into", "over", "under", "exceeds", "exceeding",
                "limit", "required", "tight", "close", "below", "above",
            }
            tech = [t.lower() for t in tech_terms if t.lower() not in stop_words_ext]
            tech_matches = sum(1 for t in tech if t in actual_lower) if tech else 0
            tech_ratio = tech_matches / len(tech) if tech else 0

            if num_ratio >= 0.5 and tech_ratio >= 0.3:
                return (
                    f"CORRECT\n{num_ratio:.0%} number match + "
                    f"{tech_ratio:.0%} tech match"
                )
            if num_ratio >= 0.7:
                return f"CORRECT\n{num_ratio:.0%} number match (strong)"

        # Signal 4: keyword overlap (standard)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "with",
            "and", "or", "for", "in", "on", "at", "to", "of", "it",
            "its", "by", "as", "that", "this", "from", "has", "have",
            "be", "been", "we", "our", "they", "do", "does", "not",
            "but", "so", "if", "than", "then", "about", "up", "out",
            "yes", "no",
        }
        gold_words = set(gold_lower.split()) - stop_words
        if not gold_words:
            return "CORRECT\nNo significant words to check"

        matches = sum(1 for w in gold_words if w in actual_lower)
        ratio = matches / len(gold_words) if gold_words else 0

        if ratio >= 0.75:
            return f"CORRECT\n{ratio:.0%} keyword match"

        # Signal 4b: character trigram overlap (catch paraphrased content)
        def _get_char_trigrams(s: str) -> set:
            s = s.replace(" ", "")
            return set(s[i:i+3] for i in range(len(s) - 2)) if len(s) >= 3 else set()

        gold_trigrams = _get_char_trigrams(gold_lower)
        actual_trigrams = _get_char_trigrams(actual_lower)

        if gold_trigrams:
            trigram_overlap = (
                len(gold_trigrams & actual_trigrams) /
                len(gold_trigrams | actual_trigrams)
            )
            if trigram_overlap >= 0.4:
                return f"CORRECT\n{trigram_overlap:.0%} char-trigram overlap"

        # Signal 5: partial identifier + keyword combined
        if identifiers:
            id_ratio_val = (
                sum(1 for ident in identifiers if ident.lower() in actual_lower)
                / len(identifiers)
            )
            if id_ratio_val >= 0.5 and ratio >= 0.4:
                return (
                    f"CORRECT\n{id_ratio_val:.0%} identifier + "
                    f"{ratio:.0%} keyword combined match"
                )
            if id_ratio_val == 0:
                return f"INCORRECT\nNo key identifiers found in actual ({identifiers})"

        return f"INCORRECT\n{ratio:.0%} keyword match (threshold: 75%)"

    def _parse_verdict(self, response: str) -> bool:
        """Parse CORRECT/INCORRECT from judge response."""
        first_line = response.strip().split("\n")[0].strip().upper()
        return "CORRECT" in first_line and "INCORRECT" not in first_line

    @property
    def stats(self):
        return {
            "calls": self._call_count,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# HeuristicJudge
# ---------------------------------------------------------------------------

class HeuristicJudge(MemoryJudge):
    """Substring-matching judge for fast iteration without LLM calls.

    Uses keyword overlap between gold answer and actual answer.
    Threshold: 75% of significant words must match.

    NOTE: This heuristic judge favors backends that return verbatim stored text.
    For semantic evaluation, use --judge llm. The 75% keyword threshold was chosen
    to reduce bias toward keyword-matching backends while still catching obvious
    matches. Character n-gram overlap provides a secondary signal for paraphrased
    content.

    In addition to the binary verdict, judge_answer now also populates the
    rubric scores dict so HeuristicJudge output is structurally comparable
    to MemoryJudge.judge_answer_structured output:

        relevance            — keyword overlap ratio (gold vs actual)
        factual_accuracy     — identifier match ratio
        completeness         — compound-answer part coverage ratio
        temporal_correctness — always 1.0 (heuristic cannot assess this)
        confidence           — mean of the four scores above
    """

    def _call_llm(self, prompt: str, system: str = JUDGE_SYSTEM_PROMPT,
                  max_tokens: int = 150) -> str:
        """Use keyword matching instead of LLM."""
        return self._heuristic_fallback(prompt)

    def judge_answer(self, question: str, gold_answer: str,
                     actual_answer: str) -> JudgeResult:
        """Binary judge with rubric scores derived from heuristic signals."""
        t0 = time.monotonic()

        if not actual_answer or not actual_answer.strip():
            return JudgeResult(
                correct=False,
                raw_response="Empty answer",
                question_type="",
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        gold_lower = gold_answer.lower()
        actual_lower = actual_answer.lower()

        # --- Rubric dimension 1: relevance — keyword overlap ---
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "with",
            "and", "or", "for", "in", "on", "at", "to", "of", "it",
            "its", "by", "as", "that", "this", "from", "has", "have",
            "be", "been", "we", "our", "they", "do", "does", "not",
            "but", "so", "if", "than", "then", "about", "up", "out",
            "yes", "no",
        }
        gold_words = set(gold_lower.split()) - stop_words
        keyword_matches = (
            sum(1 for w in gold_words if w in actual_lower)
            if gold_words else 0
        )
        relevance = (keyword_matches / len(gold_words)) if gold_words else 1.0

        # --- Rubric dimension 2: factual_accuracy — identifier match ---
        identifiers = re.findall(
            r'\b(?:\d+(?:\.\d+)*|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,})\b',
            gold_answer,
        )
        if identifiers:
            id_hits = sum(1 for i in identifiers if i.lower() in actual_lower)
            factual_accuracy = id_hits / len(identifiers)
        else:
            # Fall back to keyword ratio when no identifiers are present
            factual_accuracy = relevance

        # --- Rubric dimension 3: completeness — compound part coverage ---
        compound_seps = [" / ", ", plus ", " and ", " + "]
        all_parts_covered = False
        compound_ratio = 1.0  # optimistic default when no compound structure
        for sep in compound_seps:
            if sep in gold_lower:
                parts = [p.strip() for p in gold_lower.split(sep)]
                covered = sum(
                    1 for part in parts
                    if any(kw in actual_lower for kw in part.split() if len(kw) > 3)
                )
                compound_ratio = covered / len(parts) if parts else 1.0
                all_parts_covered = compound_ratio == 1.0
                break
        completeness = compound_ratio if compound_ratio < 1.0 else relevance

        # --- Rubric dimension 4: temporal_correctness — heuristic default ---
        temporal_correctness = 1.0  # heuristic cannot assess temporal ordering

        scores = {
            "relevance": max(0.0, min(1.0, relevance)),
            "factual_accuracy": max(0.0, min(1.0, factual_accuracy)),
            "completeness": max(0.0, min(1.0, completeness)),
            "temporal_correctness": temporal_correctness,
        }
        confidence = sum(scores.values()) / len(scores)

        # --- Binary verdict via existing heuristic logic ---
        prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            gold_answer=gold_answer,
            actual_answer=actual_answer,
        )
        response = self._heuristic_fallback(prompt)
        correct = self._parse_verdict(response)

        return JudgeResult(
            correct=correct,
            raw_response=response,
            scores=scores,
            confidence=confidence,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
