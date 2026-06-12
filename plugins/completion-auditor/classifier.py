"""Deterministic completion-claim classifier for completion-auditor.

This module intentionally stays conservative. It detects explicit claims in the
assistant's final response, while filtering common false positives such as
planning-only text, future-tense promises, conditionals, and blocker reports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Claim:
    claim_text: str
    claim_type: str
    claim_scope: str | None = None


_CLAIM_TYPES = {
    "tested",
    "modified",
    "created",
    "verified",
    "deployed",
    "implemented",
    "completed",
    "other",
}

# Short-circuit phrases that are usually not completion claims.
_BLOCKER_RE = re.compile(
    r"\b(blocked|cannot|can't|could not|unable to|missing credentials|no credentials)\b"
    r"|막혔|불가능|할 수 없|できません|できない|ブロック",
    re.IGNORECASE,
)
_PLAN_ONLY_RE = re.compile(
    r"\b(plan|proposal|next steps?|would|should|recommend)\b"
    r"|계획|제안|다음 단계|おすすめ|推奨|予定",
    re.IGNORECASE,
)
_FUTURE_RE = re.compile(
    r"\b(i will|i'll|will run|will update|will create|will implement|going to)\b"
    r"|하겠|할게|할 예정|진행할|実行します|します予定|予定です",
    re.IGNORECASE,
)
_CONDITIONAL_RE = re.compile(
    r"\b(if|when|assuming|provided that|once)\b.{0,80}\b(pass|passes|succeed|success|works)\b"
    r"|통과하면|성공하면|되면|なら|場合",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？\n])\s+|\n+")
_PATH_RE = re.compile(r"(?:[\w.-]+/)+[\w.@-]+|[\w.-]+\.(?:py|js|ts|tsx|json|ya?ml|md|txt|toml|ini|sh|bash|css|html)")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "verified",
        re.compile(
            r"\b(verif(?:y|ied|ication)|checked|validated|confirmed)\b.{0,80}\b(pass(?:ed)?|ok|success|working|clean)\b"
            r"|검증.{0,40}(완료|통과|성공)|확인.{0,40}(완료|통과|성공)|確認.{0,40}(完了|成功)",
            re.IGNORECASE,
        ),
    ),
    (
        "tested",
        re.compile(
            r"\b(test(?:ed|s)?|pytest|unit tests?|integration tests?)\b.{0,80}\b(pass(?:ed|es)?|green|ok|success|succeeded)\b"
            r"|\b(pass(?:ed|es)?)\b.{0,80}\b(test(?:ed|s)?|pytest|unit tests?)\b"
            r"|테스트.{0,40}(통과|성공|완료)|テスト.{0,40}(通過|成功|完了)",
            re.IGNORECASE,
        ),
    ),
    (
        "deployed",
        re.compile(
            r"\b(deployed|published|released|sent|posted|uploaded)\b"
            r"|배포(했| 완료)|공개(했| 완료)|전송(했| 완료)|보냈|送信|デプロイ|公開しました",
            re.IGNORECASE,
        ),
    ),
    (
        "modified",
        re.compile(
            r"\b(updated|modified|edited|changed|patched|rewrote|refactored)\b"
            r"|수정(했| 완료)|변경(했| 완료)|업데이트(했| 완료)|更新しました|修正しました|変更しました",
            re.IGNORECASE,
        ),
    ),
    (
        "created",
        re.compile(
            r"\b(created|added|wrote|generated|scaffolded)\b"
            r"|생성(했| 완료)|추가(했| 완료)|작성(했| 완료)|作成しました|追加しました",
            re.IGNORECASE,
        ),
    ),
    (
        "implemented",
        re.compile(
            r"\b(implemented|built|fixed|resolved)\b"
            r"|구현(했| 완료)|빌드(했| 완료)|고쳤|해결(했| 완료)|実装しました|修正しました|解決しました",
            re.IGNORECASE,
        ),
    ),
    (
        "completed",
        re.compile(
            r"\b(done|completed|finished|all set)\b"
            r"|완료(했|됐|입니다)?|끝났|完了しました|完了です",
            re.IGNORECASE,
        ),
    ),
]


def _sentences(text: str) -> Iterable[str]:
    text = _INLINE_CODE_RE.sub(" ", _FENCED_CODE_RE.sub(" ", text))
    for part in _SENTENCE_SPLIT_RE.split(text.strip()):
        sentence = part.strip(" \t-•*")
        if sentence:
            yield sentence


def _is_false_positive(sentence: str) -> bool:
    if _BLOCKER_RE.search(sentence):
        return True
    if _FUTURE_RE.search(sentence):
        return True
    if _CONDITIONAL_RE.search(sentence):
        return True
    # Treat plan/proposal words as false positives only when no explicit past
    # completion marker is present in the same sentence.
    if _PLAN_ONLY_RE.search(sentence) and not re.search(
        r"\b(done|completed|finished|updated|created|implemented|fixed|passed)\b|완료|수정했|구현했|作成しました|完了",
        sentence,
        re.IGNORECASE,
    ):
        return True
    return False


def _scope_from_sentence(sentence: str) -> str | None:
    match = _PATH_RE.search(sentence)
    if match:
        return match.group(0).rstrip(".,;:。！？!)）]")
    return None


def classify_response(text: str | None) -> Claim | None:
    """Return the first explicit completion claim, or ``None``.

    The classifier is deterministic and intentionally low-recall/high-precision:
    it prefers missing a vague claim over flagging explanations or plans.
    """
    if not text or not text.strip():
        return None
    for sentence in _sentences(text):
        if _is_false_positive(sentence):
            continue
        for claim_type, pattern in _PATTERNS:
            if pattern.search(sentence):
                if claim_type not in _CLAIM_TYPES:
                    claim_type = "other"
                return Claim(
                    claim_text=sentence,
                    claim_type=claim_type,
                    claim_scope=_scope_from_sentence(sentence),
                )
    return None
