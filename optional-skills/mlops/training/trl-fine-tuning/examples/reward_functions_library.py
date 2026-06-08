"""
Reusable GRPO reward function examples for TRL.

Each reward function accepts the broad signature used by ``GRPOTrainer`` and
returns one float per completion. The helpers handle both common completion
shapes:

- ``["plain text", ...]``
- ``[[{"role": "assistant", "content": "plain text"}], ...]``

Copy the functions you need into your training script, then tune the weights
for your task. Keep reward magnitudes small and combine 3-5 complementary
signals instead of relying on a single reward.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

RewardFunction = Callable[..., list[float]]

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_XML_PATTERN = re.compile(
    r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$",
    re.DOTALL,
)
_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n.*?```", re.DOTALL)


def _completion_text(completion: Any) -> str:
    """Normalize a TRL completion object into plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        return str(completion.get("content", completion))
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, Mapping):
            return str(first.get("content", first))
        return str(first)
    return str(completion)


def _texts(completions: Sequence[Any]) -> list[str]:
    return [_completion_text(completion) for completion in completions]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _reference_values(kwargs: dict[str, Any], count: int) -> list[str]:
    """Return per-example reference strings from common TRL dataset columns."""
    for key in ("answer", "answers", "reference", "references", "target", "targets"):
        value = kwargs.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value] * count
        if isinstance(value, Iterable):
            values = [str(item) for item in value]
            if len(values) == count:
                return values
    return [""] * count


def _extract_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


# ==================== General-purpose rewards ====================


def constant_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Return 1.0 for every completion; useful for debugging plumbing."""
    return [1.0] * len(completions)


def non_empty_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward completions that contain non-whitespace text."""
    return [1.0 if text.strip() else 0.0 for text in _texts(completions)]


def word_count_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward longer responses by normalized word count, capped at 1.0."""
    return [min(len(_words(text)) / 200.0, 1.0) for text in _texts(completions)]


def length_window_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward responses between 50 and 250 words."""
    rewards = []
    for text in _texts(completions):
        n_words = len(_words(text))
        rewards.append(1.0 if 50 <= n_words <= 250 else 0.0)
    return rewards


def brevity_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward concise answers; highest below 80 words, fades by 240 words."""
    rewards = []
    for text in _texts(completions):
        n_words = len(_words(text))
        rewards.append(max(0.0, 1.0 - max(0, n_words - 80) / 160.0))
    return rewards


def unique_word_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward vocabulary variety, capped at 1.0."""
    return [min(len(set(_words(text))) / 120.0, 1.0) for text in _texts(completions)]


def lexical_diversity_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward type-token ratio while avoiding divide-by-zero."""
    rewards = []
    for text in _texts(completions):
        words = _words(text)
        rewards.append(len(set(words)) / max(1, len(words)))
    return rewards


def no_apology_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward answers that avoid apology boilerplate."""
    apology_terms = ("sorry", "apologize", "apologies", "i cannot", "i can't")
    return [0.0 if any(term in text.lower() for term in apology_terms) else 1.0 for text in _texts(completions)]


def no_refusal_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward answers that avoid generic refusal phrasing."""
    refusal_terms = (
        "i am unable",
        "i'm unable",
        "i cannot assist",
        "i can't assist",
        "not able to help",
    )
    return [0.0 if any(term in text.lower() for term in refusal_terms) else 1.0 for text in _texts(completions)]


# ==================== Format rewards ====================


def xml_format_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward exact ``<reasoning>...</reasoning><answer>...</answer>`` format."""
    return [1.0 if _XML_PATTERN.search(text) else 0.0 for text in _texts(completions)]


def incremental_xml_format_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Give partial credit for each required XML tag."""
    rewards = []
    for text in _texts(completions):
        score = 0.0
        for tag in ("<reasoning>", "</reasoning>", "<answer>", "</answer>"):
            if tag in text:
                score += 0.25
        rewards.append(score)
    return rewards


def reasoning_tag_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward completions that include a non-empty reasoning block."""
    rewards = []
    for text in _texts(completions):
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        rewards.append(1.0 if reasoning and reasoning.group(1).strip() else 0.0)
    return rewards


def answer_tag_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward completions that include a non-empty answer block."""
    rewards = []
    for text in _texts(completions):
        answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        rewards.append(1.0 if answer and answer.group(1).strip() else 0.0)
    return rewards


def no_extra_text_after_answer_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward completions with no trailing text after ``</answer>``."""
    rewards = []
    for text in _texts(completions):
        if "</answer>" not in text:
            rewards.append(0.0)
            continue
        trailing = text.split("</answer>", 1)[1].strip()
        rewards.append(1.0 if not trailing else 0.0)
    return rewards


def json_parse_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward completions that are valid JSON objects or arrays."""
    rewards = []
    for text in _texts(completions):
        try:
            json.loads(text)
        except json.JSONDecodeError:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def markdown_code_block_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward completions containing a fenced Markdown code block."""
    return [1.0 if _CODE_BLOCK_RE.search(text) else 0.0 for text in _texts(completions)]


def bullet_list_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward answers with at least three Markdown bullet points."""
    rewards = []
    for text in _texts(completions):
        bullets = re.findall(r"(?m)^\s*[-*]\s+\S+", text)
        rewards.append(1.0 if len(bullets) >= 3 else 0.0)
    return rewards


# ==================== Task-specific answer rewards ====================


def exact_match_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward exact match between extracted answer and reference answer."""
    texts = _texts(completions)
    references = _reference_values(kwargs, len(texts))
    return [1.0 if _extract_answer(text) == ref else 0.0 for text, ref in zip(texts, references)]


def case_insensitive_match_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward case-insensitive answer/reference matches."""
    texts = _texts(completions)
    references = _reference_values(kwargs, len(texts))
    return [
        1.0 if _extract_answer(text).casefold() == ref.strip().casefold() else 0.0
        for text, ref in zip(texts, references)
    ]


def contains_reference_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward answers that contain the reference text as a substring."""
    texts = _texts(completions)
    references = _reference_values(kwargs, len(texts))
    return [
        1.0 if ref and ref.casefold() in _extract_answer(text).casefold() else 0.0
        for text, ref in zip(texts, references)
    ]


def numeric_match_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward answers whose final number matches the reference final number."""
    texts = _texts(completions)
    references = _reference_values(kwargs, len(texts))
    rewards = []
    for text, ref in zip(texts, references):
        predicted = _last_number(_extract_answer(text))
        expected = _last_number(ref)
        rewards.append(1.0 if predicted is not None and predicted == expected else 0.0)
    return rewards


def numeric_close_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward numerical answers by closeness to the reference value."""
    texts = _texts(completions)
    references = _reference_values(kwargs, len(texts))
    rewards = []
    for text, ref in zip(texts, references):
        predicted = _last_number(_extract_answer(text))
        expected = _last_number(ref)
        if predicted is None or expected is None:
            rewards.append(0.0)
            continue
        scale = max(abs(expected), 1.0)
        relative_error = abs(predicted - expected) / scale
        rewards.append(max(0.0, 1.0 - relative_error))
    return rewards


def finite_number_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward answers that end in a finite numeric value."""
    rewards = []
    for text in _texts(completions):
        value = _last_number(_extract_answer(text))
        rewards.append(1.0 if value is not None and math.isfinite(value) else 0.0)
    return rewards


def work_shown_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Reward math-style solutions that show intermediate work."""
    work_markers = ("=", "+", "-", "*", "/", "therefore", "so ")
    rewards = []
    for text in _texts(completions):
        lower = text.lower()
        has_work = sum(marker in lower for marker in work_markers) >= 2
        rewards.append(1.0 if has_work else 0.0)
    return rewards


# ==================== Reward factories ====================


def make_regex_reward(pattern: str, reward: float = 1.0, flags: int = 0) -> RewardFunction:
    """Create a reward function that scores completions matching ``pattern``."""
    compiled = re.compile(pattern, flags)

    def regex_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        return [reward if compiled.search(text) else 0.0 for text in _texts(completions)]

    return regex_reward


def make_keyword_coverage_reward(keywords: Sequence[str]) -> RewardFunction:
    """Create a reward based on the fraction of keywords present."""
    lowered = [keyword.casefold() for keyword in keywords]

    def keyword_coverage_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        rewards = []
        for text in _texts(completions):
            haystack = text.casefold()
            hits = sum(keyword in haystack for keyword in lowered)
            rewards.append(hits / max(1, len(lowered)))
        return rewards

    return keyword_coverage_reward


def make_json_keys_reward(required_keys: Sequence[str]) -> RewardFunction:
    """Create a reward for JSON objects containing all required keys."""
    required = set(required_keys)

    def json_keys_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        rewards = []
        for text in _texts(completions):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                rewards.append(0.0)
                continue
            if not isinstance(parsed, Mapping):
                rewards.append(0.0)
                continue
            rewards.append(1.0 if required.issubset(parsed.keys()) else 0.0)
        return rewards

    return json_keys_reward


def make_length_range_reward(min_words: int, max_words: int) -> RewardFunction:
    """Create a reward for answers inside a custom word-count range."""

    def custom_length_range_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        rewards = []
        for text in _texts(completions):
            n_words = len(_words(text))
            rewards.append(1.0 if min_words <= n_words <= max_words else 0.0)
        return rewards

    return custom_length_range_reward


def combine_rewards(weighted_rewards: Sequence[tuple[RewardFunction, float]]) -> RewardFunction:
    """Combine multiple reward functions into one weighted reward function."""

    def combined_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        totals = [0.0] * len(completions)
        for reward_fn, weight in weighted_rewards:
            values = reward_fn(completions, **kwargs)
            totals = [total + weight * value for total, value in zip(totals, values)]
        return totals

    return combined_reward


# Example bundle for XML-answer math tasks similar to ``basic_grpo_training.py``.
math_xml_reward = combine_rewards(
    [
        (incremental_xml_format_reward, 0.25),
        (xml_format_reward, 0.5),
        (numeric_match_reward, 2.0),
        (work_shown_reward, 0.25),
    ]
)
