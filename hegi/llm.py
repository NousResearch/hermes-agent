"""Hermes-native LLM adapter with hierarchical meeting analysis."""

from __future__ import annotations

import json
import re
import signal
import threading
from pathlib import Path
from typing import Any, Callable

from .models import MeetingEpisode


PROMPT_ROOT = Path(__file__).with_name("prompts") / "v2"


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content")
            if isinstance(content, str):
                return content.strip()
    raise ValueError("LLM 응답에서 text content를 찾지 못했습니다.")


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL | re.I)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(candidate[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("structured output은 JSON object여야 합니다.")
    return value


def chunk_messages(episode: MeetingEpisode, max_chars: int) -> list[str]:
    """Split only at message boundaries."""
    chunks: list[list[str]] = [[]]
    current_size = 0
    for message in episode.messages:
        line = (
            f"[message_id={message.message_id} agent={message.source_agent} "
            f"role={message.role} timestamp={message.timestamp:.3f}]\n{message.content}"
        )
        if chunks[-1] and current_size + len(line) + 2 > max_chars:
            chunks.append([])
            current_size = 0
        chunks[-1].append(line)
        current_size += len(line) + 2
    return ["\n\n".join(chunk) for chunk in chunks if chunk]


class HermesLLMClient:
    """Thin wrapper around Hermes's shared auxiliary provider resolution."""

    def __init__(
        self,
        *,
        provider: str = "",
        model: str = "",
        max_tokens: int = 10000,
        timeout_seconds: int = 180,
        call: Callable[..., Any] | None = None,
    ):
        self.provider = provider or None
        self.model = model or None
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self._call = call

    def complete(self, system: str, user: str) -> str:
        call = self._call
        if call is None:
            from agent.auxiliary_client import call_llm

            call = call_llm
        def invoke() -> Any:
            return call(
                task="hegi",
                provider=self.provider,
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=self.max_tokens,
            )

        if (
            self.timeout_seconds > 0
            and threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
        ):
            def timeout(_signum, _frame):
                raise TimeoutError(
                    f"HEGI LLM 호출이 {self.timeout_seconds}초를 초과했습니다."
                )

            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout)
            previous_timer = signal.setitimer(signal.ITIMER_REAL, self.timeout_seconds)
            try:
                response = invoke()
            finally:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
                signal.signal(signal.SIGALRM, previous_handler)
        else:
            response = invoke()
        return _response_text(response)

    def structured(self, system: str, user: str) -> dict[str, Any]:
        first = self.complete(system, user)
        try:
            return parse_json_object(first)
        except (json.JSONDecodeError, ValueError):
            repair = self.complete(
                "너는 손상된 JSON을 복구한다. 설명이나 Markdown 없이 유효한 JSON object만 출력한다.",
                f"다음 출력을 의미를 바꾸지 말고 JSON object로 복구하라:\n\n{first}",
            )
            return parse_json_object(repair)


class HierarchicalMeetingAnalyzer:
    def __init__(
        self,
        client: HermesLLMClient,
        *,
        max_input_chars: int = 100000,
        chunk_chars: int = 30000,
    ):
        self.client = client
        self.max_input_chars = max_input_chars
        self.chunk_chars = chunk_chars

    @staticmethod
    def _prompt(name: str) -> str:
        return (PROMPT_ROOT / name).read_text(encoding="utf-8")

    def analyze_payload(self, episode: MeetingEpisode) -> dict[str, Any]:
        chunks = chunk_messages(episode, self.chunk_chars)
        total_chars = sum(len(chunk) for chunk in chunks)
        if total_chars <= self.max_input_chars:
            source = "\n\n".join(chunks)
        else:
            summaries: list[str] = []
            chunk_prompt = self._prompt("chunk_summary.md")
            for index, chunk in enumerate(chunks, start=1):
                summaries.append(
                    self.client.complete(
                        chunk_prompt,
                        f"Chunk {index}/{len(chunks)}\n\n{chunk}",
                    )
                )
            source = (
                "다음은 발언 경계로 나눈 chunk별 논증 중심 요약이다. "
                "각 요약 안의 source message ID를 보존하라.\n\n"
                + "\n\n".join(
                    f"## Chunk {index}\n{summary}"
                    for index, summary in enumerate(summaries, start=1)
                )
            )
        return self.client.structured(
            self._prompt("meeting_minutes.md"),
            f"meeting_id: {episode.meeting_id}\n"
            f"episode_hash: {episode.episode_hash}\n"
            f"participants: {episode.participants}\n\n{source}",
        )
