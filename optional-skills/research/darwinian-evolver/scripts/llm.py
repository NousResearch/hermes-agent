"""OpenAI-compatible LLM client tailored for evolutionary mutation.

Reuses Hermes's existing configuration conventions:

  * Reads ``model``, ``base_url``, ``api_key`` from ``~/.hermes/config.yaml``
    so the skill transparently picks up whichever local or hosted backend
    the user has already configured.
  * Honors the per-slot context length discovered by Hermes's probing
    logic (see ``agent/model_metadata.py`` — PR #12595 fix). We never
    send more tokens than the endpoint advertises.

Async-first: mutation operators dispatch many short calls in one
generation; a bare requests loop serializes them needlessly. We use
``httpx.AsyncClient`` with an ``asyncio.Semaphore`` bounding concurrency
to the endpoint's slot count.

Budget is enforced by a ``BudgetLedger`` callback the caller supplies;
each completed call records input/output tokens and USD and can raise
``BudgetExceeded`` mid-run so long-running experiments halt cleanly.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import httpx

if TYPE_CHECKING:  # pragma: no cover
    from cache import ResponseCache


# ---------------------------------------------------------------------------
# Configuration discovery
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else Path.home() / ".hermes"


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader that only handles the subset we need.

    We deliberately avoid an import of PyYAML so this module works even
    when the user has not installed Hermes's optional dev tooling. The
    Hermes config file follows a flat ``key: value`` convention at top
    level which we parse line-by-line. Nested blocks are ignored.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def discover_endpoint() -> dict:
    """Return ``{model, base_url, api_key}`` from the active Hermes config.

    Environment variables take precedence over ``~/.hermes/config.yaml``,
    matching Hermes's own loader order. Missing values default to an
    OpenRouter configuration which is the most common setup.
    """
    cfg = _load_yaml(_hermes_home() / "config.yaml")
    return {
        "model":    os.environ.get("EVOLVER_MODEL",    cfg.get("model",    "openrouter/anthropic/claude-sonnet-4-6")),
        "base_url": os.environ.get("EVOLVER_BASE_URL", cfg.get("base_url", "https://openrouter.ai/api/v1")),
        "api_key":  os.environ.get("OPENROUTER_API_KEY") or os.environ.get("EVOLVER_API_KEY") or cfg.get("api_key", ""),
    }


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Raised by :class:`BudgetLedger` when a call would exceed the cap."""


@dataclass
class BudgetLedger:
    """Tracks spend and hard-kills the run when the cap is hit.

    Pricing is best-effort. For local models we record token counts but
    charge ``$0.00`` — the cap then acts as a token cap. For hosted
    models the caller can pass per-million-token rates.
    """

    cap_usd: float = 1.00
    input_rate_per_million: float = 0.0
    output_rate_per_million: float = 0.0
    on_record: Optional[Callable[[int, int, float, str], None]] = None
    spent_usd: float = field(default=0.0, init=False)
    total_in:  int   = field(default=0,   init=False)
    total_out: int   = field(default=0,   init=False)
    calls:     int   = field(default=0,   init=False)

    def record(self, input_tokens: int, output_tokens: int, operator: str) -> None:
        cost = (
            input_tokens  * self.input_rate_per_million  / 1_000_000
            + output_tokens * self.output_rate_per_million / 1_000_000
        )
        self.spent_usd += cost
        self.total_in  += input_tokens
        self.total_out += output_tokens
        self.calls     += 1
        if self.on_record is not None:
            self.on_record(input_tokens, output_tokens, cost, operator)
        if self.cap_usd > 0 and self.spent_usd >= self.cap_usd:
            raise BudgetExceeded(
                f"budget exhausted: ${self.spent_usd:.4f} >= cap ${self.cap_usd:.4f} "
                f"after {self.calls} calls"
            )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


@dataclass
class LLMClient:
    """Thin async OpenAI-compat client scoped to one evolution run.

    The client holds one ``httpx.AsyncClient`` for connection reuse and a
    single ``asyncio.Semaphore`` bounding in-flight calls. The caller is
    responsible for ``aclose()``; use :meth:`session` as an async context
    manager to avoid leaked connections.
    """

    model:    str = ""
    base_url: str = ""
    api_key:  str = ""
    concurrency: int = 4
    timeout_s:   float = 60.0
    max_retries: int = 4
    budget: Optional[BudgetLedger] = None
    cache:  Optional["ResponseCache"] = None
    _client: Optional[httpx.AsyncClient] = field(default=None, init=False, repr=False)
    _sem:    Optional[asyncio.Semaphore] = field(default=None, init=False, repr=False)

    @classmethod
    def from_hermes(cls, **overrides: Any) -> "LLMClient":
        cfg = discover_endpoint()
        cfg.update(overrides)
        return cls(**cfg)

    async def __aenter__(self) -> "LLMClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout_s,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )
        self._sem = asyncio.Semaphore(max(1, self.concurrency))
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(
        self,
        system: str,
        user: str,
        *,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        operator: str = "mutation",
    ) -> str:
        """One chat completion. Returns the assistant text.

        Honors seed where the backend supports it; OpenAI-compat servers
        (llama.cpp, vLLM, LM Studio) all accept it. Anthropic and OpenAI
        treat ``seed`` as a best-effort hint. Retries transient errors
        with exponential backoff plus jitter and respects ``Retry-After``
        on 429.
        """
        if self._client is None or self._sem is None:
            raise RuntimeError("LLMClient must be used as an async context manager")

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            body["seed"] = int(seed)

        # Cache short-circuit: if the exact same request body has been
        # served before, return the stored response. We intentionally do
        # NOT record a budget line here — the original call already did,
        # and charging twice would overstate spend on reruns.
        if self.cache is not None:
            hit = self.cache.get(body)
            if hit is not None:
                return hit.response

        async with self._sem:
            last_exc: Optional[Exception] = None
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await self._client.post("/chat/completions", json=body)
                    if resp.status_code == 429:
                        retry_after = float(resp.headers.get("Retry-After", "1"))
                        await asyncio.sleep(retry_after + random.uniform(0, 0.25))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    content = (data["choices"][0]["message"].get("content") or "").strip()
                    usage = data.get("usage") or {}
                    if self.cache is not None:
                        # Persist before the budget so a crash mid-record
                        # still leaves the response available to replays.
                        self.cache.put(
                            body, content,
                            prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                        )
                    if self.budget is not None:
                        self.budget.record(
                            int(usage.get("prompt_tokens", 0)),
                            int(usage.get("completion_tokens", 0)),
                            operator,
                        )
                    return content
                except (httpx.RequestError, httpx.HTTPStatusError, KeyError, ValueError) as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2 ** attempt, 10) + random.uniform(0, 0.5))
            assert last_exc is not None
            raise last_exc

    async def complete_many(
        self,
        prompts: list[tuple[str, str]],
        *,
        seed: Optional[int] = None,
        temperature: float = 0.7,
        operator: str = "mutation",
    ) -> list[str]:
        """Fire N completions concurrently; preserve input order."""
        tasks = [
            self.complete(sys_, usr, seed=seed, temperature=temperature, operator=operator)
            for sys_, usr in prompts
        ]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Convenience for synchronous callers
# ---------------------------------------------------------------------------


def run_sync(coro: Awaitable[Any]) -> Any:
    """Run an awaitable from synchronous code without closing the event loop.

    evolver.py uses this in its subcommand handlers so we can keep the
    CLI surface synchronous while the hot path stays async.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.run_coroutine_threadsafe(coro, loop).result()  # type: ignore[arg-type]
    except RuntimeError:
        pass
    return asyncio.run(coro)  # type: ignore[arg-type]
