"""LoCoMo Benchmark Adapter for Hermes.

Evaluates memory providers against the LoCoMo long-conversation QA benchmark.
LoCoMo tests a system's ability to recall facts from long multi-turn
conversations, with single-hop, multi-hop, and temporal question types.

Usage:
    python -m benchmarks.locomo_adapter --dataset path/to/locomo.json --provider hindsight

LoCoMo dataset format::

    {
      "conversation_id": "...",
      "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      "questions": [
        {"question": "...", "answer": "...", "type": "single_hop|multi_hop|temporal"}
      ]
    }
"""

from __future__ import annotations
import argparse
import asyncio
import importlib
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def _tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric tokenisation, dropping articles and punctuation."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _ARTICLES]


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------


def _compute_f1(predicted: str, expected: str) -> float:
    """Token-level F1 between *predicted* and *expected* strings.

    Uses simple whitespace/punctuation tokenisation with article removal,
    matching the standard QA-F1 convention used by SQuAD / LoCoMo.
    """
    pred_tokens = _tokenize(predicted)
    gold_tokens = _tokenize(expected)

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_f1_by_type(
    results: List[Dict[str, Any]], qtype: str
) -> float:
    """Average F1 for questions whose "type" field equals *qtype*."""
    subset = [r for r in results if r.get("type") == qtype]
    if not subset:
        return 0.0
    return sum(r["f1"] for r in subset) / len(subset)

# ---------------------------------------------------------------------------
# LLM answer generation
# ---------------------------------------------------------------------------


def _build_answer_prompt(question: str, context: str) -> str:
    return (
        "You are given retrieved context from a conversation history. "
        "Answer the following question using ONLY the information in the context. "
        "Be concise -- give a short factual answer, not a full sentence.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


async def _generate_answer(
    question: str,
    context: str,
    llm_client: Any,
    model: str,
) -> str:
    """Call an OpenAI-compatible chat completion to answer *question* given *context*."""
    prompt = _build_answer_prompt(question, context)

    # Support both sync and async OpenAI clients.
    if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
        resp = await asyncio.to_thread(
            llm_client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()

    # Fallback: assume the client exposes a simple async generate() method.
    if hasattr(llm_client, "generate"):
        result = llm_client.generate(prompt)
        if asyncio.iscoroutine(result):
            result = await result
        return str(result).strip()

    raise TypeError(
        f"Unsupported LLM client type: {type(llm_client).__name__}. "
        "Pass an openai.OpenAI / openai.AsyncOpenAI instance or an object "
        "with a .generate(prompt) method."
    )

# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------


class LocomoBenchmark:
    """Run the LoCoMo long-conversation QA benchmark against a Hermes MemoryProvider.

    Parameters
    ----------
    memory_system:
        Any ""MemoryProvider"" instance (from ""agent.memory_provider"").
    llm_client:
        An OpenAI-compatible client used to generate answers from recalled
        context.  If *None* the adapter will create a default ""openai.OpenAI""
        client (requires ""OPENAI_API_KEY"").
    llm_model:
        Model name for answer generation (default 'gpt-4o-mini').
    session_prefix:
        Prefix for session IDs created during evaluation.  Each conversation
        gets f'{session_prefix}_{convo_id}'.
    """

    def __init__(
        self,
        memory_system: Any,
        llm_client: Any | None = None,
        llm_model: str = "gpt-4o-mini",
        session_prefix: str = "locomo",
    ) -> None:
        self.memory = memory_system
        self.llm_model = llm_model
        self.session_prefix = session_prefix

        # Lazy-init the LLM client.
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                import openai

                self.llm_client = openai.OpenAI()
            except Exception as exc:
                raise RuntimeError(
                    "No llm_client supplied and openai.OpenAI() failed to "
                    "initialise.  Pass an explicit client or set OPENAI_API_KEY."
                ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the full benchmark loop over *dataset*.

        Returns a results dict with per-question scores and aggregates::

            {
              "num_conversations": int,
              "num_questions": int,
              "overall_f1": float,
              "single_hop_f1": float,
              "multi_hop_f1": float,
              "temporal_f1": float,
              "per_question": [...],
            }
        """
        all_results: List[Dict[str, Any]] = []
        start = time.monotonic()

        for convo in dataset:
            results = await self._evaluate_conversation(convo)
            all_results.extend(results)

        elapsed = time.monotonic() - start

        return {
            "num_conversations": len(dataset),
            "num_questions": len(all_results),
            "elapsed_seconds": round(elapsed, 2),
            "overall_f1": (
                sum(r["f1"] for r in all_results) / len(all_results)
                if all_results
                else 0.0
            ),
            "single_hop_f1": _compute_f1_by_type(all_results, "single_hop"),
            "multi_hop_f1": _compute_f1_by_type(all_results, "multi_hop"),
            "temporal_f1": _compute_f1_by_type(all_results, "temporal"),
            "per_question": all_results,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _evaluate_conversation(
        self, convo: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Inject turns for one conversation, then evaluate its questions."""
        convo_id: str = convo.get("conversation_id", "unknown")
        session_id = f"{self.session_prefix}_{convo_id}"
        turns: List[Dict[str, str]] = convo.get("turns", [])
        questions: List[Dict[str, str]] = convo.get("questions", [])

        # Ensure memory is initialised for this session.
        # MemoryProvider.initialize is sync; call in thread if needed.
        try:
            result = self.memory.initialize(session_id)
            if asyncio.iscoroutine(result):
                await result
        except TypeError:
            # Some providers accept extra kwargs; ignore.
            pass

        # Inject conversation turns pairwise (user, assistant).
        for i in range(0, len(turns) - 1, 2):
            user_turn = turns[i]
            asst_turn = turns[i + 1] if i + 1 < len(turns) else None

            user_content = user_turn.get("content", "")
            asst_content = asst_turn.get("content", "") if asst_turn else ""

            # sync_turn is designed to be non-blocking / async-safe.
            try:
                result = self.memory.sync_turn(
                    user_content,
                    asst_content,
                    session_id=session_id,
                )
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.warning(
                    "sync_turn failed for conversation %s turn %d",
                    convo_id,
                    i,
                    exc_info=True,
                )

        # Small delay to let background writes settle.
        await asyncio.sleep(0.5)

        # Evaluate each question.
        results: List[Dict[str, Any]] = []
        for qa in questions:
            question = qa.get("question", "")
            expected = qa.get("answer", "")
            qtype = qa.get("type", "unknown")

            # Prefetch relevant context from memory.
            context = ""
            try:
                ctx_result = self.memory.prefetch(
                    question, session_id=session_id
                )
                if asyncio.iscoroutine(ctx_result):
                    ctx_result = await ctx_result
                context = ctx_result or ""
            except Exception:
                logger.warning(
                    "prefetch failed for question in conversation %s",
                    convo_id,
                    exc_info=True,
                )

            # Generate an answer from the recalled context.
            try:
                predicted = await _generate_answer(
                    question, context, self.llm_client, self.llm_model
                )
            except Exception:
                logger.warning(
                    "LLM answer generation failed for conversation %s",
                    convo_id,
                    exc_info=True,
                )
                predicted = ""

            f1 = _compute_f1(predicted, expected)

            results.append(
                {
                    "conversation_id": convo_id,
                    "question": question,
                    "expected": expected,
                    "predicted": predicted,
                    "type": qtype,
                    "f1": round(f1, 4),
                    "context_length": len(context),
                }
            )

        return results

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_dataset(path: str) -> List[Dict[str, Any]]:
        """Load a LoCoMo-format JSON dataset from *path*.

        Supports both a single conversation object and a list of them.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data

        raise ValueError(
            f"Expected a JSON object or array at top level, got {type(data).__name__}"
        )


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------


def _resolve_provider(name: str) -> Any:
    """Try to import and instantiate a MemoryProvider by short name.

    Checks ""plugins.memory.<name>"" for a ""create_provider()"" factory or
    a class whose name ends with ""Provider"".
    """
    module_path = f"plugins.memory.{name}"
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        # Try the name as a fully-qualified module path.
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import memory provider '{name}'. "
                f"Tried plugins.memory.{name} and {name}."
            ) from exc

    # Prefer a create_provider() factory.
    if hasattr(mod, "create_provider"):
        return mod.create_provider()

    # Fall back to first class ending in "Provider".
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if (
            isinstance(obj, type)
            and attr_name.endswith("Provider")
            and attr_name != "MemoryProvider"
        ):
            return obj()

    raise AttributeError(
        f"Module {module_path} has no create_provider() or *Provider class."
    )


def main() -> None:
    """CLI entry point for running the LoCoMo benchmark."""
    parser = argparse.ArgumentParser(
        description="Run the LoCoMo benchmark against a Hermes memory provider."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to LoCoMo JSON dataset file.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help=(
            "Memory provider name (e.g. 'hindsight', 'mem0') or fully-qualified "
            "module path (e.g. 'plugins.memory.hindsight')."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model for answer generation (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (default: print to stdout).",
    )
    parser.add_argument(
        "--session-prefix",
        default="locomo",
        help="Prefix for session IDs (default: 'locomo').",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load dataset.
    logger.info("Loading dataset from %s", args.dataset)
    dataset = LocomoBenchmark.load_dataset(args.dataset)
    logger.info("Loaded %d conversation(s)", len(dataset))

    # Resolve memory provider.
    logger.info("Initialising memory provider: %s", args.provider)
    memory = _resolve_provider(args.provider)

    # Run benchmark.
    benchmark = LocomoBenchmark(
        memory_system=memory,
        llm_model=args.model,
        session_prefix=args.session_prefix,
    )

    results = asyncio.run(benchmark.evaluate(dataset))

    # Output results.
    output_json = json.dumps(results, indent=2, ensure_ascii=False)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output_json, encoding="utf-8")
        logger.info("Results written to %s", out_path)
    else:
        print(output_json)

    # Print summary to stderr for quick inspection.
    print(
        f"\n--- LoCoMo Benchmark Summary ---\n"
        f"Conversations : {results['num_conversations']}\n"
        f"Questions     : {results['num_questions']}\n"
        f"Overall F1    : {results['overall_f1']:.4f}\n"
        f"Single-hop F1 : {results['single_hop_f1']:.4f}\n"
        f"Multi-hop  F1 : {results['multi_hop_f1']:.4f}\n"
        f"Temporal   F1 : {results['temporal_f1']:.4f}\n"
        f"Elapsed       : {results['elapsed_seconds']:.2f}s\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
