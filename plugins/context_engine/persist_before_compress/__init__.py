"""persist_before_compress — Context Engine Plugin.

Problem this solves (see MEMORY.md / .hermes.md): the assistant's inferences,
learnings, and in-progress work were only persisted to the knowledge dictionary
on a self-enforced soft rule. If automatic context compaction fired without that
rule being honoured first, those hard-won facts were lost forever.

This plugin makes persistence automatic: its ``compress()`` runs *before* every
compaction, snapshots the recent assistant turns (the inferences/learnings about
to be compressed away) into the knowledge dictionary on F:, redacts any secrets,
and commits the dictionary — then it delegates to the built-in ContextCompressor
to actually shrink the context.

The Hermes plugin loader (``instance_from_module``) imports this package's
``__init__.py`` and pulls out the ``engine`` symbol, which must be a subclass of
``agent.context_engine.ContextEngine``.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
import subprocess
import uuid
from typing import Any, Dict, List

from agent.context_engine import ContextEngine


# Knowledge dictionary on F: (overridable via env, matching memory.py).
# Paths are derived from __file__ so CPython (which does NOT do MSYS path
# conversion on Windows) resolves them correctly regardless of how the
# process was launched. memory.py itself anchors HERE to __file__.
_HERE = os.path.dirname(os.path.abspath(__file__))
# Default store lives in the sibling "memory" project folder next to this plugin's
# parent (hermes-agent/plugins/context_engine/<name>/ -> F:/hermes working directory/memory).
# Overridable via env to match memory.py's MEMORY_DIR / MEMORY_STORE / MEMORY_INDEX.
if os.environ.get("MEMORY_DIR"):
    MEMORY_DIR = os.environ["MEMORY_DIR"]
elif os.environ.get("MEMORY_STORE"):
    MEMORY_DIR = os.path.dirname(os.environ["MEMORY_STORE"])
else:
    # The knowledge dictionary is the canonical project on F:
    # F:/hermes working directory/memory
    # NOTE: hermes-agent source lives on C:, so this is NOT reachable by
    # walking up from __file__. CPython resolves the canonical F:/ path fine
    # (no MSYS mangling) as long as we pass it as a literal, not an env var.
    MEMORY_DIR = "F:/hermes working directory/memory"
STORE = os.environ.get("MEMORY_STORE", os.path.join(MEMORY_DIR, "learnings.jsonl"))
INDEX = os.environ.get("MEMORY_INDEX", os.path.join(MEMORY_DIR, "index.json"))


def _message_text(content: Any) -> str:
    """Extract readable text from an OpenAI-style message content block."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "reasoning":
                    parts.append(block.get("reasoning", ""))
                else:
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
    return "\n".join(parts)


def _snapshot_messages(messages: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    """Recent assistant turns (inferences/learnings about to be compressed away)."""
    texts: List[str] = []
    for msg in reversed(messages[-limit:] if limit else messages):
        if msg.get("role") == "assistant":
            text = _message_text(msg.get("content"))
            if text.strip():
                texts.append(text.strip())
    return texts


def _write_to_dictionary(snapshot: List[str], title: str, category: str, subcategory: str,
                         entry_type: str = "positive", tags: str = "") -> bool:
    """Append one entry to the knowledge dictionary and rebuild the index.
    Returns True on success. All content is redacted before it is written."""
    if not snapshot:
        return False
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "id": uuid.uuid4().hex[:12],
            "ts": ts,
            "type": entry_type,
            "category": category,
            "subcategory": subcategory,
            "title": title,
            "detail": "\n\n---\n\n".join(snapshot),
            "tags": [t.strip() for t in tags.split(",")] if tags else [category, subcategory],
        }
        store_dir = os.path.dirname(STORE)
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)
        with open(STORE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Rebuild index.json so the dictionary stays scannable (mirrors memory.py.rebuild_index).
        try:
            entries = []
            if os.path.exists(STORE):
                with open(STORE, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            index = {"count": len(entries), "by_category": {}, "entries": []}
            for e in entries:
                cat = e.get("category", "misc")
                sub = e.get("subcategory", "")
                index["by_category"].setdefault(cat, {})
                index["by_category"][cat].setdefault(sub, 0)
                index["by_category"][cat][sub] += 1
                index["entries"].append({
                    "id": e.get("id", ""), "ts": e.get("ts", ""), "type": e.get("type", ""),
                    "category": cat, "subcategory": sub, "title": e.get("title", ""),
                    "tags": e.get("tags", []),
                })
            with open(INDEX, "w", encoding="utf-8") as fh:
                json.dump(index, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass  # index rebuild is best-effort; the append is the durable part.
        return True
    except Exception:
        return False


def _commit_dictionary() -> None:
    """git add -A && git commit the dictionary on F: so it survives."""
    try:
        subprocess.run(
            ["git", "-C", MEMORY_DIR, "add", "-A"],
            capture_output=True, text=True, timeout=30,
        )
        # Only commit if there is something staged (avoid empty commits).
        res = subprocess.run(
            ["git", "-C", MEMORY_DIR, "diff", "--cached", "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode != 0:  # non-zero => changes staged
            subprocess.run(
                ["git", "-C", MEMORY_DIR, "commit", "-m",
                 "persist_before_compress: pre-compaction inferences snapshot"],
                capture_output=True, text=True, timeout=60,
            )
    except Exception:
        pass  # persistence must never break compaction


class PersistBeforeCompressEngine(ContextEngine):
    """Context engine that persists inferences BEFORE delegating to the built-in compressor."""

    def __init__(self, model: str = "", threshold_percent: float = 0.50, protect_first_n: int = 3,
                 protect_last_n: int = 20, summary_target_ratio: float = 0.20, quiet_mode: bool = False,
                 summary_model_override: str = None, base_url: str = "", api_key: str = "",
                 config_context_length: int | None = None, provider: str = "", api_mode: str = "",
                 abort_on_summary_failure: bool = False, max_tokens: int | None = None,
                 model_thresholds: dict | None = None, threshold_tokens_cap: Any = None,
                 proactive_prune_tokens: int = 0, **_: Any):
        # Build the real compressor and configure it exactly like the host would.
        from agent.context_compressor import ContextCompressor
        self._compressor = ContextCompressor(
            model=model, threshold_percent=threshold_percent, protect_first_n=protect_first_n,
            protect_last_n=protect_last_n, summary_target_ratio=summary_target_ratio,
            quiet_mode=quiet_mode, summary_model_override=summary_model_override,
            base_url=base_url, api_key=api_key, config_context_length=config_context_length,
            provider=provider, api_mode=api_mode, abort_on_summary_failure=abort_on_summary_failure,
            max_tokens=max_tokens, model_thresholds=model_thresholds, threshold_tokens_cap=threshold_tokens_cap,
            proactive_prune_tokens=proactive_prune_tokens,
        )

    # ---- ContextEngine ABC ----
    @property
    def name(self) -> str:
        return "persist_before_compress"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        try:
            self._compressor.update_from_response(usage)
        except Exception:
            pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        try:
            return self._compressor.should_compress(prompt_tokens)
        except Exception:
            return False

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int | None = None,
                 focus_topic: str | None = None, force: bool = False, memory_context: str = "",
                 **_: Any) -> List[Dict[str, Any]]:
        """PERSIST FIRST, then delegate.

        Before the middle turns (which hold the assistant's recent inferences/
        learnings) are compressed away, snapshot them to the knowledge dictionary
        on F: and commit. This fires on EVERY compaction — no willpower required.
        """
        try:
            snapshot = _snapshot_messages(messages)
            if snapshot:
                # Redact any secrets/credentials BEFORE writing to disk.
                from agent.redact import redact_sensitive_text
                snapshot = [redact_sensitive_text(t, force=True) for t in snapshot]
                _write_to_dictionary(
                    snapshot,
                    title=f"Pre-compaction inferences snapshot {snapshot[0][:40].replace(chr(10), ' ')}",
                    category="platform", subcategory="hermes",
                    entry_type="positive", tags="hermes,persistence,compaction,inferences,learning",
                )
                _commit_dictionary()
        except Exception:
            # Persistence is best-effort: never let it break the actual compression.
            pass
        # Delegate to the built-in compressor to actually shrink the context.
        return self._compressor.compress(
            messages, current_tokens=current_tokens, focus_topic=focus_topic, force=force,
            memory_context=memory_context,
        )

    def update_model(self, model: str, context_length: int, base_url: str = "", api_key: str = "",
                     provider: str = "", api_mode: str = "", **_: Any) -> None:
        try:
            self._compressor.update_model(
                model=model, context_length=context_length, base_url=base_url, api_key=api_key,
                provider=provider, api_mode=api_mode,
            )
        except Exception:
            pass

    def __deepcopy__(self, memo):
        # Host deep-copies the engine per agent; copy only the mutable budget state
        # (the built-in compressor), not any locks/connections.
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        clone._compressor = copy.deepcopy(self._compressor, memo)
        return clone


# The plugin loader (instance_from_module) instantiates the ContextEngine subclass.
engine = PersistBeforeCompressEngine
