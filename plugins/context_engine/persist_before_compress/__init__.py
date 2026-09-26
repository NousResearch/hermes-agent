"""persist_before_compress — Context Engine Plugin.

Problem this solves (see MEMORY.md / .hermes.md): the assistant's inferences,
learnings, and in-progress work were only persisted to the knowledge dictionary
on a self-enforced soft rule. If automatic context compaction fired without that
rule being honoured first, those hard-won facts were lost forever.

This plugin makes persistence automatic: its ``compress()`` runs *before* every
compaction, snapshots the recent assistant turns (the inferences/learnings about
to be compressed away) into the knowledge dictionary, redacts any secrets, and
commits the dictionary — then it delegates to the built-in ContextCompressor to
actually shrink the context.

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


# Knowledge dictionary directory (overridable via env). Resolved at runtime.
def _resolve_memory_dir() -> str:
    """Return the knowledge-dictionary directory.

    Resolution order (mirrors memory.py's env support):
      1. Explicit ``MEMORY_DIR`` env var (preferred).
      2. ``MEMORY_STORE`` env var -> its containing directory.
      3. The sibling ``memory`` project next to ``HERMES_HOME``.
    Fail closed when none resolve, so a misconfigured host cannot silently
    create a relative knowledge-dictionary tree under the current working
    directory on POSIX.
    """
    explicit = os.environ.get("MEMORY_DIR")
    if explicit:
        return explicit
    store_env = os.environ.get("MEMORY_STORE")
    if store_env:
        return os.path.dirname(store_env)
    try:
        from hermes_constants import get_hermes_home
        home = os.path.dirname(str(get_hermes_home()))
        candidate = os.path.join(home, "memory")
        if os.path.isdir(candidate):
            return candidate
    except Exception:
        pass
    raise RuntimeError(
        "persist_before_compress: knowledge dictionary directory is not "
        "configured. Set the MEMORY_DIR environment variable (e.g. to the "
        "'memory' project)."
    )


MEMORY_DIR = _resolve_memory_dir()
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
    """git commit the dictionary so it survives.

    Only the two owned dictionary files are staged (not ``git add -A``), so
    committing here never sweeps in unrelated changes from the rest of the
    repository.
    """
    try:
        staged = subprocess.run(
            ["git", "-C", MEMORY_DIR, "diff", "--cached", "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["git", "-C", MEMORY_DIR, "add",
             os.path.basename(STORE), os.path.basename(INDEX)],
            capture_output=True, text=True, timeout=30,
        )
        # Only commit if there is something staged (avoid empty commits).
        res = subprocess.run(
            ["git", "-C", MEMORY_DIR, "diff", "--cached", "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode != 0:
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
        and commit. This fires on EVERY compaction — no willpower required.
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
        # Delegate to the built-in compressor to shrink the context.
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
