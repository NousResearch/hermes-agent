"""RLM (Recursive Language Model) agent loop over a Python REPL.

Architecture:

1. A Jupyter kernel is started via ``jupyter_client.KernelManager``.
2. The corpus is pickled to a temp file and unpickled inside the kernel.
3. Helper functions (``list_papers``, ``search``, ``get_section``, ``get_paper``,
   ``llm_query``) are defined in the kernel namespace.
4. The root LM is given a system prompt and chats in a loop. Each turn we:

      - extract the first ```repl fenced block and execute it
      - OR detect ``FINAL(...)`` / ``FINAL_VAR(varname)`` and return

5. Sub-LLM calls (``llm_query``) happen INSIDE the kernel — the API keys in
   the parent environment are inherited. This keeps the parent agnostic to
   how many sub-calls are issued, and avoids an IPC layer.

See ``skill.py`` for the top-level entry point.
"""
from __future__ import annotations

import logging
import os
import pickle
import queue
import re
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import RLMConfig
from llm_clients import ChatMessage, LLMClient
from prompts import build_root_system_prompt

log = logging.getLogger("rlm_corpus.engine")


# ---------------------------------------------------------------------------
# Protocol parsing: ```repl blocks, FINAL(...), FINAL_VAR(name)
# ---------------------------------------------------------------------------


_REPL_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)
_FINAL_VAR_RE = re.compile(r"FINAL_VAR\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")


def extract_code_block(text: str) -> str | None:
    m = _REPL_BLOCK_RE.search(text)
    return m.group(1) if m else None


def extract_final(text: str) -> tuple[str, str] | None:
    """Return ``(kind, payload)`` where kind is ``"answer"`` or ``"var"``.

    ``FINAL(...)`` wins; if absent, ``FINAL_VAR(name)`` is used.
    Parses balanced parens so an answer containing ``)`` is preserved.
    """
    m = _FINAL_VAR_RE.search(text)
    if m:
        return ("var", m.group(1))

    marker = "FINAL("
    idx = text.rfind(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return ("answer", text[start:i].strip())
            depth -= 1
        i += 1
    # Unbalanced — treat everything after FINAL( as the answer so we don't
    # strand the model in a loop over a typo.
    return ("answer", text[start:].strip())


# ---------------------------------------------------------------------------
# REPL execution result
# ---------------------------------------------------------------------------


@dataclass
class ReplResult:
    ok: bool
    output: str          # stdout/stderr/return-repr, truncated
    error: str | None    # traceback text if any


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = limit // 2
    tail = limit - head - 80
    return (
        s[:head]
        + f"\n\n[... {len(s) - head - tail} chars truncated ...]\n\n"
        + s[-tail:]
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RLMEngine:
    """Drives a root LLM in a loop against a live Jupyter kernel."""

    def __init__(
        self,
        corpus: dict[str, dict[str, Any]],
        root_llm: LLMClient,
        sub_llm_spec: dict[str, Any],
        config: RLMConfig,
    ) -> None:
        self.corpus = corpus
        self.root_llm = root_llm
        self.sub_llm_spec = sub_llm_spec  # passed into the kernel, not used here
        self.config = config

        self._km = None  # jupyter KernelManager
        self._kc = None  # jupyter KernelClient
        self._corpus_pickle: Path | None = None

    # ---- kernel lifecycle ------------------------------------------------

    def start(self) -> None:
        from jupyter_client import KernelManager  # lazy import

        self._km = KernelManager(kernel_name="python3")
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._init_repl_state()

    def stop(self) -> None:
        try:
            if self._kc is not None:
                self._kc.stop_channels()
        finally:
            if self._km is not None:
                self._km.shutdown_kernel(now=True)
            self._km = None
            self._kc = None
        if self._corpus_pickle and self._corpus_pickle.exists():
            try:
                self._corpus_pickle.unlink()
            except OSError:
                pass

    def __enter__(self) -> "RLMEngine":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ---- kernel bootstrap ------------------------------------------------

    def _init_repl_state(self) -> None:
        tmp = Path(tempfile.mkstemp(prefix="rlm_corpus_", suffix=".pkl")[1])
        with tmp.open("wb") as fh:
            pickle.dump(self.corpus, fh, protocol=pickle.HIGHEST_PROTOCOL)
        self._corpus_pickle = tmp

        bootstrap = self._build_bootstrap(tmp)
        result = self._exec(bootstrap, timeout=60)
        if not result.ok:
            raise RuntimeError(
                "RLM bootstrap failed:\n" + (result.error or result.output)
            )

    def _build_bootstrap(self, corpus_pickle: Path) -> str:
        spec = self.sub_llm_spec
        endpoint = spec.get("endpoint", "anthropic")
        model = spec.get("model")
        base_url = spec.get("base_url")
        max_chars_default = self.config.max_sub_llm_chars
        enable = self.config.enable_sub_calls

        # NOTE: double braces for any literal `{` `}` in generated Python.
        return textwrap.dedent(
            f"""
            import pickle, re as _re, json as _json, os as _os, sys as _sys

            with open({str(corpus_pickle)!r}, "rb") as _fh:
                corpus = pickle.load(_fh)

            _RLM_SUB_ENABLED = {enable!r}
            _RLM_SUB_ENDPOINT = {endpoint!r}
            _RLM_SUB_MODEL = {model!r}
            _RLM_SUB_BASE_URL = {base_url!r}
            _RLM_SUB_DEFAULT_MAX_CHARS = {max_chars_default!r}

            def list_papers():
                rows = []
                for fname, d in corpus.items():
                    md = d.get("metadata") or {{}}
                    st = d.get("stats") or {{}}
                    rows.append({{
                        "filename": fname,
                        "title": md.get("title"),
                        "authors": md.get("authors") or [],
                        "year": md.get("year"),
                        "char_count": st.get("char_count", 0),
                        "section_count": st.get("section_count", len(d.get("sections") or [])),
                    }})
                return rows

            def get_paper(filename):
                if filename in corpus:
                    return corpus[filename]
                for k in corpus:
                    if k.endswith(filename) or filename in k:
                        return corpus[k]
                raise KeyError(f"no such paper: {{filename!r}}. Try list_papers().")

            def get_section(filename, section_heading):
                doc = get_paper(filename)
                needle = section_heading.strip().lower()
                for s in doc.get("sections") or []:
                    if s.get("heading", "").strip().lower() == needle:
                        return s.get("text", "")
                for s in doc.get("sections") or []:
                    if needle in s.get("heading", "").lower():
                        return s.get("text", "")
                raise KeyError(
                    f"no section matching {{section_heading!r}} in {{filename!r}}. "
                    f"Available: {{[s.get('heading') for s in doc.get('sections') or []]}}"
                )

            def search(pattern, regex=False, case_sensitive=False, max_results=200, window=120):
                flags = 0 if case_sensitive else _re.IGNORECASE
                if regex:
                    rx = _re.compile(pattern, flags)
                else:
                    rx = _re.compile(_re.escape(pattern), flags)
                hits = []
                for fname, d in corpus.items():
                    for s in d.get("sections") or []:
                        for m in rx.finditer(s.get("text", "")):
                            start = max(0, m.start() - window)
                            end = min(len(s["text"]), m.end() + window)
                            hits.append({{
                                "filename": fname,
                                "section": s.get("heading"),
                                "match": m.group(0),
                                "context": s["text"][start:end],
                            }})
                            if len(hits) >= max_results:
                                return hits
                return hits

            def llm_query(prompt, max_chars=None):
                if not _RLM_SUB_ENABLED:
                    raise RuntimeError(
                        "sub-LLM calls are disabled for this run "
                        "(enable_sub_calls=False). Use search/get_section/etc instead."
                    )
                if max_chars is None:
                    max_chars = _RLM_SUB_DEFAULT_MAX_CHARS
                if isinstance(prompt, str) and len(prompt) > max_chars:
                    prompt = prompt[:max_chars] + f"\\n\\n[...truncated to {{max_chars}} chars]"
                wrapped = (
                    "You are assisting a research query. Answer the following based "
                    "ONLY on the provided text. If the text doesn't support an answer, "
                    "say so explicitly. Be concise.\\n\\n"
                ) + prompt
                if _RLM_SUB_ENDPOINT == "anthropic":
                    import anthropic as _a
                    _cli = _a.Anthropic()
                    _resp = _cli.messages.create(
                        model=_RLM_SUB_MODEL,
                        max_tokens=4096,
                        messages=[{{"role": "user", "content": wrapped}}],
                    )
                    return "".join(
                        getattr(b, "text", "") or "" for b in _resp.content
                    )
                else:
                    import openai as _o
                    _cli = _o.OpenAI(base_url=_RLM_SUB_BASE_URL) if _RLM_SUB_BASE_URL else _o.OpenAI()
                    _resp = _cli.chat.completions.create(
                        model=_RLM_SUB_MODEL,
                        messages=[{{"role": "user", "content": wrapped}}],
                        max_tokens=4096,
                    )
                    return _resp.choices[0].message.content or ""

            print("RLM kernel ready:", len(corpus), "docs,", sum((d.get('stats') or {{}}).get('char_count', 0) for d in corpus.values()), "chars")
            """
        )

    # ---- kernel exec -----------------------------------------------------

    def _exec(self, code: str, timeout: int | None = None) -> ReplResult:
        if self._kc is None:
            raise RuntimeError("engine not started")
        timeout = timeout if timeout is not None else self.config.kernel_exec_timeout

        msg_id = self._kc.execute(code, store_history=False)
        stdout_parts: list[str] = []
        error_text: str | None = None

        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                # Hard timeout; interrupt and return a partial result.
                if self._km is not None:
                    try:
                        self._km.interrupt_kernel()
                    except Exception:  # noqa: BLE001
                        pass
                return ReplResult(
                    ok=False,
                    output=_truncate("".join(stdout_parts), self.config.max_repl_output_chars),
                    error=f"[execution exceeded {timeout}s and was interrupted]",
                )

            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            msg_type = msg["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                stdout_parts.append(content.get("text", ""))
            elif msg_type == "execute_result":
                data = content.get("data") or {}
                if "text/plain" in data:
                    stdout_parts.append(data["text/plain"])
            elif msg_type == "display_data":
                data = content.get("data") or {}
                if "text/plain" in data:
                    stdout_parts.append(data["text/plain"])
            elif msg_type == "error":
                error_text = "\n".join(content.get("traceback") or [])
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break

        output = _truncate(
            "".join(stdout_parts), self.config.max_repl_output_chars
        )
        return ReplResult(ok=error_text is None, output=output, error=error_text)

    def _fetch_variable(self, name: str) -> str:
        # Round-trip through the kernel to materialise the variable as text.
        # `print(...)` (not stdout.write) avoids Jupyter echoing the write's
        # return value as an execute_result.
        code = f"print({name}, end='')"
        result = self._exec(code, timeout=30)
        return result.output

    # ---- main loop -------------------------------------------------------

    def answer(self, query: str) -> dict[str, Any]:
        system = build_root_system_prompt(
            self.corpus, query, enable_sub_calls=self.config.enable_sub_calls
        )
        messages: list[ChatMessage] = []
        trajectory: list[dict[str, Any]] = []

        final_payload: str | None = None
        final_kind: str | None = None
        stopped_because: str = "max_iterations"

        for step in range(self.config.max_iterations):
            # Seed the first turn with a short user kickoff so the model
            # always has a user message to respond to.
            if not messages:
                messages.append(ChatMessage(role="user", content="Begin."))

            response = self.root_llm.chat(
                messages,
                system=system,
                temperature=self.config.temperature,
                max_tokens=4096,
            )
            messages.append(ChatMessage(role="assistant", content=response))
            trajectory.append({"step": step, "role": "assistant", "content": response})

            final = extract_final(response)
            if final is not None:
                kind, payload = final
                if kind == "var":
                    payload = self._fetch_variable(payload)
                final_kind, final_payload = kind, payload
                stopped_because = "final"
                break

            code = extract_code_block(response)
            if code is None:
                nudge = (
                    "No ```repl block and no FINAL(...) found. Respond with either a "
                    "```repl code block OR FINAL(answer)."
                )
                messages.append(ChatMessage(role="user", content=nudge))
                trajectory.append({"step": step, "role": "user", "content": nudge})
                continue

            result = self._exec(code)
            feedback_lines = ["REPL output:"]
            if result.output:
                feedback_lines.append(result.output)
            if result.error:
                feedback_lines.append("TRACEBACK:\n" + result.error)
            if not result.output and not result.error:
                feedback_lines.append("(no output)")
            feedback = "\n".join(feedback_lines)
            messages.append(ChatMessage(role="user", content=feedback))
            trajectory.append({
                "step": step,
                "role": "user",
                "content": feedback,
                "ok": result.ok,
            })

        return {
            "answer": final_payload,
            "final_kind": final_kind,
            "stopped_because": stopped_because,
            "trajectory": trajectory,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }


# ---------------------------------------------------------------------------
# Citation post-processing
# ---------------------------------------------------------------------------


_BRACKET_CITE_RE = re.compile(r"\[([^\]\n]{1,200}?)\]")


def format_answer_with_references(
    answer: str,
    corpus: dict[str, dict[str, Any]],
) -> str:
    """Detect ``[filename]`` citations in the answer and append a References block."""
    if not answer:
        return answer

    seen: list[str] = []
    for m in _BRACKET_CITE_RE.finditer(answer):
        token = m.group(1).strip()
        if token in corpus and token not in seen:
            seen.append(token)

    if not seen:
        return answer

    lines = ["", "---", "", "**References**", ""]
    for fname in seen:
        md = corpus[fname].get("metadata") or {}
        title = md.get("title") or fname
        authors = ", ".join(md.get("authors") or []) or "—"
        year = md.get("year") or "n.d."
        lines.append(f"- [{fname}] **{title}** · {authors} · {year}")
    return answer.rstrip() + "\n" + "\n".join(lines)
