"""ByteRover memory plugin — V4 build.

Replaces the cli-binary backend with ByteRover V4's bundled .mjs scripts
(``recall.mjs``, ``record.mjs``, ``brv.mjs``).  No more ``brv curate``
session protocol — record is one-shot via ``--html``.  Storage is
resolved by ByteRover's space registry from the subprocess cwd; Hermes uses
``$HERMES_HOME/byterover/`` as that workspace cwd, and the bound tree lives
under ByteRover's data dir.

Per-question integration decisions (carried over from the planning round):

- **Scripts location**: env override ``BYTEROVER_V4_SCRIPTS_DIR`` →
  the Hermes-native skill install at ``$HERMES_HOME/skills/byterover/scripts/``.
  (No ``.openclaw`` or dev-workspace fallback — Hermes ships its own copy.)
- **System prompt block**: ships the current ByteRover skill curation rules
  every turn, ported to Hermes' tool-call shape (calls ``brv_record`` instead
  of shelling ``node record.mjs``).
- **Tool name**: ``brv_record`` (not ``brv_curate``).  ByteRover V4's primitive is
  ``record.mjs``; the tool name should be honest about it.
- **Curation model**: agent-tool-driven only.  No ``on_pre_compress``,
  ``sync_turn``, or ``on_memory_write`` auto-curation — those were always
  guessing what to save from raw message text; the agent picks better when
  it explicitly decides via the ``brv_record`` tool.

External runtime dep: Node.js (any modern version) on ``PATH``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeouts / thresholds
# ---------------------------------------------------------------------------

_RECALL_TIMEOUT_S = 10
_RECORD_TIMEOUT_S = 30
_MIN_QUERY_LEN = 5
_RECALL_LIMIT = 5

# ---------------------------------------------------------------------------
# Scripts dir resolution (cached)
# ---------------------------------------------------------------------------

def _hermes_scripts_dir() -> Path:
    """Hermes-native skill install: ``$HERMES_HOME/skills/byterover/scripts``."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "skills" / "byterover" / "scripts"


def _resolve_scripts_dir() -> Optional[Path]:
    """Locate the directory containing recall.mjs / record.mjs / brv.mjs.

    Highest precedence: ``BYTEROVER_V4_SCRIPTS_DIR`` env var.  Otherwise the
    Hermes-native skill install at ``$HERMES_HOME/skills/byterover/scripts`` —
    no ``.openclaw`` or dev-workspace fallback.  Returns ``None`` if nothing
    valid is found; the caller degrades gracefully (``is_available() == False``).
    """
    env = os.environ.get("BYTEROVER_V4_SCRIPTS_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
        if (p / "recall.mjs").is_file():
            return p
    p = _hermes_scripts_dir()
    if (p / "recall.mjs").is_file():
        return p
    return None


def _resolve_node() -> Optional[str]:
    return shutil.which("node")


def _get_byterover_cwd() -> Path:
    """Profile-scoped working directory for V4.

    Hermes runs all V4 subprocesses from ``$HERMES_HOME/byterover/``.
    ByteRover V4's ``resolveContextRoot`` maps that cwd through ByteRover's space
    registry. The context tree itself lives under ByteRover's data dir; this
    cwd is the stable Hermes workspace identity.
    """
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "byterover"


def _run_v4(
    script_name: str,
    args: List[str],
    *,
    timeout: int,
    cwd: Path,
) -> Dict[str, Any]:
    """Invoke a V4 .mjs entry as a subprocess.

    Best-effort throughout: any failure (missing node, missing script,
    timeout, non-zero exit) collapses to ``{"success": False, "error": ...}``
    so the caller can degrade rather than crash the agent loop.

    Returns ``{"success": True, "output": <stdout-stripped>}`` on a clean
    exit.  Callers that expect JSON envelopes from the .mjs script (recall,
    record) should still parse ``output`` themselves.
    """
    scripts_dir = _resolve_scripts_dir()
    if scripts_dir is None:
        return {
            "success": False,
            "error": (
                "ByteRover V4 scripts not found; install the byterover skill "
                "at $HERMES_HOME/skills/byterover/scripts/ "
                "(or set BYTEROVER_V4_SCRIPTS_DIR)"
            ),
        }
    node = _resolve_node()
    if not node:
        return {"success": False, "error": "node not on PATH"}

    script = scripts_dir / script_name
    if not script.is_file():
        return {"success": False, "error": f"{script} not found"}

    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [node, str(script), *args]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"V4 {script_name} timed out after {timeout}s"}
    except Exception as exc:  # noqa: BLE001 — defensive at the subprocess boundary
        return {"success": False, "error": f"V4 {script_name}: {exc}"}

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return {"success": False, "error": err or f"V4 {script_name} exited {result.returncode}"}

    return {"success": True, "output": (result.stdout or "").strip()}


# ---------------------------------------------------------------------------
# Tool schema — single tool: brv_record
# ---------------------------------------------------------------------------

RECORD_SCHEMA: Dict[str, Any] = {
    "name": "brv_record",
    "description": (
        "Save knowledge to ByteRover as a structured <bv-topic> HTML "
        "document. Use for decisions, rules, bug+fix pairs, conventions, "
        "non-obvious gotchas, or facts the user asks you to remember. The "
        "system prompt's <byterover-curate-guidance> block lists the full "
        "<bv-*> vocabulary and the structural rules every topic must "
        "follow. The agent authors the HTML; this tool just persists it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Topic path, slash-separated snake_case (e.g. "
                    "'security/auth'). Do NOT include '.html' — the writer "
                    "appends it. Must match the <bv-topic path=\"...\"> "
                    "attribute on the HTML you pass."
                ),
            },
            "html": {
                "type": "string",
                "description": (
                    "One bare <bv-topic>...</bv-topic> HTML document. No "
                    "code fences, no markdown wrapper, no <!doctype> / "
                    "<html> / <body>. See the curate guidance for required "
                    "attributes and the 19-element vocabulary."
                ),
            },
            "overwrite": {
                "type": "boolean",
                "description": (
                    "Replace an existing topic at this path. Default false. "
                    "Set to true ONLY after reading the existing topic and "
                    "merging its facts into the new HTML — overwriting "
                    "without merging loses prior knowledge."
                ),
            },
        },
        "required": ["path", "html"],
    },
}


# ---------------------------------------------------------------------------
# Curate guidance template (Hermes-flavored tool-call shape)
# ---------------------------------------------------------------------------

# Substitution sentinels (no f-strings — the body contains literal { } from
# JSON / code examples and f-string brace-escaping would be unreadable).
_RECORD_SCRIPT_SENTINEL = "@@RECORD_SCRIPT@@"
_BRV_SCRIPT_SENTINEL = "@@BRV_SCRIPT@@"

_GUIDANCE_TEMPLATE = """<byterover-curate-guidance>
# IRON LAW — read and obey

You are talking to a user inside Hermes. ByteRover is the user's durable
memory across all Hermes sessions. Two rules govern every substantive turn:

1. **USE the retrieved context** when a `<byterover-context>` block is
   present above (injected by the recall step before this turn started).
   Cite the topic path when your answer relies on it. Do not contradict a
   retrieved `<bv-rule severity="must">` without explicitly flagging it.

2. **CURATE new knowledge** at the END of your turn, before the final
   reply, whenever your work this turn produced any of these:
     - A decision the team / user should remember (e.g. "we use OAuth 2.0
       with PKCE")
     - A rule or convention worth pinning ("never log full JWTs")
     - A bug root cause + the fix that resolved it
     - A non-obvious gotcha, constraint, or pattern
     - A fact the user explicitly asked you to save

   You curate by calling the `brv_record` tool (NOT by running shell
   commands). This is NOT optional for substantive turns — the user
   expects ByteRover to learn from this conversation; that only happens
   if you save now.

## When to SKIP curation

- The answer was a one-word reply, a greeting, or a clarifying question.
- The retrieved context already covered everything; you added no new fact.
- The user explicitly said "don't record this" or equivalent.

If any of those apply, do not curate. Otherwise: curate.

# HOW to curate — call the brv_record tool

The tool takes two required args (`path`, `html`) and an optional
`overwrite` flag. Author one bare `<bv-topic>...</bv-topic>` HTML document
per topic and pass it as the `html` arg.

```json
{
  "name": "brv_record",
  "arguments": {
    "path": "security/auth",
    "html": "<bv-topic path=\\"security/auth\\" title=\\"Login flow\\" summary=\\"How users authenticate\\" keywords=\\"oauth,session,tokens\\" tags=\\"auth,security\\"><bv-reason>Authentication decisions affect credential custody and token handling across clients.</bv-reason><bv-task>End-user authentication design.</bv-task><bv-decision id=\\"d-oauth\\">Use OAuth 2.0 with PKCE.</bv-decision><bv-rule severity=\\"must\\">Tokens MUST be short-lived; refresh is handled by the gateway, not the client.</bv-rule><bv-fact subject=\\"auth_provider\\" category=\\"project\\" value=\\"OAuth 2.0 with PKCE\\" disclosure=\\"public\\">Production uses OAuth 2.0 with PKCE.</bv-fact></bv-topic>"
  }
}
```

# The full <bv-*> vocabulary — 19 elements, pick the right tag

Topics are HTML built from a closed set of 19 structured element types.
The engine indexes, ranks, and surfaces knowledge **by element type** —
putting a rule inside `<bv-fact>` or a decision inside `<bv-highlights>`
makes it unfindable. Use the element that matches the *kind* of knowledge.

**The most common anti-pattern: stuffing everything into `<bv-highlights>`
`<li>` items.** That produces flat topics with no structural index. Use
the specialized elements below instead.

## Container

| Element | Purpose |
|---|---|
| `<bv-topic>` | The root container — exactly one per file. Carries path, title, summary, keywords, tags, related. |

## Decisions and rules

| Element | Attributes | When |
|---|---|---|
| `<bv-decision id="d-...">` | `id` kebab-case | A discrete decision the team / user made. Pair with `<bv-reason>`. Body = one sentence, the decision. |
| `<bv-reason>` | none | The WHY behind a decision. 1-2 sentences. A decision without a reason rots fast. |
| `<bv-rule severity="must|should|may" id="r-...">` | `severity` (RFC 2119), `id` | Binding rule. Body = verbatim rule text. |

## Facts (structured, queryable)

| Element | Attributes | When |
|---|---|---|
| `<bv-fact subject="snake_case_subject" category="..." value="extracted-form">` | `subject`, `category`, `value` | One discrete fact per element. `category` ∈ {`convention`, `preference`, `project`, `environment`, `team`, `personal`, `other`}. Body = canonical natural-language statement (NOT a label — the statement itself). |

## Action items

| Element | When |
|---|---|
| `<bv-task>` | The scoping element. "What this topic is about", one sentence. Almost every topic has one. |
| `<bv-changes>` | Things that changed in this work. `<li>change</li>` per item. |

## Bugs and fixes

| Element | Attributes | When |
|---|---|---|
| `<bv-bug severity="low|medium|high|critical" id="b-...">` | `severity`, `id` | Bug record. Body = symptom + root cause. |
| `<bv-fix id="f-...">` | `id` | Fix for a bug. Body = ordered list of steps. |

## Patterns

| Element | Attributes | When |
|---|---|---|
| `<bv-pattern id="p-...">` | `id` | A reusable pattern (e.g. retry-with-backoff). Body = pattern description + when to apply. |

## Structure and process

| Element | When |
|---|---|
| `<bv-flow>` | A process or sequence of steps. Body = natural-language description or numbered list. |
| `<bv-structure>` | Architecture / system shape / hierarchy. Body = `<h3>` + `<ul>` / `<ol>`. |
| `<bv-dependencies>` | Dependency relationships. `<li>dep</li>` per item. |
| `<bv-highlights>` | Key takeaways AT A GLANCE — use SPARINGLY. If you have 5+ `<li>` items, ask yourself if they should be structured `<bv-fact>` / `<bv-rule>` instead. |

## References and metadata

| Element | When |
|---|---|
| `<bv-files>` | Source files this topic touches / references. `<li>src/path/to/file.ts</li>` per file. |
| `<bv-timestamp>` | Reference date (ISO 8601). Use when the topic captures a point-in-time fact. |
| `<bv-author>` | Person who authored / decided. Optional. |

## Illustrative content

| Element | Attributes | When |
|---|---|---|
| `<bv-examples>` | none | Worked examples, code snippets. Wrap code in `<pre><code>...</code></pre>`. |
| `<bv-diagram type="mermaid|plantuml|ascii|dot|graphviz|other">` | `type` | Verbatim diagram source. Body = the diagram text exactly as given. |

# Required structure (every topic you record)

1. A `<bv-reason>` explaining the WHY of this curation. Missing this is the
   most common authoring failure.
2. A scoping element: `<bv-task>` (or `<h1>` + intro paragraph).
3. At least one structural element beyond the task from:
   `<bv-decision>`, `<bv-bug>`, `<bv-fix>`, `<bv-changes>`, `<bv-files>`,
   `<bv-flow>`, `<bv-structure>`, `<bv-dependencies>`, `<bv-highlights>`,
   `<bv-pattern>`, `<bv-examples>`, `<bv-diagram>`.
4. `<bv-timestamp>` in ISO 8601 if the content has a reference date.
5. One `<bv-fact subject="snake_case" category="..." value="...">canonical
   statement</bv-fact>` for each discrete queryable fact.

A topic containing ONLY `<bv-fact>` siblings is a placeholder. Same goes
for ONLY `<bv-highlights><li>...</li></bv-highlights>` — that's flat, not
structured.

# <bv-topic> attributes

## Required
- `path` — slash-separated snake_case (e.g. `security/auth`). NO `.html`.
  Must match the `path` arg you pass to `brv_record`.
- `title` — human-readable short title.

## Recommended
- `summary` — one-line semantic summary. Drives the retrieval snippet.
- `keywords` — CSV of retrieval terms; drives BM25 search ranking.
- `tags` — CSV of categories.
- `related` — CSV of cross-references: `"@security/cookies.html"` for file
  targets, `"@ops"` for domain targets.

## NEVER author these (system-managed; writer rejects them)

`createdat`, `updatedat`, `id`, `importance`, `maturity`, `recency`.

# Output contract (hard rules)

- All attribute values are double-quoted strings, not single-quoted.
- All attribute names are lowercase.
- Path segments are snake_case with underscores between words:
  `security/oauth_pkce`, NOT `security/oauth-pkce`.
- `related=` uses `@path.html` for file targets and `@path` for folder/domain
  targets. The `@` prefix is required.
- The `html` argument is bare HTML: first character `<`, last characters
  `</bv-topic>`. No code fence wrapper.
- Do not invent custom elements outside the 19-element `<bv-*>` vocabulary, or
  attributes outside each element's documented schema.

# Sensitivity — mark facts you intend to share

A topic can be shared at three views: full, redacted, and metadata. Mark a fact
you intend to share with `disclosure="public"`; otherwise the fact defaults to
restricted and is stripped from the redacted view.

- `<bv-fact>` is the sole unit of per-item restriction.
- Topic `title` and prose text inside `<bv-structure>`, `<bv-flow>`,
  `<bv-highlights>`, etc. are public-by-contract; never put secrets there.
- Absent or misspelled `disclosure` is treated as restricted.
- `<bv-topic visibility>` does NOT make facts public; redaction consults each
  fact's own `disclosure` attribute.

# Record form used by Hermes

ByteRover V4 supports a simple CLI record form (`--title --body`) and the rich form
(`--html`). Hermes exposes only the rich form through `brv_record`, so always
author the full `<bv-topic>...</bv-topic>` HTML yourself.

# Preservation (when the user gave you primary-source material)

- Exact rules → `<bv-rule severity="must|should">` verbatim.
- Code snippets → `<pre><code>` inside `<bv-examples>`.
- Diagrams → `<bv-diagram type="...">` verbatim.
- Dates → resolve relative ("last Thursday") to absolute when possible.

# Path-exists collision

If `brv_record` returns `ok: false` with an "already exists" error:

1. The path is taken. Pick one of two recovery paths:
   - **Merge** (default — preserves prior knowledge):
     1. Don't call `brv_record` blindly with `overwrite: true`. The
        existing topic likely contains `<bv-rule>` / `<bv-fact>` /
        `<bv-decision>` you'll DROP if you don't merge them.
     2. (For now, with the current tool surface, the safest move is to
        pick a slightly different `path`, e.g.
        `security/auth_pkce_refresh` instead of `security/auth`.)
   - **Replace** (only when the user explicitly asks for replacement):
     Retry `brv_record` with `overwrite: true`. If the result includes a
     `structural-loss` warning, your HTML dropped element types — add
     them back and retry.

# After-curate behavior

When `brv_record` returns `ok: true`, briefly mention to the user that you
saved the knowledge (e.g. "Saved to byterover at `security/auth`."). Do
NOT dump the full HTML back at them — the file path is enough.

If `brv_record` returns `ok: false`, surface the error message to the
user plainly. Do not silently retry more than once.
</byterover-curate-guidance>"""


def _build_curate_guidance(scripts_dir: Optional[Path]) -> str:
    """Render the guidance with the resolved scripts paths interpolated.

    Even when scripts are missing we render with placeholder paths so the
    agent still sees the structural rules and tool contract — recall and
    record will simply fail at runtime with a clear error, which is better
    than a silent empty guidance block.
    """
    base = scripts_dir if scripts_dir else Path("/byterover-v4-scripts-missing")
    return (
        _GUIDANCE_TEMPLATE
        .replace(_RECORD_SCRIPT_SENTINEL, str(base / "record.mjs"))
        .replace(_BRV_SCRIPT_SENTINEL, str(base / "brv.mjs"))
    )


# ---------------------------------------------------------------------------
# Recalled-context wrapper (returned by prefetch)
# ---------------------------------------------------------------------------

_RECALL_BLOCK_HEADER = (
    "<byterover-context>\n"
    "# Project knowledge retrieved from ByteRover (authoritative — use it)\n\n"
    "The topics below are facts, decisions, and rules the user has already\n"
    "curated. Treat them as ground truth for anything they cover.\n\n"
    "**Instructions for using this context:**\n"
    "1. READ each <bv-topic> before drafting your answer.\n"
    "2. ALIGN your answer with retrieved decisions and rules — if a\n"
    "   <bv-rule severity=\"must\"> exists, do not contradict it.\n"
    "3. CITE the topic path (e.g. \"per security/auth\") when relying on it.\n"
    "4. SUPPLEMENT — don't duplicate. If the context already covers the\n"
    "   question, lean on it; don't re-derive from scratch.\n"
    "5. FLAG conflicts. If the user's request contradicts a retrieved rule,\n"
    "   surface the conflict explicitly rather than silently overriding.\n\n"
    "---\n\n"
)


def _wrap_recalled_content(content: str) -> str:
    return f"{_RECALL_BLOCK_HEADER}{content}\n</byterover-context>"


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------


class ByteRoverMemoryProvider(MemoryProvider):
    """ByteRover persistent memory via ByteRover V4 (.mjs scripts)."""

    def __init__(self) -> None:
        self._cwd: Path = Path()
        self._session_id: str = ""
        self._scripts_dir: Optional[Path] = None
        self._curate_guidance: str = ""

    @property
    def name(self) -> str:
        return "byterover"

    def is_available(self) -> bool:
        """Plugin only activates if BOTH node and the V4 scripts dir are
        discoverable. No network calls — pure filesystem checks."""
        return _resolve_scripts_dir() is not None and _resolve_node() is not None

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._cwd = _get_byterover_cwd()
        self._scripts_dir = _resolve_scripts_dir()
        self._curate_guidance = _build_curate_guidance(self._scripts_dir)
        self._cwd.mkdir(parents=True, exist_ok=True)
        logger.debug("byterover-v4 cwd prepared for cwd=%s", self._cwd)

    def system_prompt_block(self) -> str:
        """Return the full curate guidance every turn (per the integration
        plan's Q2: ship full guidance, no smart-debounce)."""
        if not self.is_available():
            return ""
        return self._curate_guidance

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Spawn recall.mjs and return the rendered <byterover-context>
        block, or empty string if nothing relevant.

        Best-effort: any failure path returns "" so the agent loop is never
        blocked by a recall outage.
        """
        if not query or len(query.strip()) < _MIN_QUERY_LEN:
            return ""

        result = _run_v4(
            "recall.mjs",
            [query.strip()[:5000], "--cwd", str(self._cwd), "--limit", str(_RECALL_LIMIT)],
            timeout=_RECALL_TIMEOUT_S,
            cwd=self._cwd,
        )
        if not result["success"]:
            logger.debug("byterover-v4 recall failed: %s", result.get("error"))
            return ""

        # recall.mjs always emits the envelope on stdout, even on no-match.
        try:
            envelope = json.loads(result["output"]) if result["output"] else {}
        except json.JSONDecodeError as exc:
            logger.debug("byterover-v4 recall returned non-JSON: %s", exc)
            return ""

        content = ((envelope.get("data") or {}).get("content") or "").strip()
        if not content:
            return ""
        return _wrap_recalled_content(content)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECORD_SCHEMA]

    def handle_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Route brv_record tool calls to record.mjs.

        Returns a JSON string. record.mjs's stdout is already a JSON
        envelope (`{ok: true, data: {created, path, warnings}}` on
        success, `{ok: false, error: "..."}` on failure), so we return its
        output verbatim. Any subprocess-level failure is wrapped in the
        same envelope shape so the agent sees a consistent contract.
        """
        if tool_name != "brv_record":
            return json.dumps({"ok": False, "error": f"Unknown tool: {tool_name}"})

        path = args.get("path") or ""
        html = args.get("html") or ""
        overwrite = bool(args.get("overwrite", False))

        if not isinstance(path, str) or not path.strip():
            return json.dumps({"ok": False, "error": "path is required and must be non-empty"})
        if not isinstance(html, str) or not html.strip():
            return json.dumps({"ok": False, "error": "html is required and must be non-empty"})

        cmd_args = [path.strip(), "--html", html]
        if overwrite:
            cmd_args.append("--overwrite")

        result = _run_v4(
            "record.mjs",
            cmd_args,
            timeout=_RECORD_TIMEOUT_S,
            cwd=self._cwd,
        )
        if not result["success"]:
            return json.dumps({"ok": False, "error": result.get("error", "unknown error")})

        # record.mjs returns a JSON envelope on stdout. Return it as-is.
        output = result.get("output") or ""
        if not output:
            return json.dumps({"ok": True, "data": {"created": True, "warnings": []}})
        return output

    def shutdown(self) -> None:
        # No daemon, no threads, no resources to release.
        return None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        # No API key, no remote auth. The only environment knob is the
        # scripts dir override, which is process-level (env var) — not a
        # per-profile setting.
        return []
