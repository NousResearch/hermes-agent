"""Enzyme — vault intelligence memory provider.

Builds a concept graph from markdown notes (tags, links, folders, timestamps).
Queries resolve against catalysts — pre-computed thematic questions that bridge
content — not raw text. 8ms queries, local embeddings, no runtime LLM calls.

Config in $HERMES_HOME/config.yaml:
  memory:
    provider: enzyme
    enzyme:
      vault_path: /path/to/vault
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

ENZYME_PETRI_SCHEMA = {
    "name": "enzyme_petri",
    "description": (
        "What is this vault about? Returns the main topics, how recently active "
        "they are, and thematic questions running through them. Use this when the "
        "user asks what's here, what they've been thinking about, or to orient "
        "yourself in a workspace of notes. The thematic questions (catalysts) are "
        "the vocabulary for enzyme_catalyze queries — pass them as-is or adapt them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Rank topics by relevance to this query. "
                    "Pass the user's message to focus results."
                ),
            },
            "top": {
                "type": "integer",
                "description": "Number of top entities to return. Default: 10.",
                "default": 10,
            },
        },
        "required": [],
    },
}

ENZYME_CATALYZE_SCHEMA = {
    "name": "enzyme_catalyze",
    "description": (
        "Search the user's vault by concept. Finds notes they forgot they wrote, "
        "connects entries from different time periods, surfaces patterns across "
        "hundreds of files. Returns quoted excerpts with file paths and dates.\n\n"
        "USE THIS PROACTIVELY when:\n"
        "- The user asks about their ideas, past thinking, or decisions\n"
        "- The user says 'I wrote about', 'I was thinking about', 'what did I decide'\n"
        "- You want to ground a response in the user's own words\n"
        "- The user asks about a topic visible in the petri entities or catalysts\n"
        "- You need context on any theme, project, or concept from their notes\n\n"
        "Compose queries using catalyst language from enzyme_petri, not the user's "
        "raw words — catalysts reach content that keyword search misses."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A conceptual query inspired by the user's intent. Use the "
                    "petri catalysts and entity themes as context to infer a few "
                    "different angles — don't copy catalyst phrases verbatim."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default: 10.",
                "default": 10,
            },
            "register": {
                "type": "string",
                "enum": ["explore", "continuity", "reference"],
                "description": (
                    "'explore' (default): surface patterns and tensions. "
                    "'continuity': restore prior decisions and context. "
                    "'reference': show what the user chose to capture."
                ),
                "default": "explore",
            },
        },
        "required": ["query"],
    },
}

ENZYME_REFRESH_SCHEMA = {
    "name": "enzyme_refresh",
    "description": (
        "Re-index vault content. Fast: skips if nothing changed. "
        "Use full=true to force complete re-index if results seem off."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "full": {
                "type": "boolean",
                "description": "Force full re-index. Default: false.",
                "default": False,
            },
        },
        "required": [],
    },
}

ENZYME_STATUS_SCHEMA = {
    "name": "enzyme_status",
    "description": "Show vault stats: doc count, entity count, catalyst count, embedding coverage.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

ENZYME_INIT_SCHEMA = {
    "name": "enzyme_init",
    "description": (
        "Initialize enzyme on a vault. Run after scanning vault structure and "
        "composing a guide (entity list). Takes 10-30s depending on vault size."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "guide": {
                "type": "string",
                "description": (
                    "Freeform guide string: entity names, folder: prefixes, "
                    "excludedTags: block. Tells enzyme which entities to focus on."
                ),
            },
            "quiet": {
                "type": "boolean",
                "description": "Return compact JSON with petri data. Default: true.",
                "default": True,
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_enzyme(args: list[str], vault_path: str | None = None,
                timeout: int = 30) -> str:
    """Run an enzyme CLI command and return raw stdout."""
    enzyme_bin = _find_enzyme_bin() or "enzyme"
    cmd = [enzyme_bin] + args
    if vault_path:
        cmd.extend(["-p", vault_path])
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"enzyme exited {result.returncode}")
    return result.stdout.strip()


def _find_enzyme_bin() -> str:
    """Find the enzyme binary — PATH first, then pip package managed location."""
    # Check PATH
    try:
        subprocess.run(["enzyme", "--version"], capture_output=True, timeout=5)
        return "enzyme"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check standard user binary location
    managed = os.path.join(os.path.expanduser("~"), ".local", "bin", "enzyme")
    if os.path.isfile(managed):
        try:
            subprocess.run([managed, "--version"], capture_output=True, timeout=5)
            return managed
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return ""


def _is_enzyme_available() -> bool:
    """Check if enzyme binary or pip package is available."""
    if _find_enzyme_bin():
        return True
    try:
        import enzyme_python_package
        return True
    except ImportError:
        return False


def _vault_is_initialized(vault_path: str | None) -> bool:
    """Check if the vault has an enzyme index."""
    base = vault_path or os.getcwd()
    return os.path.exists(os.path.join(base, ".enzyme", "enzyme.db"))


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class EnzymeMemoryProvider(MemoryProvider):
    """Vault intelligence via enzyme's concept graph and semantic search."""

    def __init__(self):
        self._vault_path: str | None = None
        self._petri_cache: str | None = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "enzyme"

    def is_available(self) -> bool:
        if not _is_enzyme_available():
            logger.info(
                "enzyme binary not found. Install: "
                "curl -fsSL https://enzyme.garden/install.sh | bash"
            )
            return False
        return True

    def get_config_schema(self) -> list:
        return [
            {
                "key": "vault_path",
                "description": "Path to markdown vault",
                "secret": False,
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        """Ensure the vault is indexed. Run init or refresh as needed."""
        # Vault path from config.yaml, or None (enzyme defaults to cwd)
        self._vault_path = kwargs.get("provider_config", {}).get("vault_path") or None

        # If pip package is installed but binary isn't downloaded yet, do it now
        if not _find_enzyme_bin():
            try:
                import enzyme_python_package
                enzyme_python_package.install()
            except ImportError:
                logger.warning("enzyme not available — skipping initialization")
                return
            except Exception as e:
                logger.warning("enzyme binary install failed: %s", e)
                return

        if not _find_enzyme_bin():
            logger.warning("enzyme binary not found — skipping initialization")
            return

        try:
            if not _vault_is_initialized(self._vault_path):
                print("  Indexing vault with enzyme...")
                _run_enzyme(["init", "--quiet"], self._vault_path, timeout=120)
            else:
                _run_enzyme(["refresh", "--quiet"], self._vault_path, timeout=60)
            self._initialized = True
        except Exception as e:
            logger.warning("enzyme initialize failed: %s", e)

    def system_prompt_block(self) -> str:
        """Return vault context from petri, baked into the system prompt."""
        if not self._initialized:
            return ""

        # Run petri and cache the result
        petri_output = self._run_petri()
        if not petri_output:
            return ""

        return (
            "# Enzyme — Vault Memory\n\n"
            "Active. This user has a personal knowledge vault — their notes, decisions, "
            "reflections, and accumulated thinking — indexed by enzyme.\n\n"
            "You have enzyme_petri and enzyme_catalyze tools available. Use them — do "
            "NOT call the enzyme CLI via terminal. The tools are faster and return "
            "structured results.\n\n"
            "The vault landscape below is ORIENTATION, not evidence. It tells you what "
            "topics exist and gives you catalyst vocabulary for searching. To make claims "
            "about what the user wrote or thought, you MUST call enzyme_catalyze first "
            "to retrieve actual excerpts. Do not elaborate on vault topics without "
            "grounding in retrieved content.\n\n"
            "Compose enzyme_catalyze queries using catalyst language below — catalysts "
            "reach content the user's raw words won't find. Use enzyme_petri with a "
            "query to re-rank topics for a specific question.\n\n"
            "The vault is the user's long-term memory of their own thinking. "
            "session_search covers past Hermes conversations; enzyme covers everything "
            "the user has written outside of Hermes.\n\n"
            "When presenting enzyme_catalyze results:\n"
            "- The results include a presentation_guidance field — READ IT and follow it.\n"
            "- Quote the user's own words first (blockquotes with file attribution), "
            "then add your observations.\n"
            "- Don't summarize analytically — ground every claim in a specific excerpt.\n"
            "- Never expose tool names to the user.\n\n"
            "## Vault landscape\n\n"
            f"{petri_output}\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Re-rank petri by the user's message for per-turn context."""
        if not self._initialized or not query:
            return ""
        try:
            raw = _run_enzyme(
                ["petri", "--query", query, "-n", "5"],
                self._vault_path, timeout=10,
            )
            if raw:
                return f"## Vault context (query-ranked)\n{raw}"
        except Exception as e:
            logger.debug("enzyme prefetch failed: %s", e)
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            ENZYME_PETRI_SCHEMA,
            ENZYME_CATALYZE_SCHEMA,
            ENZYME_REFRESH_SCHEMA,
            ENZYME_STATUS_SCHEMA,
            ENZYME_INIT_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "enzyme_petri":
                return self._handle_petri(args)
            elif tool_name == "enzyme_catalyze":
                return self._handle_catalyze(args)
            elif tool_name == "enzyme_refresh":
                return self._handle_refresh(args)
            elif tool_name == "enzyme_status":
                return self._handle_status(args)
            elif tool_name == "enzyme_init":
                return self._handle_init(args)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Refresh the index so notes written this session are picked up."""
        if not self._initialized:
            return
        try:
            _run_enzyme(["refresh", "--quiet"], self._vault_path, timeout=30)
        except Exception as e:
            logger.debug("enzyme session-end refresh failed: %s", e)

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "") -> None:
        # Enzyme doesn't auto-ingest turns — the vault is the source of truth.
        pass

    def shutdown(self) -> None:
        self._petri_cache = None
        self._initialized = False

    # -- Internal ------------------------------------------------------------

    def _run_petri(self, query: str | None = None, top: int = 10) -> str:
        """Run enzyme petri and return raw output."""
        try:
            cmd = ["petri", "-n", str(top)]
            if query:
                cmd.extend(["--query", query])
            return _run_enzyme(cmd, self._vault_path, timeout=10)
        except Exception as e:
            logger.debug("enzyme petri failed: %s", e)
            return ""

    def _handle_petri(self, args: dict) -> str:
        if not self._initialized:
            self.initialize(session_id="", hermes_home="")
        cmd = ["petri", "-n", str(args.get("top", 10))]
        query = args.get("query")
        if query:
            cmd.extend(["--query", query])
        raw = _run_enzyme(cmd, self._vault_path)
        return json.dumps({"output": raw}) if raw else json.dumps({"output": ""})

    def _handle_catalyze(self, args: dict) -> str:
        if not self._initialized:
            self.initialize(session_id="", hermes_home="")
        query = args.get("query", "")
        cmd = ["catalyze", query, "-n", str(args.get("limit", 10))]
        register = args.get("register", "explore")
        if register != "explore":
            cmd.extend(["--register", register])
        raw = _run_enzyme(cmd, self._vault_path)
        return json.dumps({"output": raw}) if raw else json.dumps({"output": ""})

    def _handle_refresh(self, args: dict) -> str:
        cmd = ["refresh", "--quiet"]
        if args.get("full", False):
            cmd.append("--full")
        raw = _run_enzyme(cmd, self._vault_path, timeout=120)
        self._initialized = True
        return json.dumps({"ok": True, "output": raw})

    def _handle_status(self, args: dict) -> str:
        raw = _run_enzyme(["status"], self._vault_path)
        return json.dumps({"output": raw})

    def _handle_init(self, args: dict) -> str:
        cmd = ["init"]
        if args.get("quiet", True):
            cmd.append("--quiet")
        guide = args.get("guide")
        if guide:
            cmd.extend(["--guide", guide])
        raw = _run_enzyme(cmd, self._vault_path, timeout=120)
        self._initialized = True
        return json.dumps({"ok": True, "output": raw})


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the enzyme memory provider with Hermes."""
    ctx.register_memory_provider(EnzymeMemoryProvider())
