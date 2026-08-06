"""Brain memory provider for Hermes Agent.

Wraps the EXISTING brain engine (brain-mcp/dist/cli.js — kuzu graph + sqlite,
namespaced recall with rerank) as a first-class Hermes MemoryProvider. The brain
engine is unchanged; this is a thin subprocess adapter.

Lifecycle (called by MemoryManager, per agent/memory_provider.py ABC):
  initialize()          — resolve BRAIN_ROOT, node, namespace from active profile
  system_prompt_block()  — static "brain active" note (kept tiny — recall is on-demand)
  prefetch(query)        — brain recall → inject ONLY relevant hits (not full MEMORY.md)
  get_tool_schemas()     — expose recall/write plus read-only decision/receipt checks
  handle_tool_call()     — dispatch those tools to the CLI
  sync_turn()            — persist salient turns to brain (async)
  on_memory_write()      — mirror built-in MEMORY.md writes into brain
  on_pre_compress()      — checkpoint brain-worthy insight before compaction
  on_session_end()       — final flush
  shutdown()

Config (hermes memory setup → brain, or ~/.hermes/config.yaml `memory: {provider: brain}`):
  BRAIN_ROOT env or brain.json: path to the brain workspace (contains brain-mcp/, brain.yaml)
  namespace: one of [global, apex, 5am, personal, brain] — resolved per-profile (see _resolve_namespace)

Namespaces map from Hermes profile/agent → brain namespace so each per-group
profile reads/writes its OWN memory space (the token-overhead fix). Instances in
brain.yaml (main, main-tg, coder, researcher, qa-claude) correspond to profiles.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

try:
    from agent.memory_provider import MemoryProvider
except Exception:  # allows standalone lint before Hermes is installed
    class MemoryProvider:  # minimal shim
        pass

BRAIN_NAMESPACES = ["global", "apex", "5am", "personal", "brain"]

# Profile/agent → brain namespace. Extend as per-group profiles are defined in P2.
# Group profiles get their project namespace; personal/DM → personal; shared → global.
PROFILE_NAMESPACE = {
    "main": "personal",
    "main-tg": "personal",
    "apex": "apex",
    "5am": "5am",
    # group profiles added during P2, e.g. "grp-5am": "5am", "grp-dpb": "global", ...
}
DEFAULT_NAMESPACE = "global"


class BrainMemoryProvider(MemoryProvider):
    # Hermes v2026.7.20 bounds external-provider prefetch at 8 seconds. Stop
    # our subprocess first so one slow lexical recall cannot suppress later
    # turns until the manager's detached worker finally exits.
    _PREFETCH_TIMEOUT_SECONDS = 6

    # ---- identity ----
    @property
    def name(self) -> str:
        return "brain"

    def is_available(self) -> bool:
        """No network calls. True if the brain engine is present."""
        root = os.environ.get("BRAIN_ROOT", "")
        if not root:
            return False
        return (Path(root) / "brain-mcp" / "dist" / "cli.js").is_file()

    # ---- config ----
    def get_config_schema(self):
        return [
            {
                "key": "brain_root",
                "description": "Path to the brain workspace (contains brain-mcp/ and brain.yaml)",
                "required": True,
                "env_var": "BRAIN_ROOT",
                "secret": False,
            },
            {
                "key": "node_bin",
                "description": "node binary path",
                "default": "node",
            },
            {
                "key": "prefetch_top_k",
                "description": "Max recall hits injected per turn",
                "default": 6,
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        path = Path(hermes_home) / "brain.json"
        path.write_text(json.dumps(values, indent=2))

    # ---- lifecycle ----
    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        cfg = {}
        cfg_path = Path(self._hermes_home) / "brain.json"
        if cfg_path.is_file():
            try:
                cfg = json.loads(cfg_path.read_text())
            except Exception:
                cfg = {}
        self._brain_root = os.environ.get("BRAIN_ROOT") or cfg.get("brain_root", "")
        self._node = cfg.get("node_bin", "node")
        self._top_k = int(cfg.get("prefetch_top_k", 6))
        # Read-only mode for sandbox/staging: skip all brain WRITES (checkpoint/
        # remember/mirror) so testing against a live brain never mutates it.
        self._readonly = os.environ.get("HERMES_BRAIN_READONLY", "").lower() in ("1", "true", "yes")
        # profile/agent id is injected by MemoryManager via kwargs when multi-agent
        self._agent_id = kwargs.get("agent_identity") or kwargs.get("agent_id") or kwargs.get("profile") or "main"
        self._namespace = str(cfg.get("namespace") or self._resolve_namespace(self._agent_id))
        self._synced_turns: set[str] = set()

    def _resolve_namespace(self, agent_id: str) -> str:
        return PROFILE_NAMESPACE.get(agent_id, DEFAULT_NAMESPACE)

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        self._session_id = new_session_id
        agent_identity = kwargs.get("agent_identity") or kwargs.get("agent_id") or kwargs.get("profile")
        if agent_identity:
            self._agent_id = agent_identity
            cfg_path = Path(self._hermes_home) / "brain.json"
            cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
            self._namespace = str(cfg.get("namespace") or self._resolve_namespace(agent_identity))

    # ---- brain CLI bridge ----
    def _run_brain(self, *args, timeout: int = 30) -> dict:
        """Invoke brain-mcp CLI, return parsed --json result. Never raises to caller."""
        if not self._brain_root:
            return {"error": "BRAIN_ROOT unset"}
        cmd = [self._node, "brain-mcp/dist/cli.js", *args, "--json"]
        env = {**os.environ, "BRAIN_ROOT": self._brain_root}
        try:
            out = subprocess.run(
                cmd, cwd=self._brain_root, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
            if out.returncode != 0:
                return {"error": out.stderr.strip()[:500]}
            return json.loads(out.stdout or "{}")
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)[:300]}

    # ---- recall / prefetch ----
    # Max chars per injected hit — keeps prefetch lean (the whole cost point).
    _HIT_MAX_CHARS = 500

    @staticmethod
    def _clean(content: str, limit: int) -> str:
        """Strip YAML frontmatter + collapse, then truncate — inject signal, not full docs."""
        c = content or ""
        if c.startswith("---"):
            end = c.find("---", 3)
            if end != -1:
                c = c[end + 3:]
        c = " ".join(c.split())
        return c[:limit].rstrip()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Background recall before each turn. Returns ONLY relevant, trimmed hits.

        Merges brain's always_loaded (permanent identity/boundaries) with the
        top-k query matches. Real recall JSON shape: {context, namespace,
        always_loaded: [...], retrieved: [...]} — each item has path + content.
        """
        if not query.strip():
            return ""
        res = self._run_brain(
            "recall",
            query,
            "--namespace",
            self._namespace,
            "--budget-tokens",
            str(max(500, self._top_k * 200)),
            timeout=self._PREFETCH_TIMEOUT_SECONDS,
        )
        if res.get("error"):
            return ""
        always = res.get("always_loaded") or []
        retrieved = (res.get("retrieved") or res.get("results") or [])[: self._top_k]
        if not always and not retrieved:
            return ""
        lines = []
        for h in always:
            lines.append(f"- [{h.get('path','identity')}] {self._clean(h.get('content',''), self._HIT_MAX_CHARS)}")
        for h in retrieved:
            txt = self._clean(h.get("content") or h.get("text") or str(h), self._HIT_MAX_CHARS)
            lines.append(f"- {txt}")
        return (
            "## Brain recall (namespace: %s; best-effort indexed results — verify relevance)\n%s"
            % (self._namespace, "\n".join(lines))
        )

    def system_prompt_block(self) -> str:
        # Kept intentionally tiny — recall is on-demand, not a static dump.
        return (
            "[brain memory active · namespace=%s · recall is best-effort indexed/lexical; "
            "do not claim semantic retrieval without embed_provider=up health evidence · "
            "check decisions and receipts before changing established work]"
            % getattr(self, "_namespace", DEFAULT_NAMESPACE)
        )

    # ---- tools exposed to the model ----
    def get_tool_schemas(self):
        return [
            {
                "name": "brain_recall",
                "description": (
                    "Search indexed long-term Brain memory. When embeddings are unavailable, "
                    "results are best-effort lexical matches; verify relevance and receipts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "brain_remember",
                "description": "Persist a durable fact/decision into brain memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "brain_decisions_check",
                "description": (
                    "Read the active profile's indexed decision ledger before proposing or "
                    "repeating work. This checks ledger entries only; a clear verdict is not "
                    "proof that no unindexed directive exists."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "proposal": {
                            "type": "string",
                            "description": "The concrete plan or change to compare with decisions",
                        },
                        "as_of": {
                            "type": "string",
                            "description": "Optional YYYY-MM-DD validity date",
                            "pattern": r"^\d{4}-\d{2}-\d{2}$",
                        },
                    },
                    "required": ["proposal"],
                },
            },
            {
                "name": "brain_receipt",
                "description": (
                    "Read receipt/assertion evidence for a Brain note or event ID. "
                    "This is a verification lookup and does not write memory."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Brain note ID or receipt/event ID to verify",
                        },
                    },
                    "required": ["id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        # Namespace is locked to the active profile. Model-supplied overrides
        # would break group isolation and permit cross-project recall/writes.
        ns = self._namespace
        if tool_name == "brain_recall":
            res = self._run_brain("recall", args.get("query", ""), "--namespace", ns)
            return json.dumps(res)
        if tool_name == "brain_remember":
            if self._readonly:
                return json.dumps({"skipped": "read-only mode (HERMES_BRAIN_READONLY)"})
            res = self._run_brain("remember", args.get("content", ""), "--namespace", ns)
            return json.dumps(res)
        if tool_name == "brain_decisions_check":
            cli_args = [
                "decisions-check",
                "--proposal",
                args.get("proposal", ""),
                "--namespace",
                ns,
            ]
            if args.get("as_of"):
                cli_args.extend(["--as-of", args["as_of"]])
            return json.dumps(self._run_brain(*cli_args))
        if tool_name == "brain_receipt":
            return json.dumps(self._run_brain("receipt", "--id", args.get("id", "")))
        return json.dumps({"error": f"unknown tool {tool_name}"})

    # ---- write-side sync ----
    def sync_turn(self, user: str, assistant: str, *, session_id: str = "") -> None:
        if self._readonly or not (user.strip() or assistant.strip()):
            return
        import hashlib
        payload = (
            f"Hermes turn session={session_id or self._session_id}\n"
            f"User: {user.strip()[:4000]}\nAssistant: {assistant.strip()[:6000]}"
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        if digest in self._synced_turns:
            return
        result = self._run_brain("remember", payload, "--namespace", self._namespace)
        if not result.get("error"):
            self._synced_turns.add(digest)

    def on_memory_write(self, action: str, target: str, content: str, **kwargs) -> None:
        """Mirror built-in MEMORY.md/USER.md writes into brain."""
        if self._readonly:
            return
        if action in ("add", "replace") and content.strip():
            self._run_brain("remember", content, "--namespace", self._namespace)

    def on_pre_compress(self, messages) -> str:
        """Checkpoint brain-worthy state before context compression."""
        if self._readonly:
            return ""
        content = "Hermes pre-compress checkpoint; durable turns synced to Brain."
        self._run_brain("checkpoint", "write", "--instance", self._agent_id, "--content", content)
        return ""

    def on_session_end(self, messages) -> None:
        if self._readonly:
            return
        self._run_brain("checkpoint", "write", "--instance", self._agent_id,
                        "--content", "Hermes session finalized; durable turns synced to Brain.")

    def shutdown(self) -> None:
        return None


def register(ctx) -> None:
    """Entry point — called by the memory plugin discovery system."""
    ctx.register_memory_provider(BrainMemoryProvider())
