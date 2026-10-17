"""Task-local skill resolution, independent of disposable reasoning and compression.

Only compact metadata enters the transcript. Usable tool results stay in memory so a
compacted instruction can be restored without rerunning templates or setup side effects.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field

from agent.tool_guardrails import ToolCallSignature

LOOKUP_TOOLS = frozenset({"skills_list", "skill_view"})
CONTINUE_GUIDANCE = (
    "Skill lookup is preparation, not completion of the user task. Continue the parent task "
    "using the retained instructions and ordinary available tools. Do not retry this unchanged "
    "lookup or ask the user what to do solely because discovery was blocked. Stop only if specific "
    "unavailable instructions are indispensable and no alternative exists; report that dependency."
)


def _payload(result):
    try:
        value = json.loads(result)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def catalog_generation() -> str:
    """Hash source bytes, not timestamps: same-size edits in one clock tick matter.

    Resolve roots in the caller's profile scope. Plugin skills have separate roots;
    their metadata and source participate in the same generation.
    """
    from agent.skill_utils import get_disabled_skill_names, iter_skill_index_files
    from hermes_cli.plugins import get_plugin_manager
    from tools.skills_tool import _skill_search_dirs

    roots, _ = _skill_search_dirs()
    digest = hashlib.sha256()
    paths = set()
    for tier, root in roots:
        digest.update(f"{tier}:{root}".encode())
        paths.update(iter_skill_index_files(root, "SKILL.md"))
    manager = get_plugin_manager()
    for row in manager.list_plugin_skill_metadata():
        digest.update(json.dumps(row, sort_keys=True, default=str).encode())
        path = manager.find_plugin_skill(row["name"])
        if path:
            paths.add(path)
    for path in sorted(paths, key=str):
        digest.update(str(path).encode())
        try:
            digest.update(path.read_bytes())
        except OSError as exc:
            digest.update(type(exc).__name__.encode())
    digest.update(repr(sorted(get_disabled_skill_names())).encode())
    return digest.hexdigest()


@dataclass
class SkillResolution:
    generation: str = ""
    catalog_loaded: bool = False
    candidates: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    viewed: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    blocked_calls: dict = field(default_factory=dict)
    request_args: dict = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def metadata(self):
        return {
            "catalog_loaded": self.catalog_loaded, "generation": self.generation,
            "candidates_considered": self.candidates, "selected": self.selected,
            "viewed": self.viewed, "blocked_calls": list(self.blocked_calls),
            "discovery_complete": bool(self.selected), "next_action": "execute_parent_task",
        }

    def prepare(self, controller, name, args):
        generation = catalog_generation()
        if self.generation and self.generation != generation:
            for (tool, _), request in self.request_args.items():
                controller.forget_changed_call(tool, request)
            self.results.clear()
            self.request_args.clear()
            self.blocked_calls.clear()
            self.catalog_loaded = False
            self.viewed.clear()
            self.selected.clear()
        self.generation = generation
        signature = ToolCallSignature.from_call(name, args)
        key = (name, signature.args_hash)
        self.request_args[key] = dict(args)
        old = self.results.get(key)
        if old and args.get("file_path") and old[1].get("skill_dir"):
            if _resource_digest(old[1], args) != old[2]:
                controller.forget_changed_call(name, args)
                del self.results[key]
                self.blocked_calls.pop(key, None)

    def resolve(self, name, args, execute):
        signature = ToolCallSignature.from_call(name, args)
        key = (name, signature.args_hash)
        with self.lock:
            if key in self.blocked_calls:
                return self.blocked_result(self.blocked_calls[key])
            if key in self.results:
                payload = dict(self.results[key][1])
                payload.update(skill_resolution=self.metadata(), resolution_hint=CONTINUE_GUIDANCE)
                return json.dumps(payload, ensure_ascii=False)
            result = self.catalog_subset(name, args) or execute(args)
            payload = _payload(result)
            self.record(name, args, payload)
            if payload:
                payload.update(skill_resolution=self.metadata(), resolution_hint=CONTINUE_GUIDANCE)
                return json.dumps(payload, ensure_ascii=False)
            return result

    def catalog_subset(self, name, args):
        """A complete catalog already answers subsequent category questions."""
        if name != "skills_list" or not args.get("category") or args.get("reload_reason"):
            return None
        for key, (_, payload, _) in self.results.items():
            request = self.request_args.get(key, {})
            if key[0] != name or request.get("category") or payload.get("success") is not True:
                continue
            skills = [s for s in payload.get("skills", []) if s.get("category") == args["category"]]
            return json.dumps({**payload, "skills": skills, "count": len(skills),
                               "categories": sorted({s["category"] for s in skills})})
        return None

    def record(self, name, args, payload):
        signature = ToolCallSignature.from_call(name, args)
        # Incomplete/setup-needed responses must remain eligible for a concrete retry.
        if (not payload or payload.get("content_returned") is False or payload.get("truncated")
                or payload.get("is_binary") or payload.get("setup_needed")
                or payload.get("readiness_status") == "setup_needed"):
            return
        usable = payload.get("success") is True and not payload.get("error")
        if usable and name == "skills_list":
            self.catalog_loaded = True
            self.candidates = [s["name"] for s in payload.get("skills", []) if "name" in s]
        elif usable and name == "skill_view" and isinstance(payload.get("content"), str):
            skill = args.get("name", "")
            if not args.get("file_path"):
                if skill not in self.selected:
                    self.selected.append(skill)
                self.viewed[skill] = {
                    "status": "loaded",
                    "version_or_digest": hashlib.sha256(payload["content"].encode()).hexdigest(),
                }
        elif usable:
            return
        # Failures are retained too: changing reload_reason or the source generation
        # makes a new experiment; an unchanged retry only repeats the known failure.
        served_digest = hashlib.sha256(payload["content"].encode()).hexdigest() if isinstance(payload.get("content"), str) else None
        self.results[(name, signature.args_hash)] = (self.generation, payload, served_digest)

    def block(self, decision):
        if decision.signature:
            self.blocked_calls[(decision.tool_name, decision.signature.args_hash)] = decision

    def blocked_result(self, decision):
        payload = {"error": decision.message, "guardrail": decision.to_metadata(),
                   "skill_resolution": self.metadata(), "recovery": CONTINUE_GUIDANCE}
        if decision.signature:
            cached = self.results.get((decision.tool_name, decision.signature.args_hash))
            if cached:
                payload["retained_result"] = cached[1]
        return json.dumps(payload, ensure_ascii=False)


def resolution_for(agent) -> SkillResolution:
    return agent.__dict__.setdefault("_skill_resolution", SkillResolution())


def prepare_lookup(agent, name, args):
    if name in LOOKUP_TOOLS:
        resolution_for(agent).prepare(agent._tool_guardrails, name, args)


def execute_resolved_lookup(agent, name, args, execute):
    if name in LOOKUP_TOOLS:
        return resolution_for(agent).resolve(name, args, execute)
    return execute(args)


def _resource_digest(payload, args):
    if not (args.get("file_path") and payload.get("skill_dir")):
        return None
    from pathlib import Path
    try:
        content = (Path(payload["skill_dir"]) / args["file_path"]).read_text(encoding="utf-8-sig", errors="replace")
        return hashlib.sha256(content.encode()).hexdigest()
    except OSError:
        return None


def preserve_resolution_summary(agent, compressed):
    """Compression is the sanctioned rewrite boundary; keep only resolution metadata."""
    state = getattr(agent, "_skill_resolution", None)
    if not isinstance(state, SkillResolution) or not state.results:
        return compressed
    for index, message in enumerate(compressed):
        if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str):
            result = list(compressed)
            result[index] = {**message, "content": message["content"] + "\n\n<skill_resolution>"
                             + json.dumps(state.metadata()) + "</skill_resolution>\n"
                             + CONTINUE_GUIDANCE + " If required skill text was pruned, an exact skill_view "
                             "restores the retained result without another read. Historical reload markers "
                             "do not require another catalog discovery."}
            return result
    return compressed


def handle_lookup_halt(agent, decision) -> bool:
    """A lookup loop disables that action; it does not make the parent task impossible."""
    if decision.tool_name not in LOOKUP_TOOLS:
        return False
    resolution_for(agent).block(decision)
    return True
