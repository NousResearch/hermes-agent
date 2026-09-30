"""Task-boundary router, ModelProfile catalog, and lean role toolsets (05-final-plan.md §6, §8, §12)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from samagent.ledger.store import ProjectLedger

LEAN_ROLE_TOOLS: Dict[str, List[str]] = {
    "worker": [
        "read_file",
        "search_files",
        "patch",
        "write_file",
        "terminal",
        "todo_list",
    ],
    "orchestrator": [
        "read_file",
        "search_files",
        "patch",
        "write_file",
        "terminal",
        "todo_list",
        "delegate_task",
        "clarify",
    ],
    "verifier": [
        "read_file",
        "search_files",
        "terminal",
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_console",
    ],
}


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    provider: str
    family: str
    is_local: bool
    tier: str  # "small_local" | "capable_local" | "mid_cloud" | "strong_cloud"
    max_visible_tools: int = 8
    edit_format: str = "patch"  # "patch" | "search_replace" | "whole_file"
    grammar_mode: str = "json_schema"  # "none" | "json_schema" | "gbnf"
    ctx_budget: int = 32768
    vision: bool = False
    cost_per_m_in: float = 0.0
    cost_per_m_out: float = 0.0
    known_quirks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_MODEL_PROFILES: Dict[str, ModelProfile] = {
    # Local models from hermes_cli/local_runtime/catalog.json
    "qwen3.8-flash-next": ModelProfile(
        model_id="qwen3.8-flash-next",
        provider="local",
        family="qwen",
        is_local=True,
        tier="small_local",
        max_visible_tools=6,
        edit_format="search_replace",
        grammar_mode="json_schema",
        ctx_budget=32768,
        vision=False,
    ),
    "qwen3.8-27b": ModelProfile(
        model_id="qwen3.8-27b",
        provider="local",
        family="qwen",
        is_local=True,
        tier="capable_local",
        max_visible_tools=8,
        edit_format="patch",
        grammar_mode="json_schema",
        ctx_budget=65536,
        vision=True,
    ),
    "deepseek-v4-flash": ModelProfile(
        model_id="deepseek-v4-flash",
        provider="local",
        family="deepseek",
        is_local=True,
        tier="capable_local",
        max_visible_tools=8,
        edit_format="patch",
        grammar_mode="json_schema",
        ctx_budget=65536,
        vision=False,
    ),
    # Real Industry Cloud & Local models
    "claude-3.7-sonnet": ModelProfile(
        model_id="claude-3.7-sonnet",
        provider="anthropic",
        family="anthropic",
        is_local=False,
        tier="strong_cloud",
        max_visible_tools=16,
        edit_format="patch",
        grammar_mode="none",
        ctx_budget=200000,
        vision=True,
        cost_per_m_in=3.0,
        cost_per_m_out=15.0,
    ),
    "gpt-4o": ModelProfile(
        model_id="gpt-4o",
        provider="openai",
        family="openai",
        is_local=False,
        tier="strong_cloud",
        max_visible_tools=16,
        edit_format="patch",
        grammar_mode="json_schema",
        ctx_budget=128000,
        vision=True,
        cost_per_m_in=2.5,
        cost_per_m_out=10.0,
    ),
    "o3-mini": ModelProfile(
        model_id="o3-mini",
        provider="openai",
        family="openai",
        is_local=False,
        tier="strong_cloud",
        max_visible_tools=12,
        edit_format="patch",
        grammar_mode="json_schema",
        ctx_budget=128000,
        vision=False,
        cost_per_m_in=1.1,
        cost_per_m_out=4.4,
    ),
    "claude-3.5-sonnet": ModelProfile(
        model_id="claude-3.5-sonnet",
        provider="anthropic",
        family="anthropic",
        is_local=False,
        tier="strong_cloud",
        max_visible_tools=16,
        edit_format="patch",
        grammar_mode="none",
        ctx_budget=200000,
        vision=True,
        cost_per_m_in=3.0,
        cost_per_m_out=15.0,
    ),
    "gemini-2.0-flash": ModelProfile(
        model_id="gemini-2.0-flash",
        provider="google",
        family="gemini",
        is_local=False,
        tier="mid_cloud",
        max_visible_tools=12,
        edit_format="patch",
        grammar_mode="json_schema",
        ctx_budget=1048576,
        vision=True,
        cost_per_m_in=0.1,
        cost_per_m_out=0.4,
    ),
    "deepseek-r1": ModelProfile(
        model_id="deepseek-r1",
        provider="deepseek",
        family="deepseek",
        is_local=False,
        tier="strong_cloud",
        max_visible_tools=10,
        edit_format="patch",
        grammar_mode="none",
        ctx_budget=65536,
        vision=False,
        cost_per_m_in=0.55,
        cost_per_m_out=2.19,
    ),
    "qwen2.5-coder-32b": ModelProfile(
        model_id="qwen2.5-coder-32b",
        provider="local",
        family="qwen",
        is_local=True,
        tier="capable_local",
        max_visible_tools=8,
        edit_format="patch",
        grammar_mode="json_schema",
        ctx_budget=65536,
        vision=False,
    ),
    # Legacy & reference cloud models
    "gpt-5-mini": ModelProfile(
        model_id="gpt-5-mini",
        provider="openai",
        family="openai",
        is_local=False,
        tier="mid_cloud",
        max_visible_tools=12,
        edit_format="patch",
        grammar_mode="json_schema",
        ctx_budget=131072,
        vision=True,
        cost_per_m_in=0.25,
        cost_per_m_out=2.0,
    ),
    "claude-sonnet-4-5": ModelProfile(
        model_id="claude-sonnet-4-5",
        provider="anthropic",
        family="anthropic",
        is_local=False,
        tier="strong_cloud",
        max_visible_tools=16,
        edit_format="patch",
        grammar_mode="none",
        ctx_budget=200000,
        vision=True,
        cost_per_m_in=3.0,
        cost_per_m_out=15.0,
    ),
}



@dataclass
class RouteDecision:
    task_kind: str
    model: Optional[ModelProfile]
    uses_llm: bool
    reason: str
    sticky_preserved: bool = False
    visible_tools: List[str] = field(default_factory=list)
    request_overrides: Dict[str, Any] = field(default_factory=dict)
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_kind": self.task_kind,
            "model": self.model.to_dict() if self.model else None,
            "uses_llm": self.uses_llm,
            "reason": self.reason,
            "sticky_preserved": self.sticky_preserved,
            "visible_tools": list(self.visible_tools),
            "request_overrides": dict(self.request_overrides),
            "warning": self.warning,
        }


class TaskBoundaryRouter:
    """Static task-boundary router with sticky continuations, privacy guard, and scorecard gating."""

    def __init__(
        self,
        *,
        policy: str = "default",  # "default" | "local_strict"
        cloud_available: bool = True,
        profiles: Optional[Dict[str, ModelProfile]] = None,
        ledger: Optional[ProjectLedger] = None,
        preferred_model: Optional[str] = None,
    ) -> None:
        self.policy = policy if policy in ("default", "local_strict") else "default"
        self.cloud_available = bool(cloud_available) and (self.policy != "local_strict")
        self.profiles = dict(profiles or DEFAULT_MODEL_PROFILES)
        self.ledger = ledger
        self.preferred_model = preferred_model

    def _by_tier(self, tier: str, *, exclude_family: Optional[str] = None, local_only: bool = False) -> Optional[ModelProfile]:
        candidates = [
            p
            for p in self.profiles.values()
            if p.tier == tier and (not local_only or p.is_local)
        ]
        if exclude_family:
            diff_fam = [p for p in candidates if p.family != exclude_family]
            if diff_fam:
                return diff_fam[0]
        return candidates[0] if candidates else None

    def _local_passes_scorecard(self, model_id: str) -> bool:
        if self.ledger is None:
            return True  # optimistic default when no scorecard has been recorded yet
        card = self.ledger.get_scorecard(model_id)
        if not card:
            return True
        return float(card.get("tool_validity_pct", 0.0)) >= 95.0 and float(card.get("pass_pct", 0.0)) >= 70.0

    @staticmethod
    def _build_request_overrides(profile: ModelProfile) -> Dict[str, Any]:
        if profile.is_local and profile.grammar_mode == "json_schema":
            return {"extra_body": {"cache_prompt": True}}
        return {}

    def route(
        self,
        task_kind: str,
        *,
        request_reason: str = "task_start",  # "task_start" | "continuation" | "retry"
        active_model_id: Optional[str] = None,
        sensitivity: str = "public",
        writer_family: Optional[str] = None,
        verified_failures: int = 0,
    ) -> RouteDecision:
        """Select the model and visible tool surface at a task boundary."""
        # 1. Deterministic scaffold uses NO LLM
        if task_kind == "scaffold":
            return RouteDecision(
                task_kind=task_kind,
                model=None,
                uses_llm=False,
                reason="Deterministic template scaffold executes without an LLM call.",
                visible_tools=[],
            )

        # 2. Sticky routing on continuation / retry so prompt cache stays byte-stable
        if request_reason in ("continuation", "retry") and active_model_id and active_model_id in self.profiles:
            prof = self.profiles[active_model_id]
            return RouteDecision(
                task_kind=task_kind,
                model=prof,
                uses_llm=True,
                reason=f"Sticky {request_reason}: keeping '{prof.model_id}' to preserve prompt cache prefix.",
                sticky_preserved=True,
                visible_tools=self._tools_for_task(task_kind, prof),
                request_overrides=self._build_request_overrides(prof),
            )

        force_local = (not self.cloud_available) or (self.policy == "local_strict") or (sensitivity == "private")
        capable_local = self._by_tier("capable_local", local_only=True)
        small_local = self._by_tier("small_local", local_only=True) or capable_local
        mid_cloud = self._by_tier("mid_cloud")
        strong_cloud = self._by_tier("strong_cloud")

        warning: Optional[str] = None

        # 3. High-leverage reasoning phases: interview, spec_critique, contract_freeze, judge
        if task_kind in ("interview", "spec_critique", "contract_freeze", "judge"):
            if force_local:
                chosen = (
                    self._by_tier("capable_local", exclude_family=writer_family if task_kind == "judge" else None, local_only=True)
                    or capable_local
                )
                if self.policy == "local_strict" or not self.cloud_available:
                    warning = (
                        f"Running high-leverage '{task_kind}' on local model '{chosen.model_id if chosen else 'none'}' "
                        "(local_strict / no cloud key). Recommended autonomy: milestones."
                    )
                reason = f"Local-only constraint active for '{task_kind}'."
            else:
                if self.preferred_model and self.preferred_model in self.profiles and (task_kind != "judge" or self.profiles[self.preferred_model].family != writer_family):
                    chosen = self.profiles[self.preferred_model]
                    reason = f"Selected model '{chosen.model_id}' chosen for high-leverage phase '{task_kind}'."
                elif task_kind == "judge":
                    chosen = (
                        self._by_tier("strong_cloud", exclude_family=writer_family)
                        or self._by_tier("mid_cloud", exclude_family=writer_family)
                        or strong_cloud
                    )
                    reason = f"Independent judge routed to '{chosen.model_id}' (family '{chosen.family}' != writer '{writer_family}')."
                else:
                    chosen = strong_cloud or mid_cloud or capable_local
                    reason = f"Strongest available model selected for high-leverage phase '{task_kind}'."


        # 4. Explore, log triage, memory extraction -> small/capable local first
        elif task_kind in ("explore", "memory_extract"):
            chosen = small_local if task_kind == "explore" else capable_local
            reason = f"Local model selected for '{task_kind}' (cost & privacy)."

        # 5. Fix loop -> local first; escalate at task boundary after >= 2 verified failures
        elif task_kind == "fix_loop":
            if verified_failures >= 2 and not force_local and (mid_cloud or strong_cloud):
                chosen = mid_cloud or strong_cloud
                reason = f"Escalated to '{chosen.model_id}' at task boundary after {verified_failures} verified local failures."
            else:
                chosen = capable_local
                reason = f"Local-first fix loop (verified_failures={verified_failures} < 2)."

        # 6. Browser verification -> vision-capable model
        elif task_kind == "browser_verify":
            if capable_local and capable_local.vision:
                chosen = capable_local
                reason = "Vision-capable local model selected for L2 browser verification."
            elif not force_local and mid_cloud and mid_cloud.vision:
                chosen = mid_cloud
                reason = "Vision-capable cloud model selected for L2 browser verification."
            else:
                chosen = capable_local
                reason = "Fallback local model selected for browser verification."

        # 7. Module implementation -> local-capable if scorecard passes, else mid_cloud
        else:
            if force_local:
                chosen = capable_local
                reason = "Local model selected (local_strict / private / no cloud)."
            elif capable_local and self._local_passes_scorecard(capable_local.model_id):
                chosen = capable_local
                reason = f"Local-first worker '{capable_local.model_id}' passed bench-model scorecard."
            else:
                chosen = mid_cloud or strong_cloud or capable_local
                reason = f"Local scorecard below threshold; routed module implementation to '{chosen.model_id}'."

        if chosen is None:
            raise RuntimeError(f"No model profile available for task_kind={task_kind!r}")

        return RouteDecision(
            task_kind=task_kind,
            model=chosen,
            uses_llm=True,
            reason=reason,
            sticky_preserved=False,
            visible_tools=self._tools_for_task(task_kind, chosen),
            request_overrides=self._build_request_overrides(chosen),
            warning=warning,
        )

    @staticmethod
    def _tools_for_task(task_kind: str, profile: ModelProfile) -> List[str]:
        if task_kind in ("interview", "spec_critique", "contract_freeze"):
            base = LEAN_ROLE_TOOLS["orchestrator"]
        elif task_kind in ("browser_verify", "judge"):
            base = LEAN_ROLE_TOOLS["verifier"]
        else:
            base = LEAN_ROLE_TOOLS["worker"]
        return list(base[: profile.max_visible_tools])

    def build_escalation_handoff(
        self,
        *,
        task_id: str,
        module_name: str,
        goal: str,
        file_path: str,
        verified_failures: int,
    ) -> Dict[str, Any]:
        """Spawn a fresh-context escalation package with Ledger history (never swap mid-conversation)."""
        decision = self.route("fix_loop", verified_failures=max(2, verified_failures))
        attempts = self.ledger.list_attempts(task_id=task_id, limit=5) if self.ledger else []
        failed_notes = [
            f"- Attempt on `{a.file_path}` ({a.approach}) -> FAILED with `{a.error_signature}`"
            for a in attempts
            if a.outcome == "failed"
        ]
        handoff_context = "\n".join(
            [
                f"## Escalation Handoff for Module `{module_name}` (Task `{task_id}`)",
                f"Goal: {goal}",
                f"Previous worker hit {verified_failures} verified failures on `{file_path}`.",
                "Do NOT repeat these failed approaches:",
                *(failed_notes or ["- (No prior attempt details recorded)"]),
            ]
        )
        return {
            "route": decision.to_dict(),
            "credentials_cfg": {
                "model": decision.model.model_id if decision.model else "",
                "provider": decision.model.provider if decision.model else "",
                "request_overrides": decision.request_overrides,
            },
            "handoff_context": handoff_context,
        }
