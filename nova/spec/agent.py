"""AgentSpec — one declarative agent definition.

Modelled on the ``agent.cordis.yml`` concept: a single file names an agent's model,
tools, instructions, limits and approval requirements, and the runtime adapter composes
a runnable agent from it. The spec is expressed entirely in NOVA vocabulary; nothing
here knows what a runtime calls these things.

Unknown fields are rejected. A customer bundle that misspells a key fails at load
rather than producing an agent that silently lacks the setting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from nova._fields import Doc
from nova.errors import SpecError

if TYPE_CHECKING:  # pragma: no cover
    from nova.spec.deployment import ProviderSpec

#: Reasoning depths NOVA accepts. Mapped per runtime by the adapter; a runtime that
#: cannot express a level is expected to say so rather than silently ignore it.
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "max")


@dataclass(frozen=True)
class ModelSpec:
    """Which model an agent runs on, and how to reach it.

    Deployment fields (``endpoint``, ``api_key_env``, ``context_window``, ``region``) are
    the per-agent half of :class:`nova.spec.deployment.ProviderSpec`. Most agents set none
    of them and inherit the tenant's; an agent that needs a cheaper model, or a different
    region, overrides only the field that differs.

    ``api_key_env`` names a variable and never carries a credential — refused at parse time
    if it looks like one. See :mod:`nova.spec.deployment` for why that refusal is the only
    moment it is worth anything.
    """

    provider: str = ""
    name: str = ""
    reasoning_effort: Optional[str] = None
    endpoint: str = ""
    api_key_env: str = ""
    context_window: Optional[int] = None
    region: str = ""

    @property
    def deployment(self) -> "ProviderSpec":
        """This agent's overrides, in the shared provider vocabulary."""
        from nova.spec.deployment import ProviderSpec

        return ProviderSpec(
            provider=self.provider,
            model=self.name,
            endpoint=self.endpoint,
            api_key_env=self.api_key_env,
            context_window=self.context_window,
            region=self.region,
        )

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "ModelSpec":
        if doc is None:
            return cls()
        from nova.spec.deployment import check_not_a_secret

        spec = cls(
            provider=doc.str_("provider"),
            name=doc.str_("name"),
            reasoning_effort=doc.choice("reasoning_effort", REASONING_EFFORTS),
            # Unexpanded: a ${VAR} must reach the worker as a reference and resolve in the
            # worker's environment, not in whoever ran `nova apply`.
            endpoint=doc.str_("endpoint", expand=False),
            api_key_env=check_not_a_secret(
                doc.str_("api_key_env", expand=False),
                field="model.api_key_env",
                source=doc.source,
            ),
            context_window=doc.int_("context_window", minimum=1),
            region=doc.str_("region", expand=False),
        )
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (
            ("provider", self.provider), ("name", self.name),
            ("reasoning_effort", self.reasoning_effort), ("endpoint", self.endpoint),
            ("api_key_env", self.api_key_env), ("region", self.region),
        ):
            if value:
                out[key] = value
        if self.context_window is not None:
            out["context_window"] = self.context_window
        return out


@dataclass(frozen=True)
class ToolsSpec:
    """What an agent may call.

    ``toolsets`` names groups the runtime already understands; ``allow`` adds
    individual capabilities; ``deny`` removes them and wins over both. Deny is
    deliberately last-word: a customer writing ``deny: [terminal]`` means it, whatever
    a toolset would otherwise have granted.
    """

    toolsets: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "ToolsSpec":
        if doc is None:
            return cls()
        spec = cls(
            toolsets=tuple(doc.str_list("toolsets")),
            allow=tuple(doc.str_list("allow")),
            deny=tuple(doc.str_list("deny")),
        )
        doc.reject_unknown()
        overlap = sorted(set(spec.allow) & set(spec.deny))
        if overlap:
            raise SpecError(
                f"{', '.join(overlap)} appear(s) in both allow and deny — deny wins, so "
                "remove the allow entry to make the intent explicit",
                field="tools",
                source=doc.source,
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in (("toolsets", self.toolsets), ("allow", self.allow), ("deny", self.deny)):
            if value:
                out[key] = list(value)
        return out


@dataclass(frozen=True)
class KnowledgeSpec:
    """Knowledge sources this agent may read.

    Phase 1 validates and carries these through; no retrieval is implemented. An
    adapter that cannot serve them reports it through
    :meth:`nova.runtime.base.AgentRuntime.capabilities` rather than failing the
    materialization, so a bundle authored for a later phase still loads today.
    """

    sources: tuple[str, ...] = ()

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "KnowledgeSpec":
        if doc is None:
            return cls()
        spec = cls(sources=tuple(doc.str_list("sources")))
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {"sources": list(self.sources)} if self.sources else {}


@dataclass(frozen=True)
class ApprovalSpec:
    """Business actions that require a human decision before the agent proceeds.

    Phase 1 records the requirement and exposes it; enforcement is the policy layer's
    job in a later phase. Recording it now means a bundle is authored once and the
    enforcement arrives underneath it.
    """

    required_for: tuple[str, ...] = ()

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "ApprovalSpec":
        if doc is None:
            return cls()
        spec = cls(required_for=tuple(doc.str_list("required_for")))
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {"required_for": list(self.required_for)} if self.required_for else {}


@dataclass(frozen=True)
class DelegationLimits:
    """Caps on subagent fan-out.

    Every field compiles to a runtime configuration key whose enforcement was verified at
    its call site; see :mod:`nova.policy.limits`. The runtime's own configuration warns
    that child concurrency multiplies cost linearly, which is why these are here.
    """

    max_concurrent_children: Optional[int] = None
    max_depth: Optional[int] = None
    max_child_turns: Optional[int] = None
    child_timeout_seconds: Optional[int] = None
    orchestrator_enabled: Optional[bool] = None

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "DelegationLimits":
        if doc is None:
            return cls()
        spec = cls(
            # Floors mirror the runtime's own clamps, so a value NOVA accepts is a value
            # the runtime will honour rather than silently raise.
            max_concurrent_children=doc.int_("max_concurrent_children", minimum=1),
            max_depth=doc.int_("max_depth", minimum=1),
            max_child_turns=doc.int_("max_child_turns", minimum=1),
            child_timeout_seconds=doc.int_("child_timeout_seconds", minimum=0),
            orchestrator_enabled=(
                doc.bool_("orchestrator_enabled") if doc.has("orchestrator_enabled") else None
            ),
        )
        doc.reject_unknown()
        if spec.child_timeout_seconds is not None and 0 < spec.child_timeout_seconds < 30:
            raise SpecError(
                f"child_timeout_seconds must be 0 (no timeout) or at least 30; the runtime "
                f"floors it at 30 and {spec.child_timeout_seconds} would be silently raised",
                field="limits.delegation.child_timeout_seconds",
                source=doc.source,
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("max_concurrent_children", self.max_concurrent_children),
                ("max_depth", self.max_depth),
                ("max_child_turns", self.max_child_turns),
                ("child_timeout_seconds", self.child_timeout_seconds),
                ("orchestrator_enabled", self.orchestrator_enabled),
            )
            if value is not None
        }


@dataclass(frozen=True)
class LimitsSpec:
    """Bounds on an agent's blast radius.

    Every value is optional; an unset limit means "use the runtime's default" rather than
    "unlimited", because a runtime default is usually the safer of the two.

    **No limit here vetoes a model call, and none claims to.** No plugin can veto a model
    call in this runtime. ``monthly_budget_usd`` is enforced where the runtime *does* give
    NOVA a veto: a task is refused before it starts, and every tool call but the ones that
    close the task is refused, once the agent's month-to-date spend reaches it. A model
    reply already in flight can still land, so a run can overshoot by that one reply — see
    :mod:`nova.policy.limits` for what each control actually does.
    """

    max_concurrent_tasks: Optional[int] = None
    max_task_runtime_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    max_turns: Optional[int] = None
    max_tool_calls_per_run: Optional[int] = None
    #: Injects a wrap-up request. NOT a limit — nothing terminates if it is ignored.
    soft_wrapup_after_seconds: Optional[int] = None
    #: Month-to-date spend (the runtime's own cost estimate, USD) at which new work and
    #: tool calls stop. Calendar month, UTC.
    monthly_budget_usd: Optional[float] = None
    delegation: "DelegationLimits" = field(default_factory=lambda: DelegationLimits())

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "LimitsSpec":
        if doc is None:
            return cls()
        if doc.has("daily_token_budget"):
            raise SpecError(
                "daily_token_budget is not supported: no plugin can veto a model call in "
                "this runtime, so a token ceiling cannot be enforced and NOVA will not "
                "pretend otherwise. Use max_turns and max_tool_calls_per_run to bound "
                "spend, and read reported usage to observe it",
                field="limits.daily_token_budget",
                source=doc.source,
            )
        spec = cls(
            max_concurrent_tasks=doc.int_("max_concurrent_tasks", minimum=0),
            max_task_runtime_seconds=doc.int_("max_task_runtime_seconds", minimum=1),
            max_retries=doc.int_("max_retries", minimum=0),
            max_turns=doc.int_("max_turns", minimum=1),
            max_tool_calls_per_run=doc.int_("max_tool_calls_per_run", minimum=1),
            soft_wrapup_after_seconds=doc.int_("soft_wrapup_after_seconds", minimum=1),
            monthly_budget_usd=doc.money("monthly_budget_usd"),
            delegation=DelegationLimits.parse(doc.child("delegation")),
        )
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        out = {
            key: value
            for key, value in (
                ("max_concurrent_tasks", self.max_concurrent_tasks),
                ("max_task_runtime_seconds", self.max_task_runtime_seconds),
                ("max_retries", self.max_retries),
                ("max_turns", self.max_turns),
                ("max_tool_calls_per_run", self.max_tool_calls_per_run),
                ("soft_wrapup_after_seconds", self.soft_wrapup_after_seconds),
                ("monthly_budget_usd", self.monthly_budget_usd),
            )
            if value is not None
        }
        delegation = self.delegation.to_dict()
        if delegation:
            out["delegation"] = delegation
        return out


@dataclass(frozen=True)
class DelegationSpec:
    """Which other agents this agent may hand work to.

    Validated against the bundle's roster at load, so a typo names a missing teammate
    at configuration time rather than at 3am in a customer's deployment.
    """

    may_assign_to: tuple[str, ...] = ()

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "DelegationSpec":
        if doc is None:
            return cls()
        spec = cls(may_assign_to=tuple(doc.str_list("may_assign_to")))
        doc.reject_unknown()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {"may_assign_to": list(self.may_assign_to)} if self.may_assign_to else {}


@dataclass(frozen=True)
class ExtensionsSpec:
    """MCP servers and runtime plugins this agent is granted.

    Both name things the runtime already ships. ``mcp`` entries are ids from the runtime's
    curated catalogue and nothing else — a server is a command or a URL that the agent's
    data flows through, and accepting an arbitrary one from a control plane would be remote
    code execution with a form around it. ``plugins.enable`` and ``plugins.disable`` are
    keys from the runtime's own plugin registry, with disable winning, because that is the
    precedence the runtime's gate already applies.

    Validated against the discovered catalogue at load, so a typo names a missing server at
    configuration time rather than as a tool that silently never appears.
    """

    mcp: tuple[str, ...] = ()
    plugins_enable: tuple[str, ...] = ()
    plugins_disable: tuple[str, ...] = ()

    @classmethod
    def parse(cls, doc: Optional[Doc]) -> "ExtensionsSpec":
        if doc is None:
            return cls()
        plugins = doc.child("plugins")
        spec = cls(
            mcp=tuple(doc.str_list("mcp")),
            plugins_enable=tuple(plugins.str_list("enable")) if plugins else (),
            plugins_disable=tuple(plugins.str_list("disable")) if plugins else (),
        )
        if plugins is not None:
            plugins.reject_unknown()
        doc.reject_unknown()
        overlap = sorted(set(spec.plugins_enable) & set(spec.plugins_disable))
        if overlap:
            raise SpecError(
                f"{', '.join(overlap)} appear(s) in both enable and disable — disable wins, "
                "so remove the enable entry to make the intent explicit",
                field="extensions.plugins",
                source=doc.source,
            )
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.mcp:
            out["mcp"] = list(self.mcp)
        plugins = {}
        if self.plugins_enable:
            plugins["enable"] = list(self.plugins_enable)
        if self.plugins_disable:
            plugins["disable"] = list(self.plugins_disable)
        if plugins:
            out["plugins"] = plugins
        return out


@dataclass(frozen=True)
class AgentSpec:
    """One agent, declared.

    ``instructions_path`` is a bundle-relative path to the agent's persona; the loader
    resolves and reads it so the spec carries the text, not a path the runtime adapter
    would have to resolve for itself.
    """

    id: str
    name: str
    role: str = ""
    description: str = ""
    enabled: bool = True
    instructions: str = ""
    instructions_path: Optional[str] = None
    model: ModelSpec = field(default_factory=ModelSpec)
    tools: ToolsSpec = field(default_factory=ToolsSpec)
    knowledge: KnowledgeSpec = field(default_factory=KnowledgeSpec)
    permissions: tuple[str, ...] = ()
    approval: ApprovalSpec = field(default_factory=ApprovalSpec)
    limits: LimitsSpec = field(default_factory=LimitsSpec)
    delegation: DelegationSpec = field(default_factory=DelegationSpec)
    extensions: ExtensionsSpec = field(default_factory=ExtensionsSpec)
    source: Optional[Path] = None

    @classmethod
    def parse(
        cls,
        data: Mapping[str, Any],
        *,
        source: Optional[Path] = None,
        base_dir: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "AgentSpec":
        doc = Doc(data, source=source, env=env)
        agent_id = doc.identifier("id")
        instructions_path = doc.str_("instructions")
        inline = doc.str_("instructions_text", allow_empty=True, expand=False)

        if instructions_path and inline:
            raise SpecError(
                "set either instructions (a file) or instructions_text (inline), not both",
                field="instructions",
                source=source,
            )

        instructions = inline
        if instructions_path:
            instructions = cls._read_instructions(instructions_path, base_dir, source)

        spec = cls(
            id=agent_id,
            name=doc.str_("name", default=agent_id),
            role=doc.str_("role"),
            description=doc.str_("description"),
            enabled=doc.bool_("enabled", default=True),
            instructions=instructions,
            instructions_path=instructions_path or None,
            model=ModelSpec.parse(doc.child("model")),
            tools=ToolsSpec.parse(doc.child("tools")),
            knowledge=KnowledgeSpec.parse(doc.child("knowledge")),
            permissions=tuple(doc.str_list("permissions")),
            approval=ApprovalSpec.parse(doc.child("approval")),
            limits=LimitsSpec.parse(doc.child("limits")),
            delegation=DelegationSpec.parse(doc.child("delegation")),
            extensions=ExtensionsSpec.parse(doc.child("extensions")),
            source=source,
        )
        doc.reject_unknown()
        if spec.id in spec.delegation.may_assign_to:
            raise SpecError(
                f"{spec.id!r} may not delegate to itself",
                field="delegation.may_assign_to",
                source=source,
            )
        return spec

    @staticmethod
    def _read_instructions(
        rel_path: str, base_dir: Optional[Path], source: Optional[Path]
    ) -> str:
        if base_dir is None:
            raise SpecError(
                "instructions names a file but this spec was parsed without a bundle "
                "directory to resolve it against",
                field="instructions",
                source=source,
            )
        candidate = Path(rel_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SpecError(
                "must be a path inside the bundle (no absolute paths, no '..')",
                field="instructions",
                source=source,
            )
        resolved = (base_dir / candidate).resolve()
        if not str(resolved).startswith(str(base_dir.resolve())):
            raise SpecError("resolves outside the bundle", field="instructions", source=source)
        if not resolved.is_file():
            raise SpecError(
                f"instructions file not found: {rel_path}", field="instructions", source=source
            )
        return resolved.read_text(encoding="utf-8").strip()

    def to_dict(self) -> dict[str, Any]:
        """Canonical dictionary form. Used for digests and for the Control API later."""
        out: dict[str, Any] = {"id": self.id, "name": self.name, "enabled": self.enabled}
        for key, value in (
            ("role", self.role),
            ("description", self.description),
            ("instructions", self.instructions),
        ):
            if value:
                out[key] = value
        for key, section in (
            ("model", self.model),
            ("tools", self.tools),
            ("knowledge", self.knowledge),
            ("approval", self.approval),
            ("limits", self.limits),
            ("delegation", self.delegation),
            ("extensions", self.extensions),
        ):
            rendered = section.to_dict()
            if rendered:
                out[key] = rendered
        if self.permissions:
            out["permissions"] = list(self.permissions)
        return out

    def digest(self) -> str:
        """Stable content hash of the agent's meaning.

        Deliberately excludes ``source``: the same agent declared from a different path
        is the same agent. Used to detect drift between a spec and what a runtime
        currently has materialized, and recorded in the audit log.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
