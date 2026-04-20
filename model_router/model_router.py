from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml


class TaskType(str, Enum):
    CHAT = "chat"
    CODING = "coding"
    WRITING = "writing"
    RESEARCH = "research"
    BATCH = "batch"
    TRIVIAL = "trivial"


class Mode(str, Enum):
    DRAFT = "draft"
    EXECUTE = "execute"
    REVIEW = "review"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Privacy(str, Enum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"
    LOCAL_ONLY = "local_only"


class Quota(str, Enum):
    NORMAL = "normal"
    LOW = "low"
    CRITICAL = "critical"


class Speed(str, Enum):
    NORMAL = "normal"
    FAST = "fast"


class Model(str, Enum):
    CLAUDE = "claude-sonnet-4.6"
    GPT = "gpt-5.4"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    FLASH = "flash_or_o4_mini"


@dataclass
class RouterInput:
    task_type: TaskType
    mode: Mode = Mode.DRAFT
    priority: Priority = Priority.MEDIUM
    privacy: Privacy = Privacy.NORMAL
    quota: Quota = Quota.NORMAL
    speed: Speed = Speed.NORMAL
    has_code: bool = False
    has_logs: bool = False


@dataclass
class RouterDecision:
    primary_model: Model
    fallback_models: list[Model]
    reviewer: Optional[Model]
    reason: str
    trace: list[str] = field(default_factory=list)


@dataclass
class RouterConfig:
    default_model: Model
    base_by_task: dict[TaskType, Model]
    fallbacks: dict[Model, list[Model]]
    mode_overrides: dict[Mode, dict[TaskType, Model]]
    reviewers: dict[Priority, dict[TaskType, Model]]
    policy_overrides: list[dict[str, Any]]
    router_version: str = "0.3"
    config_path: str = ""


def _to_model(value: str) -> Model:
    return Model(value)


def _to_task_type(value: str) -> TaskType:
    return TaskType(value)


def _to_mode(value: str) -> Mode:
    return Mode(value)


def _to_priority(value: str) -> Priority:
    return Priority(value)


def load_config(path: str | Path) -> RouterConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    router = raw["router"]

    base_by_task = {
        _to_task_type(key): _to_model(value)
        for key, value in raw["base_by_task"].items()
    }
    fallbacks = {
        _to_model(key): [_to_model(item) for item in value]
        for key, value in raw["fallbacks"].items()
    }
    mode_overrides = {
        _to_mode(mode): {_to_task_type(key): _to_model(value) for key, value in mapping.items()}
        for mode, mapping in raw.get("mode_overrides", {}).items()
    }
    reviewers = {
        _to_priority(priority): {_to_task_type(key): _to_model(value) for key, value in mapping.items()}
        for priority, mapping in raw.get("reviewers", {}).items()
    }

    return RouterConfig(
        default_model=_to_model(router["default_model"]),
        base_by_task=base_by_task,
        fallbacks=fallbacks,
        mode_overrides=mode_overrides,
        reviewers=reviewers,
        policy_overrides=raw.get("policy_overrides", []) or [],
        router_version=router.get("version", "0.3"),
        config_path=str(Path(path)),
    )


def load_default_config() -> RouterConfig:
    return load_config(Path(__file__).resolve().parent / "router_config.yaml")


def normalize(ctx: RouterInput, trace: list[str]) -> RouterInput:
    if ctx.has_code or ctx.has_logs:
        ctx.task_type = TaskType.CODING
        trace.append("normalize: has_code/has_logs -> coding")
    else:
        trace.append(f"normalize: keep task_type={ctx.task_type.value}")
    return ctx


def select_base_model(ctx: RouterInput, config: RouterConfig, trace: list[str]) -> Model:
    model = config.base_by_task.get(ctx.task_type, config.default_model)
    trace.append(f"base: task_type={ctx.task_type.value} -> {model.value}")
    return model


def apply_mode_override(model: Model, ctx: RouterInput, config: RouterConfig, trace: list[str]) -> Model:
    mode_map = config.mode_overrides.get(ctx.mode, {})
    if ctx.task_type in mode_map:
        overridden = mode_map[ctx.task_type]
        trace.append(f"mode_override: {ctx.mode.value}+{ctx.task_type.value} -> {overridden.value}")
        return overridden

    trace.append(f"mode_override: none -> {model.value}")
    return model


def _ctx_value_for_match(ctx: RouterInput, field_name: str) -> str | bool | None:
    value = getattr(ctx, field_name, None)
    if value is None:
        return None
    if hasattr(value, "value"):
        return value.value
    return value


def _override_matches(ctx: RouterInput, override: dict[str, Any]) -> bool:
    when = override.get("when", {}) or {}
    if not when:
        return False

    for field_name, expected in when.items():
        actual = _ctx_value_for_match(ctx, field_name)
        if actual != expected:
            return False

    return True


def apply_policy_overrides(model: Model, ctx: RouterInput, config: RouterConfig, trace: list[str]) -> Model:
    for override in config.policy_overrides:
        if not _override_matches(ctx, override):
            continue

        forced = override.get("force")
        if forced:
            forced_model = Model(forced)
            name = override.get("name", "unnamed")
            reason = override.get("reason", "")
            trace.append(
                f"policy_override: {name} -> {forced_model.value}" + (f" ({reason})" if reason else "")
            )
            return forced_model

    trace.append(f"policy_override: none -> {model.value}")
    return model


def apply_hard_safety_overrides(model: Model, ctx: RouterInput, trace: list[str]) -> Model:
    if ctx.privacy == Privacy.LOCAL_ONLY:
        trace.append(f"override: privacy=local_only -> {Model.OLLAMA.value}")
        return Model.OLLAMA

    if ctx.privacy == Privacy.SENSITIVE and ctx.task_type == TaskType.BATCH:
        trace.append(f"override: privacy=sensitive + batch -> {Model.OLLAMA.value}")
        return Model.OLLAMA

    trace.append(f"override_safety: none -> {model.value}")
    return model


def apply_soft_routing_overrides(model: Model, ctx: RouterInput, trace: list[str]) -> Model:
    if ctx.quota == Quota.LOW and ctx.priority == Priority.LOW and model in (Model.CLAUDE, Model.GPT):
        trace.append(f"override: quota=low + priority=low -> {Model.DEEPSEEK.value}")
        return Model.DEEPSEEK

    if ctx.quota == Quota.CRITICAL and ctx.priority == Priority.LOW:
        trace.append(f"override: quota=critical + priority=low -> {Model.DEEPSEEK.value}")
        return Model.DEEPSEEK

    if ctx.speed == Speed.FAST and ctx.task_type == TaskType.TRIVIAL and ctx.priority == Priority.LOW:
        trace.append(f"override: fast + trivial + low -> {Model.FLASH.value}")
        return Model.FLASH

    trace.append(f"override_soft: none -> {model.value}")
    return model


def enforce_constraints(model: Model, ctx: RouterInput, trace: list[str]) -> Model:
    if ctx.priority == Priority.HIGH and model in (Model.FLASH, Model.DEEPSEEK):
        if ctx.task_type == TaskType.CODING:
            upgraded = Model.GPT
        elif ctx.task_type == TaskType.BATCH and ctx.privacy in (Privacy.SENSITIVE, Privacy.LOCAL_ONLY):
            upgraded = Model.OLLAMA
        else:
            upgraded = Model.CLAUDE
        trace.append(f"constraint: high priority forbids cheap primary -> {upgraded.value}")
        return upgraded

    trace.append(f"constraint: none -> {model.value}")
    return model


def select_reviewer(ctx: RouterInput, config: RouterConfig, trace: list[str]) -> Optional[Model]:
    priority_map = config.reviewers.get(ctx.priority, {})
    reviewer = priority_map.get(ctx.task_type)
    if reviewer:
        trace.append(f"reviewer: {ctx.priority.value}+{ctx.task_type.value} -> {reviewer.value}")
        return reviewer

    trace.append("reviewer: none")
    return None


def build_reason(ctx: RouterInput, model: Model, trace: list[str]) -> str:
    return (
        f"task_type={ctx.task_type.value}, mode={ctx.mode.value}, priority={ctx.priority.value}, "
        f"privacy={ctx.privacy.value}, quota={ctx.quota.value} | selected={model.value} | "
        f"trace={' ; '.join(trace)}"
    )


def route_model(ctx: RouterInput, config: RouterConfig) -> RouterDecision:
    trace: list[str] = []

    ctx = normalize(ctx, trace)
    model = select_base_model(ctx, config, trace)
    model = apply_mode_override(model, ctx, config, trace)
    model = apply_hard_safety_overrides(model, ctx, trace)
    model = apply_soft_routing_overrides(model, ctx, trace)
    model = apply_policy_overrides(model, ctx, config, trace)
    model = apply_hard_safety_overrides(model, ctx, trace)
    model = enforce_constraints(model, ctx, trace)

    reviewer = select_reviewer(ctx, config, trace)
    fallbacks = config.fallbacks[model]
    reason = build_reason(ctx, model, trace)

    return RouterDecision(
        primary_model=model,
        fallback_models=fallbacks,
        reviewer=reviewer,
        reason=reason,
        trace=trace,
    )
