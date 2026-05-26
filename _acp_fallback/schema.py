"""Small ACP schema fallback used when agent-client-protocol is not installed."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Literal

ToolKind = Literal["read", "edit", "search", "execute", "fetch", "think", "other"]


def _camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


class ACPModel:
    def __init__(self, **kwargs: Any) -> None:
        model_fields = fields(self) if is_dataclass(self) else ()
        for f in model_fields:
            if f.default_factory is not MISSING:  # type: ignore[attr-defined]
                setattr(self, f.name, f.default_factory())  # type: ignore[misc]
            elif f.default is not MISSING:
                setattr(self, f.name, f.default)
            else:
                setattr(self, f.name, None)
        aliases = {_camel(f.name): f.name for f in model_fields}
        for key, value in list(kwargs.items()):
            setattr(self, aliases.get(key, key), value)

    def __getattr__(self, name: str) -> Any:
        if is_dataclass(self):
            for f in fields(self):
                if _camel(f.name) == name:
                    return getattr(self, f.name)
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    def model_dump(self, *, by_alias: bool = False, exclude_none: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if exclude_none and value is None:
                continue
            key = _camel(f.name) if by_alias else f.name
            result[key] = _dump_value(value, by_alias=by_alias, exclude_none=exclude_none)
        return result


def _dump_value(value: Any, *, by_alias: bool, exclude_none: bool) -> Any:
    if isinstance(value, ACPModel):
        return value.model_dump(by_alias=by_alias, exclude_none=exclude_none)
    if isinstance(value, list):
        return [_dump_value(v, by_alias=by_alias, exclude_none=exclude_none) for v in value]
    if isinstance(value, dict):
        return {
            k: _dump_value(v, by_alias=by_alias, exclude_none=exclude_none)
            for k, v in value.items()
            if not (exclude_none and v is None)
        }
    return value


@dataclass(init=False)
class TextContentBlock(ACPModel):
    type: str = "text"
    text: str = ""


@dataclass(init=False)
class ImageContentBlock(ACPModel):
    type: str = "image"
    data: str | None = None
    uri: str | None = None
    mime_type: str = "image/png"


@dataclass(init=False)
class AudioContentBlock(ACPModel):
    type: str = "audio"
    data: str | None = None
    uri: str | None = None
    mime_type: str = "audio/wav"


@dataclass(init=False)
class ResourceContentBlock(ACPModel):
    type: str = "resource"
    uri: str | None = None
    name: str | None = None
    title: str | None = None
    mime_type: str | None = None
    text: str | None = None


@dataclass(init=False)
class EmbeddedResourceContentBlock(ACPModel):
    type: str = "resource"
    resource: Any = None
    text: str | None = None


@dataclass(init=False)
class TextResourceContents(ACPModel):
    uri: str = ""
    mime_type: str = "text/plain"
    text: str = ""


@dataclass(init=False)
class BlobResourceContents(ACPModel):
    uri: str = ""
    mime_type: str = "application/octet-stream"
    blob: str = ""


@dataclass(init=False)
class ContentToolCallContent(ACPModel):
    type: str = "content"
    content: TextContentBlock | Any = None


@dataclass(init=False)
class FileEditToolCallContent(ACPModel):
    type: str = "diff"
    path: str = ""
    old_text: str | None = None
    new_text: str | None = None


@dataclass(init=False)
class ToolCallLocation(ACPModel):
    path: str = ""
    line: int | None = None


@dataclass(init=False)
class ToolCallStart(ACPModel):
    session_update: str = "tool_call"
    tool_call_id: str = ""
    title: str = ""
    kind: str = "other"
    content: list[Any] | None = None
    locations: list[ToolCallLocation] | None = None
    raw_input: Any = None


@dataclass(init=False)
class ToolCallProgress(ACPModel):
    session_update: str = "tool_call_update"
    tool_call_id: str = ""
    title: str | None = None
    kind: str = "other"
    status: str = "completed"
    content: list[Any] | None = None
    raw_input: Any = None
    raw_output: Any = None


@dataclass(init=False)
class AgentThoughtChunk(ACPModel):
    session_update: str = "agent_thought_chunk"
    content: TextContentBlock | Any = None


@dataclass(init=False)
class AgentMessageChunk(ACPModel):
    session_update: str = "agent_message_chunk"
    content: TextContentBlock | Any = None


@dataclass(init=False)
class UserMessageChunk(ACPModel):
    session_update: str = "user_message_chunk"
    content: TextContentBlock | Any = None


@dataclass(init=False)
class PlanEntry(ACPModel):
    content: str = ""
    priority: str = "medium"
    status: str = "pending"


@dataclass(init=False)
class AgentPlanUpdate(ACPModel):
    session_update: str = "plan"
    entries: list[PlanEntry] = field(default_factory=list)


@dataclass(init=False)
class Implementation(ACPModel):
    name: str = ""
    version: str = ""


@dataclass(init=False)
class PromptCapabilities(ACPModel):
    image: bool = False


@dataclass(init=False)
class SessionForkCapabilities(ACPModel):
    enabled: bool = True


@dataclass(init=False)
class SessionListCapabilities(ACPModel):
    enabled: bool = True


@dataclass(init=False)
class SessionResumeCapabilities(ACPModel):
    enabled: bool = True


@dataclass(init=False)
class SessionCapabilities(ACPModel):
    fork: SessionForkCapabilities | None = None
    list: SessionListCapabilities | None = None
    resume: SessionResumeCapabilities | None = None


@dataclass(init=False)
class AgentCapabilities(ACPModel):
    load_session: bool = False
    prompt_capabilities: PromptCapabilities | None = None
    session_capabilities: SessionCapabilities | None = None


@dataclass(init=False)
class AuthMethodAgent(ACPModel):
    id: str = ""
    name: str = ""
    description: str | None = None


@dataclass(init=False)
class TerminalAuthMethod(ACPModel):
    id: str = ""
    name: str = ""
    description: str | None = None
    type: str = "terminal"
    args: list[str] = field(default_factory=list)


AuthMethod = AuthMethodAgent


@dataclass(init=False)
class InitializeResponse(ACPModel):
    protocol_version: int = 1
    agent_info: Implementation | None = None
    agent_capabilities: AgentCapabilities | None = None
    auth_methods: list[AuthMethodAgent] | None = None


@dataclass(init=False)
class AuthenticateResponse(ACPModel):
    ok: bool = True


@dataclass(init=False)
class ModelInfo(ACPModel):
    model_id: str = ""
    name: str | None = None
    description: str | None = None


@dataclass(init=False)
class SessionModelState(ACPModel):
    current_model_id: str = ""
    available_models: list[ModelInfo] = field(default_factory=list)


@dataclass(init=False)
class SessionMode(ACPModel):
    id: str = ""
    name: str = ""
    description: str | None = None


@dataclass(init=False)
class SessionModeState(ACPModel):
    current_mode_id: str = "default"
    available_modes: list[SessionMode] = field(default_factory=list)


@dataclass(init=False)
class NewSessionResponse(ACPModel):
    session_id: str = ""
    models: SessionModelState | None = None
    modes: SessionModeState | None = None
    config_options: list[Any] | None = None


@dataclass(init=False)
class LoadSessionResponse(ACPModel):
    models: SessionModelState | None = None
    modes: SessionModeState | None = None
    config_options: list[Any] | None = None


@dataclass(init=False)
class ResumeSessionResponse(ACPModel):
    models: SessionModelState | None = None
    modes: SessionModeState | None = None
    config_options: list[Any] | None = None


@dataclass(init=False)
class ForkSessionResponse(ACPModel):
    session_id: str = ""


@dataclass(init=False)
class SessionInfo(ACPModel):
    session_id: str = ""
    title: str | None = None
    updated_at: str | None = None


@dataclass(init=False)
class SessionInfoUpdate(ACPModel):
    session_update: str = "session_info"
    session_info: SessionInfo | None = None


@dataclass(init=False)
class ListSessionsResponse(ACPModel):
    sessions: list[SessionInfo] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass(init=False)
class Usage(ACPModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thought_tokens: int | None = None
    cached_read_tokens: int | None = None


@dataclass(init=False)
class PromptResponse(ACPModel):
    stop_reason: str = "end_turn"
    usage: Usage | None = None


@dataclass(init=False)
class UnstructuredCommandInput(ACPModel):
    hint: str | None = None

    @property
    def root(self) -> "UnstructuredCommandInput":
        return self


@dataclass(init=False)
class AvailableCommand(ACPModel):
    name: str = ""
    description: str = ""
    input: UnstructuredCommandInput | None = None


@dataclass(init=False)
class AvailableCommandsUpdate(ACPModel):
    session_update: str = "available_commands_update"
    available_commands: list[AvailableCommand] = field(default_factory=list)


@dataclass(init=False)
class UsageUpdate(ACPModel):
    session_update: str = "usage_update"
    size: int = 0
    used: int = 0


@dataclass(init=False)
class SetSessionModeResponse(ACPModel):
    pass


@dataclass(init=False)
class SetSessionModelResponse(ACPModel):
    pass


@dataclass(init=False)
class SetSessionConfigOptionResponse(ACPModel):
    config_options: list[Any] = field(default_factory=list)


@dataclass(init=False)
class ClientCapabilities(ACPModel):
    pass


@dataclass(init=False)
class EnvVariable(ACPModel):
    name: str = ""
    value: str = ""


@dataclass(init=False)
class HttpHeader(ACPModel):
    name: str = ""
    value: str = ""


@dataclass(init=False)
class McpServerStdio(ACPModel):
    name: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: list[EnvVariable] = field(default_factory=list)


@dataclass(init=False)
class McpServerHttp(ACPModel):
    name: str = ""
    url: str = ""
    headers: list[HttpHeader] = field(default_factory=list)


@dataclass(init=False)
class McpServerSse(ACPModel):
    name: str = ""
    url: str = ""
    headers: list[HttpHeader] = field(default_factory=list)


@dataclass(init=False)
class PermissionOption(ACPModel):
    option_id: str = ""
    kind: str = ""
    name: str = ""


@dataclass(init=False)
class AllowedOutcome(ACPModel):
    option_id: str = ""
    outcome: str = "selected"


@dataclass(init=False)
class DeniedOutcome(ACPModel):
    outcome: str = "cancelled"


@dataclass(init=False)
class RequestPermissionResponse(ACPModel):
    outcome: AllowedOutcome | DeniedOutcome | Any = None
