"""
AG-UI Protocol types for Hermes.

Implements the Agent-User Interaction Protocol event model:
https://github.com/ag-ui-protocol/ag-ui

Maps Hermes run lifecycle events to AG-UI wire format so any
AG-UI-compatible frontend (CopilotKit, custom React UIs, mobile apps)
can connect to Hermes without a custom integration.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        alias_generator=to_camel,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class AGUIEventType(str, Enum):
    RUN_STARTED              = "RUN_STARTED"
    RUN_FINISHED             = "RUN_FINISHED"
    RUN_ERROR                = "RUN_ERROR"
    STEP_STARTED             = "STEP_STARTED"
    STEP_FINISHED            = "STEP_FINISHED"
    TEXT_MESSAGE_START       = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT     = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END         = "TEXT_MESSAGE_END"
    TOOL_CALL_START          = "TOOL_CALL_START"
    TOOL_CALL_ARGS           = "TOOL_CALL_ARGS"
    TOOL_CALL_END            = "TOOL_CALL_END"
    TOOL_CALL_RESULT         = "TOOL_CALL_RESULT"
    STATE_SNAPSHOT           = "STATE_SNAPSHOT"
    STATE_DELTA              = "STATE_DELTA"
    CUSTOM                   = "CUSTOM"


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

class AGUIBaseEvent(_CamelModel):
    type: AGUIEventType
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

class AGUIRunStartedEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.RUN_STARTED] = AGUIEventType.RUN_STARTED
    thread_id: str
    run_id: str


class AGUIRunFinishedEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.RUN_FINISHED] = AGUIEventType.RUN_FINISHED
    thread_id: str
    run_id: str


class AGUIRunErrorEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.RUN_ERROR] = AGUIEventType.RUN_ERROR
    message: str
    code: Optional[str] = None


class AGUIStepStartedEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.STEP_STARTED] = AGUIEventType.STEP_STARTED
    step_name: str


class AGUIStepFinishedEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.STEP_FINISHED] = AGUIEventType.STEP_FINISHED
    step_name: str


# ---------------------------------------------------------------------------
# Text messages
# ---------------------------------------------------------------------------

class AGUITextMessageStartEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.TEXT_MESSAGE_START] = AGUIEventType.TEXT_MESSAGE_START
    message_id: str
    role: str = "assistant"


class AGUITextMessageContentEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.TEXT_MESSAGE_CONTENT] = AGUIEventType.TEXT_MESSAGE_CONTENT
    message_id: str
    delta: str


class AGUITextMessageEndEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.TEXT_MESSAGE_END] = AGUIEventType.TEXT_MESSAGE_END
    message_id: str


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

class AGUIToolCallStartEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.TOOL_CALL_START] = AGUIEventType.TOOL_CALL_START
    tool_call_id: str
    tool_call_name: str
    parent_message_id: Optional[str] = None


class AGUIToolCallArgsEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.TOOL_CALL_ARGS] = AGUIEventType.TOOL_CALL_ARGS
    tool_call_id: str
    delta: str


class AGUIToolCallEndEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.TOOL_CALL_END] = AGUIEventType.TOOL_CALL_END
    tool_call_id: str


class AGUIToolCallResultEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.TOOL_CALL_RESULT] = AGUIEventType.TOOL_CALL_RESULT
    message_id: str
    tool_call_id: str
    content: str
    role: Literal["tool"] = "tool"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AGUIStateSnapshotEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.STATE_SNAPSHOT] = AGUIEventType.STATE_SNAPSHOT
    snapshot: Dict[str, Any]


class AGUIStateDeltaEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.STATE_DELTA] = AGUIEventType.STATE_DELTA
    delta: List[Any]  # JSON Patch RFC 6902


# ---------------------------------------------------------------------------
# Custom
# ---------------------------------------------------------------------------

class AGUICustomEvent(AGUIBaseEvent):
    type: Literal[AGUIEventType.CUSTOM] = AGUIEventType.CUSTOM
    name: str
    value: Any = None


# ---------------------------------------------------------------------------
# Input model (what the frontend sends to start a run)
# ---------------------------------------------------------------------------

class AGUIMessage(_CamelModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    role: str
    content: Optional[str] = None


class AGUIRunAgentInput(_CamelModel):
    thread_id: str
    run_id: str
    messages: List[AGUIMessage] = Field(default_factory=list)
    state: Optional[Dict[str, Any]] = None
    tools: List[Any] = Field(default_factory=list)
    context: List[Any] = Field(default_factory=list)
    forwarded_props: Optional[Dict[str, Any]] = None
