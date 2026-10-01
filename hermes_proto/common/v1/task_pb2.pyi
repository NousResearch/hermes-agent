from common.v1 import event_pb2 as _event_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskRequest(_message.Message):
    __slots__ = ("header", "task_type", "payload_json")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    TASK_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    header: _event_pb2.EventHeader
    task_type: str
    payload_json: str
    def __init__(self, header: _Optional[_Union[_event_pb2.EventHeader, _Mapping]] = ..., task_type: _Optional[str] = ..., payload_json: _Optional[str] = ...) -> None: ...

class TaskResponse(_message.Message):
    __slots__ = ("header", "success", "result_json", "error_message")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    header: _event_pb2.EventHeader
    success: bool
    result_json: str
    error_message: str
    def __init__(self, header: _Optional[_Union[_event_pb2.EventHeader, _Mapping]] = ..., success: _Optional[bool] = ..., result_json: _Optional[str] = ..., error_message: _Optional[str] = ...) -> None: ...

class HITLChallenge(_message.Message):
    __slots__ = ("header", "short_task_id", "title", "description", "screenshot_path", "timeout_seconds", "actions")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    SHORT_TASK_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SCREENSHOT_PATH_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    header: _event_pb2.EventHeader
    short_task_id: str
    title: str
    description: str
    screenshot_path: str
    timeout_seconds: int
    actions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, header: _Optional[_Union[_event_pb2.EventHeader, _Mapping]] = ..., short_task_id: _Optional[str] = ..., title: _Optional[str] = ..., description: _Optional[str] = ..., screenshot_path: _Optional[str] = ..., timeout_seconds: _Optional[int] = ..., actions: _Optional[_Iterable[str]] = ...) -> None: ...
