from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TaskStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TASK_STATUS_UNSPECIFIED: _ClassVar[TaskStatus]
    TASK_STATUS_PENDING: _ClassVar[TaskStatus]
    TASK_STATUS_RUNNING: _ClassVar[TaskStatus]
    TASK_STATUS_WAITING_USER_APPROVAL: _ClassVar[TaskStatus]
    TASK_STATUS_APPROVED: _ClassVar[TaskStatus]
    TASK_STATUS_REJECTED: _ClassVar[TaskStatus]
    TASK_STATUS_SUCCESS: _ClassVar[TaskStatus]
    TASK_STATUS_FAILED: _ClassVar[TaskStatus]
    TASK_STATUS_EXPIRED: _ClassVar[TaskStatus]
    TASK_STATUS_RETRYING: _ClassVar[TaskStatus]
TASK_STATUS_UNSPECIFIED: TaskStatus
TASK_STATUS_PENDING: TaskStatus
TASK_STATUS_RUNNING: TaskStatus
TASK_STATUS_WAITING_USER_APPROVAL: TaskStatus
TASK_STATUS_APPROVED: TaskStatus
TASK_STATUS_REJECTED: TaskStatus
TASK_STATUS_SUCCESS: TaskStatus
TASK_STATUS_FAILED: TaskStatus
TASK_STATUS_EXPIRED: TaskStatus
TASK_STATUS_RETRYING: TaskStatus

class EventHeader(_message.Message):
    __slots__ = ("event_id", "trace_id", "idempotency_key", "origin_service", "timestamp", "status", "retry_count", "target_worker")
    EVENT_ID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    IDEMPOTENCY_KEY_FIELD_NUMBER: _ClassVar[int]
    ORIGIN_SERVICE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    RETRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    TARGET_WORKER_FIELD_NUMBER: _ClassVar[int]
    event_id: str
    trace_id: str
    idempotency_key: str
    origin_service: str
    timestamp: int
    status: TaskStatus
    retry_count: int
    target_worker: str
    def __init__(self, event_id: _Optional[str] = ..., trace_id: _Optional[str] = ..., idempotency_key: _Optional[str] = ..., origin_service: _Optional[str] = ..., timestamp: _Optional[int] = ..., status: _Optional[_Union[TaskStatus, str]] = ..., retry_count: _Optional[int] = ..., target_worker: _Optional[str] = ...) -> None: ...
