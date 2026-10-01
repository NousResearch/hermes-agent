from common.v1 import event_pb2 as _event_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CalendarSyncRequest(_message.Message):
    __slots__ = ("header", "user_id", "year", "week_number", "force_full_resync", "date_range_start", "date_range_end")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    YEAR_FIELD_NUMBER: _ClassVar[int]
    WEEK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    FORCE_FULL_RESYNC_FIELD_NUMBER: _ClassVar[int]
    DATE_RANGE_START_FIELD_NUMBER: _ClassVar[int]
    DATE_RANGE_END_FIELD_NUMBER: _ClassVar[int]
    header: _event_pb2.EventHeader
    user_id: str
    year: int
    week_number: int
    force_full_resync: bool
    date_range_start: str
    date_range_end: str
    def __init__(self, header: _Optional[_Union[_event_pb2.EventHeader, _Mapping]] = ..., user_id: _Optional[str] = ..., year: _Optional[int] = ..., week_number: _Optional[int] = ..., force_full_resync: _Optional[bool] = ..., date_range_start: _Optional[str] = ..., date_range_end: _Optional[str] = ...) -> None: ...

class CourseEvent(_message.Message):
    __slots__ = ("external_id", "deterministic_id", "title", "description", "location", "start_time_ms", "end_time_ms", "is_exam")
    EXTERNAL_ID_FIELD_NUMBER: _ClassVar[int]
    DETERMINISTIC_ID_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    START_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    END_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    IS_EXAM_FIELD_NUMBER: _ClassVar[int]
    external_id: str
    deterministic_id: str
    title: str
    description: str
    location: str
    start_time_ms: int
    end_time_ms: int
    is_exam: bool
    def __init__(self, external_id: _Optional[str] = ..., deterministic_id: _Optional[str] = ..., title: _Optional[str] = ..., description: _Optional[str] = ..., location: _Optional[str] = ..., start_time_ms: _Optional[int] = ..., end_time_ms: _Optional[int] = ..., is_exam: _Optional[bool] = ...) -> None: ...

class CalendarSyncResult(_message.Message):
    __slots__ = ("header", "total_events", "inserted_events", "updated_events", "upcoming_exams", "conflict_severity")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    TOTAL_EVENTS_FIELD_NUMBER: _ClassVar[int]
    INSERTED_EVENTS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_EVENTS_FIELD_NUMBER: _ClassVar[int]
    UPCOMING_EXAMS_FIELD_NUMBER: _ClassVar[int]
    CONFLICT_SEVERITY_FIELD_NUMBER: _ClassVar[int]
    header: _event_pb2.EventHeader
    total_events: int
    inserted_events: int
    updated_events: int
    upcoming_exams: _containers.RepeatedCompositeFieldContainer[CourseEvent]
    conflict_severity: str
    def __init__(self, header: _Optional[_Union[_event_pb2.EventHeader, _Mapping]] = ..., total_events: _Optional[int] = ..., inserted_events: _Optional[int] = ..., updated_events: _Optional[int] = ..., upcoming_exams: _Optional[_Iterable[_Union[CourseEvent, _Mapping]]] = ..., conflict_severity: _Optional[str] = ...) -> None: ...
