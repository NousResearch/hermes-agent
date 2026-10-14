"""Versioned workspace I/O protocol; no caller-selected runtime or owner."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_FILE_BYTES = 262_144
MAX_OUTPUT_BYTES = 131_072
MAX_CHUNK_BYTES = 16_384
MAX_TIMEOUT_SECONDS = 300
OperationStatus = Literal["reserved", "running", "stopping", "completed", "failed", "cancelled", "interrupted"]


class WorkspaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(min_length=1, max_length=256)


class PathRequest(WorkspaceRequest):
    path: str = Field(min_length=1, max_length=4096)


class ReadRequest(PathRequest):
    max_bytes: int = Field(default=MAX_FILE_BYTES, ge=1, le=MAX_FILE_BYTES)


class WriteRequest(PathRequest):
    data_base64: str = Field(max_length=4 * ((MAX_FILE_BYTES + 2) // 3))


class RemoveRequest(PathRequest):
    recursive: bool = False
    force: bool = False


class StartRequest(WorkspaceRequest):
    command: str = Field(min_length=1, max_length=32_768)
    cwd: str = Field(default=".", min_length=1, max_length=4096)
    env: dict[str, str] = Field(default_factory=dict, max_length=64)
    timeout_seconds: int = Field(default=60, ge=1, le=MAX_TIMEOUT_SECONDS)
    max_output_bytes: int = Field(default=MAX_OUTPUT_BYTES, ge=1, le=MAX_OUTPUT_BYTES)

    @field_validator("env")
    @classmethod
    def child_environment(cls, values):
        for key, value in values.items():
            if not key or "=" in key or "\x00" in key + value or len(key) > 128 or len(value) > 8192:
                raise ValueError("Invalid child environment entry")
            if key in {"HOME", "HERMES_HOME", "HERMES_PROFILE"} or key.startswith(("TERMINAL_", "HERMES_")):
                raise ValueError("Runtime selection variables are not child overrides")
        if sum(len(k) + len(v) for k, v in values.items()) > 32_768:
            raise ValueError("Child environment is too large")
        return values


class OutputRequest(WorkspaceRequest):
    stdout_cursor: int = Field(default=0, ge=0, le=MAX_OUTPUT_BYTES)
    stderr_cursor: int = Field(default=0, ge=0, le=MAX_OUTPUT_BYTES)
    max_bytes: int = Field(default=MAX_CHUNK_BYTES, ge=1, le=MAX_CHUNK_BYTES)


class OperationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str
    session_id: str
    kind: Literal["write", "remove", "process"]
    status: OperationStatus
    process_id: str = ""
    exit_code: int | None = None
    reason: str = ""
    termination_confirmed: bool = True
    acknowledged: bool = False
    output_available: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class StreamChunk(BaseModel):
    data_base64: str = Field(max_length=4 * ((MAX_CHUNK_BYTES + 2) // 3))
    next_cursor: int = Field(ge=0, le=MAX_OUTPUT_BYTES)
    truncated: bool


class OutputResponse(BaseModel):
    operation_id: str
    status: OperationStatus
    complete: bool
    partial: bool
    stdout: StreamChunk
    stderr: StreamChunk


class FileData(BaseModel):
    kind: Literal["file"]
    data_base64: str
    size_bytes: int


class MissingFile(BaseModel):
    kind: Literal["missing"]


class ResolvedPath(BaseModel):
    path: str
