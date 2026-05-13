from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------- dp_cli subprocess result ----------

class Citation(BaseModel):
    filename: str
    page: int | None = None
    cell: str | None = None


class DPCLIParseResult(BaseModel):
    """Parsed from dp file upload --trigger-parse --wait-ready JSON output."""
    file_id: str
    folder_id: str
    file_name: str
    extension: str
    pages: int | None = None
    tables: int | None = None


class DPCLIAskResult(BaseModel):
    """Parsed from dp chat ask JSON output."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)


# ---------- API request / response ----------

class ParseJobStatus(BaseModel):
    job_id: str
    status: str  # QUEUED | PARSING | READY | PARSE_FAILED | TIMEOUT
    created_at: str
    completed_at: str | None = None
    result: dict[str, Any] | None = None  # DPCLIAskResult shape when READY
    error_detail: str | None = None


class ParseSubmitResponse(BaseModel):
    job_id: str
    status: str = "QUEUED"


class SyncParseResponse(BaseModel):
    """Returned inline when sync mode completes in time."""
    job_id: str
    status: str
    result: dict[str, Any] | None = None


class AskRequest(BaseModel):
    job_id: str
    question: str


class AskResponse(BaseModel):
    job_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class KeyRegistrationRequest(BaseModel):
    email: str
    intended_use: str | None = None


class KeyRegistrationResponse(BaseModel):
    api_key: str
    created_at: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class AdminStatsResponse(BaseModel):
    keys_registered: int
    keys_activated: int
    activation_rate: float
    requests_today: int
    parse_jobs: int
    storage_bytes: int
    errors_today: int


class ErrorResponse(BaseModel):
    code: str
    message: str
    detail: str | None = None
    doc_url: str = "https://github.com/ysh145/hermes-agent/tree/main/deepparser"
