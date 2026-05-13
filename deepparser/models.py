from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    filename: str
    page: int | None = None
    cell: str | None = None


class ParseResult(BaseModel):
    file_id: str
    folder_id: str | None = None
    file_name: str
    extension: str
    pages: int | None = None
    tables: int | None = None


class ParseJob(BaseModel):
    job_id: str
    status: str  # QUEUED | PARSING | READY | PARSE_FAILED | TIMEOUT
    created_at: str
    completed_at: str | None = None
    result: ParseResult | None = None
    error_detail: str | None = None


class AskResult(BaseModel):
    job_id: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class KeyInfo(BaseModel):
    api_key: str
    created_at: str
