from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class IssueType(str, Enum):
    DATA_UNCERTAINTY = "DATA_UNCERTAINTY"
    LEGAL_INTERPRETATION = "LEGAL_INTERPRETATION"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    MISSING_REFERENCE = "MISSING_REFERENCE"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Evidence(BaseModel):
    document_id: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    document_hash: str = Field(..., min_length=1)
    hash_algorithm: str = "SHA-256"
    extracted_text: str = Field(..., min_length=1, max_length=500)
    source_type: str | None = None
    invoice_number: str | None = None
    counterparty: str | None = None
    document_date: str | None = None
    service_date: str | None = None
    payment_date: str | None = None

    @field_validator("hash_algorithm")
    @classmethod
    def normalize_hash_algorithm(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"SHA-256", "SHA256"}:
            raise ValueError("hash_algorithm must be SHA-256")
        return "SHA-256"


class FieldMapping(BaseModel):
    value: Any
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    mapping_confidence: float = Field(..., ge=0.0, le=1.0)
    validation_confidence: float = Field(..., ge=0.0, le=1.0)
    field_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: Evidence
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_field_confidence(self) -> "FieldMapping":
        self.field_confidence = min(
            self.extraction_confidence,
            self.mapping_confidence,
            self.validation_confidence,
        )
        return self


class ClarificationRequest(BaseModel):
    field_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    missing_information: list[str] = Field(default_factory=list)
    issue_type: IssueType
    evidence_refs: list[str] = Field(default_factory=list)


class DepreciationSuggestion(BaseModel):
    asset_name: str = Field(..., min_length=1)
    amount: float = Field(..., ge=0.0)
    useful_life_years: int = Field(..., ge=1)
    afa_table_version: str = Field(..., min_length=1)
    evidence: list[Evidence] = Field(default_factory=list)


class ValidationIssue(BaseModel):
    issue_id: str = Field(..., min_length=1)
    issue_type: IssueType
    severity: Severity
    description: str = Field(..., min_length=1)
    field_id: str | None = None


class TaxPipelineOutput(BaseModel):
    tax_year: int
    form_id: str = Field(..., min_length=1)
    form_version: str = Field(..., min_length=1)
    form_schema_version: str | None = None
    validation_rules_version: str | None = None
    pipeline_version: str = Field(..., min_length=1)
    processing_id: str = Field(..., min_length=1)
    execution_timestamp: str = Field(..., min_length=1)
    field_mappings: dict[str, FieldMapping] = Field(default_factory=dict)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    depreciation_suggestions: list[DepreciationSuggestion] = Field(default_factory=list)
    clarifications: list[ClarificationRequest] = Field(default_factory=list)
    requires_clarification: bool = False

    @model_validator(mode="after")
    def set_requires_clarification(self) -> "TaxPipelineOutput":
        self.requires_clarification = self.requires_clarification or bool(self.clarifications)
        return self
