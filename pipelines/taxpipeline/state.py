from typing import Any, TypedDict

from .models import ClarificationRequest, DepreciationSuggestion, FieldMapping, ValidationIssue


class TaxPipelineState(TypedDict, total=False):
    tax_year: int
    form_id: str
    form_version: str
    form_schema_version: str
    validation_rules_version: str
    afa_table_version: str
    coco_chunks: list[dict[str, Any]]
    prior_year_tax_state: dict[str, Any]

    raw_extracted_data: dict[str, Any]

    field_mapping_candidates: dict[str, list[FieldMapping]]
    field_mappings: dict[str, FieldMapping]
    validation_issues: list[ValidationIssue]
    clarifications: list[ClarificationRequest]
    depreciation_suggestions: list[DepreciationSuggestion]
    requires_clarification: bool
