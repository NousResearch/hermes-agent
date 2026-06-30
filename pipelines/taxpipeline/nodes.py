import hashlib
import json
import pathlib
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from json import JSONDecodeError
from typing import Any

from .models import ClarificationRequest, FieldMapping, IssueType, Severity, ValidationIssue
from .state import TaxPipelineState

CANONICAL_DOC_ROOT = pathlib.Path("/home/tobi/Dokumente/hermes-dokuments")
REF_SCHEMA_DIR = pathlib.Path("/home/tobi/.hermes/profiles/taxpipeline/reference/schemas")
REF_AFA_DIR = pathlib.Path("/home/tobi/.hermes/profiles/taxpipeline/reference/afa_tables")
SOURCE_PRIORITY = {
    "tax_assessment": 1,
    "invoice": 2,
    "receipt": 2,
    "bank_statement": 3,
    "transaction_system": 4,
    "memo": 5,
    "user_input": 6,
}
CENT_TOLERANCE = Decimal("0.01")


def _clarification(
    *,
    field_id: str,
    question: str,
    missing_information: list[str],
    issue_type: IssueType,
    evidence_refs: list[str] | None = None,
) -> ClarificationRequest:
    return ClarificationRequest(
        field_id=field_id,
        question=question,
        missing_information=missing_information,
        issue_type=issue_type,
        evidence_refs=evidence_refs or [],
    )


def _append_clarification(
    state: TaxPipelineState, request: ClarificationRequest
) -> dict[str, Any]:
    clarifications = [*state.get("clarifications", []), request]
    return {"clarifications": clarifications, "requires_clarification": True}


def _schema_path(form_id: str, form_version: str) -> pathlib.Path:
    return REF_SCHEMA_DIR / f"{form_id}_{form_version}.json"


def _load_json_reference(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Reference file must contain a JSON object: {path}")
    return data


def _is_canonical_document(document_id: str) -> bool:
    path = pathlib.Path(document_id)
    if not path.is_absolute():
        return True
    try:
        path.resolve().relative_to(CANONICAL_DOC_ROOT.resolve())
    except ValueError:
        return False
    return True


def source_rank(mapping: FieldMapping) -> int:
    source_type = mapping.evidence.source_type or mapping.metadata.get("source_type") or "user_input"
    return SOURCE_PRIORITY.get(str(source_type), 99)


def _numeric_value(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int | float):
        return Decimal(str(value))
    if isinstance(value, str):
        normalized = (
            value.strip()
            .replace("EUR", "")
            .replace("€", "")
            .replace(" ", "")
        )
        if "," in normalized and "." in normalized:
            normalized = normalized.replace(".", "").replace(",", ".")
        elif "," in normalized:
            normalized = normalized.replace(",", ".")
        try:
            return Decimal(normalized)
        except InvalidOperation:
            return None
    return None


def _date_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
    return None


def _mapping_date(mapping: FieldMapping, key: str) -> date | None:
    return _date_value(mapping.metadata.get(key) or getattr(mapping.evidence, key, None))


def _evidence_ref(mapping: FieldMapping) -> str:
    return mapping.evidence.document_id


def _validation_issue(
    *,
    issue_id: str,
    issue_type: IssueType,
    severity: Severity,
    description: str,
    field_id: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        issue_id=issue_id,
        issue_type=issue_type,
        severity=severity,
        description=description,
        field_id=field_id,
    )


def _add_validation_error(
    *,
    field_id: str,
    question: str,
    missing_information: list[str],
    evidence_refs: list[str],
    clarifications: list[ClarificationRequest],
    validation_issues: list[ValidationIssue],
    issue_id: str,
    issue_type: IssueType = IssueType.VALIDATION_ERROR,
    severity: Severity = Severity.ERROR,
) -> None:
    validation_issues.append(
        _validation_issue(
            issue_id=issue_id,
            issue_type=issue_type,
            severity=severity,
            description=question,
            field_id=field_id,
        )
    )
    clarifications.append(
        _clarification(
            field_id=field_id,
            question=question,
            missing_information=missing_information,
            issue_type=issue_type,
            evidence_refs=evidence_refs,
        )
    )


def _field_mapping_lookup(mappings: dict[str, FieldMapping]) -> dict[str, FieldMapping]:
    return {field_id.lower(): mapping for field_id, mapping in mappings.items()}


def _find_mapping(
    mappings: dict[str, FieldMapping],
    lookup: dict[str, FieldMapping],
    *candidates: str,
) -> FieldMapping | None:
    for candidate in candidates:
        direct = mappings.get(candidate)
        if direct is not None:
            return direct
        lowered = lookup.get(candidate.lower())
        if lowered is not None:
            return lowered
    return None


def _validate_amount_relationships(
    mappings: dict[str, FieldMapping],
    clarifications: list[ClarificationRequest],
    validation_issues: list[ValidationIssue],
) -> None:
    lookup = _field_mapping_lookup(mappings)
    gross = _find_mapping(mappings, lookup, "gross_amount", "gross", "brutto", "brutto_betrag")
    net = _find_mapping(mappings, lookup, "net_amount", "net", "netto", "netto_betrag")
    vat = _find_mapping(mappings, lookup, "vat_amount", "vat", "ust", "umsatzsteuer")
    acquisition = _find_mapping(mappings, lookup, "acquisition_cost", "anschaffungskosten")
    depreciation = _find_mapping(mappings, lookup, "depreciation_amount", "afa_amount", "afa")

    gross_value = _numeric_value(gross.value) if gross else None
    net_value = _numeric_value(net.value) if net else None
    vat_value = _numeric_value(vat.value) if vat else None
    acquisition_value = _numeric_value(acquisition.value) if acquisition else None
    depreciation_value = _numeric_value(depreciation.value) if depreciation else None

    if gross and net and gross_value is not None and net_value is not None and net_value > gross_value:
        _add_validation_error(
            field_id="net_amount",
            question="Nettobetrag ist groesser als der Bruttobetrag.",
            missing_information=["Korrektur Betrag"],
            evidence_refs=[_evidence_ref(net), _evidence_ref(gross)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id="net_amount:greater_than_gross",
        )
    if gross and vat and gross_value is not None and vat_value is not None and vat_value > gross_value:
        _add_validation_error(
            field_id="vat_amount",
            question="Umsatzsteuerbetrag ist groesser als der Bruttobetrag.",
            missing_information=["Korrektur Betrag"],
            evidence_refs=[_evidence_ref(vat), _evidence_ref(gross)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id="vat_amount:greater_than_gross",
        )
    if (
        gross
        and net
        and vat
        and gross_value is not None
        and net_value is not None
        and vat_value is not None
        and abs(gross_value - net_value - vat_value) > CENT_TOLERANCE
    ):
        _add_validation_error(
            field_id="gross_amount",
            question="Bruttobetrag entspricht nicht Netto plus Umsatzsteuer.",
            missing_information=["Korrektur Brutto/Netto/Umsatzsteuer"],
            evidence_refs=[_evidence_ref(gross), _evidence_ref(net), _evidence_ref(vat)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id="gross_amount:net_vat_mismatch",
        )
    if (
        acquisition
        and depreciation
        and acquisition_value is not None
        and depreciation_value is not None
        and depreciation_value > acquisition_value
    ):
        _add_validation_error(
            field_id="depreciation_amount",
            question="AfA-Betrag ist groesser als die Anschaffungskosten.",
            missing_information=["Korrektur AfA oder Anschaffungskosten"],
            evidence_refs=[_evidence_ref(depreciation), _evidence_ref(acquisition)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id="depreciation_amount:greater_than_acquisition_cost",
        )


def _validate_scalar_field(
    field_id: str,
    mapping: FieldMapping,
    clarifications: list[ClarificationRequest],
    validation_issues: list[ValidationIssue],
) -> None:
    lowered = field_id.lower()
    numeric = _numeric_value(mapping.value)
    if numeric is None:
        return
    if "carryforward" in lowered and numeric < 0:
        _add_validation_error(
            field_id=field_id,
            question="Verlustvortrag oder Carryforward darf nicht negativ sein.",
            missing_information=["Korrektur Carryforward"],
            evidence_refs=[_evidence_ref(mapping)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id=f"{field_id}:negative_carryforward",
        )
    if any(token in lowered for token in ("percent", "percentage", "prozentsatz", "quote")) and not (
        Decimal("0") <= numeric <= Decimal("100")
    ):
        _add_validation_error(
            field_id=field_id,
            question="Prozentwert liegt ausserhalb des gueltigen Bereichs 0 bis 100.",
            missing_information=["Korrektur Prozentwert"],
            evidence_refs=[_evidence_ref(mapping)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id=f"{field_id}:percent_out_of_range",
        )


def _validate_dates(
    field_id: str,
    mapping: FieldMapping,
    tax_year: int | None,
    clarifications: list[ClarificationRequest],
    validation_issues: list[ValidationIssue],
) -> None:
    today = date.today()
    relevant_dates = {
        "document_date": _mapping_date(mapping, "document_date"),
        "service_date": _mapping_date(mapping, "service_date"),
        "payment_date": _mapping_date(mapping, "payment_date"),
    }
    for label, parsed in relevant_dates.items():
        if parsed and parsed > today:
            _add_validation_error(
                field_id=field_id,
                question=f"{label} liegt in der Zukunft.",
                missing_information=[label],
                evidence_refs=[_evidence_ref(mapping)],
                clarifications=clarifications,
                validation_issues=validation_issues,
                issue_id=f"{field_id}:{label}_in_future",
            )

    if tax_year is None:
        return

    payment_date = relevant_dates["payment_date"]
    document_date = relevant_dates["document_date"]
    service_date = relevant_dates["service_date"]
    if payment_date and payment_date.year != tax_year:
        _add_validation_error(
            field_id=field_id,
            question="Zahlungsdatum liegt ausserhalb des Steuerjahres und muss nach Zufluss-/Abflussprinzip geklaert werden.",
            missing_information=["payment_date", "tax_year"],
            evidence_refs=[_evidence_ref(mapping)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id=f"{field_id}:payment_date_outside_tax_year",
            issue_type=IssueType.DATA_UNCERTAINTY,
            severity=Severity.WARNING,
        )
    elif not payment_date and any(
        parsed and parsed.year != tax_year for parsed in (document_date, service_date)
    ):
        _add_validation_error(
            field_id=field_id,
            question="Beleg- oder Leistungsdatum liegt ausserhalb des Steuerjahres; Zahlungsdatum fehlt fuer die Zuordnung.",
            missing_information=["payment_date"],
            evidence_refs=[_evidence_ref(mapping)],
            clarifications=clarifications,
            validation_issues=validation_issues,
            issue_id=f"{field_id}:missing_payment_date_for_tax_year",
            issue_type=IssueType.DATA_UNCERTAINTY,
            severity=Severity.WARNING,
        )


def _detect_duplicate_transactions(
    state: TaxPipelineState,
    clarifications: list[ClarificationRequest],
    validation_issues: list[ValidationIssue],
) -> None:
    raw_data = state.get("raw_extracted_data", {})
    extracted = raw_data.get("extracted", []) if isinstance(raw_data, dict) else []
    if not isinstance(extracted, list):
        return

    seen: dict[tuple[str, Any], dict[str, Any]] = {}
    for item in extracted:
        if not isinstance(item, dict):
            continue
        amount = _numeric_value(item.get("amount") or item.get("gross_amount") or item.get("value"))
        date_key = item.get("payment_date") or item.get("document_date") or item.get("date")
        invoice_number = item.get("invoice_number")
        counterparty = item.get("counterparty")
        document_hash = item.get("document_hash")
        keys: list[tuple[str, Any]] = []
        if document_hash:
            keys.append(("document_hash", document_hash))
        if invoice_number:
            keys.append(("invoice_number", str(invoice_number).strip().lower()))
        if amount is not None and date_key:
            keys.append(("amount_date", (str(amount), str(date_key))))
        if amount is not None and date_key and counterparty:
            keys.append(
                (
                    "counterparty_amount_date",
                    (str(counterparty).strip().lower(), str(amount), str(date_key)),
                )
            )

        for key in keys:
            previous = seen.get(key)
            if previous is None:
                seen[key] = item
                continue
            if previous is item:
                continue
            previous_doc = str(previous.get("document_id") or previous.get("absolute_path") or "")
            current_doc = str(item.get("document_id") or item.get("absolute_path") or "")
            if previous_doc == current_doc and key[0] != "document_hash":
                continue
            _add_validation_error(
                field_id="GLOBAL",
                question="Moegliche Doppelzaehlung desselben wirtschaftlichen Vorgangs erkannt.",
                missing_information=["Dedupe-Entscheidung", "Aggregation"],
                evidence_refs=[ref for ref in (previous_doc, current_doc) if ref],
                clarifications=clarifications,
                validation_issues=validation_issues,
                issue_id=f"dedupe:{key[0]}:{hashlib.sha256(str(key[1]).encode('utf-8')).hexdigest()[:12]}",
                issue_type=IssueType.DATA_UNCERTAINTY,
                severity=Severity.WARNING,
            )
            break


def _resolve_mapping_candidates(
    state: TaxPipelineState,
    clarifications: list[ClarificationRequest],
    validation_issues: list[ValidationIssue],
) -> dict[str, FieldMapping]:
    resolved = dict(state.get("field_mappings", {}))
    candidates_by_field = state.get("field_mapping_candidates", {})
    for field_id, candidates in candidates_by_field.items():
        if not candidates:
            continue
        ranked = sorted(candidates, key=source_rank)
        best_rank = source_rank(ranked[0])
        best = [candidate for candidate in ranked if source_rank(candidate) == best_rank]
        if len(best) > 1:
            values = {_json_stable(candidate.value) for candidate in best}
            if len(values) > 1:
                _add_validation_error(
                    field_id=field_id,
                    question="Gleichrangige Quellen widersprechen sich fuer dasselbe Formularfeld.",
                    missing_information=["Konfliktaufloesung"],
                    evidence_refs=[_evidence_ref(candidate) for candidate in best],
                    clarifications=clarifications,
                    validation_issues=validation_issues,
                    issue_id=f"{field_id}:same_priority_conflict",
                    issue_type=IssueType.DATA_UNCERTAINTY,
                    severity=Severity.ERROR if best_rank <= 3 else Severity.WARNING,
                )
                continue
        resolved[field_id] = ranked[0]
    return resolved


def _json_stable(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(value)


def tax_data_extractor_node(state: TaxPipelineState) -> dict[str, Any]:
    """Step 1: prepare CocoIndex chunks for deterministic extraction.

    The production LLM extraction hook belongs here. This skeleton deliberately
    avoids inventing tax values and only preserves chunk metadata for audit.
    """
    chunks = state.get("coco_chunks", [])
    prepared_chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        prepared_chunks.append(
            {
                "index": index,
                "document_id": chunk.get("document_id") or chunk.get("absolute_path"),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "text": chunk.get("text") or chunk.get("content") or "",
                "metadata": chunk.get("metadata", {}),
            }
        )
    return {
        "raw_extracted_data": {
            "extracted": [],
            "source_chunk_count": len(chunks),
            "available_chunks": prepared_chunks,
        }
    }


def tax_field_mapper_node(state: TaxPipelineState) -> dict[str, Any]:
    """Step 2: map extracted values to the loaded ELSTER schema."""
    form_id = state.get("form_id")
    form_version = state.get("form_version")
    if not form_id or not form_version:
        return _append_clarification(
            state,
            _clarification(
                field_id="GLOBAL",
                question="Formular-ID oder Formularversion fehlt.",
                missing_information=["form_id", "form_version"],
                issue_type=IssueType.MISSING_REFERENCE,
            ),
        )

    schema_file = _schema_path(form_id, form_version)
    if not schema_file.exists():
        return _append_clarification(
            state,
            _clarification(
                field_id="GLOBAL",
                question=f"Formular-Schema fuer {form_id}_{form_version} nicht gefunden.",
                missing_information=["form_schema_version", str(schema_file)],
                issue_type=IssueType.MISSING_REFERENCE,
            ),
        )

    try:
        schema = _load_json_reference(schema_file)
    except (JSONDecodeError, ValueError) as exc:
        return _append_clarification(
            state,
            _clarification(
                field_id="GLOBAL",
                question=f"Formular-Schema ist nicht lesbar: {exc}",
                missing_information=[str(schema_file)],
                issue_type=IssueType.VALIDATION_ERROR,
            ),
        )

    raw_data = state.get("raw_extracted_data", {})
    extracted = raw_data.get("extracted", [])
    if not extracted:
        return {
            "field_mappings": {},
            "form_schema_version": str(schema.get("schema_version") or state.get("form_schema_version") or ""),
        }

    # Production mapping belongs here. It must create FieldMapping instances
    # with evidence and may only use fields defined by the loaded schema.
    return {
        "field_mappings": {},
        "form_schema_version": str(schema.get("schema_version") or state.get("form_schema_version") or ""),
    }


def tax_validator_node(state: TaxPipelineState) -> dict[str, Any]:
    """Step 3: validate evidence, source roots, and blocking conflicts."""
    clarifications = [*state.get("clarifications", [])]
    validation_issues = [*state.get("validation_issues", [])]
    mappings = _resolve_mapping_candidates(state, clarifications, validation_issues)
    tax_year = state.get("tax_year")

    _validate_amount_relationships(mappings, clarifications, validation_issues)
    _detect_duplicate_transactions(state, clarifications, validation_issues)

    for field_id, mapping in mappings.items():
        evidence = mapping.evidence
        if not _is_canonical_document(evidence.document_id):
            issue_id = f"{field_id}:non_canonical_evidence"
            validation_issues.append(
                ValidationIssue(
                    issue_id=issue_id,
                    issue_type=IssueType.VALIDATION_ERROR,
                    severity=Severity.ERROR,
                    description="Evidence document is outside the canonical Hermes document root.",
                    field_id=field_id,
                )
            )
            clarifications.append(
                _clarification(
                    field_id=field_id,
                    question="Evidenz liegt ausserhalb des kanonischen Hermes-Dokumentenpfads.",
                    missing_information=[str(CANONICAL_DOC_ROOT)],
                    issue_type=IssueType.VALIDATION_ERROR,
                    evidence_refs=[evidence.document_id],
                )
            )
        _validate_scalar_field(field_id, mapping, clarifications, validation_issues)
        _validate_dates(field_id, mapping, tax_year, clarifications, validation_issues)
        confidence_failures: list[str] = []
        if mapping.extraction_confidence < 0.80:
            confidence_failures.append("extraction_confidence >= 0.80")
        if mapping.mapping_confidence < 0.90:
            confidence_failures.append("mapping_confidence >= 0.90")
        if mapping.validation_confidence < 0.95:
            confidence_failures.append("validation_confidence >= 0.95")
        if confidence_failures:
            validation_issues.append(
                ValidationIssue(
                    issue_id=f"{field_id}:confidence_threshold",
                    issue_type=IssueType.DATA_UNCERTAINTY,
                    severity=Severity.WARNING,
                    description="Confidence thresholds from the legacy TaxPipeline skill were not met.",
                    field_id=field_id,
                )
            )
            clarifications.append(
                _clarification(
                    field_id=field_id,
                    question="Die Konfidenzwerte reichen fuer eine automatische Uebernahme nicht aus.",
                    missing_information=confidence_failures,
                    issue_type=IssueType.DATA_UNCERTAINTY,
                    evidence_refs=[evidence.document_id],
                )
            )

    requires_clarification = bool(clarifications)
    return {
        "field_mappings": mappings,
        "validation_issues": validation_issues,
        "clarifications": clarifications,
        "requires_clarification": requires_clarification,
    }


def tax_optimizer_node(state: TaxPipelineState) -> dict[str, Any]:
    """Step 4: create AfA suggestions only from versioned reference tables."""
    if state.get("requires_clarification", False):
        return {}

    afa_table_version = state.get("afa_table_version")
    if not afa_table_version:
        return _append_clarification(
            state,
            _clarification(
                field_id="GLOBAL",
                question="AfA-Tabellenversion fehlt.",
                missing_information=["afa_table_version"],
                issue_type=IssueType.MISSING_REFERENCE,
            ),
        )

    afa_table = REF_AFA_DIR / f"{afa_table_version}.json"
    if not afa_table.exists():
        return _append_clarification(
            state,
            _clarification(
                field_id="GLOBAL",
                question=f"AfA-Referenztabelle nicht gefunden: {afa_table_version}.",
                missing_information=[str(afa_table)],
                issue_type=IssueType.MISSING_REFERENCE,
            ),
        )

    try:
        _load_json_reference(afa_table)
    except (JSONDecodeError, ValueError) as exc:
        return _append_clarification(
            state,
            _clarification(
                field_id="GLOBAL",
                question=f"AfA-Referenztabelle ist nicht lesbar: {exc}",
                missing_information=[str(afa_table)],
                issue_type=IssueType.VALIDATION_ERROR,
            ),
        )

    # Production AfA matching belongs here. This skeleton never infers useful
    # lives from model knowledge.
    return {"depreciation_suggestions": []}
