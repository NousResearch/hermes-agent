"""Plugin-shaped payment candidate classifier/extractor wrappers.

These wrappers preserve the current payments adapter semantics while exposing
explicit classifier/extractor steps that a generalized inbox-ops runtime can
compose during shadow mode and cutover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PaymentCandidateClassification:
    workflow_type: str
    status: str
    confidence: Any
    action_required: bool
    receipt_like: bool
    invoice_like: bool


class PaymentCandidateClassifier:
    """Classify an existing payment-shaped record without changing thresholds."""

    def classify(self, request: dict[str, Any]) -> PaymentCandidateClassification:
        status = str(request.get("status") or "needs_review")
        receipt_like = bool(request.get("looks_paid")) or status == "paid"
        return PaymentCandidateClassification(
            workflow_type="payment_receipt" if receipt_like else "payment_request",
            status=status,
            confidence=request.get("confidence"),
            action_required=status != "paid",
            receipt_like=receipt_like,
            invoice_like=bool(request.get("invoice_number")),
        )


class PaymentCandidateExtractor:
    """Extract canonical inbox-ops fields from an existing payment-shaped record."""

    def extract(self, request: dict[str, Any], classification: PaymentCandidateClassification) -> dict[str, Any]:
        original = request.get("original") or {}
        return {
            "id": str(request.get("id") or request.get("payment_id") or ""),
            "source": str(request.get("source") or "gmail"),
            "source_account": "",
            "source_thread_id": str(original.get("thread_id") or ""),
            "source_message_id": str(original.get("message_id") or ""),
            "source_url": str(original.get("url") or ""),
            "captured_at": request.get("received_at"),
            "updated_at": request.get("updated_at"),
            "last_classified_at": request.get("updated_at"),
            "workflow_type": classification.workflow_type,
            "queue": "payments",
            "status": classification.status,
            "title": str(request.get("title") or ""),
            "summary": str(request.get("preview_text") or ""),
            "counterparty": str(request.get("vendor") or ""),
            "action_required": classification.action_required,
            "confidence": classification.confidence,
            "amount": dict(request.get("amount") or {}),
            "due_date": request.get("due_date"),
            "meeting_dates": [],
            "meeting_timezone": "",
            "reference_id": str(request.get("payment_reference") or ""),
            "receipt_like": classification.receipt_like,
            "invoice_like": classification.invoice_like,
            "calendar_like": False,
            "warning_flags": [str(item) for item in (request.get("warnings") or [])],
            "operator_notes": str(request.get("review_note") or ""),
            "raw_payload_path": request.get("raw_text_path"),
            "materialized_path": request.get("materialized_path"),
            "manual_status": False,
            "manual_notes": bool(request.get("review_note")),
            "reviewed_at": None,
            "reviewed_by": None,
            "entities": {
                "sender": str(original.get("label") or ""),
                "organization": str(request.get("vendor") or ""),
                "payee_name": str(request.get("payee_name") or ""),
                "account_holder": str(request.get("account_holder") or ""),
                "account_number": str(request.get("account_number") or ""),
                "sort_code": str(request.get("sort_code") or ""),
                "iban": str(request.get("iban") or ""),
                "swift": str(request.get("swift") or ""),
                "routing_number": str(request.get("routing_number") or ""),
                "payment_reference": str(request.get("payment_reference") or ""),
                "invoice_number": str(request.get("invoice_number") or ""),
                "billing_address": str(request.get("billing_address") or ""),
                "tax_details": str(request.get("tax_details") or ""),
            },
            "artifacts": {
                "label": str(original.get("label") or ""),
                "attachments": [str(item) for item in (request.get("attachments") or [])],
                "original_thread_url": str(original.get("url") or ""),
            },
        }

