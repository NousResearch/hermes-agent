"""Offline pathology archive and calibrated credit gates for Hermes traces."""

from .archive import PathologyArchive, PathologyRecord, Span, Task, Trace
from .battery import BatteryScore, SealedBatteryClient
from .gates import (
    ActivationEvidence,
    CreditPolicy,
    GateDecision,
    GateReport,
    PairedScore,
    ValidityEvidence,
    evaluate_activation,
    evaluate_credit,
    evaluate_gates,
    evaluate_validity,
)

__all__ = [
    "ActivationEvidence",
    "BatteryScore",
    "CreditPolicy",
    "GateDecision",
    "GateReport",
    "PairedScore",
    "PathologyArchive",
    "PathologyRecord",
    "SealedBatteryClient",
    "Span",
    "Task",
    "Trace",
    "ValidityEvidence",
    "evaluate_activation",
    "evaluate_credit",
    "evaluate_gates",
    "evaluate_validity",
]
