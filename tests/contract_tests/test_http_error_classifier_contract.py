"""Pytest entry-point that runs the HttpErrorClassifier contract against
MiniMaxErrorClassifier and CodexErrorClassifier.

If either contract run fails, the test fails — the contract suite is
the definition of the architectural contract, so a contract failure
is a STOP condition.
"""

from __future__ import annotations

import pytest

from agent.provider_errors import CodexErrorClassifier, MiniMaxErrorClassifier
from tests.contract_tests.http_error_classifier_contract import (
    run_http_error_classifier_contract,
)


def test_minimax_classifier_passes_http_error_contract():
    assert run_http_error_classifier_contract(
        MiniMaxErrorClassifier, name="MiniMaxErrorClassifier"
    )


def test_codex_classifier_passes_http_error_contract():
    assert run_http_error_classifier_contract(
        CodexErrorClassifier, name="CodexErrorClassifier"
    )