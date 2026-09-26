from pathlib import Path

import tools.quarantine_dev_human_approval as dev_approval
import tools.quarantine_lab_simulation as lab_sim


def test_development_modules_do_not_import_production_signer_verifier_or_ledger():
    texts = [
        Path(dev_approval.__file__).read_text(encoding="utf-8"),
        Path(lab_sim.__file__).read_text(encoding="utf-8"),
    ]
    combined = "\n".join(texts)
    forbidden = [
        "ApprovalArtifact",
        "SignatureVerifier",
        "NonceLedger",
        "build_production_approval_runtime",
        "WindowsWebAuthnApprovalProvider",
        "WindowsWebAuthnApprovalVerifier",
        "TrustedVerifierRegistry",
    ]
    for symbol in forbidden:
        assert symbol not in combined


def test_development_modules_have_no_active_install_or_execution_sink():
    texts = [
        Path(dev_approval.__file__).read_text(encoding="utf-8"),
        Path(lab_sim.__file__).read_text(encoding="utf-8"),
    ]
    combined = "\n".join(texts)
    forbidden = [
        "install_from_quarantine",
        "subprocess.",
        "requests.",
        "webbrowser.",
        "os.system(",
        "MakeCredential",
        "GetAssertion",
    ]
    for symbol in forbidden:
        assert symbol not in combined


def test_simulation_result_literal_is_not_production_allow_token():
    assert lab_sim.ALLOW_TO_LAB_SIMULATION == "ALLOW_TO_LAB_SIMULATION"
    assert lab_sim.ALLOW_TO_LAB_SIMULATION != "ALLOW_TO_LAB"
