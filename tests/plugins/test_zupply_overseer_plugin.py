"""Tests for the Zupply Overseer deterministic preflight gate."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "zupply-overseer"


def _load_plugin():
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.zupply_overseer",
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.zupply_overseer"
    mod.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["hermes_plugins.zupply_overseer"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCourierCollectionGate:
    def test_read_only_tools_are_never_blocked(self):
        mod = _load_plugin()
        assert mod._on_pre_tool_call(
            tool_name="read_file",
            args={"path": "/tmp/QUOTE-70001.pdf", "note": "NZC pickup request margin: 15%"},
        ) is None

    def test_blocks_nzc_collection_email(self):
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="terminal",
            args={"command": "himalaya message send --to russell --subject 'NZC pickup request' --body 'Please collect MTX0002597'"},
        )
        assert out["action"] == "block"
        assert "must not be emailed" in out["message"]
        assert "NZCouriers" in out["message"]
        assert "bookpickup" in out["message"]

    def test_allows_gosweetspot_bookpickup_for_nzc(self):
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="execute_code",
            args={"code": "requests.post('https://api.gosweetspot.com/api/bookpickup', json={'Carrier':'NZCouriers','Consignments':['MTX0002597']})"},
        )
        assert out is None


class TestPaymentGate:
    def test_blocks_wise_submit_transfer(self):
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="browser_click",
            args={"button": "Submit transfer in Wise payment screen"},
        )
        assert out["action"] == "block"
        assert "payment/bill execution" in out["message"]

    def test_allows_wise_review_pack(self):
        mod = _load_plugin()
        assert mod._on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/tmp/wise-review.md", "content": "Wise payment review pack only; prepare details, do not submit."},
        ) is None


class TestCommercialGate:
    def test_blocks_customer_quote_below_min_margin(self):
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/tmp/quote.md", "content": "Customer quote 70001\nMargin: 15%\nSell price: 100"},
        )
        assert out["action"] == "block"
        assert "below 20%" in out["message"]

    def test_allows_quote_at_min_margin(self):
        mod = _load_plugin()
        assert mod._on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/tmp/quote.md", "content": "Customer quote 70001\nMargin: 28%\nSell price: 100"},
        ) is None


class TestPdfGate:
    def test_blocks_plain_final_zupply_pdf(self):
        mod = _load_plugin()
        out = mod._on_pre_tool_call(
            tool_name="terminal",
            args={"command": "python make_pdf.py --title 'Zupply Quote 70001' --output QUOTE-70001.pdf"},
        )
        assert out["action"] == "block"
        assert "branded Ghostmark/PDF-generator" in out["message"]

    def test_allows_ghostmark_pdf_generator_workflow(self):
        mod = _load_plugin()
        assert mod._on_pre_tool_call(
            tool_name="terminal",
            args={"command": "python zupply-pdf-generator --ghostmark --input job.json --output QUOTE-70001.pdf"},
        ) is None

    def test_allows_obsidian_job_note_that_references_source_pdf(self):
        mod = _load_plugin()
        assert mod._on_pre_tool_call(
            tool_name="write_file",
            args={
                "path": "/Users/logandubray/Zupply/Zupply-OS-Vault/Operations/Jobs/72946 - Kim Freight.md",
                "content": "Freight-only comparison. Source PDF: /Users/logandubray/.hermes/cache/documents/dsv_quote.pdf",
            },
        ) is None

    def test_allows_readme_documenting_pdf_workflow(self):
        mod = _load_plugin()
        assert mod._on_pre_tool_call(
            tool_name="write_file",
            args={
                "path": "/Users/logandubray/Zupply/Zupply-OS-Vault/Operations/PDF Quote PO Workflow.md",
                "content": "Final Zupply PDFs must use Ghostmark. Source docs may include quote PDFs from suppliers.",
            },
        ) is None
