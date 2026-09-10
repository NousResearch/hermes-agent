"""Tests for implementation_annex_generator (Phase 1 — ADHD-I user-facing quality)."""

from __future__ import annotations

import hashlib
import json
import py_compile
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOOLS = Path(__file__).resolve().parent.parent / "tools"
CONFIG = Path(__file__).resolve().parent.parent / "config"


# ── py_compile ────────────────────────────────────────────────────────────


class TestPyCompile:
    def test_generator_compiles(self):
        py_compile.compile(str(TOOLS / "implementation_annex_generator.py"),
                           doraise=True)

    def test_quality_gate_compiles(self):
        py_compile.compile(
            str(TOOLS / "implementation_annex_quality_gate.py"),
            doraise=True)


# ── generate_annex smoke test ─────────────────────────────────────────────


class TestGenerateAnnex:
    """End-to-end smoke test using ADHD-I fixtures with separate contract."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "annex_output"

    def _generate(self, output_dir):
        from tools.implementation_annex_generator import generate_annex
        return generate_annex(
            final_report=FIXTURES / "adhd_i_decision_report.md",
            external_calibration=FIXTURES / "adhd_i_external_calibration.md",
            contract=FIXTURES / "adhd_i_decision_context_contract.md",
            output_dir=output_dir,
            domain="child_adhd_education",
        )

    def test_generate_returns_expected_keys(self, output_dir):
        result = self._generate(output_dir)
        assert "annex_path" in result
        assert "manifest_path" in result
        assert "annex_sha256" in result
        assert "manifest" in result
        assert "quality_verdict" in result

    def test_generate_creates_files(self, output_dir):
        result = self._generate(output_dir)
        annex = Path(result["annex_path"])
        manifest = Path(result["manifest_path"])
        assert annex.exists(), f"Annex not found: {annex}"
        assert manifest.exists(), f"Manifest not found: {manifest}"
        assert annex.stat().st_size > 100, "Annex too small"
        assert manifest.stat().st_size > 50, "Manifest too small"

    def test_generate_annex_has_required_sections(self, output_dir):
        result = self._generate(output_dir)
        annex_text = Path(result["annex_path"]).read_text(encoding="utf-8")
        required = [
            "当前判断",
            "未来 2 周启动方案",
            "作业流程",
            "家长行为支持动作",
            "学校沟通策略",
            "每周观察表",
            "维持、升级、复评信号",
            "不要做的事",
            "证据边界",
        ]
        for section in required:
            assert section in annex_text, f"Missing section: {section}"

    def test_generate_manifest_has_provenance(self, output_dir):
        result = self._generate(output_dir)
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        assert "domain" in manifest
        assert manifest["domain"] == "child_adhd_education"
        assert "provenance" in manifest
        prov = manifest["provenance"]
        assert "inputs" in prov
        assert "outputs" in prov
        assert "generated_at" in prov
        assert "final_report" in prov["inputs"]
        assert prov["inputs"]["final_report"]["sha256"] is not None

    def test_manifest_contract_distinct_from_final_report(self, output_dir):
        """Verify contract fixture is NOT the same file as final report."""
        result = self._generate(output_dir)
        manifest = json.loads(Path(result["manifest_path"]).read_text())
        prov = manifest["provenance"]["inputs"]
        final_sha = prov["final_report"]["sha256"]
        contract_sha = prov["contract"]["sha256"]
        assert final_sha != contract_sha, (
            "Contract sha256 equals final_report sha256 — "
            "contract fixture must be distinct from final report"
        )

    def test_quality_gate_passes_on_generated_annex(self, output_dir):
        result = self._generate(output_dir)
        verdict = result["quality_verdict"]
        # Print failures for debugging
        failed_checks = [c for c in verdict["checks"] if not c["passed"]]
        assert verdict["passed"] is True, (
            f"Quality gate failed: {verdict['summary']}\n"
            f"Failures: {failed_checks}"
        )

    def test_generate_no_drug_names(self, output_dir):
        result = self._generate(output_dir)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        failures = []
        import re
        drug_patterns = [
            r"\b(methylphenidate|ritalin|concerta|哌甲酯|利他林|专注达)\b",
            r"\b(amphetamine|adderall|vyvanse|苯丙胺|阿德拉)\b",
            r"\b(atomoxetine|strattera|托莫西汀)\b",
            r"\b\d+\s*mg\b",
        ]
        for pat in drug_patterns:
            if re.search(pat, text, re.IGNORECASE):
                failures.append(pat)
        assert failures == [], f"Drug names found: {failures}"

    def test_generate_no_treatment_instructions(self, output_dir):
        result = self._generate(output_dir)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        import re
        treatment = [
            r"\b(治疗方案|治疗计划|医嘱|临床建议)\b",
            r"\b(必须.*治疗|应当.*服药)\b",
        ]
        failures = []
        for pat in treatment:
            if re.search(pat, text):
                failures.append(pat)
        assert failures == [], f"Treatment instructions found: {failures}"

    def test_generate_no_internal_terms_in_annex(self, output_dir):
        """Verify no internal evidence-tier labels leak."""
        result = self._generate(output_dir)
        from tools.implementation_annex_quality_gate import check_internal_terms
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        failures = check_internal_terms(text, "child_adhd_education", CONFIG)
        assert failures == [], f"Internal terms leaked: {failures}"

    def test_generate_no_user_facing_internal_terms(self, output_dir):
        """Verify no 收敛/外部校准/pipeline/artifact/scenario_branches etc."""
        result = self._generate(output_dir)
        from tools.implementation_annex_quality_gate import (
            check_no_user_facing_internal_terms)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        failures = check_no_user_facing_internal_terms(
            text, "child_adhd_education", CONFIG)
        assert failures == [], (
            f"User-facing internal terms leaked: {failures}")

    def test_generate_no_empty_placeholders(self, output_dir):
        """Verify no 无明确/未提取/TODO/placeholder in output."""
        result = self._generate(output_dir)
        from tools.implementation_annex_quality_gate import (
            check_no_empty_placeholders)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        failures = check_no_empty_placeholders(
            text, "child_adhd_education", CONFIG)
        assert failures == [], f"Empty placeholders found: {failures}"

    def test_generate_concrete_actions(self, output_dir):
        """Verify concrete actions are real (not heading-only)."""
        result = self._generate(output_dir)
        from tools.implementation_annex_quality_gate import (
            check_concrete_actions_present)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        failures = check_concrete_actions_present(text)
        assert failures == [], f"Concrete actions check failed: {failures}"

    def test_generate_contains_homework_flow(self, output_dir):
        """Verify 作业流程 contains 作业前/作业中/作业后."""
        result = self._generate(output_dir)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        for phase in ["作业前", "作业中", "作业后"]:
            assert phase in text, f"Missing '{phase}' in homework flow"

    def test_generate_contains_school_script(self, output_dir):
        """Verify 学校沟通策略 contains concrete talk scripts."""
        result = self._generate(output_dir)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        assert "参考话术" in text, "Missing school talk scripts"

    def test_generate_contains_do_not_do_list(self, output_dir):
        """Verify 不要做的事 has at least 6 items."""
        result = self._generate(output_dir)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        # Count numbered items (1. 2. etc.)
        import re
        items = re.findall(r"^\s*\d+\.\s*\*", text, re.MULTILINE)
        assert len(items) >= 6, f"Only {len(items)} do-not-do items, need >= 6"

    def test_generate_contains_evidence_boundary(self, output_dir):
        """Verify 证据边界 contains both '我们知道的' and '我们不确定的'."""
        result = self._generate(output_dir)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        assert "我们知道的" in text
        assert "我们不确定的" in text

    def test_generate_contains_escalation_categories(self, output_dir):
        """Verify 维持、升级、复评信号 contains all three categories."""
        result = self._generate(output_dir)
        text = Path(result["annex_path"]).read_text(encoding="utf-8")
        for category in ["维持", "升级", "复评"]:
            assert category in text, (
                f"Missing '{category}' in escalation signals")

    def test_generator_unknown_domain_raises(self, output_dir):
        from tools.implementation_annex_generator import generate_annex
        with pytest.raises(ValueError, match="Unknown domain profile"):
            generate_annex(
                final_report=FIXTURES / "adhd_i_decision_report.md",
                external_calibration=FIXTURES / "adhd_i_external_calibration.md",
                contract=FIXTURES / "adhd_i_decision_context_contract.md",
                output_dir=output_dir,
                domain="nonexistent_domain",
            )


# ── quality gate unit tests ───────────────────────────────────────────────


class TestQualityGate:
    def test_check_no_fffd_clean(self):
        from tools.implementation_annex_quality_gate import check_no_fffd
        assert check_no_fffd("hello world") == []

    def test_check_no_fffd_dirty(self):
        from tools.implementation_annex_quality_gate import check_no_fffd
        failures = check_no_fffd("hello \ufffd world")
        assert len(failures) == 1
        assert "U+FFFD" in failures[0]

    def test_check_internal_terms_clean(self):
        from tools.implementation_annex_quality_gate import check_internal_terms
        text = "这是一个面向用户的内容"
        failures = check_internal_terms(text, "child_adhd_education", CONFIG)
        assert failures == []

    def test_check_internal_terms_dirty(self):
        from tools.implementation_annex_quality_gate import check_internal_terms
        text = "这基于 evidence_supported 的证据"
        failures = check_internal_terms(text, "child_adhd_education", CONFIG)
        assert len(failures) >= 1
        assert "evidence_supported" in failures[0]

    def test_check_user_facing_internal_terms_clean(self):
        from tools.implementation_annex_quality_gate import (
            check_no_user_facing_internal_terms)
        text = "这是一个面向用户的执行指南"
        failures = check_no_user_facing_internal_terms(
            text, "child_adhd_education", CONFIG)
        assert failures == []

    def test_check_user_facing_internal_terms_dirty_convergence(self):
        from tools.implementation_annex_quality_gate import (
            check_no_user_facing_internal_terms)
        text = "本指南基于决策收敛报告"
        failures = check_no_user_facing_internal_terms(
            text, "child_adhd_education", CONFIG)
        assert len(failures) >= 1

    def test_check_user_facing_internal_terms_dirty_scenario_branches(self):
        from tools.implementation_annex_quality_gate import (
            check_no_user_facing_internal_terms)
        text = "参考 scenario_branches 判断"
        failures = check_no_user_facing_internal_terms(
            text, "child_adhd_education", CONFIG)
        assert len(failures) >= 1

    def test_check_empty_placeholders_clean(self):
        from tools.implementation_annex_quality_gate import (
            check_no_empty_placeholders)
        text = "根据评估和证据，需要主动干预"
        failures = check_no_empty_placeholders(
            text, "child_adhd_education", CONFIG)
        assert failures == []

    def test_check_empty_placeholders_dirty(self):
        from tools.implementation_annex_quality_gate import (
            check_no_empty_placeholders)
        text = "无明确的共识点提取"
        failures = check_no_empty_placeholders(
            text, "child_adhd_education", CONFIG)
        assert len(failures) >= 1

    def test_check_no_drug_names_clean(self):
        from tools.implementation_annex_quality_gate import check_no_drug_names
        assert check_no_drug_names("家长应关注孩子的注意力表现") == []

    def test_check_no_drug_names_dirty(self):
        from tools.implementation_annex_quality_gate import check_no_drug_names
        failures = check_no_drug_names("考虑使用利他林 5mg")
        assert len(failures) >= 1

    def test_check_no_treatment_instructions_clean(self):
        from tools.implementation_annex_quality_gate import (
            check_no_treatment_instructions)
        assert check_no_treatment_instructions("家长可以观察行为模式") == []

    def test_check_no_treatment_instructions_dirty(self):
        from tools.implementation_annex_quality_gate import (
            check_no_treatment_instructions)
        failures = check_no_treatment_instructions("建议进行药物治疗")
        assert len(failures) >= 1

    def test_check_grade3_forward_only_clean(self):
        from tools.implementation_annex_quality_gate import (
            check_grade3_forward_only)
        text = "三年级是一个需要关注的学业转折点（前瞻性推断）"
        assert check_grade3_forward_only(text) == []

    def test_check_grade3_forward_only_dirty(self):
        from tools.implementation_annex_quality_gate import (
            check_grade3_forward_only)
        text = "三年级应当制定具体训练计划"
        failures = check_grade3_forward_only(text)
        assert len(failures) >= 1

    def test_check_overcommit_bpt_clean(self):
        from tools.implementation_annex_quality_gate import (
            check_overcommit_guards)
        text = "BPT 是组成部分，不是单独充分方案"
        failures = check_overcommit_guards(text, "child_adhd_education", CONFIG)
        assert failures == []

    def test_check_overcommit_bpt_dirty(self):
        from tools.implementation_annex_quality_gate import (
            check_overcommit_guards)
        text = "BPT 一定有效改善孩子的注意力问题"
        failures = check_overcommit_guards(text, "child_adhd_education", CONFIG)
        assert len(failures) >= 1

    def test_check_concrete_actions_empty(self):
        """Verify that a heading-only annex fails concrete_actions check."""
        from tools.implementation_annex_quality_gate import (
            check_concrete_actions_present)
        # Minimal annex with headings but no real content
        text = """
## 1. 当前判断

## 2. 未来 2 周启动方案

## 3. 作业流程

## 4. 家长行为支持动作

## 5. 学校沟通策略

## 6. 每周观察表

## 7. 维持、升级、复评信号

## 8. 不要做的事

## 9. 证据边界
"""
        failures = check_concrete_actions_present(text)
        assert len(failures) >= 1, (
            "Empty sections should trigger concrete_actions failures")

    def test_check_pseudo_threshold_clean(self):
        """Verify that softened thresholds pass."""
        from tools.implementation_annex_quality_gate import (
            check_pseudo_threshold)
        text = "每周观察时间变化，参考信号，不是固定标准"
        failures = check_pseudo_threshold(text)
        assert failures == []

    def test_check_manifest_contract_distinct(self, tmp_path):
        """Verify manifest check detects same-sha contract."""
        from tools.implementation_annex_quality_gate import (
            check_manifest_contract_distinct)
        # Create a manifest where contract sha == final_report sha
        manifest = {
            "provenance": {
                "inputs": {
                    "final_report": {"sha256": "abc123"},
                    "contract": {"sha256": "abc123"},
                }
            }
        }
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        failures = check_manifest_contract_distinct(mpath)
        assert len(failures) >= 1

    def test_check_manifest_contract_distinct_clean(self, tmp_path):
        """Verify clean manifest passes."""
        from tools.implementation_annex_quality_gate import (
            check_manifest_contract_distinct)
        manifest = {
            "provenance": {
                "inputs": {
                    "final_report": {"sha256": "abc123"},
                    "contract": {"sha256": "def456"},
                }
            }
        }
        mpath = tmp_path / "manifest.json"
        mpath.write_text(json.dumps(manifest))
        failures = check_manifest_contract_distinct(mpath)
        assert failures == []


# ── CLI smoke test ────────────────────────────────────────────────────────


class TestCLI:
    def test_quality_gate_cli_help(self):
        result = subprocess.run(
            ["python3", "-m", "tools.implementation_annex_quality_gate",
             "--help"],
            capture_output=True, text=True,
            cwd=TOOLS.parent,
        )
        assert result.returncode == 0
        assert "--annex" in result.stdout

    def test_generator_cli_help(self):
        result = subprocess.run(
            ["python3", "-m", "tools.implementation_annex_generator",
             "--help"],
            capture_output=True, text=True,
            cwd=TOOLS.parent,
        )
        assert result.returncode == 0
        assert "--final-report" in result.stdout

    def test_quality_gate_passes_on_generated_annex(self, tmp_path):
        """Verify the quality gate passes on fresh generation output."""
        from tools.implementation_annex_generator import generate_annex
        result = generate_annex(
            final_report=FIXTURES / "adhd_i_decision_report.md",
            external_calibration=FIXTURES / "adhd_i_external_calibration.md",
            contract=FIXTURES / "adhd_i_decision_context_contract.md",
            output_dir=tmp_path / "annex_cli",
            domain="child_adhd_education",
        )
        verdict = result["quality_verdict"]
        assert verdict["passed"] is True, (
            f"Quality gate failed on generated annex: {verdict['summary']}\n"
            f"Failed checks: {[c for c in verdict['checks'] if not c['passed']]}")


# ── fixture safety checks ─────────────────────────────────────────────────


class TestFixtureSafety:
    def test_no_fffd_in_fixtures(self):
        """Fixture files must not contain U+FFFD."""
        for f in FIXTURES.glob("adhd_i_*.md"):
            text = f.read_text(encoding="utf-8")
            assert "\ufffd" not in text, f"U+FFFD found in {f.name}"

    def test_fixture_decision_report_has_contract_preamble(self):
        text = (FIXTURES / "adhd_i_decision_report.md").read_text()
        assert "decision_context_contract_preamble:start" in text

    def test_fixture_contract_is_separate_file(self):
        """Verify the separate contract fixture exists and differs from report."""
        contract = FIXTURES / "adhd_i_decision_context_contract.md"
        report = FIXTURES / "adhd_i_decision_report.md"
        assert contract.exists(), "Contract fixture missing"
        assert contract.stat().st_size > 100
        # Verify they are different files (different SHA256)
        import hashlib
        c_sha = hashlib.sha256(contract.read_bytes()).hexdigest()
        r_sha = hashlib.sha256(report.read_bytes()).hexdigest()
        assert c_sha != r_sha, "Contract fixture must differ from report"

    def test_fixture_external_calibration_has_keys(self):
        text = (FIXTURES / "adhd_i_external_calibration.md").read_text()
        assert "calibration_verdict" in text
        assert "agreement_points" in text
        assert "disagreement_or_risk_points" in text

    def test_source_files_mtime_unchanged(self):
        """Verify source files exist and aren't being modified by tests."""
        sources = [
            FIXTURES / "adhd_i_decision_report.md",
            FIXTURES / "adhd_i_external_calibration.md",
            FIXTURES / "adhd_i_decision_context_contract.md",
        ]
        for s in sources:
            assert s.exists(), f"Source file missing: {s}"
            assert s.stat().st_size > 100, f"Source too small: {s}"
