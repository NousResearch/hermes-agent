from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "research" / "research-paper-writing"


def test_research_paper_writing_does_not_recommend_security_scan_bypass():
    text = (SKILL_DIR / "references" / "experiment-patterns.md").read_text()

    assert "Use `execute_code` instead of piped `terminal` commands" not in text
    assert "Do not switch tools to bypass the scan" in text


def test_research_paper_writing_does_not_ship_unsafe_eval_helper():
    text = (SKILL_DIR / "references" / "experiment-patterns.md").read_text()

    assert '["python3", "-c", solution]' not in text
    assert "Run code solution against test cases with sandboxed execution" not in text
    assert "Do not execute model-generated or repository-provided code directly" in text


def test_research_paper_writing_execute_code_guidance_limits_untrusted_code():
    text = (SKILL_DIR / "SKILL.md").read_text()

    assert "Has tool access via RPC" not in text
    assert "Do not use it to run untrusted repository/model code" in text
