import importlib.util
import json
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "hooks", "ai-fail-stats-v2.py")


def load_hook():
    spec = importlib.util.spec_from_file_location("ai_fail_stats_v2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def keyword_book(disabled=None):
    return {
        "version": 2,
        "repeat_warn_threshold": 3,
        "targets": {
            "hermes": ["fuck you hermes", "fuck u hermes", "f u hermes"],
            "agent": ["fuck you agent", "fuck u agent", "f u agent"],
            "opus": ["fuck you opus", "fuck u opus", "f u opus"],
            "codex": ["fuck you codex", "fuck u codex", "f u codex"],
            "claude": ["fuck you claude", "fuck u claude", "fuck you ai", "fuck u ai", "f u ai"],
        },
        "generic_curse": [
            "fuck",
            "fuk",
            "fck",
            "stfu",
            "wtf",
            "bullshit",
            "เหี้ย",
            "เชี่ย",
            "มึง",
            "ควย",
            "สัส",
            "สัด",
            "แม่ง",
            "มั่ว",
        ],
        "jargon_markers": ["ภาษาคน"],
        "disabled": disabled or [],
    }


def categories_for(text, book=None):
    module = load_hook()
    hits = module.detect_hits(text, book or keyword_book())
    return [hit["category"] for hit in hits]


def test_generic_english_curse():
    assert categories_for("Fuck you") == ["curse-generic"]


def test_generic_english_curse_with_other_word():
    assert categories_for("Fuck ass") == ["curse-generic"]


def test_ai_target_keeps_old_category_name():
    assert categories_for("Fuck you AI") == ["ai-fail"]


def test_opus_target_uses_target_category():
    assert categories_for("Fuck you Opus") == ["target:opus"]


def test_hermes_target_keeps_old_category_name():
    assert categories_for("fuck you hermes") == ["hermes-fail"]


def test_generic_thai_curse():
    assert categories_for("เหี้ยจริง ๆ ทำไมพลาดอีก") == ["curse-generic"]


def test_jargon_marker():
    assert categories_for("พูดภาษาคนหน่อย") == ["jargon"]


def test_target_phrase_does_not_count_generic_duplicate():
    assert categories_for("Fuck you Opus") == ["target:opus"]


def test_jargon_can_be_counted_with_one_curse_category():
    assert categories_for("Fuck you Opus พูดภาษาคนหน่อย") == ["target:opus", "jargon"]


def test_plain_prompt_has_no_hits():
    assert categories_for("สวัสดีครับ ช่วยตรวจไฟล์หน่อย") == []


def test_disabled_keyword_is_not_used():
    book = keyword_book(disabled=["badword"])
    book["generic_curse"].append("badword")
    assert categories_for("badword", book) == []


def test_missing_keyword_files_falls_back_without_throwing(tmp_path):
    module = load_hook()
    book = module.load_keywords(
        stats_dir=str(tmp_path),
        fallback_path=os.path.join(str(tmp_path), "missing-curse-keywords.json"),
    )
    assert categories_for("fuck you hermes", book) == ["hermes-fail"]


def test_hook_subprocess_writes_log_to_env_stats_dir(tmp_path):
    stats_dir = tmp_path / "stats"
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    env = os.environ.copy()
    env["AI_FAIL_STATS_DIR"] = str(stats_dir)
    env["HOME"] = str(home_dir)

    proc = subprocess.run(
        [sys.executable, SCRIPT],
        input=json.dumps({"prompt": "Fuck you Opus", "cwd": "/tmp/work"}),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )

    assert proc.returncode == 0
    assert proc.stderr == ""
    assert "target:opus" in proc.stdout

    log_path = stats_dir / "log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["category"] == "target:opus"
    assert row["target"] == "opus"
    assert row["cwd"] == "/tmp/work"

    counts = json.loads((stats_dir / "counts.json").read_text(encoding="utf-8"))
    assert counts["target:opus"] == 1
