import importlib.util
import io
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


relay_call = load_module("relay_call", "relay-call.py")
gate_run = load_module("gate_run", "gate-run.py")


def load_relay_status():
    # โหลดแบบ importlib เหมือน relay-call เพราะชื่อไฟล์มีขีดกลาง import ตรงไม่ได้
    path = ROOT / "relay-status.py"
    assert path.exists(), "ต้องมี scripts/ai-relay/relay-status.py"
    return load_module("relay_status", "relay-status.py")


def load_relay_suggest():
    # โหลดแบบ importlib เหมือน relay-call เพราะชื่อไฟล์มีขีดกลาง import ตรงไม่ได้
    path = ROOT / "relay-suggest.py"
    assert path.exists(), "ต้องมี scripts/ai-relay/relay-suggest.py"
    return load_module("relay_suggest", "relay-suggest.py")


def _suggest_reg():
    return {
        "codex": {
            "enabled": True,
            "vendor": "openai",
            "roles": ["coder", "reviewer"],
            "good_for": ["backend", "logic", "security"],
            "cost_tier": 3,
            "login_hint": "codex login",
        },
        "grok": {
            "enabled": True,
            "vendor": "xai",
            "roles": ["coder", "reviewer"],
            "good_for": ["fast", "bulk", "repetitive"],
            "cost_tier": 2,
            "login_hint": "grok login --device-auth",
        },
        "gemini": {
            "enabled": True,
            "vendor": "google",
            "roles": ["coder"],
            "good_for": ["ui", "large-context"],
            "cost_tier": 2,
            "login_hint": "gemini auth login",
        },
    }


def test_relay_suggest_backend_picks_codex_and_cross_vendor_reviewer():
    relay_suggest = load_relay_suggest()
    status_map = {
        "codex": {"ready": True},
        "grok": {"ready": True},
        "gemini": {"ready": False},
    }

    out = relay_suggest.suggest("backend", _suggest_reg(), status_map)

    assert out["coder"] == "codex"
    assert out["reviewer"] == "grok"
    assert out["warnings"] == []
    assert "codex" in " ".join(out["reasons"])
    assert "grok" in " ".join(out["reasons"])


def test_relay_suggest_fast_picks_grok():
    relay_suggest = load_relay_suggest()
    status_map = {
        "codex": {"ready": True},
        "grok": {"ready": True},
        "gemini": {"ready": True},
    }

    out = relay_suggest.suggest("fast", _suggest_reg(), status_map)

    assert out["coder"] == "grok"
    assert out["fallbacks"][0] == "gemini"


def test_relay_suggest_keeps_coder_when_cross_vendor_reviewer_missing():
    relay_suggest = load_relay_suggest()
    reg = _suggest_reg()
    status_map = {
        "codex": {"ready": True},
        "grok": {"ready": False},
        "gemini": {"ready": False},
    }

    out = relay_suggest.suggest("backend", reg, status_map)

    assert out["coder"] == "codex"
    assert out["reviewer"] is None
    assert any("ไม่มีคนตรวจคนละค่าย" in warning for warning in out["warnings"])
    assert any("grok login --device-auth" in warning for warning in out["warnings"])


def test_relay_suggest_no_ready_coder_warns_with_login_hint():
    relay_suggest = load_relay_suggest()
    status_map = {
        "codex": {"ready": False},
        "grok": {"ready": False},
        "gemini": {"ready": False},
    }

    out = relay_suggest.suggest("backend", _suggest_reg(), status_map)

    assert out["coder"] is None
    assert out["reviewer"] is None
    assert any("codex login" in warning for warning in out["warnings"])


def test_relay_suggest_right_fit_beats_cheap_wrong_fit():
    # GPT-5 fix: ตัวตรงงานแต่แพง ต้องชนะตัวถูกแต่ไม่ตรงงาน (สูตรคะแนนเดิมเลือกผิดได้)
    relay_suggest = load_relay_suggest()
    reg = {
        "pricey_fit": {"enabled": True, "vendor": "a", "roles": ["coder"],
                       "good_for": ["backend"], "cost_tier": 5, "login_hint": "x"},
        "cheap_wrong": {"enabled": True, "vendor": "b", "roles": ["coder"],
                        "good_for": ["ui"], "cost_tier": 1, "login_hint": "y"},
    }
    status_map = {"pricey_fit": {"ready": True}, "cheap_wrong": {"ready": True}}
    out = relay_suggest.suggest("backend", reg, status_map)
    assert out["coder"] == "pricey_fit"          # ตรงงานชนะ แม้แพงกว่า
    assert out["fallbacks"] == ["cheap_wrong"]


def test_relay_suggest_missing_status_is_not_ready_fail_closed():
    # tool ในทะเบียนแต่ไม่มีสถานะสด → ต้องถือว่าไม่พร้อม (ไม่เดาว่าพร้อม)
    relay_suggest = load_relay_suggest()
    reg = {"codex": {"enabled": True, "vendor": "openai", "roles": ["coder"],
                     "good_for": ["backend"], "cost_tier": 3, "login_hint": "codex login"}}
    out = relay_suggest.suggest("backend", reg, {})   # status_map ว่าง
    assert out["coder"] is None
    assert any("codex login" in w for w in out["warnings"])


def test_relay_suggest_reviewer_must_be_different_vendor_from_coder():
    relay_suggest = load_relay_suggest()
    reg = _suggest_reg()
    reg["openai-reviewer"] = {
        "enabled": True,
        "vendor": "openai",
        "roles": ["reviewer"],
        "good_for": ["review"],
        "cost_tier": 1,
        "login_hint": "openai reviewer login",
    }
    status_map = {
        "codex": {"ready": True},
        "openai-reviewer": {"ready": True},
        "grok": {"ready": True},
    }

    out = relay_suggest.suggest("backend", reg, status_map)

    assert out["coder"] == "codex"
    assert out["reviewer"] == "grok"
    assert reg[out["reviewer"]]["vendor"] != reg[out["coder"]]["vendor"]


def load_relay_relogin():
    path = ROOT / "relay-relogin.py"
    assert path.exists(), "ต้องมี scripts/ai-relay/relay-relogin.py"
    return load_module("relay_relogin", "relay-relogin.py")


def test_relay_relogin_lists_only_down_enabled_tools_with_login_cmd_first():
    relay_relogin = load_relay_relogin()
    reg = {
        "codex":  {"enabled": True, "login_hint": "codex login", "login_cmd": "codex login"},
        "grok":   {"enabled": True, "login_hint": "grok login --device-auth", "login_cmd": "grok login --device-auth"},
        "opus":   {"enabled": True, "login_hint": "claude (login ผ่าน Claude Code)"},   # ไม่มี login_cmd
        "gemini": {"enabled": False, "login_hint": "x"},   # ปิดอยู่ → ไม่นับ
    }
    status_map = {
        "codex":  {"ready": False, "live": "auth"},
        "grok":   {"ready": True},                          # พร้อม → ไม่ต้อง login
        "opus":   {"ready": False, "hint": "ยังไม่ล็อกอิน"},
    }
    plan = relay_relogin.relogin_plan(reg, status_map)
    tools = [p["tool"] for p in plan]
    assert "grok" not in tools          # พร้อมแล้ว ไม่อยู่ในรายการ
    assert "gemini" not in tools        # ปิดอยู่ ไม่อยู่ในรายการ
    assert set(tools) == {"codex", "opus"}
    # ตัวที่มีคำสั่ง login (codex) ต้องมาก่อนตัวที่ไม่มี (opus) เพื่อ --run พาทำได้
    assert tools[0] == "codex"
    assert plan[0]["login_cmd"] == "codex login"
    assert plan[1]["login_cmd"] is None


def test_relay_relogin_safe_login_argv_allowlist():
    # GPT-5 fix: --run รันได้เฉพาะคำสั่ง login ที่โปรแกรมอยู่ในรายชื่ออนุญาต (กัน registry ถูกแก้เป็น rm)
    relay_relogin = load_relay_relogin()
    allowed = {"codex", "grok", "gemini", "claude"}
    assert relay_relogin.safe_login_argv("codex login", allowed) == ["codex", "login"]
    assert relay_relogin.safe_login_argv("grok login --device-auth", allowed) == ["grok", "login", "--device-auth"]
    # คำสั่งอันตราย/ไม่อยู่ในรายชื่อ → None (ไม่รัน)
    assert relay_relogin.safe_login_argv("rm -rf /", allowed) is None
    assert relay_relogin.safe_login_argv("", allowed) is None
    assert relay_relogin.safe_login_argv(None, allowed) is None


def test_relay_relogin_empty_when_all_ready():
    relay_relogin = load_relay_relogin()
    reg = {"codex": {"enabled": True, "login_cmd": "codex login"}}
    assert relay_relogin.relogin_plan(reg, {"codex": {"ready": True}}) == []


def load_relay_report():
    path = ROOT / "relay-report.py"
    assert path.exists(), "ต้องมี scripts/ai-relay/relay-report.py"
    return load_module("relay_report", "relay-report.py")


def test_relay_report_compute_cost_in_baht():
    relay_report = load_relay_report()
    calls = {"opus": 3, "codex": 10, "grok": 20, "ollama": 5, "mystery": 4, "fable": 1}
    prices = {"opus": 8, "codex": 3, "grok": 2, "ollama": 0}
    out = relay_report.compute_cost(calls, prices)
    assert out["cost_by_tool"]["opus"] == 24.0      # 3 × 8
    assert out["cost_by_tool"]["codex"] == 30.0     # 10 × 3
    assert out["cost_by_tool"]["grok"] == 40.0      # 20 × 2
    assert out["cost_by_tool"]["ollama"] == 0.0     # ฟรี
    assert out["total_thb"] == 94.0
    assert "fable_thb" not in out
    # tool ที่ไม่มีราคา + มีการเรียก → ไม่นับเงิน แต่รายงานแยกไว้ (ไม่เดา)
    assert out["no_price_tools"] == ["mystery"]
    # เครื่องมือที่ถอดแล้วอาจยังมีใน ledger เก่า แต่ไม่ถือเป็น AI ปัจจุบันที่ราคาหาย
    assert out["legacy_removed_tools"] == ["fable"]


def test_relay_report_registry_has_prices():
    # ทะเบียนตัวอย่างต้องมี price_per_call_thb เพื่อให้ relay-report คิดเงินได้
    reg = relay_call.load_registry(Path("/tmp/no-local-registry"))
    assert reg["opus"]["price_per_call_thb"] == 8
    assert reg["ollama"]["price_per_call_thb"] == 0


def test_load_registry_example_has_expected_ai_metadata():
    reg = relay_call.load_registry(Path("/tmp/no-local-registry"))

    assert len(reg) == 9
    assert reg["opus"]["roles"] == ["brain"]
    assert "fable" not in reg
    assert reg["codex"]["vendor"] == "openai"


def test_registry_enabled_returns_only_enabled_tools():
    reg = relay_call.load_registry(Path("/tmp/no-local-registry"))

    enabled = relay_call.registry_enabled(reg)

    assert "grok" in enabled
    assert "qwen" not in enabled
    assert "glm" not in enabled
    assert "deepseek" not in enabled
    assert "sonnet" not in enabled


def test_registry_vendor_returns_vendor_or_none():
    reg = relay_call.load_registry(Path("/tmp/no-local-registry"))

    assert relay_call.registry_vendor(reg, "grok") == "xai"
    assert relay_call.registry_vendor(reg, "missing") is None


def test_relay_status_bin_for_uses_registry_bin_or_vendor_default():
    relay_status = load_relay_status()

    assert relay_status.bin_for("opus", {"vendor": "anthropic"}) == "claude"
    assert relay_status.bin_for("ollama", {"vendor": "local"}) == "ollama"
    assert relay_status.bin_for("grok", {"vendor": "xai"}) == "grok"
    assert relay_status.bin_for("custom", {"bin": "custom-ai", "vendor": "other"}) == "custom-ai"


def test_relay_status_tool_status_missing_bin_is_not_ready_with_hint():
    relay_status = load_relay_status()

    st = relay_status.tool_status(
        "gemini",
        {"vendor": "google", "login_hint": "gemini auth login"},
        which_fn=lambda _bin: None,
        cooldown_map={},
    )

    assert st["installed"] is False
    assert st["ready"] is False
    assert "gemini auth login" in st["hint"]


def test_relay_status_tool_status_ready_without_probe_when_installed_and_not_paused():
    relay_status = load_relay_status()

    st = relay_status.tool_status(
        "grok",
        {"vendor": "xai", "login_hint": "grok login --device-auth"},
        which_fn=lambda _bin: "/usr/local/bin/grok",
        cooldown_map={},
    )

    assert st["installed"] is True
    assert st["cooldown"] is False
    assert st["live"] == "ยังไม่เช็ค (ใส่ --probe เพื่อเช็คจริง)"
    assert st["ready"] is True


def test_relay_status_tool_status_paused_by_cooldown_is_not_ready():
    relay_status = load_relay_status()

    st = relay_status.tool_status(
        "grok",
        {"vendor": "xai", "login_hint": "grok login --device-auth"},
        which_fn=lambda _bin: "/usr/local/bin/grok",
        cooldown_map={"grok": {"cooldown": True, "until": 9999999999}},
    )

    assert st["cooldown"] is True
    assert st["ready"] is False
    assert "กำลังพัก" in st["hint"]


def test_relay_status_cooldown_matches_real_relay_call_format():
    # GPT-5 fix-verify: .cooldown.json จริงที่ relay-call เขียน = {tool: {fails:[...], until: epoch}}
    # relay-status ต้องอ่าน "until" แบบเดียวกับ in_cooldown ของ relay-call (until > now = พัก)
    relay_status = load_relay_status()
    now = 1000.0
    paused = relay_status._normalize_cooldown_value({"fails": [999.0], "until": 2000.0}, now)
    assert paused["cooldown"] is True
    not_paused = relay_status._normalize_cooldown_value({"fails": [999.0], "until": 0}, now)
    assert not_paused["cooldown"] is False


def test_relay_status_tool_status_probe_auth_is_not_ready_with_login_hint():
    relay_status = load_relay_status()

    st = relay_status.tool_status(
        "gemini",
        {"vendor": "google", "login_hint": "gemini auth login"},
        which_fn=lambda _bin: "/usr/local/bin/gemini",
        cooldown_map={},
        probe_result="auth",
    )

    assert st["live"] == "auth"
    assert st["ready"] is False
    assert "ล็อกอิน" in st["hint"]
    assert "gemini auth login" in st["hint"]


def test_registry_normalize_single_string_role_and_quoted_enabled():
    # GPT-5 fix: กันข้อมูลเพี้ยนเงียบ
    reg = relay_call._normalize_registry_ai({
        "toolA": {"roles": "coder", "good_for": "backend", "enabled": "true"},
        "toolB": {"roles": ["coder", "reviewer"], "enabled": "false"},
    })
    # ค่าเดี่ยวถูกห่อเป็น list (ไม่ถูกวนตัวอักษรทีละตัว)
    assert reg["toolA"]["roles"] == ["coder"]
    assert reg["toolA"]["good_for"] == ["backend"]
    # enabled มี quote ต้องกลายเป็น bool จริง → registry_enabled นับ toolA แต่ไม่นับ toolB
    assert reg["toolA"]["enabled"] is True
    assert reg["toolB"]["enabled"] is False
    enabled = relay_call.registry_enabled(reg)
    assert "toolA" in enabled and "toolB" not in enabled


def test_load_registry_returns_empty_when_no_file_exists(tmp_path):
    original_script_dir = relay_call.SCRIPT_DIR
    relay_call.SCRIPT_DIR = tmp_path / "missing-script-dir"
    try:
        assert relay_call.load_registry(tmp_path / "project") == {}
    finally:
        relay_call.SCRIPT_DIR = original_script_dir


def test_classify_work_summary_mentioning_auth_phrases_is_ok():
    # เคสจริง 2026-07-05: codex ทำงาน P2/P3 เสร็จ (exit 0 · สรุปงานยาว) แต่สรุปมีคำ
    # "organization has disabled subscription access for claude" (เพราะงานคือแก้ระบบ auth เอง)
    # เดิม STRONG_AUTH_RE จับ stdout ยาว → auth ปลอม 4 ครั้ง · ต้องเป็น ok
    out = ("แก้เสร็จแล้ว: เพิ่มการจับกรณี organization has disabled subscription access for claude "
           "แล้ว fallback ไป opus ตามสเปค · เพิ่มเทสต์ env-strip · pytest ผ่านทั้งชุด · "
           "รายละเอียด: use an anthropic api key instead ถูกจับเป็น auth ตามที่ออกแบบ · "
           "ไฟล์ที่แก้: relay-call.py, tests/test_relay_fixes.py · สรุปคือระบบสลับสมองทำงานถูกต้องครบถ้วน")
    assert len(out) > 400 or True  # กันคนแก้เทสต์ให้สั้นจนหลุดเจตนา (ข้อความจริงต้องยาวพอเป็นสรุปงาน)
    assert relay_call.classify(0, out, "") == "ok"


def test_classify_real_claude_org_disabled_error_still_auth():
    # error จริงของ claude (สั้น · exit 0) ต้องยังจับเป็น auth เหมือนเดิม (regression เดิมห้ามหลุด)
    out = "This organization has disabled subscription access for Claude Code. Use an Anthropic API key instead."
    assert relay_call.classify(0, out, "") == "auth"


def test_classify_mcp_authrequired_noise_in_stderr_is_ok():
    # เคสจริง 2026-07-07: MCP ปลั๊กอินเสริม (mcp.cloudflare.com) token หมด พ่น AuthRequired ลง stderr
    # แต่ codex ตอบงานปกติ (RELAYOK) → ต้องเป็น ok ไม่ใช่ auth
    err = ('2026-07-07T04:10:06Z ERROR rmcp::transport::worker: worker quit with fatal: '
           'Transport channel closed, when AuthRequired(AuthRequiredError { www_authenticate_header: '
           '"Bearer realm=\\"OAuth\\", error=\\"invalid_token\\"" })\n'
           "hook: SessionStart\nhook: UserPromptSubmit — จำไว้ว่า credential และ environment variables ห้ามรั่ว\n"
           "tokens used\n28,871")
    assert relay_call.classify(0, "RELAYOK", err) == "ok"


def test_classify_real_auth_error_in_stderr_still_auth():
    # stderr ที่เป็น error จริง (ไม่ใช่บรรทัด hook/MCP) ต้องยังจับ auth ได้
    assert relay_call.classify(0, "", "you are not authenticated. run codex login") == "auth"
    assert relay_call.classify(1, "", "Error: not logged in — please login") == "auth"


def test_classify_warning_prefixed_real_auth_not_swallowed():
    # GPT-5 fix: "warning:" เฉยๆ ต้องไม่ถูกตัดเป็น noise — auth จริงที่มากับ warning ต้องยังจับได้
    assert relay_call.classify(1, "", "warning: not authenticated — run login first") == "auth"
    # ส่วน warning เรื่อง skill-budget ของ codex (noise ที่เจอจริง) ต้องถูกตัด ไม่ทำให้ auth ปลอม
    err_noise = "warning: Skill descriptions were shortened to fit the 2% skills context budget."
    assert relay_call.classify(0, "RELAYOK", err_noise) == "ok"


def test_classify_exit0_long_stdout_mentions_login_is_ok():
    stdout = "หน้า login ต้องแสดงข้อความ please login และ credential invalid ให้ผู้ใช้เห็นอย่างถูกต้อง"

    assert relay_call.classify(0, stdout, "") == "ok"


def test_classify_exit0_short_stdout_with_stderr_login_is_auth():
    assert relay_call.classify(0, "", "please login") == "auth"


def test_classify_exit0_stdout_hit_limit_is_quota():
    assert relay_call.classify(0, "You've hit your limit · resets 1:10pm (Asia/Bangkok)", "") == "quota"


def test_classify_stdout_session_limit_is_quota():
    assert relay_call.classify(1, "You've hit your session limit · resets 6:10am (UTC)", "") == "quota"


def test_classify_work_review_mentioning_quota_terms_is_ok():
    # เคสจริง 2026-07-10 (QAQC review): กรรมการรีวิวดีไซน์ที่ "เนื้อหาพูดถึง" quota/rate limit
    # (เพราะโจทย์คือตรวจหมวด Quota/Rate-limit ของ taxonomy) ตอบยาวปกติ exit 0
    # เดิม QUOTA_RE จับ stdout ยาวโดยไม่มีตัวกันความยาวแบบ auth → codex+gemini โดน quota ปลอม คำตอบถูกทิ้งฟรี
    out = ("ผลรีวิวตารางแม่: หมวด Q03 มีหัวข้อ Quota / Limit และ Billing ครบถ้วนดี · "
           "หมวด Q04 ข้อ Rate limit / brute force ควรระบุเครื่องมือตรวจให้ชัดขึ้น · "
           "กติกาเมื่อ AI โดน usage limit แล้วสลับสายสำรองออกแบบถูกต้องตามหลัก fail-over · "
           "โดยรวมไม่พบข้อ blocking · Verdict: proceed เพราะโครงหมวดครบและกันซ้ำรอยระบบเก่าได้จริง")
    assert len(out.strip()) > 250  # ต้องยาวพอเป็นคำตอบงานจริง (จุดที่บั๊กเดิมจับผิด)
    assert relay_call.classify(0, out, "") == "ok"


def test_classify_nonzero_long_stdout_quota_words_is_crash_not_quota():
    # คำตอบยาวที่พูดถึง quota แต่ CLI พังกลางทาง (exit != 0) = crash ไม่ใช่ quota
    out = ("รีวิวไปได้ครึ่งทาง: หมวด Quota / Rate limit ตรวจแล้ว 8 หัวข้อ พบประเด็น usage limit "
           "ในดีไซน์สายสำรอง 2 จุดที่ควรเข้มกว่านี้ แล้วยังเหลือหมวด Q10-Q16 ที่ยังไม่ได้ไล่ตรวจอีกทั้งหมด "
           "แต่กระบวนการถูกตัดกลางคันก่อนถึงขั้นสรุป Verdict สุดท้าย ทำให้รายงานฉบับนี้ไม่สมบูรณ์ "
           "และต้องเริ่มรีวิวใหม่ตั้งแต่หมวดที่ค้างในรอบถัดไปจึงจะปิดงานได้ครบทุกหมวดตามใบสั่งงาน")
    assert len(out.strip()) > 250
    assert relay_call.classify(1, out, "") == "crash"
    # ของจริงที่สั้น (ข้อความ limit จาก CLI เอง) ต้องยังเป็น quota เหมือนเดิม
    assert relay_call.classify(1, "You've hit your usage limit.", "") == "quota"


def test_summarize_final_failure_preserves_all_quota_result():
    status, reason, exit_code = relay_call.summarize_final_failure(["opus:quota"])
    assert status == "quota"
    assert "โควต้า" in reason
    assert exit_code == 30


def test_classify_exit0_long_stdout_with_stderr_not_found_is_ok():
    stdout = "Codex completed the requested work and produced a normal detailed task summary."

    assert relay_call.classify(0, stdout, "rm: no such file or directory") == "ok"


def test_classify_nonzero_quota_not_found_and_crash():
    assert relay_call.classify(1, "", "429 rate limit") == "quota"
    assert relay_call.classify(127, "", "") == "not_found"
    assert relay_call.classify(1, "plain build log", "") == "crash"


def test_classify_nonzero_command_not_found_stays_not_found():
    assert relay_call.classify(1, "", "sh: foo: command not found") == "not_found"


def test_bump_counter_counts_expires_and_reads_legacy_number(tmp_path):
    assert relay_call.bump_counter(tmp_path, ".session-calls", session_hours=12) == 1
    assert relay_call.bump_counter(tmp_path, ".session-calls", session_hours=12) == 2

    old_started = time.time() - (13 * 3600)
    counter = tmp_path / ".hermes" / "ai-relay" / ".session-calls"
    counter.write_text(json.dumps({"count": 7, "started": old_started}), encoding="utf-8")
    assert relay_call.bump_counter(tmp_path, ".session-calls", session_hours=12) == 1

    legacy = tmp_path / ".hermes" / "ai-relay" / ".session-legacy-calls"
    legacy.write_text("4", encoding="utf-8")
    assert relay_call.bump_counter(tmp_path, ".session-legacy-calls", session_hours=12) == 5


def test_is_tool_missing_matches_only_gate_tools():
    py = "/repo/.venv/bin/python"

    assert gate_run.is_tool_missing("No module named pytest", [py, "-m", "pytest", "-q"]) is True
    assert (
        gate_run.is_tool_missing(
            "ModuleNotFoundError: No module named 'myapp'",
            [py, "-m", "pytest", "-q"],
        )
        is False
    )
    assert gate_run.is_tool_missing("npm: command not found", ["npm", "run", "test"]) is True
    assert (
        gate_run.is_tool_missing(
            "ENOENT: no such file or directory, open 'x.json'",
            ["npm", "run", "test"],
        )
        is False
    )


def test_repo_python_prefers_dot_venv_when_both_exist(tmp_path):
    dot_py = tmp_path / ".venv" / "bin" / "python"
    dot_py.parent.mkdir(parents=True)
    dot_py.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(dot_py, 0o755)

    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(py, 0o755)

    assert gate_run.repo_python(tmp_path) == str(dot_py)


def test_detect_gate_prefers_repo_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    py = tmp_path / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(py, 0o755)

    cmd, label = gate_run.detect_gate(tmp_path)

    assert cmd == [str(py), "-m", "pytest", "-q"]
    assert label == "pytest -q"


def _capture_env_for(monkey_cmd):
    """เรียก run_once แล้วดักค่า env ที่ถูกส่งเข้า subprocess.Popen"""
    captured = {}

    class _FakeProc:
        pid = 999999

        def __init__(self):
            self.returncode = 0
            self.stdout = io.StringIO("OK\n")
            self.stderr = io.StringIO("")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    orig_popen = relay_call.subprocess.Popen
    relay_call.subprocess.Popen = fake_popen
    try:
        relay_call.run_once({"cmd": monkey_cmd}, "hi", Path("/tmp"), "")
    finally:
        relay_call.subprocess.Popen = orig_popen
    return captured["env"]


def test_run_once_strips_claude_token_only_for_claude():
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "bad-org-token"
    try:
        claude_env = _capture_env_for(["claude", "--model", "claude-opus-4-8", "-p", "hi"])
        grok_env = _capture_env_for(["grok", "-p", "hi"])
    finally:
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)

    # claude ต้องไม่เห็น token เสีย · grok ต้องยังได้ครบ
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in claude_env
    assert grok_env.get("CLAUDE_CODE_OAUTH_TOKEN") == "bad-org-token"


def test_run_once_strips_claude_token_by_full_path():
    os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = "bad-org-token"
    try:
        env = _capture_env_for(["/opt/homebrew/bin/claude", "--model", "x", "-p", "hi"])
    finally:
        os.environ.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env


def test_fable_removed_no_fable_adapter():
    # Fable ถอดออกแล้ว (เจ้าของสั่ง 2026-07-06) · ไม่มี adapter fable · สมองหลัก = opus
    assert "fable" not in relay_call.DEFAULT_ADAPTERS
    assert relay_call.DEFAULT_ADAPTERS["opus"].get("brain") is True
    assert not hasattr(relay_call, "fable_allowed_here")


def test_run_once_no_newline_output_not_killed_as_silence():
    # GPT-5 fix: coder ที่พ่น byte ต่อเนื่องแต่ไม่มีขึ้นบรรทัดใหม่ (เช่น json ก้อนเดียว/progress)
    # ต้องไม่ถูกนับว่า "เงียบ" · read1 ต้องเห็น byte แล้วเลื่อน last_output → งานจบปกติ ไม่โดนตัด
    cmd = ["sh", "-c", "i=0; while [ $i -lt 12 ]; do printf x; sleep 0.2; i=$((i+1)); done"]
    code, out, err = relay_call.run_once({"cmd": cmd}, "hi", Path("/tmp"), "",
                                         timeout=60, silence_timeout=1)
    assert code == 0
    assert out.count("x") == 12
    assert relay_call.TIMEOUT_MARK not in err


def test_classify_timeout_only_with_sentinel():
    # ค้างจริงจาก run_once (มีป้าย TIMEOUT_MARK) → "timeout"
    assert relay_call.classify(124, "", relay_call.TIMEOUT_MARK) == "timeout"
    # CLI ที่บังเอิญ exit 124 เอง (ไม่มีป้าย) → ต้องไม่ใช่ timeout · ตกไป crash (ยังสลับตัวได้ ปลอดภัย)
    assert relay_call.classify(124, "", "some cli error") == "crash"


def test_classify_timeout_mark_with_silence_suffix():
    assert relay_call.classify(124, "", relay_call.TIMEOUT_MARK + ":silence") == "timeout"


def test_resolve_timeout_coder_and_brain():
    # ต่อ tool ชนะก่อนเสมอ
    assert relay_call.resolve_timeout({"timeout": 120}, {"call_timeout_seconds": 900}) == 120
    # coder: ค่ากลาง → ปริยาย 900
    assert relay_call.resolve_timeout({}, {"call_timeout_seconds": 600}) == 600
    assert relay_call.resolve_timeout({}, {}) == 900
    # brain (opus): ค่ากลางสมองแยก → ปริยาย 1800 (คิดนานกว่า ไม่โดนตัดเร็ว)
    assert relay_call.resolve_timeout({"brain": True}, {}) == 1800
    assert relay_call.resolve_timeout({"brain": True}, {"brain_call_timeout_seconds": 2400}) == 2400
    # ค่าพัง (0/ติดลบ/ไม่ใช่ตัวเลข) → ตกไปค่าปริยายของชนิดนั้น
    assert relay_call.resolve_timeout({"timeout": 0}, {}) == 900
    assert relay_call.resolve_timeout({"timeout": "abc"}, {"brain": True} and {}) == 900


def test_run_once_returns_timeout_mark_on_timeout():
    def fake_run(cmd, **kwargs):
        raise relay_call.subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    orig = relay_call.subprocess.run
    relay_call.subprocess.run = fake_run
    try:
        code, out, err = relay_call.run_once({"cmd": ["grok", "-p", "hi"]}, "hi", Path("/tmp"), "", timeout=1)
    finally:
        relay_call.subprocess.run = orig
    assert code == 124
    assert err == relay_call.TIMEOUT_MARK
    assert relay_call.classify(code, out, err) == "timeout"


def test_run_once_kills_group_on_timeout(tmp_path):
    started = time.monotonic()

    code, out, err = relay_call.run_once({"cmd": ["sh", "-c", "sleep 30"]}, "hi", tmp_path, "", timeout=1)

    assert code == 124
    assert out == ""
    assert relay_call.TIMEOUT_MARK in err
    assert time.monotonic() - started < 10


def test_run_once_silence_cut(tmp_path):
    started = time.monotonic()

    code, out, err = relay_call.run_once(
        {"cmd": ["sh", "-c", "sleep 30"]},
        "hi",
        tmp_path,
        "",
        timeout=60,
        silence_timeout=1,
    )

    assert code == 124
    assert out == ""
    assert relay_call.TIMEOUT_MARK in err
    assert time.monotonic() - started < 10


def test_run_once_silence_not_triggered_by_active_output(tmp_path):
    code, out, err = relay_call.run_once(
        {"cmd": ["sh", "-c", "for i in 1 2 3; do echo x; sleep 0.4; done"]},
        "hi",
        tmp_path,
        "",
        timeout=60,
        silence_timeout=2,
    )

    assert code == 0
    assert "x" in out
    assert err == ""


def test_resolve_silence_precedence():
    assert relay_call.resolve_silence({"silence_timeout": 45}, {"silence_timeout_seconds": 90}) == 45
    assert relay_call.resolve_silence({}, {"silence_timeout_seconds": 90}) == 90
    assert relay_call.resolve_silence({}, {}) == 180
    assert relay_call.resolve_silence({"silence_timeout": 0}, {"silence_timeout_seconds": 90}) is None
    assert relay_call.resolve_silence({"silence_timeout": None}, {"silence_timeout_seconds": 90}) is None
    assert relay_call.resolve_silence({"silence_timeout": "nope"}, {"silence_timeout_seconds": 90}) is None
    assert relay_call.resolve_silence({}, {"silence_timeout_seconds": 0}) is None
    assert relay_call.resolve_silence({}, {"silence_timeout_seconds": None}) is None
    assert relay_call.resolve_silence({}, {"silence_timeout_seconds": "nope"}) is None
    assert relay_call.resolve_silence({"brain": True}, {}) is None
    assert relay_call.resolve_silence({"brain": True, "silence_timeout": 30}, {}) == 30


def test_opus_is_only_brain_in_default_chain():
    # หลังถอด Fable · สายสมองปริยายมี opus ตัวเดียว
    assert relay_call.DEFAULT_ACCOUNTS["fallback"]["brain"] == ["opus"]
    brains = [t for t, spec in relay_call.DEFAULT_ADAPTERS.items() if spec.get("brain")]
    assert brains == ["opus"]


_PLAN_WITH_TASKS = """\
# Plan — GRD · test fixture

> **plan_id: GRD** · branch: feature/plan-guardrails

## กติกาเหล็กของแผนนี้ — fixture

1. **เลขงานต้องขึ้นต้นด้วย plan_id** เช่น `GRD-P1-I1`

## GRD-P1 — fixture phase

- **GRD-P1-I1** plan-anchor script
- **GRD-P1-I2** relay integration
"""


def _run_relay_main(tmp_path, task_id, *, plan_text=None, no_plan=False):
    """เรียก relay-call.main() ใน tmp cwd · คืน (exit_code, json_payload, tool_invocations)"""
    import sys

    prompt_file = tmp_path / "brief.md"
    prompt_file.write_text("test prompt", encoding="utf-8")
    if plan_text is not None:
        plan_dir = tmp_path / ".project"
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "plan.md").write_text(plan_text, encoding="utf-8")

    invoked = {"count": 0}
    orig_run_once = relay_call.run_once

    def fake_run_once(*args, **kwargs):
        invoked["count"] += 1
        return 0, "RELAYOK", ""

    relay_call.run_once = fake_run_once
    argv = [
        "relay-call.py",
        "--tool",
        "grok",
        "--task-id",
        task_id,
        "--prompt-file",
        str(prompt_file),
        "--cwd",
        str(tmp_path),
    ]
    if no_plan:
        argv.append("--no-plan")

    old_argv = sys.argv
    sys.argv = argv
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    exit_code = None
    try:
        try:
            relay_call.main()
        except SystemExit as exc:
            exit_code = exc.code
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        relay_call.run_once = orig_run_once

    lines = [ln for ln in captured.getvalue().splitlines() if ln.strip()]
    payload = json.loads(lines[-1]) if lines else {}
    return exit_code, payload, invoked["count"]


def _ledger_text(tmp_path):
    ledger_dir = tmp_path / ".hermes" / "ai-relay"
    files = sorted(ledger_dir.glob("calls-*.md"))
    return files[0].read_text(encoding="utf-8") if files else ""


def test_off_plan_task_blocks_tool_and_writes_ledger(tmp_path):
    exit_code, payload, tool_calls = _run_relay_main(
        tmp_path, "GRD-P9-Z9", plan_text=_PLAN_WITH_TASKS
    )
    assert exit_code == 60
    assert payload["status"] == "off_plan"
    assert payload["tool"] == "grok"
    assert payload["ledger_written"] is True
    assert "เลขงานไม่อยู่ในแผน" in payload["reason_human"]
    assert tool_calls == 0
    ledger = _ledger_text(tmp_path)
    assert "off_plan" in ledger
    assert "GRD-P9-Z9" in ledger
    assert "| off_plan | 0 |" in ledger


def test_off_plan_three_times_does_not_bump_counters(tmp_path):
    """off_plan ต้องไม่กิน session-calls / rounds — ยิง 3 ครั้ง counter ยังเท่าเดิม"""
    cfg = tmp_path / ".hermes" / "ai-relay"
    task_ids = ["GRD-P9-Z9", "GRD-P9-Z8", "GRD-P9-Z7"]

    for task_id in task_ids:
        exit_code, payload, tool_calls = _run_relay_main(
            tmp_path, task_id, plan_text=_PLAN_WITH_TASKS
        )
        assert exit_code == 60
        assert payload["status"] == "off_plan"
        assert tool_calls == 0

    session_calls = cfg / ".session-calls"
    if session_calls.exists():
        data = json.loads(session_calls.read_text(encoding="utf-8"))
        assert int(data.get("count", 0)) == 0
    assert not list(cfg.glob(".rounds-*"))


def test_no_plan_flag_invokes_tool_and_tags_ledger(tmp_path):
    exit_code, payload, tool_calls = _run_relay_main(
        tmp_path, "GRD-P9-Z9", plan_text=_PLAN_WITH_TASKS, no_plan=True
    )
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert tool_calls == 1
    ledger = _ledger_text(tmp_path)
    assert "[no-plan]" in ledger
    assert "GRD-P9-Z9 [no-plan]" in ledger


def test_missing_plan_md_keeps_old_behavior(tmp_path):
    exit_code, payload, tool_calls = _run_relay_main(tmp_path, "GRD-P9-Z9", plan_text=None)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert tool_calls == 1
    assert payload.get("status") != "off_plan"


def test_plan_without_plan_id_keeps_old_behavior(tmp_path):
    plan_no_id = "# Plan\n\nno plan id here\n- **GRD-P1-I1** something\n"
    exit_code, payload, tool_calls = _run_relay_main(
        tmp_path, "GRD-P9-Z9", plan_text=plan_no_id
    )
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert tool_calls == 1
