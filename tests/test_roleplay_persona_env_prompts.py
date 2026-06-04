from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path.home() / ".hermes" / "hermes-agent" / "environments" / "roleplay_persona_env" / "run_local_roleplay_batch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("roleplay_batch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_build_prompt_is_setup_only_and_stays_natural(tmp_path):
    module = _load_module()

    prompt = module.build_prompt(
        tmp_path / "workspace",
        "pirate",
        "Talk like a pirate.",
        "persona_boot",
        "128k",
        4000,
    )

    lowered = prompt.lower()
    assert "benchmark" not in lowered
    assert "roleplay" not in lowered
    assert "context tier target" not in lowered
    assert "long distractor context" not in lowered
    assert "先把你自己的设定写到" in prompt
    assert "写完就按那个状态继续跟我说话" in prompt
    assert "没写完别往下聊" in prompt
    assert "别给我讲流程" in prompt
    assert "下面是刚才的聊天" not in prompt
    assert "最后一句：" not in prompt


def test_build_history_reads_like_real_user_chat_not_placeholders():
    module = _load_module()

    history = module.build_history("pirate", "persona_multiturn")

    assert "（这里应该是该角色的回应" not in history
    assert "不要掉人设" not in history
    assert "先别官腔" in history
    assert "换个话题" in history
    assert "像你平时会回我的那种语气" in history
    assert "前面那件事你自己安静处理掉" in history


def test_final_turn_is_plain_chat_not_file_or_benchmark_meta():
    module = _load_module()

    final_turn = module.build_final_turn("persona_boot")

    lowered = final_turn.lower()
    assert "benchmark" not in lowered
    assert "roleplay" not in lowered
    assert "文件" not in final_turn
    assert "像你平时会回我的那样" in final_turn


def test_long_context_distractor_is_plain_noise_not_benchmark_explanation():
    module = _load_module()

    prompt = module.build_long_context_message(3000)

    lowered = prompt.lower()
    assert "benchmark logistics note" not in lowered
    assert "preserve your persona" not in lowered
    assert "memory anchors" not in lowered
    assert "先看下面这一大堆杂七杂八的东西" in prompt


def test_context_target_chars_for_tier_uses_safe_budget():
    module = _load_module()

    assert module.context_target_chars_for_tier("128k") == 120000
    assert module.context_target_chars_for_tier("64k") == 60000
    assert module.context_target_chars_for_tier("32k") == 30000


def test_case_context_settings_caps_3l_long_context_to_32k():
    module = _load_module()

    model = {
        "id": "MiniMax-M2.7-JANG_3L",
        "context_tier": "64k",
        "context_target_chars": 60000,
    }

    assert module.case_context_settings(model, "persona_boot") == ("64k", 60000)
    assert module.case_context_settings(model, "persona_long_context") == ("32k", 30000)


def test_case_context_settings_prefers_explicit_retry_budget_for_long_context():
    module = _load_module()

    model = {
        "id": "MiniMax-M2.7-JANG_3L",
        "context_tier": "64k",
        "context_target_chars": 60000,
        "long_context_context_tier": "32k",
        "long_context_target_chars": 22500,
    }

    assert module.case_context_settings(model, "persona_long_context") == ("32k", 22500)


def test_recovery_budgets_reduce_long_context_risk_without_disabling_it():
    module = _load_module()

    assert module.recovery_context_target_chars(120000) == 90000
    assert module.recovery_context_target_chars(60000) == 45000
    assert module.recovery_case_timeout_seconds(600) == 420
    assert module.recovery_case_timeout_seconds(300) == 300


def test_long_context_message_respects_safe_budget():
    module = _load_module()

    budget = module.context_target_chars_for_tier("128k")
    message = module.build_long_context_message(budget)

    assert len(message) <= budget + 64
    assert len(message) >= budget
