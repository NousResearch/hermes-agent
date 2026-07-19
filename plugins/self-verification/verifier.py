"""Verification engine for the self-verification plugin.

Provides factual accuracy, logic consistency, and output completeness
verification via a dedicated Verifier model. Also implements confidence
scoring (0-100), adversarial self-refutation, and internationalized
footnote formatting.

Architecture:
  - Layer 1: Factual accuracy + Logic consistency + Output completeness
  - Layer 2: Confidence scoring (0-100 continuous)
  - Layer 3: Self-refute (adversarial disprove)
  - Layer 4: Auto-fix retry loop (max 3, driven by __init__.py)

References:
  - Claude Code code-review: 0-100 scoring, threshold 80
  - Claude Code security-guidance: Investigate -> Self-Refute pattern
  - VMAO (AWS+HSBC ICLR 2026): sub-goal coverage checking
  - LLM-as-a-Verifier (arXiv 2607.05391): continuous scoring
  - BRSCP ECG: Evidence Capture Gate
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# i18n footnote text
# ---------------------------------------------------------------------------

FOOTNOTE_I18N: dict[str, dict[str, str]] = {
    "zh": {
        "title": "⚠️ Self-Verification 验证结果",
        "retry_hint": "💡 回复「修正」触发修正",
        "pass": "✅ 验证通过",
        "warn": "⚠️ 需关注",
        "fail": "❌ 不通过",
        "threshold_note": "（低于置信度阈值，已过滤）",
        "confidence_label": "置信度",
        "self_refute_survived": "对抗验证：问题成立",
        "self_refute_refuted": "对抗验证：误报已排除",
        "claims_label": "声明",
        "contradictions_label": "矛盾",
        "completeness_label": "完整性",
        "missing_goals": "遗漏子目标",
        "partial_goals": "部分覆盖子目标",
    },
    "en": {
        "title": "⚠️ Self-Verification Results",
        "retry_hint": "💡 Reply 'fix' to correct",
        "pass": "✅ Verification passed",
        "warn": "⚠️ Attention needed",
        "fail": "❌ Failed",
        "threshold_note": "(below confidence threshold, filtered)",
        "confidence_label": "Confidence",
        "self_refute_survived": "Self-refute: issue confirmed",
        "self_refute_refuted": "Self-refute: false positive dismissed",
        "claims_label": "Claims",
        "contradictions_label": "Contradictions",
        "completeness_label": "Completeness",
        "missing_goals": "Missing sub-goals",
        "partial_goals": "Partial sub-goals",
    },
}


def _get_i18n(lang: str = "zh") -> dict[str, str]:
    """Return the i18n dict for the given language, falling back to zh."""
    return FOOTNOTE_I18N.get(lang, FOOTNOTE_I18N["zh"])


# ---------------------------------------------------------------------------
# Prompt templates (Chinese, inherited from existing verifier.py)
# ---------------------------------------------------------------------------

FACTUAL_ACCURACY_PROMPT = """\
你是验证模型。审查以下 AI 回复，标记可能不准确的事实声明。简洁输出。

待审查回复：
---
{response}
---

{search_context}

规则：
1. 仅标记断言具体事实的声明（统计数据、日期、价格、API行为）
2. 风险等级：low（次要）、medium（用户可见）、high（业务关键）
3. 标注是否可验证、是否有来源
4. 无事实声明则返回空列表，verdict="pass"
5. 保持精简，note 不超过一句话
6. 不要标记AI模型名/版本号——你的训练数据可能不包含最新模型名，这不代表它们不存在
7. 不要仅因"数据无来源"就标记为高风险——只有当数据本身看起来不合理时才标记
8. 如果搜索参考信息与回复中的声明一致，判定为pass；如果冲突，标记并说明

仅输出 JSON：
{{"claims":[{{"text":"声明","risk_level":"low|medium|high","verifiable":true,"has_source":false,"note":"原因"}}],"verdict":"pass|warn|fail"}}
"""

LOGIC_CONSISTENCY_PROMPT = """\
你是验证模型。审查以下 AI 回复中的逻辑一致性问题，仅关注矛盾，不评价写作质量。简洁输出。

待审查回复：
---
{response}
---

规则：
1. 矛盾 = 声明A与声明B冲突
2. 数字不一致（如"3个项目"但列了4个）
3. 循环推理或无根据结论
4. 不要标记观点差异或风格选择
5. 无矛盾则返回空列表，verdict="pass"
6. 保持精简，note 不超过一句话

仅输出 JSON：
{{"contradictions":[{{"location":"位置","statement_a":"声明A","statement_b":"冲突声明B","severity":"critical|minor"}}],"verdict":"pass|warn|fail"}}
"""

COMPLETENESS_PROMPT = """\
你是验证模型。审查 AI 回复是否完整覆盖了用户的所有子目标。简洁输出。

用户原始请求：
---
{user_message}
---

AI 回复：
---
{response}
---

规则：
1. 从用户请求中提取子目标（按"和/且/另外/同时/还/然后/以及"等拆分）
2. 逐项检查每个子目标在回复中是否被处理
3. status: addressed（已处理）/ partial（部分处理）/ missing（遗漏）
4. 无子目标或全部 addressed 则 verdict="pass"
5. 有 missing 则 verdict="fail"，有 partial 则 verdict="warn"
6. 保持精简，note 不超过一句话

仅输出 JSON：
{{"sub_goals":[{{"goal":"子目标","status":"addressed|partial|missing","note":"说明"}}],"completeness_score":0,"verdict":"pass|warn|fail"}}
"""

# ---------------------------------------------------------------------------
# Risk signal detection
# ---------------------------------------------------------------------------

_RISK_PATTERNS = [
    r"\d+\.?\d*\s*%",
    r"\d+\.?\d*\s*(?:million|billion|thousand|万|亿)",
    r"\b(?:19|20)\d{2}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d",
    r"v?\d+\.\d+\.\d+",
    r"https?://",
    r"\b[\w-]+\.(?:com|org|net|io|dev|py|js|ts)\b",
    r'"\s*[^"]{20,}\s*"',
    r"\w+\.\w+\([^)]*\)",
    r"[¥$€£]\s*\d",
    r"\d+\s*(?:元|块钱|美元)",
]


def _has_risk_signals(text: str) -> bool:
    """Quick check: does the response contain patterns worth verifying?"""
    for pattern in _RISK_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


# ---------------------------------------------------------------------------
# Verifier model config and calling
# ---------------------------------------------------------------------------


def _get_verifier_config() -> dict[str, Any]:
    """Read Verifier model config from env vars and config.yaml."""
    config: dict[str, Any] = {
        "primary_model": "deepseek-v4-flash",
        "primary_provider": "deepseek",
        "fallback_model": "qwen3.6-plus",
        "fallback_provider": "zai",
        "timeout": 20,
    }

    env_model = os.environ.get("HERMES_VERIFIER_MODEL")
    if env_model:
        config["primary_model"] = env_model
    env_provider = os.environ.get("HERMES_VERIFIER_PROVIDER")
    if env_provider:
        config["primary_provider"] = env_provider

    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        sv_cfg = (cfg.get("agent") or {}).get("self_verification") or {}
        if isinstance(sv_cfg, dict):
            if sv_cfg.get("verifier_model"):
                config["primary_model"] = sv_cfg["verifier_model"]
            if sv_cfg.get("verifier_provider"):
                config["primary_provider"] = sv_cfg["verifier_provider"]
            if sv_cfg.get("fallback_model"):
                config["fallback_model"] = sv_cfg["fallback_model"]
            if sv_cfg.get("fallback_provider"):
                config["fallback_provider"] = sv_cfg["fallback_provider"]
            if sv_cfg.get("verifier_timeout"):
                config["timeout"] = sv_cfg["verifier_timeout"]
    except Exception:
        pass

    return config


def _call_verifier(prompt: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Call the Verifier model with a prompt and parse JSON output."""
    from agent.plugin_llm import PluginLlm

    plugin_llm = PluginLlm(plugin_id="self-verification")
    timeout = config.get("timeout", 20)

    models_to_try = [
        (config["primary_provider"], config["primary_model"]),
        (config["fallback_provider"], config["fallback_model"]),
    ]

    for provider, model in models_to_try:
        try:
            result = plugin_llm.complete(
                messages=[{"role": "user", "content": prompt}],
                provider=provider,
                model=model,
                timeout=timeout,
                purpose="self-verification",
            )
            result_text = result.text if hasattr(result, "text") else str(result)
            if not result_text:
                logger.debug("self-verification: %s/%s returned empty", provider, model)
                continue

            parsed = _parse_json_response(result_text)
            if parsed is not None:
                return parsed

            logger.debug(
                "self-verification: %s/%s returned unparseable JSON", provider, model
            )
        except Exception as e:
            logger.debug(
                "self-verification: %s/%s failed: %s, trying fallback",
                provider,
                model,
                e,
            )
            continue

    return None


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse JSON from Verifier model output, handling markdown fences."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    pass
                break

    logger.debug("self-verification: could not parse Verifier JSON output")
    return None


# ---------------------------------------------------------------------------
# Web search for evidence
# ---------------------------------------------------------------------------


def _get_last_user_message() -> str | None:
    """Read the most recent user message from the session DB."""
    try:
        import sqlite3

        db_path = os.path.join(
            os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
            "state.db",
        )
        if not os.path.exists(db_path):
            return None

        conn = sqlite3.connect(db_path)
        cur = conn.execute(
            "SELECT content FROM messages WHERE role='user' "
            "ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()

        if row and row[0]:
            return row[0]
    except Exception as e:
        logger.debug("self-verification: failed to read user message: %s", e)

    return None


def _extract_search_queries(response: str, max_queries: int = 3) -> list[str]:
    """Extract key verifiable claims from the response as search queries."""
    queries = []

    for m in re.finditer(
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*[:：]?\s*(\d+\.?\d*\s*(?:million|billion|thousand|万|亿|M|K|\+))",
        response,
    ):
        name = m.group(1).strip()
        number = m.group(2).strip()
        queries.append(f"{name} {number}")

    for m in re.finditer(
        r"\$(\d+\.?\d*)\s*(?:billion|B|million|M)\s*(?:ARR|revenue)",
        response,
        re.IGNORECASE,
    ):
        queries.append(f"ARR revenue ${m.group(1)} billion")

    for m in re.finditer(
        r"(\d+\.?\d*K?)\s*GitHub\s*stars", response, re.IGNORECASE
    ):
        queries.append(f"GitHub stars {m.group(1)}")

    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
        if len(unique) >= max_queries:
            break

    return unique


def _tavily_search(query: str, api_key: str, max_results: int = 3) -> str:
    """Search via Tavily REST API."""
    import urllib.request
    import urllib.error

    url = "https://api.tavily.com/search"
    payload = json.dumps(
        {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        parts = []
        answer = data.get("answer")
        if answer:
            parts.append(f"摘要: {answer}")

        for r in data.get("results", [])[:max_results]:
            title = r.get("title", "")
            snippet = r.get("content", "")[:200]
            if title and snippet:
                parts.append(f"• {title}: {snippet}")

        return "\n".join(parts) if parts else ""

    except Exception as e:
        logger.debug("self-verification: Tavily search failed for '%s': %s", query, e)
        return ""


def _exa_search(query: str, api_key: str, max_results: int = 3) -> str:
    """Search via Exa REST API."""
    import urllib.request

    url = "https://api.exa.ai/search"
    payload = json.dumps(
        {
            "query": query,
            "numResults": max_results,
            "type": "auto",
            "contents": {"text": {"maxCharacters": 300}},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        parts = []
        for r in data.get("results", [])[:max_results]:
            title = r.get("title", "")
            snippet = (
                r.get("text", "")[:200]
                if r.get("text")
                else (r.get("url", "")[:200])
            )
            if title and snippet:
                parts.append(f"• {title}: {snippet}")

        return "\n".join(parts) if parts else ""

    except Exception as e:
        logger.debug("self-verification: Exa search failed for '%s': %s", query, e)
        return ""


def _search_for_evidence(response: str) -> str:
    """Search Tavily + Exa in parallel for evidence; merge results."""
    tavily_key = None
    try:
        cfg = (
            __import__("hermes_cli.config", fromlist=["load_config"]).load_config()
            or {}
        )
        tavily_cfg = (cfg.get("mcp_servers") or {}).get("tavily") or {}
        url = tavily_cfg.get("url", "") if isinstance(tavily_cfg, dict) else ""
        m = re.search(r"tavilyApiKey=([^&\s\"]+)", url)
        if m:
            tavily_key = m.group(1)
    except Exception:
        pass

    exa_key = os.environ.get("EXA_API_KEY")

    queries = _extract_search_queries(response)
    if not queries:
        return ""

    all_results: list[str] = []
    for q in queries:
        results_box: dict[str, str] = {}

        def _tavily_worker() -> None:
            if tavily_key:
                r = _tavily_search(q, tavily_key)
                if r:
                    results_box["tavily"] = r

        def _exa_worker() -> None:
            if exa_key:
                r = _exa_search(q, exa_key)
                if r:
                    results_box["exa"] = r

        threads: list[threading.Thread] = []
        if tavily_key:
            threads.append(threading.Thread(target=_tavily_worker, daemon=True))
        if exa_key:
            threads.append(threading.Thread(target=_exa_worker, daemon=True))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=8)

        parts: list[str] = []
        if "tavily" in results_box:
            parts.append(f"【Tavily: {q}】\n{results_box['tavily']}")
        if "exa" in results_box:
            parts.append(f"【Exa: {q}】\n{results_box['exa']}")

        if parts:
            all_results.append("\n\n".join(parts))

    if not all_results:
        return ""

    return "搜索参考信息：\n" + "\n\n".join(all_results) + "\n"


# ---------------------------------------------------------------------------
# Core verification APIs
# ---------------------------------------------------------------------------


def verify_factual_accuracy(response: str) -> dict[str, Any] | None:
    """Run all three verification layers in parallel.

    Returns a merged dict with claims, contradictions, sub_goals, and verdict.
    """
    if not response or len(response.strip()) < 50:
        return None

    has_risk = _has_risk_signals(response)
    user_msg = _get_last_user_message()
    has_user_msg = user_msg and len(user_msg.strip()) > 10

    if not has_risk and not has_user_msg:
        logger.debug(
            "self-verification: no risk signals and no user message, skipping"
        )
        return None

    config = _get_verifier_config()

    factual_prompt = None
    logic_prompt = None
    completeness_prompt = None

    if has_risk:
        search_context = _search_for_evidence(response)
        if not search_context:
            search_context = "（无搜索参考信息）"
        factual_prompt = FACTUAL_ACCURACY_PROMPT.format(
            response=response, search_context=search_context
        )
        logic_prompt = LOGIC_CONSISTENCY_PROMPT.format(response=response)

    if has_user_msg:
        completeness_prompt = COMPLETENESS_PROMPT.format(
            user_message=user_msg[:2000], response=response[:3000]
        )

    results: dict[str, Any | None] = {
        "factual": None,
        "logic": None,
        "completeness": None,
    }

    def _run_factual() -> None:
        results["factual"] = _call_verifier(factual_prompt, config) if factual_prompt else None

    def _run_logic() -> None:
        results["logic"] = _call_verifier(logic_prompt, config) if logic_prompt else None

    def _run_completeness() -> None:
        if completeness_prompt:
            results["completeness"] = _call_verifier(completeness_prompt, config)

    threads: list[threading.Thread] = []
    if factual_prompt:
        threads.append(
            threading.Thread(target=_run_factual, daemon=True, name="sv-factual")
        )
    if logic_prompt:
        threads.append(
            threading.Thread(target=_run_logic, daemon=True, name="sv-logic")
        )
    if completeness_prompt:
        threads.append(
            threading.Thread(target=_run_completeness, daemon=True, name="sv-complete")
        )
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Merge results
    factual = results["factual"]
    if factual and isinstance(factual, dict) and "verdict" in factual:
        result: dict[str, Any] = {
            "claims": factual.get("claims", []),
            "verdict": factual.get("verdict", "pass"),
        }
    else:
        result = {"claims": [], "verdict": "pass"}

    logic = results["logic"]
    if logic and isinstance(logic, dict):
        result["contradictions"] = logic.get("contradictions", [])
        logic_verdict = logic.get("verdict", "pass")
        if logic_verdict == "fail":
            result["verdict"] = "fail"
        elif logic_verdict == "warn" and result["verdict"] == "pass":
            result["verdict"] = "warn"
    else:
        result["contradictions"] = []

    completeness = results["completeness"]
    if completeness and isinstance(completeness, dict):
        result["sub_goals"] = completeness.get("sub_goals", [])
        comp_verdict = completeness.get("verdict", "pass")
        if comp_verdict == "fail":
            result["verdict"] = "fail"
        elif comp_verdict == "warn" and result["verdict"] == "pass":
            result["verdict"] = "warn"
    else:
        result["sub_goals"] = []

    return result


def verify_with_timeout(
    response: str, timeout: float = 0
) -> dict[str, Any] | None:
    """Run factual accuracy + logic consistency verification.

    ``timeout`` is kept for API compatibility but ignored.
    """
    return verify_factual_accuracy(response)


# ---------------------------------------------------------------------------
# Tool result verification (NEW in v0.3.0)
# ---------------------------------------------------------------------------


def verify_tool_result(
    tool_name: str,
    args: dict[str, Any] | None,
    result: str,
) -> list[str]:
    """Verify an intermediate tool call result for common issues.

    This is a fast, local check (no LLM calls) that catches:
      - write_file: Python syntax errors via ``compile()``, JSON parse errors
      - patch: patch failure (missing "success": true, "error" field present)
      - terminal: non-zero exit codes
      - web_search: empty result arrays

    Args:
        tool_name: Name of the tool that was called.
        args: The tool call arguments (dict or None).
        result: The tool result string (may be JSON).

    Returns:
        List of warning strings (empty list = no issues).
    """
    warnings: list[str] = []

    if tool_name == "write_file":
        _check_write_file(args, result, warnings)
    elif tool_name == "patch":
        _check_patch(result, warnings)
    elif tool_name == "terminal":
        _check_terminal(result, warnings)
    elif tool_name == "web_search":
        _check_web_search(result, warnings)

    return warnings


def _check_write_file(
    args: dict[str, Any] | None,
    result: str,
    warnings: list[str],
) -> None:
    """Check write_file results: Python compile, JSON parse."""
    if not isinstance(args, dict):
        return

    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        return

    content = args.get("content", "")
    if not isinstance(content, str) or not content:
        return
    if len(content) < 2:
        return

    if path.endswith(".py"):
        try:
            compile(content, path, "exec")
        except SyntaxError as e:
            warnings.append(
                f"write_file: Python syntax error in {path}: {e}"
            )
    elif path.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            warnings.append(
                f"write_file: JSON parse error in {path}: {e}"
            )


def _check_patch(result: str, warnings: list[str]) -> None:
    """Check patch results: success field or error field."""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Result is not JSON — can't verify structurally
        return

    if not isinstance(parsed, dict):
        return

    if "error" in parsed:
        error_val = parsed["error"]
        warnings.append(f"patch: reported error — {error_val}")
        return

    if "success" in parsed:
        success_val = parsed["success"]
        if success_val is not True:
            warnings.append(f"patch: success is {success_val!r} (not True)")


def _check_terminal(result: str, warnings: list[str]) -> None:
    """Check terminal results: non-zero exit code."""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Result is not JSON — can't verify structurally
        return

    if not isinstance(parsed, dict):
        return

    exit_code = parsed.get("exit_code")
    if exit_code is not None:
        try:
            code = int(exit_code)
            if code != 0:
                warnings.append(
                    f"terminal: command exited with non-zero code {code}"
                )
        except (ValueError, TypeError):
            pass


def _check_web_search(result: str, warnings: list[str]) -> None:
    """Check web_search results: empty array."""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Result is not JSON — can't verify structurally
        return

    if isinstance(parsed, list):
        if len(parsed) == 0:
            warnings.append("web_search: returned 0 results — may need broader query")


def _format_tool_warning_block(warnings: list[str], lang: str = "zh") -> str:
    """Format tool verification warnings into a Markdown block.

    Follows the same pattern as security-guidance's _format_warning_block.
    """
    i18n = _get_i18n(lang)
    lines = [
        "",
        "---",
        f"⚠️ Self-Verification tool check — {len(warnings)} issue{'s' if len(warnings) != 1 else ''}:",
        "",
    ]
    for w in warnings:
        lines.append(f"  - {w}")
    lines.append("")
    lines.append(
        "Tool results may still be usable. Review the warnings above and "
        "retry if the issues are material."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence scoring (NEW — 0-100 continuous, not binary pass/fail)
# ---------------------------------------------------------------------------


def score_claims(
    claims: list[dict[str, Any]],
    contradictions: list[dict[str, Any]] | None = None,
    sub_goals: list[dict[str, Any]] | None = None,
) -> int:
    """Score the verification result on a 0-100 confidence scale.

    Scoring logic:
      - Each claim reduces confidence based on risk_level: low=-5, medium=-15, high=-25
      - Each critical contradiction: -20
      - Each minor contradiction: -5
      - Each missing sub-goal: -15
      - Each partial sub-goal: -5
      - Start at 100, floor at 0, cap at 100

    Returns an integer 0-100.
    """
    score = 100

    # Deduct for claims
    for claim in claims:
        risk = claim.get("risk_level", "low")
        if risk == "high":
            score -= 25
        elif risk == "medium":
            score -= 15
        else:
            score -= 5

    # Deduct for contradictions
    for contra in (contradictions or []):
        if contra.get("severity") == "critical":
            score -= 20
        else:
            score -= 5

    # Deduct for sub-goal coverage
    for goal in (sub_goals or []):
        status = goal.get("status", "")
        if status == "missing":
            score -= 15
        elif status == "partial":
            score -= 5

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Self-refute (adversarial disprove)
# ---------------------------------------------------------------------------


def self_refute(
    claims: list[dict[str, Any]],
    evidence: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Adversarially try to DISPROVE each claim.

    This implements the Investigate -> Self-Refute pattern from Claude Code's
    security-guidance plugin. For each claim, we attempt to find counter-evidence
    that would invalidate it.

    If a claim is successfully refuted (false positive), it is removed.
    If it survives refutation, it gets a ``refute_status: "survived"`` tag
    and remains in the output list.

    Args:
        claims: List of claim dicts (from verify_factual_accuracy)
        evidence: Optional dict mapping claim text -> counter-evidence string

    Returns:
        Filtered list of claims that survived self-refutation.
    """
    if not claims:
        return []

    evidence = evidence or {}
    survived: list[dict[str, Any]] = []

    for claim in claims:
        claim_text = claim.get("text", "")

        # Check for self-contradiction within the claim itself
        # A claim like "X is both A and not-A" is self-refuting
        if _is_self_contradictory(claim_text):
            logger.debug(
                "self-verification: self-refute dismissed self-contradictory claim: %s",
                claim_text[:80],
            )
            claim["refute_status"] = "self_contradictory"
            claim["refuted"] = True
            continue

        # Check against provided evidence
        refute_evidence = evidence.get(claim_text, "")
        if refute_evidence:
            # If evidence contradicts the claim, refute it
            if _evidence_contradicts(claim_text, refute_evidence):
                logger.debug(
                    "self-verification: self-refute dismissed claim with evidence: %s",
                    claim_text[:80],
                )
                claim["refute_status"] = "counter_evidence"
                claim["refuted"] = True
                continue

        # Check for obviously unknowable claims (future predictions, etc.)
        if _is_unknowable(claim_text):
            logger.debug(
                "self-verification: self-refute dismissed unknowable claim: %s",
                claim_text[:80],
            )
            claim["refute_status"] = "unknowable"
            claim["refuted"] = True
            continue

        # Claim survives
        claim["refute_status"] = "survived"
        claim["refuted"] = False
        survived.append(claim)

    return survived


def _is_self_contradictory(text: str) -> bool:
    """Check if a claim contradicts itself within its own text.

    Detects patterns like:
      - "X is always true, except when it's not"
      - "The answer is A, but it could also be B"
      - "Never do X, unless Y" (where Y covers most cases)
    """
    text_lower = text.lower()

    # Pattern: "always ... except" / "never ... unless" self-contradiction
    if ("always" in text_lower or "never" in text_lower or "绝不" in text or "总是" in text) and (
        "except" in text_lower or "unless" in text_lower or "除了" in text or "除非" in text
    ):
        return True

    # Pattern: "A but also not-A" (hedged certainty)
    hedged_patterns = [
        (r"但是?可能", r"一定"),
        (r"but\s+(it\s+)?could\s+(also\s+)?be", r"is\s+(definitely|always|certainly)"),
    ]
    for hedge, certainty in hedged_patterns:
        if re.search(hedge, text_lower) and re.search(certainty, text_lower):
            return True

    return False


def _evidence_contradicts(claim: str, evidence: str) -> bool:
    """Check if the provided evidence contradicts the claim.

    This is a heuristic check — in practice, the Verifier model handles the
    heavy lifting. This function provides a fast path for clear contradictions.

    Returns True if the evidence likely contradicts the claim.
    """
    if not evidence or not claim:
        return False

    # If evidence explicitly says the claim is wrong
    negation_patterns = [
        r"(?:不|没有|并非|错误|incorrect|wrong|false|not|cannot|doesn'?t)",
    ]
    for pat in negation_patterns:
        if re.search(pat, evidence, re.IGNORECASE):
            # Only flag if the evidence text shares key terms with the claim
            claim_words = set(re.findall(r"\w{4,}", claim.lower()))
            evidence_words = set(re.findall(r"\w{4,}", evidence.lower()))
            overlap = claim_words & evidence_words
            if len(overlap) >= 3:
                return True

    return False


def _is_unknowable(text: str) -> bool:
    """Check if a claim is inherently unverifiable.

    Examples:
      - Future predictions ("will happen next year")
      - Subjective opinions disguised as facts
      - Claims about internal state of private companies
    """
    text_lower = text.lower()

    future_patterns = [
        r"(?:will|shall|going to)\s+(?:be|happen|become)",
        r"(?:明年|未来|将来|即将|将会|预计|预测)",
        r"(?:by\s+20\d{2}|in\s+20\d{2})",
    ]
    for pat in future_patterns:
        if re.search(pat, text_lower):
            return True

    opinion_patterns = [
        r"(?:我认为|我觉得|个人认为|在我看来|i\s+(?:think|believe|feel))",
    ]
    for pat in opinion_patterns:
        if re.search(pat, text_lower):
            return True

    return False


# ---------------------------------------------------------------------------
# Internationalized footnote formatting
# ---------------------------------------------------------------------------


def format_verification_footer(
    result: dict[str, Any], lang: str = "zh", threshold: int = 50
) -> str:
    """Format a verification result as a user-visible footnote.

    Uses i18n dict for language-aware text. Only appended for warn/fail verdicts.
    Claims below the confidence threshold are filtered out.

    Args:
        result: Verification result dict from verify_factual_accuracy()
        lang: Language code ('zh' or 'en'), defaults to 'zh'
        threshold: Confidence threshold for filtering claims (0-100)

    Returns:
        Formatted footnote string, or empty string if verdict is pass.
    """
    i18n = _get_i18n(lang)

    verdict = result.get("verdict", "pass")
    if verdict == "pass":
        return ""

    claims = result.get("claims") or []
    contradictions = result.get("contradictions") or []
    sub_goals = result.get("sub_goals") or []

    if not claims and not contradictions and not sub_goals:
        return ""

    # Filter claims by risk level heuristic (high-risk claims have higher priority)
    # Full threshold filtering is done by the caller using score_claims()
    high_risk = [c for c in claims if c.get("risk_level") == "high"]
    medium_risk = [c for c in claims if c.get("risk_level") == "medium"]
    low_risk = [c for c in claims if c.get("risk_level") == "low"]

    missing_goals = [g for g in sub_goals if g.get("status") == "missing"]
    partial_goals = [g for g in sub_goals if g.get("status") == "partial"]

    # Build verdict label
    verdict_label_map = {
        "warn": i18n["warn"],
        "fail": i18n["fail"],
    }
    verdict_label = verdict_label_map.get(verdict, f"⚠️ {verdict}")

    # Score the result for display
    confidence = score_claims(claims, contradictions, sub_goals)

    summary_parts = []
    if claims:
        summary_parts.append(f"{i18n['claims_label']} {len(claims)}")
    if contradictions:
        summary_parts.append(f"{i18n['contradictions_label']} {len(contradictions)}")
    if missing_goals:
        summary_parts.append(f"{i18n['missing_goals']} {len(missing_goals)}")
    if partial_goals:
        summary_parts.append(f"{i18n['partial_goals']} {len(partial_goals)}")

    lines = [
        f"> ---",
        f"> **{verdict_label}**（{i18n['title']}）：{'，'.join(summary_parts)}。",
        f"> {i18n['confidence_label']}: {confidence}/100",
        f">",
    ]

    # Show high-risk claims first
    for claim in high_risk:
        text = (claim.get("text") or "")[:120]
        note = (claim.get("note") or "")[:100]
        source_flag = (
            "无来源" if not claim.get("has_source") else "有来源"
            if lang == "zh"
            else ("no source" if not claim.get("has_source") else "has source")
        )
        refute_info = ""
        if claim.get("refuted"):
            refute_info = f" [{i18n['self_refute_refuted']}]"
        lines.append(f"> • [高风险/{source_flag}] {text}{refute_info}")
        if note:
            lines.append(f">   → {note}")
        lines.append(">")

    for claim in medium_risk:
        text = (claim.get("text") or "")[:120]
        note = (claim.get("note") or "")[:100]
        source_flag = (
            "无来源" if not claim.get("has_source") else "有来源"
            if lang == "zh"
            else ("no source" if not claim.get("has_source") else "has source")
        )
        lines.append(f"> • [中风险/{source_flag}] {text}")
        if note:
            lines.append(f">   → {note}")
        lines.append(">")

    for claim in low_risk:
        text = (claim.get("text") or "")[:120]
        note = (claim.get("note") or "")[:100]
        source_flag = (
            "无来源" if not claim.get("has_source") else "有来源"
            if lang == "zh"
            else ("no source" if not claim.get("has_source") else "has source")
        )
        lines.append(f"> • [低风险/{source_flag}] {text}")
        if note:
            lines.append(f">   → {note}")
        lines.append(">")

    # Show critical contradictions
    for contra in contradictions[:2]:
        if contra.get("severity") == "critical":
            a = (contra.get("statement_a") or "")[:80]
            b = (contra.get("statement_b") or "")[:80]
            lines.append(f"> • [{i18n['contradictions_label']}] 「{a}」↔「{b}」")
            lines.append(">")

    # Show missing/partial sub-goals
    for goal in missing_goals:
        g = (goal.get("goal") or "")[:100]
        note = (goal.get("note") or "")[:80]
        lines.append(f"> • [{i18n['missing_goals']}] {g}")
        if note:
            lines.append(f">   → {note}")
        lines.append(">")

    for goal in partial_goals:
        g = (goal.get("goal") or "")[:100]
        note = (goal.get("note") or "")[:80]
        lines.append(f"> • [{i18n['partial_goals']}] {g}")
        if note:
            lines.append(f">   → {note}")
        lines.append(">")

    # Retry hint
    lines.append(f"> {i18n['retry_hint']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plugin lifecycle helpers
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """Check if self-verification is enabled.

    Precedence: HERMES_SELF_VERIFICATION env var > config.yaml > default (true).
    """
    env = os.environ.get("HERMES_SELF_VERIFICATION")
    if env is not None:
        return env.strip().lower() not in {"0", "false", "no", "off"}

    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        sv_cfg = (cfg.get("agent") or {}).get("self_verification") or {}
        if isinstance(sv_cfg, dict) and "enabled" in sv_cfg:
            return bool(sv_cfg["enabled"])
    except Exception:
        pass

    return True


# ---------------------------------------------------------------------------
# Result persistence for "修正" flow
# ---------------------------------------------------------------------------

_LAST_RESULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "memories",
    "self-verification-last-result.json",
)


def save_last_result(
    result: dict[str, Any], original_response: str, session_id: str
) -> None:
    """Save verification result to a temp file for the '修正' (fix) flow."""
    try:
        import time

        payload = {
            "timestamp": time.time(),
            "session_id": session_id,
            "verdict": result.get("verdict", ""),
            "claims": result.get("claims", []),
            "contradictions": result.get("contradictions", []),
            "original_response": original_response[:2000],
        }
        os.makedirs(os.path.dirname(_LAST_RESULT_PATH), exist_ok=True)
        with open(_LAST_RESULT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.debug("self-verification: saved last result to %s", _LAST_RESULT_PATH)
    except Exception as e:
        logger.debug("self-verification: failed to save last result: %s", e)


def load_last_result() -> dict[str, Any] | None:
    """Load the last verification result."""
    try:
        if not os.path.exists(_LAST_RESULT_PATH):
            return None
        with open(_LAST_RESULT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
