"""
Self Evolution Plugin — Dream Engine (Reflection Engine)
=========================================================

Runs nightly at 1:00 to analyze the previous day's sessions.

Design reference: Claude Code plugins/hookify/agents/conversation-analyzer.md
  - Analyzes conversations in reverse chronological order
  - Detects: corrections, frustrations, repeated issues, reversions
  - Extracts tool usage patterns, converts to actionable rules
  - Categorizes by severity

We extend this pattern with:
  - Full automated analysis (not just on user request)
  - Error analysis (tool failures, retries, API errors)
  - Time waste analysis (slow tools, repeated ops, inefficient sessions)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from self_evolution import db
from self_evolution.model_config import resolve_config, get_active_text_config, switch_to_fallback
from self_evolution.git_analyzer import analyze_code_changes
from self_evolution.models import (
    ErrorAnalysis, ToolFailure, RetryPattern,
    WasteAnalysis, ToolDuration, RepeatedOperation,
    CodeChangeAnalysis, CommitInfo,
    ReflectionReport,
)

logger = logging.getLogger(__name__)


# ── Backward-compatible aliases ────────────────────────────────────────────
# These are used by cron_jobs.py and other callers.
_resolve_runtime_config = resolve_config
_get_active_text_config = get_active_text_config
_switch_to_fallback = switch_to_fallback


class DreamEngine:
    """Nightly dream consolidation engine.

    Analyzes the previous day's sessions to find:
    1. Error patterns (tool failures, retries, incomplete tasks)
    2. Time waste patterns (slow tools, repeated operations, inefficient flows)
    3. Success patterns (what worked well)
    4. Generates actionable evolution proposals
    """

    def __init__(self, config: dict = None):
        self.config = config or _resolve_runtime_config()
        self._model_client = None
        self._current_prompt = ""

    def run(self, hours: int = 24, max_runtime_seconds: int = 0) -> Optional[ReflectionReport]:
        """Main dream consolidation flow.

        Args:
            hours: Analyze data from the last N hours.
            max_runtime_seconds: Hard timeout in seconds. 0 = no limit.
                If exceeded, stops at the next step boundary and returns None.
        """
        logger.info("Dream engine starting — analyzing last %d hours", hours)

        deadline = time.time() + max_runtime_seconds if max_runtime_seconds > 0 else 0

        now = time.time()
        cutoff = now - (hours * 3600)

        try:
            # 1. Load session data
            scores = db.fetch_all(
                "session_scores",
                where="created_at >= ?",
                params=(cutoff,),
                order_by="created_at DESC",
            )
            tool_invocations = db.fetch_all(
                "tool_invocations",
                where="created_at >= ?",
                params=(cutoff,),
                order_by="created_at DESC",
            )
            signals = db.fetch_all(
                "outcome_signals",
                where="created_at >= ?",
                params=(cutoff,),
            )

            if not scores:
                logger.info("No sessions to analyze")
                return None

            # 2. Error analysis
            if deadline and time.time() > deadline:
                logger.warning("Dream engine timed out before error analysis")
                return None
            error_analysis = self._analyze_errors(scores, tool_invocations, signals)
            logger.info("Error analysis: %s", error_analysis.summary())

            # 3. Time waste analysis
            if deadline and time.time() > deadline:
                logger.warning("Dream engine timed out before waste analysis")
                return None
            waste_analysis = self._analyze_time_waste(scores, tool_invocations)
            logger.info("Waste analysis: %s", waste_analysis.summary())

            # 3.5. Code change analysis
            if deadline and time.time() > deadline:
                logger.warning("Dream engine timed out before code analysis")
                return None
            code_analysis = analyze_code_changes(hours=hours)
            logger.info("Code change analysis: %d commits found", code_analysis.total_commits)

            # 4. Compute average score
            avg_score = (
                sum(s.get("composite_score", 0) for s in scores) / len(scores)
                if scores else 0
            )

            # 5. Build reflection prompt
            if deadline and time.time() > deadline:
                logger.warning("Dream engine timed out before model call")
                return None
            prompt = self._build_reflection_prompt(
                scores, tool_invocations, signals,
                error_analysis, waste_analysis, avg_score,
                code_analysis=code_analysis,
            )

            # 6. Call model for deep reflection
            reflection_text = self._call_model(prompt)
            if not reflection_text:
                logger.warning("Model returned empty reflection")
                return None

            # 7. Parse reflection report
            report = self._parse_reflection(
                reflection_text=reflection_text,
                period_start=cutoff,
                period_end=now,
                sessions_analyzed=len(scores),
                avg_score=avg_score,
                error_analysis=error_analysis,
                waste_analysis=waste_analysis,
                code_analysis=code_analysis,
            )

            # 8. Store report
            report_id = db.insert("reflection_reports", report.to_db_row())
            logger.info("Reflection report saved: id=%d, avg_score=%.3f", report_id, avg_score)

            # 9. Generate evolution proposals
            from self_evolution.evolution_proposer import generate_proposals
            proposals = generate_proposals(report, report_id)
            for p in proposals:
                db.insert("evolution_proposals", p.to_db_row())
            logger.info("Generated %d evolution proposals", len(proposals))

            # 10. Compress existing strategies
            try:
                from self_evolution.strategy_compressor import compress_strategies
                from self_evolution.strategy_store import StrategyStore
                store = StrategyStore()
                data = store.load()
                rules = data.get("rules", [])
                compressed = compress_strategies(rules)
                if len(compressed) < len(rules):
                    data["rules"] = compressed
                    store.save(data)
                    logger.info("Strategies compressed: %d → %d", len(rules), len(compressed))
            except Exception as exc:
                logger.warning("Strategy compression failed: %s", exc)

            # 11. Cleanup old data
            db.cleanup(days=30)

            return report

        except Exception as exc:
            logger.exception("Dream engine failed: %s", exc)
            return None

    # ── Error Analysis ────────────────────────────────────────────────────

    def _analyze_errors(
        self,
        scores: List[dict],
        invocations: List[dict],
        signals: List[dict],
    ) -> ErrorAnalysis:
        """Analyze all errors in the period.

        Inspired by Claude Code conversation-analyzer's signal detection.
        """
        # Tool failures
        failures = {}
        for inv in invocations:
            if not inv.get("success", True):
                tool = inv.get("tool_name", "unknown")
                error_type = inv.get("error_type", "unknown")
                key = f"{tool}:{error_type}"
                if key not in failures:
                    failures[key] = ToolFailure(
                        tool_name=tool,
                        error_type=error_type,
                        count=0,
                        sessions_affected=[],
                        example_session=inv.get("session_id", ""),
                    )
                failures[key].count += 1
                sid = inv.get("session_id", "")
                if sid and sid not in failures[key].sessions_affected:
                    failures[key].sessions_affected.append(sid)

        # Retry patterns (same tool called > 2 times in same session)
        retries = self._detect_retry_patterns(invocations)

        # Incomplete sessions
        incomplete = [
            s.get("session_id", "") for s in scores
            if s.get("completion_rate", 1.0) < 0.5
        ]

        # User corrections from signals
        corrections = [s for s in signals if s.get("signal_type") == "correction"]
        frustration = [s for s in signals if s.get("signal_type") == "frustration"]
        api_errors = [s for s in signals if s.get("signal_type") == "api_error"]

        # API error type distribution
        api_error_types: Dict[str, int] = {}
        for s in api_errors:
            meta = json.loads(s.get("metadata", "{}"))
            etype = meta.get("error_type", "unknown")
            api_error_types[etype] = api_error_types.get(etype, 0) + 1

        return ErrorAnalysis(
            tool_failures=sorted(failures.values(), key=lambda x: x.count, reverse=True),
            retry_patterns=retries,
            incomplete_sessions=incomplete,
            user_corrections=len(corrections),
            correction_examples=[s.get("metadata", "") for s in corrections[:3]],
            api_error_count=len(api_errors),
            api_error_types=api_error_types,
        )

    def _detect_retry_patterns(self, invocations: List[dict]) -> List[RetryPattern]:
        """Detect tools called > 2 times in same session."""
        session_tools: Dict[str, Dict[str, int]] = {}
        for inv in invocations:
            sid = inv.get("session_id", "")
            tool = inv.get("tool_name", "")
            if sid not in session_tools:
                session_tools[sid] = {}
            session_tools[sid][tool] = session_tools[sid].get(tool, 0) + 1

        patterns = []
        for sid, tools in session_tools.items():
            for tool, count in tools.items():
                if count > 2:
                    patterns.append(RetryPattern(
                        session_id=sid,
                        tool_name=tool,
                        attempt_count=count,
                        final_outcome="unknown",
                    ))
        return sorted(patterns, key=lambda x: x.attempt_count, reverse=True)[:20]

    # ── Time Waste Analysis ───────────────────────────────────────────────

    def _analyze_time_waste(
        self,
        scores: List[dict],
        invocations: List[dict],
    ) -> WasteAnalysis:
        """Analyze time waste patterns."""
        # Slowest tools
        tool_durations: Dict[str, List[int]] = {}
        for inv in invocations:
            tool = inv.get("tool_name", "")
            duration = inv.get("duration_ms", 0)
            if not duration:
                continue
            if tool not in tool_durations:
                tool_durations[tool] = []
            tool_durations[tool].append(duration)

        slowest = [
            ToolDuration(
                tool_name=tool,
                total_duration_ms=sum(durs),
                call_count=len(durs),
                avg_duration_ms=sum(durs) / len(durs),
            )
            for tool, durs in tool_durations.items()
        ]
        slowest.sort(key=lambda x: x.avg_duration_ms, reverse=True)

        # Repeated operations (same tool + same session > 3 times)
        session_tool_calls: Dict[str, Dict[str, int]] = {}
        for inv in invocations:
            sid = inv.get("session_id", "")
            tool = inv.get("tool_name", "")
            if sid not in session_tool_calls:
                session_tool_calls[sid] = {}
            session_tool_calls[sid][tool] = session_tool_calls[sid].get(tool, 0) + 1

        repeated = []
        for sid, tools in session_tool_calls.items():
            for tool, count in tools.items():
                if count > 3:
                    repeated.append(RepeatedOperation(
                        description=f"{tool} called {count} times",
                        count=count,
                        sessions=[sid],
                        wasted_ms=tool_durations.get(tool, [0])[0] * (count - 2) if tool in tool_durations else 0,
                    ))

        # Inefficient sessions (low efficiency score)
        inefficient = [
            s.get("session_id", "") for s in scores
            if s.get("efficiency_score", 1.0) < 0.3
        ]

        return WasteAnalysis(
            slowest_tools=slowest[:10],
            repeated_operations=sorted(repeated, key=lambda x: x.count, reverse=True)[:10],
            inefficient_sessions=inefficient,
            shortcut_opportunities=[],
        )

    # ── Reflection Prompt ─────────────────────────────────────────────────

    def _build_reflection_prompt(
        self,
        scores: List[dict],
        invocations: List[dict],
        signals: List[dict],
        errors: ErrorAnalysis,
        waste: WasteAnalysis,
        avg_score: float,
        code_analysis: CodeChangeAnalysis = None,
    ) -> str:
        """Build the reflection prompt as structured JSON data.

        All analysis results are serialized as JSON so the model receives
        lossless data instead of pre-summarized text.
        """
        # Load user prompt template (short: just overview + data placeholder)
        template_path = Path(__file__).parent / "prompts" / "reflection.md"
        if template_path.exists():
            template = template_path.read_text(encoding="utf-8")
        else:
            template = _DEFAULT_REFLECTION_PROMPT

        # Compute statistics
        total_invocations = len(invocations)
        success_rate = (
            sum(1 for i in invocations if i.get("success", True)) / total_invocations * 100
            if total_invocations else 100
        )

        # Period range
        if scores:
            ts_min = min(s.get("created_at", 0) for s in scores)
            ts_max = max(s.get("created_at", 0) for s in scores)
            period_range = (
                f"{time.strftime('%m-%d %H:%M', time.localtime(ts_min))} ~ "
                f"{time.strftime('%m-%d %H:%M', time.localtime(ts_max))}"
            )
        else:
            period_range = "N/A"

        # Build structured data JSON — compact format to save tokens
        data = {}

        # 1. Sessions — compact: [score, completion, efficiency, cost, satisfaction, category]
        data["sessions"] = [
            [
                round(s.get("composite_score", 0), 2),
                round(s.get("completion_rate", 0), 2),
                round(s.get("efficiency_score", 0), 2),
                round(s.get("cost_efficiency", 0), 2),
                round(s.get("satisfaction_proxy", 0), 2),
                s.get("task_category", ""),
            ]
            for s in scores
        ]

        # 2. Tool usage — compact: {tool: [calls, failures, avg_ms]}
        tool_stats: Dict[str, List[int]] = {}
        for inv in invocations:
            tool = inv.get("tool_name", "")
            if tool not in tool_stats:
                tool_stats[tool] = [0, 0, 0]  # calls, failures, total_ms
            tool_stats[tool][0] += 1
            if not inv.get("success", True):
                tool_stats[tool][1] += 1
            tool_stats[tool][2] += inv.get("duration_ms", 0) or 0
        data["tools"] = {
            t: [v[0], v[1], round(v[2] / max(v[0], 1))]
            for t, v in sorted(tool_stats.items(), key=lambda x: x[1][2], reverse=True)
        }

        # 3. Signals — compact: {type: count}
        signal_types = {}
        for s in signals:
            stype = s.get("signal_type", "unknown")
            signal_types[stype] = signal_types.get(stype, 0) + 1
        data["signals"] = signal_types

        # 4. Errors — only non-empty fields
        err_data = {}
        if errors.tool_failures:
            err_data["tool_failures"] = [
                f"{tf.tool_name}:{tf.error_type}x{tf.count}"
                for tf in errors.tool_failures
            ]
        if errors.retry_patterns:
            err_data["retries"] = [
                f"{rp.tool_name}x{rp.attempt_count}"
                for rp in errors.retry_patterns[:5]
            ]
        if errors.incomplete_sessions:
            err_data["incomplete"] = len(errors.incomplete_sessions)
        if errors.user_corrections:
            err_data["corrections"] = errors.user_corrections
            if errors.correction_examples:
                err_data["correction_examples"] = errors.correction_examples[:2]
        if errors.api_error_count:
            err_data["api_errors"] = errors.api_error_count
        if err_data:
            data["errors"] = err_data

        # 5. Waste — only non-empty
        waste_data = {}
        if waste.slowest_tools:
            waste_data["slowest"] = [
                f"{td.tool_name} {round(td.avg_duration_ms)}ms/{td.call_count}calls"
                for td in waste.slowest_tools[:5]
            ]
        if waste.repeated_operations:
            waste_data["repeated"] = [
                f"{ro.description} x{ro.count}"
                for ro in waste.repeated_operations[:3]
            ]
        if waste.inefficient_sessions:
            waste_data["inefficient"] = len(waste.inefficient_sessions)
        if waste_data:
            data["waste"] = waste_data

        # 6. Code changes — flat compact format
        if code_analysis and code_analysis.commits:
            cc = code_analysis
            commits_data = []
            for c in cc.commits[:10]:
                entry = f"{c.hash_short} {c.subject} +{c.insertions}/-{c.deletions}"
                if c.file_list:
                    entry += f" [{','.join(c.file_list[:5])}]"
                if c.body:
                    entry += f" | {c.body[:150]}"
                commits_data.append(entry)
            data["code_changes"] = {
                "stats": f"{cc.total_commits} commits +{cc.total_insertions}/-{cc.total_deletions} lines {cc.total_files_changed} files",
                "categories": cc.change_categories,
                "areas": cc.areas_touched,
                "commits": commits_data,
            }

        data_json = json.dumps(data, ensure_ascii=False, indent=2)

        # Fill template
        prompt = template.replace("{period_range}", period_range)
        prompt = prompt.replace("{sessions_count}", str(len(scores)))
        prompt = prompt.replace("{avg_score}", f"{avg_score:.3f}")
        prompt = prompt.replace("{total_invocations}", str(total_invocations))
        prompt = prompt.replace("{success_rate}", f"{success_rate:.1f}")
        prompt = prompt.replace("{data_json}", data_json)

        return prompt

    # ── Model Call ────────────────────────────────────────────────────────

    def _call_model(self, prompt: str) -> Optional[str]:
        """Call the active model with automatic failover.

        Resolution order:
          1. Primary model (glm-5.1 via zai)
          2. Fallback model (Qwen3.6 via local) — if primary fails
        Health check: when on fallback, probes primary every 30 min
        and switches back when it recovers.
        """
        self._current_prompt = prompt

        active_cfg, is_fallback = _get_active_text_config(self.config)
        base_url = active_cfg.get("base_url", "")
        api_key = active_cfg.get("api_key", "")
        model = active_cfg.get("model", "")

        if not base_url or not model:
            logger.warning("Incomplete runtime config: base_url=%s model=%s",
                           bool(base_url), model)
            return None

        result = self._call_chat_completions(base_url, api_key, model)

        # If primary failed, try fallback
        if result is None and not is_fallback:
            fallback = self.config.get("fallback", {})
            if fallback.get("base_url") and fallback.get("model"):
                logger.warning("Primary model failed, trying fallback: %s",
                               fallback.get("model"))
                result = self._call_chat_completions(
                    fallback["base_url"], fallback.get("api_key", ""),
                    fallback["model"],
                )
                if result is not None:
                    _switch_to_fallback()

        return result

    def _call_chat_completions(
        self, base_url: str, api_key: str, model: str,
    ) -> Optional[str]:
        """Call OpenAI-compatible /chat/completions endpoint."""
        try:
            import requests
            url = f"{base_url.rstrip('/')}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            resp = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": self._current_prompt or ""},
                    ],
                    "temperature": 0.3,
                },
                timeout=300,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                logger.debug("Model call failed: %d %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.debug("Chat completions call failed: %s", exc)
        return None

    # ── Multimodal Call ───────────────────────────────────────────────────

    def call_multimodal(self, prompt: str, images: list = None) -> Optional[str]:
        """Call multimodal model with text and optional images.

        Routes to local multimodal model (gemma-4-26b-a4b-it-4bit) when
        images are involved. Falls back to text model if no images.

        Args:
            prompt: Text prompt.
            images: List of image data, each item is either:
                - URL string (http/https/data:image)
                - bytes (raw image data, auto-encoded to base64)

        Returns:
            Model response text, or None on failure.
        """
        mm = self.config.get("multimodal", {})
        if not mm or not mm.get("base_url"):
            logger.debug("No multimodal model configured, falling back to text")
            return self._call_model(prompt)

        # Build content with images
        content = [{"type": "text", "text": prompt}]
        for img in (images or []):
            if isinstance(img, bytes):
                import base64
                b64 = base64.b64encode(img).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            elif isinstance(img, str):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img},
                })

        try:
            from openai import OpenAI
            client = OpenAI(
                base_url=mm["base_url"].rstrip("/") + ("/v1" if not mm["base_url"].rstrip("/").endswith("/v1") else ""),
                api_key=mm.get("api_key") or "no-key",
            )
            resp = client.chat.completions.create(
                model=mm["model"],
                messages=[{"role": "user", "content": content}],
                temperature=0.3,
                max_tokens=2000,
                timeout=120,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logger.debug("Multimodal call failed: %s", exc)
            return None

    # ── Reflection Parsing ────────────────────────────────────────────────

    def _parse_reflection(
        self,
        reflection_text: str,
        period_start: float,
        period_end: float,
        sessions_analyzed: int,
        avg_score: float,
        error_analysis: ErrorAnalysis,
        waste_analysis: WasteAnalysis,
        code_analysis: CodeChangeAnalysis = None,
    ) -> ReflectionReport:
        """Parse model output into structured ReflectionReport.

        Extraction cascade:
          1. Direct JSON parse
          2. Strip markdown ```json ... ``` wrapper, retry JSON
          3. Extract JSON object via regex (handle trailing text)
          4. Text-based section extraction (fallback)
        """
        worst_patterns = []
        best_patterns = []
        recommendations = []
        tool_insights = {}

        text = reflection_text.strip()

        # 1. Direct JSON parse
        data = _try_parse_json(text)

        if data is None:
            # 2. Strip markdown wrapper
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                data = _try_parse_json(m.group(1))

        if data is None:
            # 3. Regex extract first JSON object
            m = re.search(r'\{[^{}]*"(?:worst|best|recommendations)"[^{}]*\}', text, re.DOTALL)
            if m:
                data = _try_parse_json(m.group(0))

        if data is None:
            # 3.5. Broader regex — find outermost braces
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end > start:
                data = _try_parse_json(text[start:end + 1])

        if data is not None:
            worst_patterns = data.get("worst_patterns") or []
            best_patterns = data.get("best_patterns") or []
            recommendations = data.get("recommendations") or []
            tool_insights = data.get("tool_insights") or {}
        else:
            # 4. Text-based extraction
            section = None
            for line in text.split("\n"):
                stripped = line.strip()
                lower = stripped.lower()
                if ("worst" in lower and "pattern" in lower) or "最差" in stripped or "错误模式" in stripped:
                    section = "worst"
                elif ("best" in lower and "pattern" in lower) or "最佳" in stripped or "成功" in stripped:
                    section = "best"
                elif ("recommend" in lower) or "建议" in stripped:
                    section = "rec"
                elif stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("• "):
                    item = stripped.lstrip("-*• ").strip()
                    if section == "worst":
                        worst_patterns.append(item)
                    elif section == "best":
                        best_patterns.append(item)
                    elif section == "rec":
                        recommendations.append(item)
                elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)" and stripped[2] == " ":
                    item = stripped[3:].strip()
                    if section == "worst":
                        worst_patterns.append(item)
                    elif section == "best":
                        best_patterns.append(item)
                    elif section == "rec":
                        recommendations.append(item)

        return ReflectionReport(
            period_start=period_start,
            period_end=period_end,
            sessions_analyzed=sessions_analyzed,
            avg_score=avg_score,
            error_summary=error_analysis.summary(),
            waste_summary=waste_analysis.summary(),
            worst_patterns=worst_patterns,
            best_patterns=best_patterns,
            tool_insights=tool_insights,
            recommendations=recommendations,
            code_change_summary=code_analysis.summary() if code_analysis else "",
            model_used=self.config.get("model", "unknown"),
        )


# ── Default Prompt Template ──────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "你是 Hermes Agent 性能分析引擎。分析运行数据+代码变更，输出严格JSON（无markdown）。\n"
    "格式:\n"
    '{"worst_patterns":["模式(工具+场景+根因)"],"best_patterns":["成功经验"],'
    '"tool_insights":{"工具":{"sr":0.95,"ms":500,"rec":"建议"}},'
    '"recommendations":["做什么|效果|风险(l/m/h)|验证"]}\n'
    "重点:系统性错误>偶发,错误连锁,策略vs工具问题,重复操作,代码设计合理性,自我进化状态,"
    "可固化流程。≤5条建议,优先高影响低风险。无数据时输出空数组。"
)


_DEFAULT_REFLECTION_PROMPT = """## 概况
- 时段: {period_range}
- Session 数: {sessions_count}, 平均质量: {avg_score}
- 工具调用: {total_invocations} 次, 成功率 {success_rate}%

## 数据
{data_json}
"""


def _try_parse_json(text: str) -> Optional[dict]:
    """Try to parse JSON, returning None on any failure."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None
