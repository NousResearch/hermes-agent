"""大熊原版飞书卡片渲染器，从原维护分支提取。

保留原卡片布局和数据契约；纯渲染，无 Hermes/SDK 导入、网络或用户目录 I/O。
来源和原始测试见 reference/provenance.json。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class CardRenderer:
    MAX_STREAMING_CARD_CONTENT_LENGTH = 100_000
    _STREAMING_CARD_ELEMENT_CHARS = 3500
    _CARDKIT_INITIAL_CONTENT = "思考中..."
    _CARDKIT_STREAM_ELEMENT_ID = "streaming_content"

    @classmethod
    def _split_streaming_card_content(cls, content: str) -> list[str]:
        """Split long content into card elements while preserving one card."""
        text = str(content or "").strip() or " "
        encoded = text.encode("utf-8")
        if len(encoded) > 120_000:
            text = (
                encoded[:60_000].decode("utf-8", errors="ignore")
                + "\n\n---\n内容过长，中间部分已省略。\n---\n\n"
                + encoded[-60_000:].decode("utf-8", errors="ignore")
            )
        if len(text) > cls.MAX_STREAMING_CARD_CONTENT_LENGTH:
            head_budget = cls.MAX_STREAMING_CARD_CONTENT_LENGTH // 2
            tail_budget = cls.MAX_STREAMING_CARD_CONTENT_LENGTH - head_budget
            text = (
                text[:head_budget].rstrip()
                + "\n\n---\n内容过长，中间部分已折叠。\n---\n\n"
                + text[-tail_budget:].lstrip()
            )

        chunks: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= cls._STREAMING_CARD_ELEMENT_CHARS:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, cls._STREAMING_CARD_ELEMENT_CHARS)
            if split_at < cls._STREAMING_CARD_ELEMENT_CHARS // 2:
                split_at = cls._STREAMING_CARD_ELEMENT_CHARS
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip("\n")
        return chunks or [" "]

    @classmethod
    def _build_streaming_card_payload(
        cls,
        content: str,
        *,
        finalize: bool = False,
    ) -> str:
        elements: list[dict[str, Any]] = [
            {"tag": "markdown", "content": chunk}
            for chunk in cls._split_streaming_card_content(content)
        ]
        return json.dumps(
            {
                "config": {
                    "wide_screen_mode": True,
                    "update_multi": True,
                    "streaming_mode": not finalize,
                },
                "elements": elements,
            },
            ensure_ascii=False,
        )

    @classmethod
    def _build_cardkit_initial_card(cls) -> dict[str, Any]:
        return {
            "schema": "2.0",
            "config": {
                "streaming_mode": True,
                "locales": ["zh_cn", "en_us"],
                "summary": {"content": "思考中..."},
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": cls._CARDKIT_INITIAL_CONTENT,
                        "text_align": "left",
                        "text_size": "normal_v2",
                        "element_id": cls._CARDKIT_STREAM_ELEMENT_ID,
                    },
                ]
            },
        }

    @staticmethod
    def _unified_stream_status_lines(status: Optional[dict[str, Any]]) -> list[str]:
        if not isinstance(status, dict):
            return []
        done = [str(item) for item in status.get("done", []) if str(item).strip()]
        running = [str(item) for item in status.get("running", []) if str(item).strip()]
        return (done + running)[-100:]

    @staticmethod
    def _unified_stream_call_records(
        status: Optional[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(status, dict):
            return []
        records: list[dict[str, Any]] = []
        for item in status.get("calls", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            try:
                round_number = max(1, int(item.get("round") or 1))
            except (TypeError, ValueError):
                round_number = 1
            record = {
                "round": round_number,
                "name": name,
                "status": str(item.get("status") or "running"),
            }
            try:
                if item.get("duration") is not None:
                    record["duration"] = max(
                        0.0,
                        float(item["duration"]),
                    )
            except (TypeError, ValueError):
                pass
            records.append(record)
        return records[-100:]

    @staticmethod
    def _unified_stream_orchestration(
        status: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not isinstance(status, dict):
            return None
        orchestration = status.get("orchestration")
        if not isinstance(orchestration, dict) or orchestration.get("kind") != "moa":
            return None
        references: list[dict[str, str]] = []
        for item in orchestration.get("references", []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            state = str(item.get("status") or "running").strip().lower()
            if state not in {"running", "done", "failed"}:
                state = "running"
            references.append({"label": label, "status": state})
        try:
            refs_done = max(0, int(orchestration.get("refs_done") or 0))
        except (TypeError, ValueError):
            refs_done = 0
        try:
            refs_total = max(0, int(orchestration.get("refs_total") or 0))
        except (TypeError, ValueError):
            refs_total = 0
        return {
            "phase": str(orchestration.get("phase") or "references"),
            "refs_done": refs_done,
            "refs_total": refs_total or len(references),
            "references": references,
            "aggregator": str(orchestration.get("aggregator") or "").strip(),
        }

    @classmethod
    def _format_moa_orchestration(
        cls,
        status: Optional[dict[str, Any]],
        *,
        streaming: bool,
    ) -> str:
        orchestration = cls._unified_stream_orchestration(status)
        if orchestration is None:
            return ""
        phase = orchestration["phase"]
        refs_done = orchestration["refs_done"]
        refs_total = orchestration["refs_total"]
        references = orchestration["references"]
        aggregator = orchestration["aggregator"]

        if streaming and phase == "references":
            heading = (
                f"**MoA 协作 · 参考分析 {refs_done}/{refs_total}**"
                if refs_total
                else "**MoA 协作 · 参考分析**"
            )
        elif streaming and phase == "aggregating":
            heading = "**MoA 协作 · 正在综合参考意见**"
        elif streaming and phase == "degraded_aggregating":
            heading = "**部分参考模型不可用，正在降级综合**"
        elif phase == "degraded":
            heading = "**部分参考模型不可用，已降级综合**"
        else:
            heading = "**MoA 协作已完成**"

        lines = [heading]
        icons = {"running": "⏳", "done": "✅", "failed": "⚠️"}
        for reference in references:
            lines.append(f"{icons[reference['status']]} `{reference['label']}`")
        if aggregator and phase != "references":
            prefix = (
                "🧩 正在综合"
                if streaming and phase in {"aggregating", "degraded_aggregating"}
                else "🧩 综合模型"
            )
            lines.append(f"{prefix}：`{aggregator}`")
        return "\n".join(lines)

    @classmethod
    def _format_tool_call_rounds(
        cls,
        status: Optional[dict[str, Any]],
        *,
        streaming: bool,
    ) -> str:
        records = cls._unified_stream_call_records(status)
        if not records:
            return "\n".join(cls._unified_stream_status_lines(status))
        grouped: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(int(record["round"]), []).append(record)
        sections: list[str] = []
        for round_number, calls in grouped.items():
            lines = [
                (
                    f"**工具调用 · 第 {round_number} 轮**"
                    if streaming
                    else f"**第 {round_number} 轮**"
                )
            ]
            for call in calls:
                state = str(call["status"])
                if state == "done":
                    icon = "✅"
                elif state == "failed":
                    icon = "❌"
                else:
                    icon = "⏳"
                duration = call.get("duration")
                duration_text = f" · {float(duration):.1f}s" if duration is not None else ""
                lines.append(f"{icon} `{call['name']}`{duration_text}")
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    @classmethod
    def _final_unified_stream_summary(
        cls,
        status: Optional[dict[str, Any]],
    ) -> str:
        """Return the compact final-state summary for ordinary streaming cards.

        During execution the card exposes individual tool calls so the user can see progress.
        Once finalized, ordinary interactive cards should not retain that unbounded trace; the
        CardKit renderer has a real collapsible panel, while the compatible card path keeps a
        single collapsed summary line.
        """
        records = cls._unified_stream_call_records(status)
        count = len(records)
        if not count and isinstance(status, dict):
            count = len([item for item in status.get("done", []) if str(item).strip()])
        if not count:
            return ""
        return f"🔧 工具调用：{count} 项（已收起）"

    @classmethod
    def _compose_unified_stream_content(
        cls,
        content: str,
        metadata: Optional[dict[str, Any]],
        *,
        finalize: bool = False,
    ) -> str:
        if not isinstance(metadata, dict):
            return content
        status = metadata.get("__hermes_stream_status")
        if finalize:
            progress_section = cls._final_unified_stream_summary(status)
        else:
            progress_sections = [
                section
                for section in (
                    cls._format_moa_orchestration(status, streaming=True),
                    cls._format_tool_call_rounds(status, streaming=True),
                )
                if section
            ]
            progress_section = "\n\n".join(progress_sections)
        footer = ""
        if finalize:
            footer_parts: list[str] = []
            model_name = str(metadata.get("__hermes_stream_model") or "").strip()
            if model_name:
                footer_parts.append(f"🤖 {model_name}")
            elapsed_seconds = metadata.get("__hermes_stream_elapsed_seconds")
            if isinstance(elapsed_seconds, (int, float)) and not isinstance(elapsed_seconds, bool):
                footer_parts.append(f"⏱ {max(0.0, float(elapsed_seconds)):.1f}s")
            if footer_parts:
                footer = " · ".join(footer_parts)

        sections = [section for section in (progress_section, content or "", footer) if section]
        return "\n\n---\n\n".join(sections)

    @classmethod
    def _strip_status_only_stream_cursor(
        cls,
        content: str,
        metadata: Optional[dict[str, Any]],
    ) -> str:
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("__hermes_stream_status"), dict
        ):
            return content
        text = str(content or "")
        return "" if not text.replace("▉", "").strip() else content

    @classmethod
    def _build_final_cardkit_card(
        cls,
        content: str,
        *,
        tool_status: Optional[dict[str, Any]] = None,
        elapsed_seconds: Optional[float] = None,
        model_name: str = "",
    ) -> dict[str, Any]:
        elements: list[dict[str, Any]] = [
            {
                "tag": "markdown",
                "content": chunk or "...",
                "text_align": "left",
                "text_size": "normal_v2",
                "element_id": f"hermes_final_{idx}",
            }
            for idx, chunk in enumerate(
                cls._split_streaming_card_content(content),
                start=1,
            )
        ]
        orchestration = cls._unified_stream_orchestration(tool_status)
        moa_log = cls._format_moa_orchestration(
            tool_status,
            streaming=False,
        )
        if orchestration is not None and moa_log:
            ref_count = len(orchestration["references"]) or orchestration["refs_total"]
            elements.append(
                {
                    "tag": "collapsible_panel",
                    "element_id": "moa_trace",
                    "expanded": False,
                    "direction": "vertical",
                    "background_color": "grey",
                    "header": {
                        "title": {
                            "tag": "markdown",
                            "content": f"**MoA 协作记录（{ref_count}）**",
                        }
                    },
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": moa_log,
                            "text_size": "notation",
                            "element_id": "moa_log",
                        }
                    ],
                }
            )
        tool_records = cls._unified_stream_call_records(tool_status)
        tool_log = cls._format_tool_call_rounds(
            tool_status,
            streaming=False,
        )
        if tool_log:
            tool_elements = [
                {
                    "tag": "markdown",
                    "content": chunk,
                    "text_size": "notation",
                    "element_id": f"tool_log_{idx}",
                }
                for idx, chunk in enumerate(
                    cls._split_streaming_card_content(tool_log),
                    start=1,
                )
            ]
            elements.append(
                {
                    "tag": "collapsible_panel",
                    "element_id": "tool_trace",
                    "expanded": False,
                    "direction": "vertical",
                    "background_color": "grey",
                    "header": {
                        "title": {
                            "tag": "markdown",
                            "content": (
                                f"**工具调用记录（{len(tool_records) or len(cls._unified_stream_status_lines(tool_status))}）**"
                            ),
                        }
                    },
                    "elements": tool_elements,
                }
            )
        footer_parts: list[str] = []
        if model_name.strip():
            footer_parts.append(f"🤖 {model_name.strip()}")
        if elapsed_seconds is not None:
            footer_parts.append(f"⏱ {max(0.0, elapsed_seconds):.1f}s")
        if isinstance(tool_status, dict):
            tool_count = len(tool_records)
            if not tool_count:
                tool_count = len(
                    [item for item in tool_status.get("done", []) if str(item).strip()]
                )
            if tool_count:
                footer_parts.append(f"🔧 {tool_count} tools")
        if footer_parts:
            elements.extend(
                [
                    {"tag": "hr"},
                    {
                        "tag": "markdown",
                        "content": ("<font color='grey'>" + " · ".join(footer_parts) + "</font>"),
                        "text_size": "notation",
                        "element_id": "final_footer",
                    },
                ]
            )
        return {
            "schema": "2.0",
            "config": {
                "streaming_mode": False,
                "wide_screen_mode": True,
                "update_multi": True,
            },
            "body": {"elements": elements},
        }

    @staticmethod
    def _build_cardkit_terminal_notice(content: str) -> dict[str, Any]:
        return {
            "schema": "2.0",
            "config": {
                "streaming_mode": False,
                "wide_screen_mode": True,
                "update_multi": True,
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": content or "已切换到兼容卡片。",
                        "element_id": "fallback_notice",
                    }
                ]
            },
        }
