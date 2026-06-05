"""代码级故障分析器 - 从日志/堆栈定位代码问题。"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StackFrame:
    """堆栈帧。"""
    file: str
    line: int
    method: str
    code_snippet: str = ""


@dataclass
class CodeAnalysis:
    """代码分析结果。"""
    exception_type: str
    exception_message: str
    stack_frames: List[StackFrame] = field(default_factory=list)
    root_cause_guess: str = ""
    fix_suggestions: List[str] = field(default_factory=list)
    diff_suggestion: str = ""


# 常见 Java 异常的诊断模式
JAVA_EXCEPTION_PATTERNS = {
    "NullPointerException": {
        "root_cause": "空指针引用：对象为 null 时调用了方法或访问了属性",
        "fix_suggestions": [
            "在调用前增加 null 检查",
            "使用 Optional 包装可能为 null 的值",
            "检查上游方法是否正确初始化了对象",
        ],
    },
    "OutOfMemoryError": {
        "root_cause": "JVM 堆内存不足",
        "fix_suggestions": [
            "增加 -Xmx 参数",
            "检查是否存在内存泄漏（对象未释放）",
            "检查缓存是否过大",
        ],
    },
    "StackOverflowError": {
        "root_cause": "递归调用过深或无限递归",
        "fix_suggestions": [
            "检查递归终止条件",
            "考虑改用迭代实现",
            "增加递归深度限制",
        ],
    },
    "SQLException": {
        "root_cause": "数据库访问异常",
        "fix_suggestions": [
            "检查 SQL 语法",
            "检查数据库连接是否正常",
            "检查表/字段是否存在",
        ],
    },
    "ConnectException": {
        "root_cause": "网络连接失败",
        "fix_suggestions": [
            "检查目标服务是否启动",
            "检查网络连通性",
            "检查防火墙规则",
        ],
    },
}


class CodeAnalyzer:
    """代码级故障分析器。"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def analyze_log(self, log_text: str) -> CodeAnalysis:
        """分析日志文本，提取异常信息和堆栈。"""
        # 提取异常类型和消息
        exception_match = re.search(
            r"([\w.]+(?:Exception|Error|Fault))\s*:\s*(.+)?", log_text
        )
        exception_type = exception_match.group(1) if exception_match else "Unknown"
        exception_message = (exception_match.group(2) or "").strip() if exception_match else ""

        # 提取堆栈帧
        frames = self._extract_stack_frames(log_text)

        # 诊断
        pattern = JAVA_EXCEPTION_PATTERNS.get(exception_type, {})
        root_cause = pattern.get("root_cause", "未知异常类型，需要人工分析")
        suggestions = pattern.get("fix_suggestions", ["收集更多信息", "联系开发团队"])

        # 生成 diff 建议
        diff = self._generate_diff_suggestion(exception_type, frames)

        return CodeAnalysis(
            exception_type=exception_type,
            exception_message=exception_message,
            stack_frames=frames,
            root_cause_guess=root_cause,
            fix_suggestions=suggestions,
            diff_suggestion=diff,
        )

    def _extract_stack_frames(self, log_text: str) -> List[StackFrame]:
        """从日志中提取 Java 堆栈帧。"""
        frames = []
        # Java 堆栈格式: at com.xxx.Class.method(File.java:123)
        pattern = re.compile(r"at\s+([\w.$]+)\.([\w<>]+)\((\w+\.java):(\d+)\)")
        for match in pattern.finditer(log_text):
            class_method = match.group(1)
            method = match.group(2)
            file_name = match.group(3)
            line = int(match.group(4))
            # 跳过框架帧
            if any(skip in class_method for skip in ["java.", "javax.", "org.springframework.", "sun.", "jdk."]):
                continue
            frames.append(StackFrame(
                file=class_method.replace(".", "/") + ".java",
                line=line,
                method=method,
            ))
        return frames[:10]  # 最多 10 帧

    def _generate_diff_suggestion(self, exception_type: str, frames: List[StackFrame]) -> str:
        """生成 diff 格式的修复建议。"""
        if not frames:
            return ""

        top_frame = frames[0]
        if exception_type == "NullPointerException":
            file_path = top_frame.file
            line_num = top_frame.line
            method_name = top_frame.method
            nl = "\n"
            return (
                f"# 修复建议: {file_path}:{line_num}{nl}"
                f"--- a/{file_path}{nl}"
                f"+++ b/{file_path}{nl}"
                f"@@ -{line_num},3 +{line_num},6 @@{nl}"
                f" # 原代码可能存在空指针{nl}"
                f"+if (obj == null) {{{nl}"
                f"+    log.error(\"对象为空: {method_name}\");{nl}"
                f"+    throw new BusinessException(\"数据不存在\");{nl}"
                f"+}}{nl}"
            )
        return ""

    def format_analysis(self, analysis: CodeAnalysis) -> str:
        """格式化分析结果为可读文本。"""
        lines = [
            f"🔍 代码故障分析",
            f"━━━━━━━━━━━━━━",
            f"异常类型: {analysis.exception_type}",
            f"异常消息: {analysis.exception_message}",
            f"",
            f"📋 堆栈帧（业务代码）:",
        ]
        for i, frame in enumerate(analysis.stack_frames[:5], 1):
            lines.append(f"  {i}. {frame.file}:{frame.line} → {frame.method}()")

        if analysis.root_cause_guess:
            lines.extend(["", f"🧠 AI 根因分析:", f"  {analysis.root_cause_guess}"])

        if analysis.fix_suggestions:
            lines.extend(["", "🛠️ 修复建议:"])
            for i, s in enumerate(analysis.fix_suggestions, 1):
                lines.append(f"  {i}. {s}")

        if analysis.diff_suggestion:
            lines.extend(["", "📝 Diff 建议:", analysis.diff_suggestion])

        return "\n".join(lines)
