"""线程堆栈分析器 — 通过 SSH 在远程节点执行 jstack，解析线程 dump。"""

import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.ssh_executor import create_executor

# 线程块头部模式: "thread-name" #nid ...
_THREAD_HEADER_RE = re.compile(r'^"([^"]+)"\s+#(\d+)')
# 线程状态模式: java.lang.Thread.State: STATE
_THREAD_STATE_RE = re.compile(r"java\.lang\.Thread\.State:\s*(\S+)")
# 死锁检测模式
_DEADLOCK_RE = re.compile(r"Found.*deadlock", re.IGNORECASE)


@dataclass
class StackMatch:
    """关键字匹配结果。"""
    thread_name: str
    state: str
    matched_line: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_name": self.thread_name,
            "state": self.state,
            "matched_line": self.matched_line,
            "context_before": self.context_before,
            "context_after": self.context_after,
        }


class ThreadStackAnalyzer:
    """线程堆栈分析器 — 通过 SSH 获取 jstack 输出并解析分析。"""

    SSH_TIMEOUT = 30

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._pid = config.get("pid")

    def _get_pid(self, executor) -> Optional[int]:
        """通过 jps -l | grep {process_name} 获取 catalina PID。"""
        process_name = self.config.get("tomcat_process_name", "catalina")
        # shlex.quote 防止命令注入
        safe_name = shlex.quote(process_name)
        cmd = f"jps -l | grep {safe_name}"
        result = executor.run(cmd, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            return None
        # 取第一行的 PID
        first_line = result.stdout.strip().splitlines()[0]
        parts = first_line.split()
        if parts and parts[0].isdigit():
            return int(parts[0])
        return None

    def _fetch_jstack_output(self) -> str:
        """获取 jstack 输出。如果 config 有 pid 直接用，否则通过 jps 查找。"""
        executor = create_executor(self.config)

        pid = self._pid
        if pid is None:
            pid = self._get_pid(executor)
            if pid is None:
                return ""

        # shlex.quote 防止命令注入
        safe_pid = shlex.quote(str(pid))
        cmd = f"jstack {safe_pid}"
        result = executor.run(cmd, timeout=self.SSH_TIMEOUT)
        if result.returncode != 0:
            return ""
        return result.stdout

    @staticmethod
    def _parse_thread_blocks(jstack_output: str) -> List[Dict[str, Any]]:
        """解析 jstack 输出为线程块列表。"""
        if not jstack_output:
            return []

        blocks = []
        lines = jstack_output.splitlines()
        current_block: Optional[Dict[str, Any]] = None

        for line in lines:
            header_match = _THREAD_HEADER_RE.match(line)
            if header_match:
                # 保存上一个块
                if current_block is not None:
                    current_block["stacktrace"] = "\n".join(current_block["lines"])
                    blocks.append(current_block)
                current_block = {
                    "thread_name": header_match.group(1),
                    "nid": header_match.group(2),
                    "state": "UNKNOWN",
                    "stacktrace": "",
                    "lines": [line],
                }
            elif current_block is not None:
                current_block["lines"].append(line)
                state_match = _THREAD_STATE_RE.search(line)
                if state_match:
                    current_block["state"] = state_match.group(1)

        # 保存最后一个块
        if current_block is not None:
            current_block["stacktrace"] = "\n".join(current_block["lines"])
            blocks.append(current_block)

        return blocks

    @staticmethod
    def filter_by_keyword(
        blocks: List[Dict[str, Any]],
        keyword: str,
        context_lines: int = 10,
    ) -> List[StackMatch]:
        """按关键字过滤线程块，返回匹配行及上下文。不区分大小写。"""
        matches = []
        keyword_lower = keyword.lower()

        for block in blocks:
            # 也搜索线程名
            if keyword_lower in block["thread_name"].lower():
                all_lines = block["lines"]
                # 线程名匹配 — 返回状态行附近上下文
                context_before = all_lines[1:1 + context_lines] if len(all_lines) > 1 else []
                matches.append(StackMatch(
                    thread_name=block["thread_name"],
                    state=block["state"],
                    matched_line=block["lines"][0] if block["lines"] else "",
                    context_before=context_before,
                    context_after=[],
                ))
                continue

            # 在堆栈行中搜索
            lines = block["lines"]
            for i, line in enumerate(lines):
                if keyword_lower in line.lower():
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context_before = lines[start:i]
                    context_after = lines[i + 1:end]
                    matches.append(StackMatch(
                        thread_name=block["thread_name"],
                        state=block["state"],
                        matched_line=line,
                        context_before=context_before,
                        context_after=context_after,
                    ))
                    break  # 每个线程块只取第一个匹配

        return matches

    @staticmethod
    def _count_by_state(blocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """按状态统计线程数。"""
        counts: Dict[str, int] = {}
        for block in blocks:
            state = block["state"]
            counts[state] = counts.get(state, 0) + 1
        return counts

    @staticmethod
    def _detect_deadlock(jstack_output: str) -> bool:
        """检测是否存在死锁。"""
        return bool(_DEADLOCK_RE.search(jstack_output))

    def analyze(self, keyword: str = None, context_lines: int = 10) -> Dict[str, Any]:
        """完整分析流程：获取 jstack → 解析 → 过滤 → 汇总。"""
        jstack_output = self._fetch_jstack_output()
        blocks = self._parse_thread_blocks(jstack_output)
        state_counts = self._count_by_state(blocks)

        matches = []
        if keyword:
            matches = self.filter_by_keyword(blocks, keyword, context_lines)

        return {
            "matches": [m.to_dict() for m in matches],
            "all_threads": [
                {
                    "thread_name": b["thread_name"],
                    "nid": b["nid"],
                    "state": b["state"],
                }
                for b in blocks
            ],
            "summary": {
                "total_threads": len(blocks),
                "state_counts": state_counts,
                "blocked_count": state_counts.get("BLOCKED", 0),
                "deadlock_detected": self._detect_deadlock(jstack_output),
                "keyword": keyword,
                "match_count": len(matches),
            },
            "config": {
                "host": self.config.get("host"),
                "pid": self._pid,
            },
        }
