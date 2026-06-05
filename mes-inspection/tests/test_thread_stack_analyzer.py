"""线程堆栈分析器测试。"""

import subprocess
from unittest.mock import patch, MagicMock
import pytest

from scripts.thread_stack_analyzer import StackMatch, ThreadStackAnalyzer


# ── 示例 jstack 输出 ──────────────────────────────────────────────────

SAMPLE_JSTACK = '''2026-06-05 10:30:45
Full thread dump Java HotSpot(TM) 64-Bit Server VM (25.292-b10 mixed mode):

"http-nio-8080-exec-1" #42 daemon prio=5 os_prio=0 tid=0x00007f8b8c00a000 nid=0x1234 waiting on condition [0x00007f8b7c5fe000]
   java.lang.Thread.State: WAITING (parking)
        at sun.misc.Unsafe.park(Native Method)
        - parking to wait for  <0x00000000e0a1b2c0> (a java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject)
        at java.util.concurrent.locks.LockSupport.park(LockSupport.java:175)
        at java.util.concurrent.locks.AbstractQueuedSynchronizer$ConditionObject.await(AbstractQueuedSynchronizer.java:2039)
        at java.util.concurrent.LinkedBlockingQueue.take(LinkedBlockingQueue.java:442)
        at org.apache.tomcat.util.threads.TaskQueue.take(TaskQueue.java:103)
        at org.apache.tomcat.util.threads.ThreadPoolExecutor.getTask(ThreadPoolExecutor.java:162)

"http-nio-8080-exec-2" #43 daemon prio=5 os_prio=0 tid=0x00007f8b8c00b000 nid=0x1235 runnable [0x00007f8b7c4fd000]
   java.lang.Thread.State: RUNNABLE
        at java.net.SocketInputStream.socketRead0(Native Method)
        at java.net.SocketInputStream.socketRead(SocketInputStream.java:116)
        at java.net.SocketInputStream.read(SocketInputStream.java:171)
        at java.net.SocketInputStream.read(SocketInputStream.java:141)
        at com.mysql.jdbc.MysqlIO.readFully(MysqlIO.java:3008)

"http-nio-8080-exec-3" #44 daemon prio=5 os_prio=0 tid=0x00007f8b8c00c000 nid=0x1236 waiting for monitor entry [0x00007f8b7c3fc000]
   java.lang.Thread.State: BLOCKED (on object monitor)
        at com.example.service.OrderService.processOrder(OrderService.java:120)
        - waiting to lock <0x00000000e0a1b2d0> (a com.example.service.OrderService)
        at com.example.controller.OrderController.create(OrderController.java:45)

"GC task thread#0 (ParallelGC)" #1 daemon prio=5 os_prio=0 tid=0x00007f8b8c00d000 nid=0x1237 runnable [0x0000000000000000]
   java.lang.Thread.State: RUNNABLE

"DestroyJavaVM" #45 prio=5 os_prio=0 tid=0x00007f8b8c00e000 nid=0x1238 waiting on condition [0x0000000000000000]
   java.lang.Thread.State: RUNNABLE
'''

SAMPLE_JSTACK_WITH_DEADLOCK = '''2026-06-05 10:30:45
Full thread dump Java HotSpot(TM) 64-Bit Server VM (25.292-b10 mixed mode):

"Thread-1" #10 prio=5 os_prio=0 tid=0x00007f8b8c00a000 nid=0x2001 waiting for monitor entry [0x00007f8b7c5fe000]
   java.lang.Thread.State: BLOCKED (on object monitor)
        at com.example.A.methodA(A.java:10)
        - waiting to lock <0x00000000e0a1b2d0> (a com.example.A)
        - locked <0x00000000e0a1b2e0> (a com.example.B)

"Thread-2" #11 prio=5 os_prio=0 tid=0x00007f8b8c00b000 nid=0x2002 waiting for monitor entry [0x00007f8b7c4fd000]
   java.lang.Thread.State: BLOCKED (on object monitor)
        at com.example.B.methodB(B.java:10)
        - waiting to lock <0x00000000e0a1b2e0> (a com.example.B)
        - locked <0x00000000e0a1b2d0> (a com.example.A)

Found one Java-level deadlock:
=============================
"Thread-1":
  waiting to lock monitor 0x00007f8b8c00a000
  object 0x00000000e0a1b2d0, a com.example.A
  which is held by "Thread-2"
"Thread-2":
  waiting to lock monitor 0x00007f8b8c00b000
  object 0x00000000e0a1b2e0, a com.example.B
  which is held by "Thread-1"

Found 1 deadlock.
'''


# ── StackMatch 测试 ──────────────────────────────────────────────────

class TestStackMatch:
    """StackMatch dataclass 测试。"""

    def test_to_dict(self):
        match = StackMatch(
            thread_name="http-nio-8080-exec-1",
            state="WAITING",
            matched_line="at java.util.concurrent.LinkedBlockingQueue.take(LinkedBlockingQueue.java:442)",
            context_before=["line1", "line2"],
            context_after=["line3", "line4"],
        )
        d = match.to_dict()
        assert d["thread_name"] == "http-nio-8080-exec-1"
        assert d["state"] == "WAITING"
        assert "LinkedBlockingQueue.take" in d["matched_line"]
        assert d["context_before"] == ["line1", "line2"]
        assert d["context_after"] == ["line3", "line4"]


# ── jstack 解析测试 ──────────────────────────────────────────────────

class TestJstackParsing:
    """ThreadStackAnalyzer._parse_thread_blocks() 测试。"""

    def _make_analyzer(self, **overrides) -> ThreadStackAnalyzer:
        config = {
            "host": "10.0.0.1",
            "ssh_user": "app",
            "ssh_key": "~/.ssh/id_rsa",
            "ssh_port": 22,
            "tomcat_process_name": "catalina",
        }
        config.update(overrides)
        return ThreadStackAnalyzer(config)

    def test_parse_thread_blocks(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        # 应该解析出 5 个线程块
        assert len(blocks) == 5
        names = [b["thread_name"] for b in blocks]
        assert "http-nio-8080-exec-1" in names
        assert "http-nio-8080-exec-2" in names
        assert "http-nio-8080-exec-3" in names

    def test_parse_thread_state(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        states = {b["thread_name"]: b["state"] for b in blocks}
        assert states["http-nio-8080-exec-1"] == "WAITING"
        assert states["http-nio-8080-exec-2"] == "RUNNABLE"
        assert states["http-nio-8080-exec-3"] == "BLOCKED"

    def test_parse_blocked_thread(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        blocked = [b for b in blocks if b["state"] == "BLOCKED"]
        assert len(blocked) == 1
        assert blocked[0]["thread_name"] == "http-nio-8080-exec-3"
        assert "OrderService.processOrder" in blocked[0]["stacktrace"]


# ── 关键字过滤测试 ──────────────────────────────────────────────────

class TestKeywordFilter:
    """ThreadStackAnalyzer.filter_by_keyword() 测试。"""

    def _make_analyzer(self, **overrides) -> ThreadStackAnalyzer:
        config = {
            "host": "10.0.0.1",
            "ssh_user": "app",
            "ssh_key": "~/.ssh/id_rsa",
            "tomcat_process_name": "catalina",
        }
        config.update(overrides)
        return ThreadStackAnalyzer(config)

    def test_filter_by_keyword(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        matches = analyzer.filter_by_keyword(blocks, "LinkedBlockingQueue")
        assert len(matches) >= 1
        assert any("LinkedBlockingQueue" in m.matched_line for m in matches)

    def test_filter_by_thread_name(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        matches = analyzer.filter_by_keyword(blocks, "exec-2")
        assert len(matches) == 1
        assert matches[0].thread_name == "http-nio-8080-exec-2"

    def test_filter_no_match(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        matches = analyzer.filter_by_keyword(blocks, "NonExistentKeyword12345")
        assert len(matches) == 0

    def test_filter_case_insensitive(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        matches_lower = analyzer.filter_by_keyword(blocks, "linkedblockingqueue")
        matches_upper = analyzer.filter_by_keyword(blocks, "LINKEDBLOCKINGQUEUE")
        assert len(matches_lower) >= 1
        assert len(matches_lower) == len(matches_upper)

    def test_filter_context_lines(self):
        analyzer = self._make_analyzer()
        blocks = analyzer._parse_thread_blocks(SAMPLE_JSTACK)
        matches = analyzer.filter_by_keyword(blocks, "LinkedBlockingQueue", context_lines=2)
        assert len(matches) >= 1
        m = matches[0]
        # 上下文行数应该不超过 2
        assert len(m.context_before) <= 2
        assert len(m.context_after) <= 2


# ── jstack 获取测试 ──────────────────────────────────────────────────

class TestThreadStackFetch:
    """ThreadStackAnalyzer._fetch_jstack_output() 测试。"""

    def _make_analyzer(self, **overrides) -> ThreadStackAnalyzer:
        config = {
            "host": "10.0.0.1",
            "ssh_user": "app",
            "ssh_key": "~/.ssh/id_rsa",
            "ssh_port": 22,
            "tomcat_process_name": "catalina",
        }
        config.update(overrides)
        return ThreadStackAnalyzer(config)

    @patch("scripts.thread_stack_analyzer.create_executor")
    def test_fetch_via_jstack(self, mock_create):
        mock_exec = MagicMock()
        # 第一次调用 jps 返回 pid，第二次调用 jstack 返回 dump
        jps_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="12345 org.apache.catalina.startup.Bootstrap\n",
            stderr="",
        )
        jstack_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=SAMPLE_JSTACK,
            stderr="",
        )
        mock_exec.run.side_effect = [jps_result, jstack_result]
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer()
        output = analyzer._fetch_jstack_output()
        assert "http-nio-8080-exec-1" in output
        # 应该调用了 jps 和 jstack
        assert mock_exec.run.call_count == 2

    @patch("scripts.thread_stack_analyzer.create_executor")
    def test_fetch_uses_pid_from_jps(self, mock_create):
        mock_exec = MagicMock()
        # 第一次调用 jps 返回 pid
        jps_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="12345 org.apache.catalina.startup.Bootstrap\n",
            stderr="",
        )
        # 第二次调用 jstack 返回 dump
        jstack_result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=SAMPLE_JSTACK,
            stderr="",
        )
        mock_exec.run.side_effect = [jps_result, jstack_result]
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer()
        output = analyzer._fetch_jstack_output()
        assert "12345" in mock_exec.run.call_args_list[1][0][0]

    @patch("scripts.thread_stack_analyzer.create_executor")
    def test_fetch_with_explicit_pid(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=SAMPLE_JSTACK,
            stderr="",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer(pid=99999)
        output = analyzer._fetch_jstack_output()
        # 只调用 jstack，不调用 jps
        assert mock_exec.run.call_count == 1
        cmd = mock_exec.run.call_args[0][0]
        assert "99999" in cmd
        assert "jstack" in cmd

    @patch("scripts.thread_stack_analyzer.create_executor")
    def test_fetch_jstack_failure(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout="",
            stderr="jstack: No such process",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer(pid=12345)
        output = analyzer._fetch_jstack_output()
        assert output == ""


# ── analyze() 完整流程测试 ──────────────────────────────────────────

class TestAnalyzerAnalyze:
    """ThreadStackAnalyzer.analyze() 完整流程测试。"""

    def _make_analyzer(self, **overrides) -> ThreadStackAnalyzer:
        config = {
            "host": "10.0.0.1",
            "ssh_user": "app",
            "ssh_key": "~/.ssh/id_rsa",
            "tomcat_process_name": "catalina",
        }
        config.update(overrides)
        return ThreadStackAnalyzer(config)

    @patch("scripts.thread_stack_analyzer.create_executor")
    def test_analyze_with_keyword(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=SAMPLE_JSTACK,
            stderr="",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer(pid=12345)
        result = analyzer.analyze(keyword="Blocked")
        assert "matches" in result
        assert "summary" in result
        assert result["summary"]["keyword"] == "Blocked"
        assert result["summary"]["total_threads"] == 5

    @patch("scripts.thread_stack_analyzer.create_executor")
    def test_analyze_blocked_threads_deadlock_detected(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=SAMPLE_JSTACK_WITH_DEADLOCK,
            stderr="",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer(pid=12345)
        result = analyzer.analyze()
        assert result["summary"]["deadlock_detected"] is True
        assert result["summary"]["blocked_count"] == 2

    @patch("scripts.thread_stack_analyzer.create_executor")
    def test_analyze_empty_jstack(self, mock_create):
        mock_exec = MagicMock()
        mock_exec.run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1,
            stdout="",
            stderr="error",
        )
        mock_create.return_value = mock_exec

        analyzer = self._make_analyzer(pid=12345)
        result = analyzer.analyze()
        assert result["summary"]["total_threads"] == 0
        assert result["matches"] == []
