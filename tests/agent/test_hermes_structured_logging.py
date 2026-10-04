"""Unit tests for agent/hermes/structured_logging.py."""

import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.hermes.structured_logging import (
    _log_context,
    set_log_context,
    get_log_context,
    clear_log_context,
    HermesJSONFormatter,
    HermesStructuredLogHandler,
    StructuredLoggerAdapter,
    get_structured_logger,
)


class TestLogContext(unittest.TestCase):
    def tearDown(self):
        clear_log_context()

    def test_set_and_get_session_id(self):
        set_log_context(session_id="sess-abc")
        sid, _ = get_log_context()
        self.assertEqual(sid, "sess-abc")

    def test_set_and_get_trace_id(self):
        set_log_context(trace_id="trace-xyz")
        _, tid = get_log_context()
        self.assertEqual(tid, "trace-xyz")

    def test_trace_id_auto_generated_when_empty(self):
        set_log_context(trace_id="")
        _, tid = get_log_context()
        self.assertNotEqual(tid, "")
        self.assertEqual(len(tid), 16)

    def test_clear_log_context(self):
        set_log_context(session_id="sess-123", trace_id="trace-456")
        clear_log_context()
        sid, tid = get_log_context()
        self.assertEqual(sid, "")
        self.assertEqual(tid, "")

    def test_both_ids_set(self):
        set_log_context(session_id="sess-xyz", trace_id="trace-abc")
        sid, tid = get_log_context()
        self.assertEqual(sid, "sess-xyz")
        self.assertEqual(tid, "trace-abc")


class TestLogContextThreadIsolation(unittest.TestCase):
    def tearDown(self):
        clear_log_context()

    def test_thread_isolation(self):
        results = {}

        def worker(session_id, trace_id, results_dict, key):
            set_log_context(session_id=session_id, trace_id=trace_id)
            results_dict[key] = get_log_context()

        t1 = threading.Thread(target=worker, args=("sess-1", "trace-1", results, "t1"))
        t2 = threading.Thread(target=worker, args=("sess-2", "trace-2", results, "t2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread's context should be independent
        self.assertEqual(results["t1"], ("sess-1", "trace-1"))
        self.assertEqual(results["t2"], ("sess-2", "trace-2"))


class TestHermesJSONFormatter(unittest.TestCase):
    def test_format_includes_required_fields(self):
        formatter = HermesJSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        line = formatter.format(record)
        data = json.loads(line)
        self.assertEqual(data["level"], "INFO")
        self.assertEqual(data["message"], "hello world")
        self.assertEqual(data["logger"], "test.logger")

    def test_format_includes_timestamp(self):
        formatter = HermesJSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="test", args=(), exc_info=None,
        )
        line = formatter.format(record)
        data = json.loads(line)
        self.assertIn("timestamp", data)

    def test_format_uses_context_session_id(self):
        set_log_context(session_id="ctx-sess")
        try:
            formatter = HermesJSONFormatter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=0, msg="test", args=(), exc_info=None,
            )
            line = formatter.format(record)
            data = json.loads(line)
            self.assertEqual(data["session_id"], "ctx-sess")
        finally:
            clear_log_context()

    def test_format_uses_context_trace_id(self):
        set_log_context(trace_id="ctx-trace")
        try:
            formatter = HermesJSONFormatter()
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=0, msg="test", args=(), exc_info=None,
            )
            line = formatter.format(record)
            data = json.loads(line)
            self.assertEqual(data["trace_id"], "ctx-trace")
        finally:
            clear_log_context()

    def test_record_level_name(self):
        formatter = HermesJSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.WARNING, pathname="",
            lineno=0, msg="warn", args=(), exc_info=None,
        )
        line = formatter.format(record)
        data = json.loads(line)
        self.assertEqual(data["level"], "WARNING")

    def test_extra_fields_included(self):
        formatter = HermesJSONFormatter(include_extra=True)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="msg", args=(), exc_info=None,
        )
        record.custom_field = "custom_value"
        line = formatter.format(record)
        data = json.loads(line)
        self.assertEqual(data["extra"]["custom_field"], "custom_value")

    def test_redaction_function_applied(self):
        redaction_fn = lambda s: s.replace("SECRET", "[REDACTED]")
        formatter = HermesJSONFormatter(redaction_fn=redaction_fn)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="SECRET data", args=(), exc_info=None,
        )
        line = formatter.format(record)
        data = json.loads(line)
        self.assertEqual(data["message"], "[REDACTED] data")

    def test_exc_info_included(self):
        formatter = HermesJSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="",
                lineno=0, msg="error", args=(), exc_info=sys.exc_info(),
            )
        line = formatter.format(record)
        data = json.loads(line)
        self.assertIn("exception", data)

    def test_module_and_function_included(self):
        formatter = HermesJSONFormatter()
        record = logging.LogRecord(
            name="my.module", level=logging.INFO, pathname="myfile.py",
            lineno=42, msg="test", args=(), exc_info=None,
        )
        record.funcName = "my_function"
        line = formatter.format(record)
        data = json.loads(line)
        self.assertEqual(data["module"], "myfile.py")
        self.assertEqual(data["function"], "my_function")
        self.assertEqual(data["line"], 42)

    def test_json_serializable(self):
        """Format must return valid JSON even with non-serializable extras."""
        formatter = HermesJSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="",
            lineno=0, msg="msg", args=(), exc_info=None,
        )
        record.bad_field = object()  # not JSON-serializable
        line = formatter.format(record)
        data = json.loads(line)
        # Should fall back gracefully
        self.assertIn("message", data)


class TestStructuredLoggerAdapter(unittest.TestCase):
    def tearDown(self):
        clear_log_context()

    def test_injects_session_id_from_context(self):
        set_log_context(session_id="adapter-sess", trace_id="adapter-trace")
        base = logging.getLogger("test.adapter")
        adapter = StructuredLoggerAdapter(base, {})
        # Process should inject session/trace IDs into extra
        msg, kwargs = adapter.process("hello", {})
        self.assertEqual(kwargs["extra"]["session_id"], "adapter-sess")
        self.assertEqual(kwargs["extra"]["trace_id"], "adapter-trace")

    def test_per_call_override_takes_precedence(self):
        set_log_context(session_id="ctx-sess")
        base = logging.getLogger("test.override")
        adapter = StructuredLoggerAdapter(base, {})
        msg, kwargs = adapter.process("hello", {"extra": {"session_id": "override-sess"}})
        # Per-call override should win
        self.assertEqual(kwargs["extra"]["session_id"], "override-sess")

    def test_does_not_mutate_caller_extra(self):
        set_log_context(session_id="sess-x")
        base = logging.getLogger("test.mutation")
        adapter = StructuredLoggerAdapter(base, {})
        caller_extra = {"my_key": "my_value"}
        adapter.process("hello", {"extra": caller_extra})
        # Original dict should not be mutated
        self.assertNotIn("session_id", caller_extra)


class TestGetStructuredLogger(unittest.TestCase):
    def tearDown(self):
        clear_log_context()

    def test_returns_adapter(self):
        logger = get_structured_logger("test.get_logger")
        self.assertIsInstance(logger, StructuredLoggerAdapter)

    def test_sets_context_when_session_id_provided(self):
        logger = get_structured_logger("test.with_ctx", session_id="logger-sess")
        # The adapter's process should now inject this
        _, kwargs = logger.process("test", {})
        self.assertEqual(kwargs["extra"]["session_id"], "logger-sess")

    def test_uses_existing_logger(self):
        base_name = "test.duplicate"
        logger1 = get_structured_logger(base_name)
        logger2 = get_structured_logger(base_name)
        # Both should wrap the same base logger
        self.assertEqual(logger1.logger, logger2.logger)


class TestHermesStructuredLogHandler(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_emit_writes_json_line_to_file(self):
        log_path = Path(self.tmpdir) / "test.jsonl"
        handler = HermesStructuredLogHandler(
            log_path,
            max_bytes=50_000,
            backup_count=1,
        )
        logger = logging.getLogger("test.emit")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        try:
            logger.info("hello from handler")
        finally:
            handler.close()
            logger.removeHandler(handler)

        lines = log_path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["message"], "hello from handler")

    def test_emit_with_session_and_trace_ids(self):
        log_path = Path(self.tmpdir) / "test2.jsonl"
        handler = HermesStructuredLogHandler(
            log_path,
            session_id="handler-sess",
            trace_id="handler-trace",
        )
        logger = logging.getLogger("test.emit2")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        try:
            # The formatter reads from thread context, not handler's own fields.
            # Use set_log_context so the formatter picks up the session/trace IDs.
            set_log_context(session_id="ctx-sess", trace_id="ctx-trace")
            logger.info("with context")
        finally:
            clear_log_context()
            handler.close()
            logger.removeHandler(handler)

        lines = log_path.read_text().strip().split("\n")
        data = json.loads(lines[0])
        self.assertEqual(data["session_id"], "ctx-sess")
        self.assertEqual(data["trace_id"], "ctx-trace")

    def test_emit_with_event_bus_does_not_raise(self):
        log_path = Path(self.tmpdir) / "test3.jsonl"
        mock_bus = MagicMock()
        handler = HermesStructuredLogHandler(
            log_path,
            event_bus=mock_bus,
            session_id="bus-sess",
        )
        logger = logging.getLogger("test.emit3")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        try:
            logger.info("with bus")
        finally:
            handler.close()
            logger.removeHandler(handler)

        # EventBus.emit should have been called
        self.assertTrue(mock_bus.emit.called)


if __name__ == "__main__":
    import sys
    sys.exit(unittest.main())
