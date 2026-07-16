from __future__ import annotations

import os
from contextlib import contextmanager

import cron.scheduler as scheduler


class _CaptureLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []

    def warning(self, message: str, *args) -> None:
        self.records.append({"level": "warning", "message": message % args})

    def debug(self, message: str, *args) -> None:
        self.records.append({"level": "debug", "message": message % args})


def run_case(case: dict):
    if case["kind"] == "timeout":
        return _run_timeout_case(case)
    if case["kind"] == "reasoning":
        return _run_reasoning_case(case)
    raise AssertionError(f"unknown scheduler_ext case kind: {case['kind']!r}")


def _run_timeout_case(case: dict):
    logger = _CaptureLogger()

    def load_config():
        if "config_error" in case:
            raise RuntimeError(case["config_error"])
        return case.get("config")

    with _patched_env(case.get("env") or {}):
        helper = _scheduler_ext_helper("get_script_timeout")
        if helper is None:
            old_timeout = scheduler._SCRIPT_TIMEOUT
            old_loader = scheduler.load_config
            old_logger = scheduler.logger
            try:
                scheduler._SCRIPT_TIMEOUT = case["script_timeout"]
                scheduler.load_config = load_config
                scheduler.logger = logger
                result = scheduler._get_script_timeout()
            finally:
                scheduler._SCRIPT_TIMEOUT = old_timeout
                scheduler.load_config = old_loader
                scheduler.logger = old_logger
        else:
            result = helper(
                case["script_timeout"],
                scheduler._DEFAULT_SCRIPT_TIMEOUT,
                load_config=load_config,
                logger=logger,
            )
    return {"return": result, "messages": logger.records, "db": []}


def _run_reasoning_case(case: dict):
    helper = _scheduler_ext_helper("resolve_cron_reasoning_config")
    if helper is None:
        from hermes_constants import parse_reasoning_effort, resolve_reasoning_config

        reasoning_config = None
        _job_effort = str(case.get("job", {}).get("reasoning_effort") or "").strip()
        if _job_effort:
            reasoning_config = parse_reasoning_effort(_job_effort)
        if reasoning_config is None:
            cfg = case.get("config")
            reasoning_config = resolve_reasoning_config(
                cfg if isinstance(cfg, dict) else {}, str(case.get("model") or "")
            )
        result = reasoning_config
    else:
        result = helper(case.get("job") or {}, case.get("config"), str(case.get("model") or ""))
    return {"return": result, "messages": [], "db": []}


def _scheduler_ext_helper(name: str):
    try:
        import cron.fork_ext.scheduler_ext as scheduler_ext
    except ModuleNotFoundError:
        return None
    return getattr(scheduler_ext, name)


@contextmanager
def _patched_env(values: dict[str, str]):
    old = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        if "HERMES_CRON_SCRIPT_TIMEOUT" not in values:
            os.environ.pop("HERMES_CRON_SCRIPT_TIMEOUT", None)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
