# Gateway timeout logs and timestamped console output

Date: 2026-05-02
Branch: fix/gateway-timeout-log-timestamps-20260502030014
Worktree: /work/.hermes-data/hermes-agent-gateway-timeout-logs-20260502030014

## Problem

A gateway log excerpt showed noisy Python stack traces ending in `httpx.ReadTimeout` / `openai.APITimeoutError` during `session_search` summarization, followed by context compaction. The docker log prefix only showed `hermes-gateway |`, making it harder to correlate incidents without timestamps.

## Facts from inspection

- `tools/session_search_tool.py::_summarize_session` retries all generic exceptions three times and logs the final failure with `exc_info=True`, which emits a large stack trace for expected upstream timeout failures.
- `session_search()` only catches `concurrent.futures.TimeoutError` around `_run_async(_summarize_all())`; per-call OpenAI/httpx timeout exceptions are handled inside `_summarize_session` as generic exceptions.
- `gateway/run.py::start_gateway` attaches a stderr `StreamHandler` when verbosity is not `None`; its formatter is currently `'%(levelname)s %(name)s: %(message)s'`, so container stdout/stderr lines have no application timestamp unless Docker is configured separately.
- `hermes_logging.py` already defines timestamped file-log formats and `setup_verbose_logging()` uses timestamped console formatting, but gateway stderr does not.

## Scope

Implement two low-risk behavior improvements:

1. Treat session-search LLM timeout failures as expected, concise warnings rather than multi-line stack traces.
2. Add a timestamp to gateway stderr log lines so docker compose logs contain time context next to the `hermes-gateway |` prefix.

Out of scope:

- Changing Docker Compose logging driver configuration.
- Changing provider/client timeout durations globally.
- Changing context compaction behavior.
- Reworking the session_search LLM summarization pipeline.

## No-ADR rationale

No ADR is needed. This is an operational logging/triage refinement with no architecture, persistence, API, or product behavior contract change. The plan and tests are sufficient documentation.

## Test strategy

RED first:

1. Add a unit test in `tests/tools/test_session_search.py` that patches `async_call_llm` to raise a timeout exception, calls `_summarize_session()`, and asserts:
   - result is `None`
   - warning is concise
   - `exc_info` is not attached for expected timeout warnings
2. Add a gateway startup test in `tests/gateway/test_runner_startup_failures.py` or logging test in `tests/test_hermes_logging.py` that exercises the gateway stderr handler and asserts the formatter output starts with an HH:MM:SS timestamp before the level/name/message.

Expected RED cause:

- Session-search timeout warning currently logs stack traces via `exc_info=True` after generic retry exhaustion.
- Gateway stderr formatter currently emits `WARNING gateway.run: message` without a timestamp.

GREEN implementation:

- Add a narrow timeout classifier/helper in `tools/session_search_tool.py` that recognizes OpenAI/httpx/httpcore/asyncio timeout exception classes without requiring all optional imports to be present.
- On expected per-session summarization timeout failures, log a concise warning and return `None` rather than retrying three times and printing a traceback. This intentionally trades retry recovery for lower gateway stalls and cleaner logs for known timeout failures; unexpected exceptions still use the existing retry/final traceback path.
- On outer `_run_async` bridge timeouts, keep the existing JSON error response but log a concise warning without `exc_info=True`.
- Update gateway stderr formatter in `gateway/run.py` to include `%(asctime)s` with a short date format (HH:MM:SS) while preserving redaction and existing level/name/message fields.

Validation commands:

- `python -m pytest tests/tools/test_session_search.py -q`
- `python -m pytest tests/gateway/test_runner_startup_failures.py -q`
- `python -m pytest tests/test_hermes_logging.py -q`
- broader relevant regression if targeted tests are green: `python -m pytest tests/tools/test_session_search.py tests/gateway/test_runner_startup_failures.py tests/test_hermes_logging.py -q`
- final code work guard with `HERMES_TDD_EVIDENCE` pointing to RED/GREEN output artifacts.

## Acceptance criteria

- Session search timeout handling no longer emits a full stack trace for expected timeout failures.
- Session search still logs unexpected summarization failures with traceback for debugging.
- Gateway stderr log lines include a timestamp before level/name/message.
- Targeted tests pass.
- Devlog is updated before PR.
