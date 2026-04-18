from pathlib import Path
import importlib.util

script_path = Path.home() / '.hermes' / 'scripts' / 'hermes_gateway_healthcheck.py'
spec = importlib.util.spec_from_file_location('healthcheck', script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_count_matches_counts_distinct_lines_once_per_line():
    lines = [
        '409 Conflict: terminated by other getUpdates request',
        'httpx.ConnectError: boom',
    ]
    assert module.count_matches(lines, module.TOKEN_CONFLICT_PATTERNS) == 1
    assert module.count_matches(lines, module.NETWORK_PATTERNS) == 1


def test_analyze_recent_conflicts_does_not_flag_old_conflict_when_tail_is_clean(tmp_path):
    error_log = tmp_path / 'gateway.error.log'
    lines = ['409 Conflict: Telegram bot token already in use'] + [f'clean line {i}' for i in range(250)]
    error_log.write_text('\n'.join(lines), encoding='utf-8')

    recent = module.tail_lines(error_log, 200)
    assert module.count_matches(recent, module.TOKEN_CONFLICT_PATTERNS) == 0


def test_tail_lines_keeps_recent_conflict_in_window(tmp_path):
    error_log = tmp_path / 'gateway.error.log'
    lines = [f'clean line {i}' for i in range(199)] + ['409 Conflict: another getUpdates request']
    error_log.write_text('\n'.join(lines), encoding='utf-8')

    recent = module.tail_lines(error_log, 200)
    assert module.count_matches(recent, module.TOKEN_CONFLICT_PATTERNS) == 1
