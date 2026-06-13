from cron.scheduler import _resolve_cron_disabled_toolsets, _resolve_cron_enabled_toolsets
from toolsets import CORP_DANGEROUS_TOOLSETS
from model_tools import get_tool_definitions

def test_cron_denylist_contains_all_dangerous_toolsets():
    disabled = set(_resolve_cron_disabled_toolsets({}))
    missing = set(CORP_DANGEROUS_TOOLSETS) - disabled
    assert not missing, f"cron denylist missing dangerous toolsets: {missing}"
    # plus the always-disabled interactive/recursive ones
    assert {"cronjob", "messaging", "clarify"} <= disabled

def test_cron_denylist_layers_user_disabled():
    disabled = set(_resolve_cron_disabled_toolsets({"agent": {"disabled_toolsets": ["spotify"]}}))
    assert "spotify" in disabled
    assert "terminal" in disabled  # corp dangerous still present


def _schema_names(defs):
    s = set()
    for d in defs or []:
        fn = d.get("function") if isinstance(d, dict) else None
        if isinstance(fn, dict) and "name" in fn:
            s.add(fn["name"])
        elif isinstance(d, dict) and "name" in d:
            s.add(d["name"])
    return s


def test_cron_per_job_enabled_terminal_cannot_escalate():
    # A Time agent schedules a cron job asking for terminal directly.
    job = {"enabled_toolsets": ["terminal", "code_execution", "delegation"]}
    cfg = {}
    enabled = _resolve_cron_enabled_toolsets(job, cfg)      # per-job wins → ["terminal", ...]
    disabled = _resolve_cron_disabled_toolsets(cfg)         # corp dangerous always denied
    names = _schema_names(get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled))
    leaked = names & {"terminal", "process", "execute_code", "delegate_task"}
    assert not leaked, f"cron escalation leaked dangerous tools: {sorted(leaked)}"
