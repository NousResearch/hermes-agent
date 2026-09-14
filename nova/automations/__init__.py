"""Governed automations: declared, compiled, then handed to the runtime's scheduler.

Phase 11 surfaced Hermes' cron engine and withheld *create*, because
``cron.jobs.create_job`` takes a free-text prompt and would have let a control plane hand
an agent a recurring instruction that no policy reviewed. This package is the answer:
an automation is a spec, it compiles against its tenant, and the compiler is the only
path to the scheduler.
"""

from nova.automations.compile import (
    CompiledAutomation,
    ScheduleValidator,
    check_automation_references,
    compile_all,
    compile_automation,
)

__all__ = [
    "CompiledAutomation",
    "ScheduleValidator",
    "check_automation_references",
    "compile_all",
    "compile_automation",
]
