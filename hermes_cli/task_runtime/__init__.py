"""Task Runtime public package.

Re-exports the TaskRuntime coordinator so callers can do:

    from hermes_cli.task_runtime import TaskRuntime
    runtime = TaskRuntime()
    result = runtime.run("Explain Producer Normalizer v1.1", execution_mode="dry-run")
"""

from hermes_cli.task_runtime.runtime import TaskRuntime
from hermes_cli.task_runtime.intent_resolver import ResolvedIntent
from hermes_cli.task_runtime.task_contract_builder import (
    CONTRACT_SCHEMA,
    CONTRACT_VERSION,
    SCHEMA_VERSION,
)
from hermes_cli.task_runtime.execution_pipeline import PipelineResult
from hermes_cli.task_runtime.result_consolidator import TaskResult

__all__ = [
    "TaskRuntime",
    "ResolvedIntent",
    "TaskResult",
    "PipelineResult",
    "CONTRACT_SCHEMA",
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
]