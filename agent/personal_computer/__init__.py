"""Shared task primitives for Hermes Personal Computer mode.

This package defines persistent task state only. It does not execute commands,
control a browser, or bypass the normal terminal/tool approval policy.
"""

from agent.personal_computer.tasks import PersonalComputerTask, PersonalComputerTaskStore, TaskStatus

__all__ = ["PersonalComputerTask", "PersonalComputerTaskStore", "TaskStatus"]
