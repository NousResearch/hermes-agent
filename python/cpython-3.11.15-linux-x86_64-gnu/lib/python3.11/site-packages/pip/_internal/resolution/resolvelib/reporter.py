from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from logging import getLogger
from typing import Any

from pip._vendor.resolvelib.reporters import BaseReporter

from .base import Candidate, Constraint, Requirement

logger = getLogger(__name__)


class PipReporter(BaseReporter[Requirement, Candidate, str]):
    def __init__(self, constraints: Mapping[str, Constraint] | None = None) -> None:
        self.reject_count_by_package: defaultdict[str, int] = defaultdict(int)
        self._constraints = constraints or {}

        self._messages_at_reject_count = {
            1: (
                "pip is looking at multiple versions of {package_name} to "
                "determine which version is compatible with other "
                "requirements. This could take a while."
            ),
            8: (
                "pip is still looking at multiple versions of {package_name} to "
                "determine which version is compatible with other "
                "requirements. This could take a while."
            ),
            13: (
                "This is taking longer than usual. You might need to provide "
                "the dependency resolver with stricter constraints to reduce "
                "runtime. See https://pip.pypa.io/warnings/backtracking for "
                "guidance. If you want to abort this run, press Ctrl + C."
            ),
        }

    def rejecting_candidate(self, criterion: Any, candidate: Candidate) -> None:
        """Report a candidate being rejected.

        Logs both the rejection count message (if applicable) and details about
        the requirements and constraints that caused the rejection.
        """
        self.reject_count_by_package[candidate.name] += 1

        count = self.reject_count_by_package[candidate.name]
        if count in self._messages_at_reject_count:
            message = self._messages_at_reject_count[count]
            logger.info("INFO: %s", message.format(package_name=candidate.name))

        msg = "Will try a different candidate, due to conflict:"
        for req_info in criterion.information:
            req, parent = req_info.requirement, req_info.parent
            msg += "\n    "
            if parent:
                msg += f"{parent.name} {parent.version} depends on "
            else:
                msg += "The user requested "
            msg += req.format_for_error()

        # Add any relevant constraints
        if self._constraints:
            name = candidate.name
            constraint = self._constraints.get(name)
            if constraint and constraint.specifier:
                constraint_text = f"{name}{constraint.format_for_error()}"
                msg += f"\n    The user requested (constraint) {constraint_text}"

        logger.debug(msg)


class PipDebuggingReporter(BaseReporter[Requirement, Candidate, str]):
    """A reporter that does an info log for every event it sees."""

    def starting(self) -> None:
        logger.info("Reporter.starting()")

    def starting_round(self, index: int) -> None:
        logger.info("Reporter.starting_round(%r)", index)

    def ending_round(self, index: int, state: Any) -> None:
        logger.info("Reporter.ending_round(%r, state)", index)
        logger.debug("Reporter.ending_round(%r, %r)", index, state)

    def ending(self, state: Any) -> None:
        logger.info("Reporter.ending(%r)", state)

    def adding_requirement(
        self, requirement: Requirement, parent: Candidate | None
    ) -> None:
        logger.info("Reporter.adding_requirement(%r, %r)", requirement, parent)

    def rejecting_candidate(self, criterion: Any, candidate: Candidate) -> None:
        logger.info("Reporter.rejecting_candidate(%r, %r)", criterion, candidate)

    def pinning(self, candidate: Candidate) -> None:
        logger.info("Reporter.pinning(%r)", candidate)
