"""Fail-closed errors for the Agent Computer control plane."""

from __future__ import annotations

from typing import Any


class AgentComputerError(Exception):
    code = "AGENT_COMPUTER_ERROR"
    http_status = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = dict(details or {})


class NotFoundError(AgentComputerError):
    code = "NOT_FOUND"
    http_status = 404


class ConflictError(AgentComputerError):
    code = "CONFLICT"
    http_status = 409


class ForbiddenError(AgentComputerError):
    code = "FORBIDDEN"
    http_status = 403


class UnauthorizedError(AgentComputerError):
    code = "UNAUTHORIZED"
    http_status = 401


class IdentityBusyError(AgentComputerError):
    code = "BROWSER_IDENTITY_BUSY"
    http_status = 409


class StaleControllerError(AgentComputerError):
    code = "STALE_CONTROLLER"
    http_status = 409


class ObserveRequiredError(AgentComputerError):
    code = "OBSERVE_REQUIRED"
    http_status = 409


class CheckpointRequiredError(AgentComputerError):
    code = "CHECKPOINT_REQUIRED"
    http_status = 409


class InvalidTokenError(AgentComputerError):
    code = "INVALID_TAKEOVER_TOKEN"
    http_status = 403


class RevokedError(AgentComputerError):
    code = "IDENTITY_REVOKED"
    http_status = 409
