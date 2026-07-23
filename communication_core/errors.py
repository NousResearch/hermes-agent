"""Typed fail-closed errors for Communication Core."""


class CommunicationError(RuntimeError):
    """Base class for domain failures safe to surface through the CLI."""


class DatabaseMissingError(CommunicationError):
    """A read path was requested before explicit storage initialization."""


class AccountRequiredError(CommunicationError):
    """An account-scoped operation omitted its exact connected account."""


class AccountUnavailableError(CommunicationError):
    """The selected account is missing, disabled, unhealthy, or needs auth."""


class CapabilityUnsupportedError(CommunicationError):
    """The selected adapter does not declare the requested capability."""


class ScopeViolationError(CommunicationError):
    """Data from different account/contact namespaces would be mixed."""


class RouteDeniedError(CommunicationError):
    """A directed account or person route is not explicitly allowed."""


class ApprovalInvalidError(CommunicationError):
    """An approval is absent, expired, consumed, or no longer exact."""


class AmbiguousIdentityError(CommunicationError):
    """An identity operation needs manual evidence or disambiguation."""
