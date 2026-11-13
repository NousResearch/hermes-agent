from __future__ import annotations

from hermes_agent.tools.connectors.gateway.errors import ToolGatewayError


class InvalidConnectorSlug(ValueError):
    pass


class PortalConnectorUnavailable(ToolGatewayError):
    pass


class PortalToolsUnavailable(PortalConnectorUnavailable):
    pass
