"""Public authorization mixin plus gateway model-authority interception.

The authorization implementation remains in one mutable module namespace so
existing imports and monkeypatches keep targeting the globals used at runtime.
"""

from __future__ import annotations

import sys

from gateway import authz_mixin_impl as _impl
from gateway.model_authority import GatewayModelAuthorityMixin


_OriginalGatewayAuthorizationMixin = _impl.GatewayAuthorizationMixin


class GatewayAuthorizationMixin(
    GatewayModelAuthorityMixin,
    _OriginalGatewayAuthorizationMixin,
):
    """Authorization behavior with durable model-selection settlement."""


_impl.GatewayAuthorizationMixin = GatewayAuthorizationMixin
sys.modules[__name__] = _impl
