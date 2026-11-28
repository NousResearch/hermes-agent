"""The memory reference does not model physical recipient generations or webhook retention.

Expose the declared repository door while refusing the optional capability explicitly;
SQLite's behavioral contract owns this implementation, not a second fake ledger.
"""
from persistence import CapabilityUnsupported
from ..backend import register_repository


class UnsupportedWebhookDeliveries:
    def __init__(self, store):
        self._store = store

    def get(self, ledger, origin_id):
        raise CapabilityUnsupported('webhook_delivery_fences')

    def publish(self, *, epoch, record, native=None, claim=False, gate=None):
        raise CapabilityUnsupported('webhook_delivery_fences')

    def candidates(self, ledger, *, after='', limit=8):
        raise CapabilityUnsupported('webhook_delivery_fences')

    def inspection(self, *, after='', routes=None, limit=51):
        raise CapabilityUnsupported('webhook_delivery_fences')

    def check_owner(self, *, epoch, gate=None):
        raise CapabilityUnsupported('webhook_delivery_fences')

    def capture_incarnation(self):
        raise CapabilityUnsupported('webhook_delivery_fences')

    def require_incarnation(self, expected):
        raise CapabilityUnsupported('webhook_delivery_fences')


register_repository('webhook_deliveries', UnsupportedWebhookDeliveries)
