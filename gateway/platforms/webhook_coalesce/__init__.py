"""Durable webhook coalescing.

config validates routes and groups events; ingress owns per-delivery decisions;
binding validates authority provenance; store retains pending batches; recovery
hands sealed batches to the existing producer ledger.
"""

__all__ = []
