"""Hardened Telegram Mini App for read-only Hermes observability.

The ASGI module is intentionally not imported here: lifecycle commands must be
able to load the package before the Mini App's dedicated environment is read.
"""
