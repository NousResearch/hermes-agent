"""Typed credential dataclasses (future home).

F-C2 step 1 scaffold — this module is intentionally empty. F-M7 will
populate it with typed credential containers (``AnthropicCredentials``,
``OpenRouterCredentials``, ``GoogleCredentials``, …) that replace the
``dict[str, Any]`` blobs currently returned by the per-provider
resolution functions.

Keeping the new types in a dedicated module (rather than
``__init__.py``) keeps the import graph narrow: providers depend on
``.types`` for their return type, callers depend on ``.types`` only
when they need to introspect credentials, and the package root stays
focused on the public resolution functions.
"""
