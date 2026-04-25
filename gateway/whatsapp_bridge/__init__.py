"""Embedded WhatsApp bridge (Node.js).

This package vendors the small Node.js daemon that talks to whatsapp-web.js.
It lives inside the ``gateway`` package so setuptools picks it up as
package-data and ships it to ``site-packages/gateway/whatsapp_bridge/`` —
which fixes #15336 (NixOS / pip-installed builds previously had no copy
on disk because ``scripts/whatsapp-bridge/`` was outside any Python
package).

The directory is intentionally a regular package (with this ``__init__``)
rather than a namespace package so package-data globs resolve cleanly
on every modern setuptools.  Nothing in here is meant to be imported
from Python — the consumer (``gateway.platforms.whatsapp``) launches
``bridge.js`` via ``subprocess``.
"""
