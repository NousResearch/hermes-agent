"""Addresses the machine's network interfaces own now (not cached: interfaces come and go).

Its own module, not ``facts``: a process started on v0.23.3 keeps the ``facts`` it loaded and
imports this lazily after ``hermes update`` moved the checkout (#158723).
"""

from __future__ import annotations

import ipaddress
import socket


def has_interface_address(host: str) -> bool:
    """Does an interface currently own this numeric IP (not just its network or broadcast)?"""
    import psutil

    wanted = ipaddress.ip_address(host)
    return any(ipaddress.ip_address(row.address.partition("%")[0]) == wanted
               for rows in psutil.net_if_addrs().values() for row in rows
               if row.family in (socket.AF_INET, socket.AF_INET6))
