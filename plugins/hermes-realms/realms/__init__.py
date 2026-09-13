"""Private compositor runtime: unsupported hosts fail before any registration.

All sibling modules rely on Linux process, socket and ownership primitives.
Keep this gate before their imports rather than weakening those checks on Windows.
"""
import platform

if platform.system() != "Linux":
    raise RuntimeError("Hermes Realms requires Linux; leave this optional plugin disabled on this host")
