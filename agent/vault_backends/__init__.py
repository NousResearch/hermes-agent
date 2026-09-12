"""Login backends for the browser credential vault.

The local Fernet store (``agent/vault_store.py``) is one backend; 1Password
(``op``), Bitwarden Password Manager (``bw``) and KeePassXC
(``keepassxc-cli``) are the others. Every backend
hands the agent the same shape — opaque handle + login metadata — and resolves
the password server-side at fill time only. Handles are namespaced by backend
(``vault_…`` local, ``op:…``, ``bw:…``, ``keepass:…``) so the browser tools need
no schema change to route to the right one.

External managers are locked until the user unlocks them for the current
session (``agent/vault_backends/unlock.py``); the master password is typed
into a masked prompt owned by the surface (CLI panel, Desktop dialog) and is
never a tool argument, never argv, never persisted. KeePassXC mints no session
token, so its database key is what the session store holds — still in-process,
profile-scoped, and never written to disk.
"""

from agent.vault_backends.base import LoginBackend, UnlockRequired, backend_for_handle, enabled_backends

__all__ = ["LoginBackend", "UnlockRequired", "backend_for_handle", "enabled_backends"]
