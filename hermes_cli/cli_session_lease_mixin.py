"""HermesCLI's session id, kept in step with its active-session lease.

The lease is claimed for the id the CLI starts with, but the id moves afterwards: the gateway
session the CLI opens has its own (``local-…``) id, and ``/new``, ``/branch``, resume and
compression rotations reassign it. A lease left on the old id fences nothing and makes every
lease reader (``hermes status --json``, the session lists' live verdict) miss the session.
"""

from __future__ import annotations


class CLISessionLeaseMixin:
    @property
    def session_id(self):
        return self._cli_session_id

    @session_id.setter
    def session_id(self, value) -> None:
        self._cli_session_id = value
        lease = getattr(self, "_active_session_lease", None)
        if lease is None or not value or lease.session_id == str(value):
            return
        from hermes_cli.active_sessions import transfer_active_session

        transfer_active_session(lease, session_id=str(value), metadata={"live_session_id": str(value)})
