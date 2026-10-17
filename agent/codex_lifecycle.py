"""Ownership of retiring Codex clients, scoped to one gateway or API host."""

import os
import threading


def codex_writer_key(thread_id, codex_home=None):
    home = codex_home or os.environ.get("CODEX_HOME") or os.path.expanduser("~/.codex")
    return os.path.realpath(home), thread_id


def close_codex_session(agent):
    """Keep the owner reachable until close confirms exit; concurrent closers join the session lock."""
    session = getattr(agent, "_codex_session", None)
    if session is not None:
        session.close()
        if getattr(agent, "_codex_session", None) is session:
            agent._codex_session = None


class CodexRetirements:
    """Capture owners before cache removal; successors may help retire only their own writer.

    Failed closes retain the concrete session/Popen for the next attempt. No failed Future
    poisons a routing key, and unrelated threads or providers never wait on these owners.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._owners = {}

    def track(self, agent):
        session = getattr(agent, "_codex_session", None)
        if session is not None:
            with self._lock:
                self._owners.setdefault(id(agent), (agent, session, session.writer_key))

    def release(self, agent):
        with self._lock:
            owner = self._owners.get(id(agent))
        if owner is None:
            return
        _, session, _ = owner
        session.close()
        if getattr(agent, "_codex_session", None) is session:
            agent._codex_session = None
        with self._lock:
            if self._owners.get(id(agent)) is owner:
                del self._owners[id(agent)]

    def wait(self, writer_key):
        if not writer_key[1]:
            return
        with self._lock:
            agents = [agent for agent, _, key in self._owners.values() if key == writer_key]
        for agent in agents:
            # Help directly: another owner's memory flush must not delay this writer.
            self.release(agent)


def codex_retirements(host):
    return host.__dict__.setdefault("_codex_retirements", CodexRetirements())


def wait_for_codex_writer(agent):
    retirements = getattr(agent, "_codex_retirements", None)
    if retirements is not None:
        retirements.wait(agent._codex_session.writer_key)
