"""Synchronous ``threading.Thread`` stand-in for tests.

Tests patch ``<module>.threading.Thread`` with this so background work runs
inside ``start()`` and assertions can follow immediately. That patches the
shared ``threading`` module, so every thread started in the meantime goes
through it too, including the stdlib's own: on Windows ``Popen.communicate()``
reads pipes via ``Thread(target=self._readerthread, args=(fh, buffer))`` and
then ``join()``s them. The fake therefore keeps the real constructor shape and
forwards ``args``/``kwargs`` instead of calling ``target()`` bare.
"""


class ImmediateThread:
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.name = name
        self.daemon = daemon

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None
