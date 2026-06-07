import os
import portalocker

class FileLock:
    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.fd = None

    def __enter__(self):
        self.fd = open(self.lock_file, 'w')
        portalocker.lock(self.fd, portalocker.LOCK_EX | portalocker.LOCK_NB)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.fd:
            portalocker.unlock(self.fd)
            self.fd.close()
