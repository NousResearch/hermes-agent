"""Helper for test_serve_profile_singleton: take an exclusive flock on the
path in sys.argv[1] and HOLD it until killed. The test reads the lock file's
existence + a peer-process probe (via /proc on POSIX) to confirm contention.
Prints nothing — the parent probes the lock state directly."""
import fcntl, os, sys, time
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
h = open(path, "a+b")
fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
# Hold until SIGTERM. Parent kills us when done.
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
