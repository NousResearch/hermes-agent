"""Matching in a child forked from a process whose other threads are matching.

RE2 guards each program's DFA cache with a mutex. If another thread holds one at ``fork()``, the
child's copy of that mutex is locked for good, and a match with the inherited program never
returns. The facade therefore drops every inherited program in the child and builds its own.
The stress runs in an interpreter of its own, so a hang can only fail this test.
"""
import pytest

from tests.hermes_re._support import fresh_interpreter

STRESS = r'''
import os, signal, sys, threading
import hermes_re as re

ROUNDS, hung, wrong = 12, 0, 0
words = ["alpha", "beta", "token", "x9", "\u00e9", "key", "=", ":", "secret", "zz"]
texts = [" ".join(words[(i * k + k) % len(words)] for i in range(400)) for k in range(1, 9)]
for number in range(ROUNDS):
    # a pattern nobody has used: its DFA is being built (under RE2's mutex) while the fork happens
    shared = re.compile(r"(?:\w+\s+){1,8}(?:secret|token|key)\s*[:=]\s*\S+" + "|round%d" % number)
    started, stop = threading.Barrier(5), threading.Event()

    def spin(shared=shared, started=started, stop=stop):
        started.wait()
        while not stop.is_set():
            for text in texts:
                shared.search(text)
    threads = [threading.Thread(target=spin, daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    started.wait()
    for _ in range(4):
        pid = os.fork()
        if pid == 0:
            signal.alarm(20)  # a child stuck on an inherited lock dies by SIGALRM
            ok = (shared._built is False and shared.search("a b token = 1") is not None
                  and re.search(r"\d+x", "12x") is not None and shared._built is True)
            os._exit(0 if ok else 3)
        status = os.waitpid(pid, 0)[1]
        hung += os.WIFSIGNALED(status)
        wrong += (not os.WIFSIGNALED(status)) and os.WEXITSTATUS(status) != 0
    stop.set()
    for thread in threads:
        thread.join()
    assert shared._built is True and shared.search("a b token = 1") is not None  # the parent is untouched
print(ROUNDS * 4, hung, wrong)
'''


@pytest.mark.platforms("posix")
def test_a_forked_child_builds_its_own_programs_and_never_hangs_on_an_inherited_one():
    forks, hung, wrong = fresh_interpreter("import warnings; warnings.simplefilter('ignore')\n" + STRESS, timeout=600)
    assert (forks, hung, wrong) == ("48", "0", "0")
