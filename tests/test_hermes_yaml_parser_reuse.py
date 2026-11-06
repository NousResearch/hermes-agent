"""Parser-instance reuse in hermes_yaml.safe_load.

Every safe_load call constructed a fresh ruamel ``YAML(typ="safe")`` — ~50us of
Resolver/scanner setup per parse, ~90ms of a cold 852-skill scan. Ruamel instances
are reusable for repeated loads on one thread; the sharing ban is cross-thread, so
a thread-local cache keeps that contract while paying construction once.
"""

import threading
from unittest.mock import patch


def test_safe_load_constructs_one_parser_per_thread():
    import hermes_yaml

    constructed = []
    real_yaml = hermes_yaml.YAML

    class CountingYAML(real_yaml):
        def __init__(self, *args, **kwargs):
            constructed.append(1)
            super().__init__(*args, **kwargs)

    with patch.object(hermes_yaml, "YAML", CountingYAML):
        for _ in range(50):
            assert hermes_yaml.safe_load("a: 1\n") == {"a": 1}

    assert len(constructed) <= 1, (
        f"50 safe_load calls constructed {len(constructed)} parsers; "
        "each thread must reuse one parser instance"
    )


def test_parser_reuse_stays_thread_partitioned():
    """One construction per thread, and results stay correct under concurrency."""
    import hermes_yaml

    constructed = []
    real_yaml = hermes_yaml.YAML

    class CountingYAML(real_yaml):
        def __init__(self, *args, **kwargs):
            constructed.append(threading.current_thread().name)
            super().__init__(*args, **kwargs)

    docs = [{"a": i, "b": "y" if i % 2 else "n"} for i in range(40)]
    results = {}
    with patch.object(hermes_yaml, "YAML", CountingYAML):
        def worker(name, chunk):
            results[name] = [hermes_yaml.safe_load(f"a: {d['a']}\nb: {d['b']}\n") for d in chunk]

        t1 = threading.Thread(target=worker, args=("t1", docs[:20]))
        t2 = threading.Thread(target=worker, args=("t2", docs[20:]))
        t1.start(); t2.start(); t1.join(); t2.join()

    assert len(constructed) <= 2, f"expected <= 2 parser constructions (one per thread), got {len(constructed)}"
    # YAML 1.1 policy intact: bare y/n load as booleans.
    assert results["t1"][0] == {"a": 0, "b": False} and results["t1"][1] == {"a": 1, "b": True}
    assert all(r == {"a": d["a"], "b": d["b"] in ("y",)} for r, d in zip(results["t1"], docs[:20]))
